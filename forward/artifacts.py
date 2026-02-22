from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from forward.schemas import SIGNALS_COLUMNS


def _to_iso(ts: Any) -> str:
    if ts is None:
        return ""
    try:
        if isinstance(ts, pd.Timestamp):
            return ts.tz_convert("UTC").isoformat()
    except Exception:
        pass
    return str(ts)


def write_jsonl(path: Path, events: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False, default=str) + "\n")


def append_jsonl(path: Path, events: List[Dict[str, Any]]) -> None:
    """Append JSON lines to an events file (creates it if missing)."""
    if not events:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False, default=str) + "\n")


def write_csv(df: pd.DataFrame, path: Path, columns: List[str]) -> None:
    out = df.copy()
    for c in columns:
        if c not in out.columns:
            out[c] = ""
    out = out[columns]
    out.to_csv(path, index=False)


def append_csv_rows(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    """Append rows to a CSV file, writing header if the file is new/empty."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        for r in rows:
            out = {c: r.get(c, "") for c in columns}
            writer.writerow(out)


def build_signals_df(df_sig: pd.DataFrame, symbol: str) -> pd.DataFrame:
    s = df_sig[df_sig["signal"].astype(int) != 0].copy()
    if s.empty:
        return pd.DataFrame(columns=SIGNALS_COLUMNS)

    def side_from_signal(x: int) -> str:
        return "LONG" if int(x) > 0 else "SHORT"

    out = pd.DataFrame(
        {
            "timestamp_utc": [t.tz_convert("UTC").isoformat() for t in s.index],
            "symbol": symbol,
            "side": [side_from_signal(x) for x in s["signal"].astype(int).tolist()],
            "reason": s["signal_type"].astype(str).tolist(),
            "adx": s.get("adx", pd.Series([None] * len(s))).tolist(),
            "orb_low": s.get("orb_low", pd.Series([None] * len(s))).tolist(),
            "orb_high": s.get("orb_high", pd.Series([None] * len(s))).tolist(),
            "close": s.get("close", pd.Series([None] * len(s))).tolist(),
        }
    )
    return out


def _trade_id(t: Dict[str, Any], i: int) -> str:
    et = t.get("entry_time")
    st = str(t.get("signal_type", ""))
    if isinstance(et, pd.Timestamp):
        key = et.tz_convert("UTC").strftime("%Y%m%dT%H%M%SZ")
    else:
        key = str(et)
    return f"T{i:05d}_{key}_{st}".replace(" ", "")


def build_orders_fills_positions(
    df_sig: pd.DataFrame,
    trades: List[Dict[str, Any]],
    equity_curve: pd.Series,
    symbol: str,
    delay_bars: int,
    valid_days: Optional[set] = None,
    risk_events: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    """Create forward-test artifacts from a deterministic shadow replay run."""
    events: List[Dict[str, Any]] = []

    sig_rows = df_sig[df_sig["signal"].astype(int) != 0].copy()

    # Map trades by entry timestamp (UTC) + signal_type.
    # NOTE: Mapping by entry *date* can mis-label signals that execute after midnight
    # (e.g., 23:30 signal with delay_bars=1 enters at 00:00 next day).
    trade_by_entry: Dict[Tuple[pd.Timestamp, str], Dict[str, Any]] = {}
    trade_by_entry_ts_only: Dict[pd.Timestamp, Dict[str, Any]] = {}
    for i, t in enumerate(trades):
        et = t.get("entry_time")
        st = str(t.get("signal_type", ""))
        if isinstance(et, pd.Timestamp):
            et_utc = et.tz_convert("UTC")
            payload = {**t, "_trade_index": i, "_trade_id": _trade_id(t, i)}
            trade_by_entry[(et_utc, st)] = payload
            # Fallback (assumes <= 1 trade per entry timestamp)
            trade_by_entry_ts_only[et_utc] = payload

    orders_out: List[Dict[str, Any]] = []
    fills_out: List[Dict[str, Any]] = []
    positions_out: List[Dict[str, Any]] = []

    
    # Pre-compute open-position intervals to explain skipped signals.
    open_intervals: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for _t in trades:
        _et = _t.get("entry_time")
        _xt = _t.get("exit_time")
        if isinstance(_et, pd.Timestamp) and isinstance(_xt, pd.Timestamp):
            try:
                open_intervals.append((_et.tz_convert("UTC"), _xt.tz_convert("UTC")))
            except Exception:
                pass

    def _position_open(at_ts_utc: pd.Timestamp) -> bool:
        for a, b in open_intervals:
            if a <= at_ts_utc < b:
                return True
        return False

    # Risk-derived hints (optional). Useful when risk controls prevent scheduling/execution.
    global_halt_ts: Optional[pd.Timestamp] = None
    day_halt_ts: Dict[str, pd.Timestamp] = {}
    reject_reason_by_ts: Dict[pd.Timestamp, str] = {}

    if isinstance(risk_events, list):
        for e in risk_events:
            kind = str(e.get("kind", ""))
            ts_s = e.get("ts")
            try:
                ts_e = pd.Timestamp(ts_s)
                if ts_e.tzinfo is None:
                    ts_e = ts_e.tz_localize("UTC")
                else:
                    ts_e = ts_e.tz_convert("UTC")
            except Exception:
                continue

            if kind == "HALT_GLOBAL":
                if global_halt_ts is None or ts_e < global_halt_ts:
                    global_halt_ts = ts_e
            elif kind == "HALT_DAY":
                d = str(e.get("day", ""))
                if d:
                    prev = day_halt_ts.get(d)
                    if prev is None or ts_e < prev:
                        day_halt_ts[d] = ts_e
            elif kind == "ORDER_REJECT":
                reject_reason_by_ts[ts_e] = str(e.get("reason", ""))

# Orders derived from signals (scheduled at signal time; filled only if a trade exists)
    for ts, row in sig_rows.iterrows():
        #sig_date = row["date"]
        sig_type = str(row.get("signal_type", ""))
        signal = int(row.get("signal", 0))
        side = "LONG" if signal > 0 else "SHORT"

        due_ts: Optional[pd.Timestamp] = None
        try:
            i = df_sig.index.get_loc(ts)
            due_i = i + int(delay_bars)
            if 0 <= due_i < len(df_sig.index):
                due_ts = df_sig.index[due_i]
        except Exception:
            due_ts = None

        trade = None
        if due_ts is not None:
            try:
                due_utc = pd.Timestamp(due_ts).tz_convert('UTC')
            except Exception:
                due_utc = None
            if due_utc is not None:
                trade = trade_by_entry.get((due_utc, sig_type)) or trade_by_entry_ts_only.get(due_utc)
        status = "scheduled"
        order_id = f"SIG_{pd.Timestamp(ts).tz_convert('UTC').strftime('%Y%m%dT%H%M%SZ')}_{sig_type}".replace(
            " ", ""
        )
        qty = ""
        if trade is None:
            status = "blocked_or_skipped"
        else:
            status = "filled"
            qty = float(trade.get("qty", 0.0))
            order_id = str(trade.get("_trade_id")) + "_ENTRY"

        status_detail = ""
        if trade is None:
            ts_utc = pd.Timestamp(ts).tz_convert("UTC")
            if due_ts is None:
                status_detail = "missing_due_bar"
            else:
                # Determine the signal day / due day for validity checks
                sig_day = row.get("date")
                try:
                    due_day = pd.Timestamp(due_ts).tz_convert("UTC").date()
                except Exception:
                    due_day = None

                # Most common: signal ignored because a prior position was open
                if _position_open(ts_utc):
                    status_detail = "position_open"

                # Next: invalid execution day (entry bar falls on invalid/missing day)
                if not status_detail and valid_days is not None and due_day is not None:
                    try:
                        if sig_day not in valid_days:
                            status_detail = "invalid_signal_day"
                        elif due_day not in valid_days:
                            status_detail = "invalid_execution_day"
                    except Exception:
                        pass

                # Risk halts (if risk events provided)
                if not status_detail:
                    sig_day_str = str(sig_day) if sig_day is not None else str(ts_utc.date())
                    due_day_str = str(due_day) if due_day is not None else ""
                    if global_halt_ts is not None and global_halt_ts <= ts_utc:
                        status_detail = "risk_halt_global"
                    elif sig_day_str in day_halt_ts and day_halt_ts[sig_day_str] <= ts_utc:
                        status_detail = "risk_halt_day"
                    else:
                        try:
                            due_utc = pd.Timestamp(due_ts).tz_convert("UTC")
                            if due_day_str in day_halt_ts and day_halt_ts[due_day_str] <= due_utc:
                                status_detail = "risk_halt_day_due"
                        except Exception:
                            pass

                # Engine-level reject reason (best-effort, only available when risk events exist)
                if not status_detail:
                    try:
                        due_utc = pd.Timestamp(due_ts).tz_convert("UTC")
                        rr = reject_reason_by_ts.get(due_utc)
                        if rr:
                            status_detail = f"engine_reject:{rr}"
                    except Exception:
                        pass

                if not status_detail:
                    status_detail = "unknown"

        orders_out.append(
            {
                "timestamp_utc": ts.tz_convert("UTC").isoformat(),
                "due_timestamp_utc": _to_iso(due_ts),
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "order_type": "market",
                "limit_price": "",
                "status": status,
                "status_detail": status_detail,
                "reason": sig_type,
            }
        )

        events.append(
            {
                "ts": ts.tz_convert("UTC").isoformat(),
                "type": "SIGNAL",
                "symbol": symbol,
                "signal": signal,
                "signal_type": sig_type,
                "order_status": status,
                "due_ts": _to_iso(due_ts),
            }
        )

    # Fills/Positions derived from realized trades
    for i, t in enumerate(trades):
        tid = _trade_id(t, i)
        et = t.get("entry_time")
        xt = t.get("exit_time")
        typ = str(t.get("type", ""))  # LONG/SHORT
        qty = float(t.get("qty", 0.0))
        entry_px = float(t.get("entry_price", 0.0))
        exit_px = float(t.get("exit_price", 0.0))
        entry_fee = float(t.get("entry_fee", 0.0))
        exit_fee = float(t.get("exit_fee", 0.0))
        lev = float(t.get("leverage", 0.0))
        margin_used = float(t.get("initial_margin_used", 0.0))

        # Entry fill
        entry_side = "buy" if typ == "LONG" else "sell"
        fills_out.append(
            {
                "timestamp_utc": _to_iso(et),
                "order_id": tid + "_ENTRY",
                "symbol": symbol,
                "side": entry_side,
                "qty": qty,
                "fill_price": entry_px,
                "fee": entry_fee,
                "slippage_bps": "",
                "exec_model": "shadow_next_open",
            }
        )
        events.append(
            {
                "ts": _to_iso(et),
                "type": "FILL",
                "symbol": symbol,
                "trade_id": tid,
                "leg": "entry",
                "side": entry_side,
                "qty": qty,
                "price": entry_px,
            }
        )

        # Entry snapshot (equity at bar close if available)
        eq_at_entry = ""
        try:
            if isinstance(et, pd.Timestamp) and et in equity_curve.index:
                eq_at_entry = float(equity_curve.loc[et])
        except Exception:
            eq_at_entry = ""

        positions_out.append(
            {
                "timestamp_utc": _to_iso(et),
                "symbol": symbol,
                "side": typ,
                "qty": qty,
                "entry_price": entry_px,
                "mark_price": entry_px,
                "unrealized_pnl": 0.0,
                "equity": eq_at_entry,
                "margin_used": margin_used,
                "leverage": lev,
            }
        )

        # Exit fill
        exit_side = "sell" if typ == "LONG" else "buy"
        fills_out.append(
            {
                "timestamp_utc": _to_iso(xt),
                "order_id": tid + "_EXIT",
                "symbol": symbol,
                "side": exit_side,
                "qty": qty,
                "fill_price": exit_px,
                "fee": exit_fee,
                "slippage_bps": "",
                "exec_model": "shadow_backtest_rules",
            }
        )
        events.append(
            {
                "ts": _to_iso(xt),
                "type": "FILL",
                "symbol": symbol,
                "trade_id": tid,
                "leg": "exit",
                "side": exit_side,
                "qty": qty,
                "price": exit_px,
                "reason": t.get("exit_reason", ""),
                "pnl_net": t.get("pnl_net", ""),
            }
        )

        eq_at_exit = ""
        try:
            if isinstance(xt, pd.Timestamp) and xt in equity_curve.index:
                eq_at_exit = float(equity_curve.loc[xt])
        except Exception:
            eq_at_exit = ""

        positions_out.append(
            {
                "timestamp_utc": _to_iso(xt),
                "symbol": symbol,
                "side": "FLAT",
                "qty": 0.0,
                "entry_price": "",
                "mark_price": exit_px,
                "unrealized_pnl": 0.0,
                "equity": eq_at_exit,
                "margin_used": 0.0,
                "leverage": "",
            }
        )

    orders_df = pd.DataFrame(orders_out)
    fills_df = pd.DataFrame(fills_out)
    positions_df = pd.DataFrame(positions_out)
    return orders_df, fills_df, positions_df, events

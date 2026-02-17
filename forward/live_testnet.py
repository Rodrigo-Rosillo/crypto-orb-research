from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from backtester.risk import RiskLimits
from forward.artifacts import append_csv_rows, append_jsonl, build_signals_df
from forward.binance_live import (
    BinanceLiveKlineSource,
    fetch_recent_klines_df,
    fetch_server_time_ms,
    interval_to_seconds,
)
from forward.shadow import build_signals
from forward.schemas import FILLS_COLUMNS, ORDERS_COLUMNS, POSITIONS_COLUMNS, SIGNALS_COLUMNS, validate_df_columns
from forward.state_store import OpenPositionState, RunnerState, load_state, save_state
from forward.testnet_broker import (
    BinanceFuturesTestnetBroker,
    RateLimitConfig,
    TestnetConfig,
    TestnetAPIError,
    TestnetAuthError,
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_rows(path, rows, columns, name: str) -> None:
    if rows:
        validate_df_columns(pd.DataFrame(rows), columns, name)
    else:
        validate_df_columns(pd.DataFrame(columns=columns), columns, name)
    append_csv_rows(path, rows, columns)


def _float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _pos_side_from_amt(amt: float) -> str:
    if amt > 0:
        return "LONG"
    if amt < 0:
        return "SHORT"
    return "FLAT"


def _extract_order_id(resp: Any) -> Optional[int]:
    # Regular order responses use orderId; Algo (conditional) responses use algoId.
    if isinstance(resp, dict):
        oid = resp.get("orderId") or resp.get("orderID") or resp.get("algoId") or resp.get("algoID")
        if oid is not None:
            try:
                return int(oid)
            except Exception:
                return None
    return None


def _order_status(resp: Any) -> str:
    if isinstance(resp, dict):
        # Regular orders: status; Algo orders: algoStatus
        return str(resp.get("status") or resp.get("algoStatus") or "")
    return ""


def _order_avg_price(resp: Any) -> float:
    if isinstance(resp, dict):
        # RESULT response often provides avgPrice; fallback to price.
        if resp.get("avgPrice") not in (None, ""):
            return _float(resp.get("avgPrice"), 0.0)
        if resp.get("price") not in (None, ""):
            return _float(resp.get("price"), 0.0)
    return 0.0


def _order_exec_qty(resp: Any) -> float:
    if isinstance(resp, dict):
        if resp.get("executedQty") not in (None, ""):
            return _float(resp.get("executedQty"), 0.0)
        if resp.get("origQty") not in (None, ""):
            return _float(resp.get("origQty"), 0.0)
    return 0.0


async def run_live_testnet(
    *,
    run_dir,  # Path
    cfg: Dict[str, Any],
    ft_cfg: Dict[str, Any],
    risk_limits: Optional[RiskLimits],
    symbol: str,
    timeframe: str,
    orb_start,
    orb_end,
    orb_cutoff,
    adx_period: int,
    adx_threshold: float,
    initial_capital: float,
    position_size: float,
    taker_fee_rate: float,
    leverage: float,
    delay_bars: int,
    slippage_bps: float,
    max_bars: Optional[int] = None,
    duration_minutes: Optional[int] = None,
    smoke_test: bool = False,
    external_stop_event: Optional[asyncio.Event] = None,
) -> int:
    """Phase 5 Step 4: live market data + TESTNET order placement.

    This runner uses live market data ingestion (Step 3) but sends orders to Binance Futures TESTNET.

    Supports two workflows:
      - smoke_test=True: place a tiny entry+flatten cycle and exit.
      - normal run: wait for strategy signals and trade them on testnet.

    State is persisted to run_dir/state.json and is used on restart to reconcile.
    """

    live_cfg = ft_cfg.get("live") if isinstance(ft_cfg.get("live"), dict) else {}
    market = str((live_cfg or {}).get("market", "futures"))
    bootstrap_limit = int((live_cfg or {}).get("bootstrap_limit", 1000))
    max_backoff_seconds = int((live_cfg or {}).get("max_backoff_seconds", 60))
    heartbeat_seconds = int((live_cfg or {}).get("heartbeat_seconds", 120))
    stale_check_interval_seconds = int((live_cfg or {}).get("stale_check_interval_seconds", 30))

    testnet_cfg = ft_cfg.get("testnet") if isinstance(ft_cfg.get("testnet"), dict) else {}
    base_url = str((testnet_cfg or {}).get("base_url", "https://demo-fapi.binance.com"))
    recv_window_ms = int((testnet_cfg or {}).get("recv_window_ms", 5000))
    poll_interval_seconds = float((testnet_cfg or {}).get("poll_interval_seconds", 2.0))
    cancel_open_orders_on_exit = bool((testnet_cfg or {}).get("cancel_open_orders_on_exit", True))
    flatten_on_mismatch = bool((testnet_cfg or {}).get("flatten_on_mismatch", False))
    # Smoke test settings
    smoke_qty = float((testnet_cfg or {}).get("smoke_qty", 0.1))
    smoke_auto_flatten = bool((testnet_cfg or {}).get("smoke_auto_flatten", True))

    # Rate limit tuning
    rl = RateLimitConfig(
        max_retries=int((testnet_cfg or {}).get("max_retries", 6)),
        base_backoff_seconds=float((testnet_cfg or {}).get("base_backoff_seconds", 0.5)),
        max_backoff_seconds=float((testnet_cfg or {}).get("max_backoff_seconds", 20.0)),
    )

    bar_seconds = interval_to_seconds(timeframe)

    # Output files
    signals_path = run_dir / "signals.csv"
    orders_path = run_dir / "orders.csv"
    fills_path = run_dir / "fills.csv"
    positions_path = run_dir / "positions.csv"
    events_path = run_dir / "events.jsonl"
    state_path = run_dir / "state.json"

    # Allow the CLI runner to inject an external stop_event (e.g., for graceful
    # Ctrl+C handling). If not provided, create an internal event.
    stop_event = external_stop_event or asyncio.Event()

    # Bootstrap history
    rest_df, rest_meta = fetch_recent_klines_df(
        symbol=symbol,
        interval=timeframe,
        limit=bootstrap_limit,
        market=market,
    )
    df_raw = rest_df.copy()

    # --- Bootstrap validation (freshness + local clock sanity) ---
    default_age_bars = int(risk_limits.kill_switch.max_data_gap_bars) if risk_limits is not None else 2
    bam = (live_cfg or {}).get("bootstrap_max_age_bars", default_age_bars)
    if bam is None:
        bam = default_age_bars
    bootstrap_max_age_bars = max(1, int(bam))
    allowed_bootstrap_age_s = float(bar_seconds * bootstrap_max_age_bars)

    last_close_iso = str((rest_meta or {}).get("last_close_time", "") or "")
    last_close_ts: Optional[pd.Timestamp] = None
    if last_close_iso:
        try:
            last_close_ts = pd.to_datetime(last_close_iso, utc=True)
        except Exception:
            last_close_ts = None
    if last_close_ts is None and len(df_raw.index):
        last_close_ts = df_raw.index[-1] + pd.Timedelta(seconds=int(bar_seconds))

    now_utc = datetime.now(timezone.utc)
    bootstrap_age_s = None
    if last_close_ts is not None:
        bootstrap_age_s = float((now_utc - last_close_ts.to_pydatetime()).total_seconds())

    # Local clock skew check vs Binance server time
    clock_skew_ms = None
    clock_meta: Dict[str, Any] = {}
    try:
        server_ms, clock_meta = fetch_server_time_ms(market=market)
        local_ms = int(now_utc.timestamp() * 1000)
        clock_skew_ms = int(local_ms - int(server_ms))
    except Exception as e:
        append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "CLOCK_SKEW_CHECK_FAILED", "error": str(e)}])

    trading_start_ts: Optional[pd.Timestamp] = df_raw.index[-1] if len(df_raw.index) else None

    append_jsonl(
        events_path,
        [
            {
                "ts": _utcnow_iso(),
                "type": "LIVE_RUN_START",
                "mode": "testnet",
                "source": "live",
                "symbol": symbol,
                "timeframe": timeframe,
                "market": market,
                "bootstrap": rest_meta,
                "trading_start_ts": trading_start_ts.isoformat() if trading_start_ts is not None else "",
            }
        ],
    )

    # Clock skew thresholds
    skew_warn_ms = 5_000
    skew_fatal_ms = 30_000
    if clock_skew_ms is not None:
        if abs(clock_skew_ms) > skew_fatal_ms:
            append_jsonl(
                events_path,
                [
                    {
                        "ts": _utcnow_iso(),
                        "type": "CLOCK_SKEW_FATAL",
                        "clock_skew_ms": int(clock_skew_ms),
                        "warn_ms": int(skew_warn_ms),
                        "fatal_ms": int(skew_fatal_ms),
                        "server_time": clock_meta,
                    }
                ],
            )
            append_jsonl(
                events_path,
                [{"ts": _utcnow_iso(), "type": "LIVE_RUN_END", "bars_processed": 0, "final_equity": float(initial_capital), "reason": "CLOCK_SKEW_FATAL"}],
            )
            return 0
        if abs(clock_skew_ms) > skew_warn_ms:
            append_jsonl(
                events_path,
                [
                    {
                        "ts": _utcnow_iso(),
                        "type": "CLOCK_SKEW_WARN",
                        "clock_skew_ms": int(clock_skew_ms),
                        "warn_ms": int(skew_warn_ms),
                        "fatal_ms": int(skew_fatal_ms),
                        "server_time": clock_meta,
                    }
                ],
            )

    if last_close_ts is None:
        append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "BOOTSTRAP_EMPTY_OR_INVALID", "bootstrap_rows": int(len(df_raw.index))}])
        append_jsonl(
            events_path,
            [{"ts": _utcnow_iso(), "type": "LIVE_RUN_END", "bars_processed": 0, "final_equity": float(initial_capital), "reason": "BOOTSTRAP_EMPTY_OR_INVALID"}],
        )
        return 0

    append_jsonl(
        events_path,
        [
            {
                "ts": _utcnow_iso(),
                "type": "BOOTSTRAP_VALIDATION",
                "last_close_time": last_close_ts.isoformat(),
                "age_seconds": float(bootstrap_age_s) if bootstrap_age_s is not None else None,
                "allowed_seconds": float(allowed_bootstrap_age_s),
                "bootstrap_max_age_bars": int(bootstrap_max_age_bars),
                "bar_seconds": int(bar_seconds),
            }
        ],
    )

    if bootstrap_age_s is not None and bootstrap_age_s > allowed_bootstrap_age_s:
        append_jsonl(
            events_path,
            [{"ts": _utcnow_iso(), "type": "BOOTSTRAP_STALE", "age_seconds": float(bootstrap_age_s), "allowed_seconds": float(allowed_bootstrap_age_s), "bootstrap_max_age_bars": int(bootstrap_max_age_bars)}],
        )
        append_jsonl(
            events_path,
            [{"ts": _utcnow_iso(), "type": "LIVE_RUN_END", "bars_processed": 0, "final_equity": float(initial_capital), "reason": "BOOTSTRAP_STALE"}],
        )
        return 0
    else:
        append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "BOOTSTRAP_OK", "age_seconds": float(bootstrap_age_s) if bootstrap_age_s is not None else None}])

    # Create broker
    try:
        broker = BinanceFuturesTestnetBroker(cfg=TestnetConfig(base_url=base_url, recv_window_ms=recv_window_ms, rate_limit=rl))
        append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "TESTNET_BROKER_CONFIG", "base_url": base_url, "recv_window_ms": int(recv_window_ms)}])
    except TestnetAuthError as e:
        append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "TESTNET_AUTH_ERROR", "error": str(e)}])
        append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "LIVE_RUN_END", "bars_processed": 0, "final_equity": float(initial_capital), "reason": "TESTNET_AUTH_ERROR"}])
        return 0

    # Best-effort set leverage
    try:
        broker.set_leverage(symbol, int(leverage))
        append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "TESTNET_SET_LEVERAGE", "leverage": int(leverage)}])
    except Exception as e:
        append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "TESTNET_SET_LEVERAGE_FAILED", "error": str(e)}])

    # Load and reconcile state
    had_state_file = bool(state_path.exists())
    state = load_state(state_path)
    append_jsonl(
        events_path,
        [
            {
                "ts": _utcnow_iso(),
                "type": "STATE_LOADED",
                "path": "state.json",
                "found": bool(had_state_file),
                "bars_processed": int(state.bars_processed or 0),
                "last_bar_open_time_utc": str(state.last_bar_open_time_utc or ""),
            }
        ],
    )

    # Persist an initial state snapshot immediately so state.json exists even if the
    # run stops before the first closed candle arrives (common for short duration runs).
    if state.last_bar_open_time_utc is None and trading_start_ts is not None:
        try:
            state.last_bar_open_time_utc = trading_start_ts.tz_convert("UTC").isoformat()
        except Exception:
            state.last_bar_open_time_utc = str(trading_start_ts)
    try:
        save_state(state_path, state)
        append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "STATE_SAVED", "path": "state.json"}])
    except Exception as e:
        append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "STATE_SAVE_FAILED", "error": str(e)}])

    # Risk kill-switch thresholds (apply even if limits.enabled is False; these are operational safety checks)
    max_order_rejects_per_day = int(risk_limits.kill_switch.max_order_rejects_per_day) if risk_limits is not None else 3
    max_margin_ratio = float(risk_limits.kill_switch.max_margin_ratio) if risk_limits is not None else 0.85

    def fetch_exchange_position() -> Tuple[str, float, float, float]:
        pr = broker.position_risk(symbol=symbol)
        amt = _float(pr.get("positionAmt"), 0.0)
        entry = _float(pr.get("entryPrice"), 0.0)
        upl = _float(pr.get("unRealizedProfit"), 0.0)
        side = _pos_side_from_amt(amt)
        return side, float(abs(amt)), float(entry), float(upl)


    try:
        ex_side, ex_qty, ex_entry, ex_upl = fetch_exchange_position()
    except TestnetAPIError as e:
        append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "TESTNET_POSITION_RISK_FAILED", "error": str(e), "payload": getattr(e, "payload", None)}])
        append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "LIVE_RUN_END", "bars_processed": 0, "final_equity": float(initial_capital), "reason": "TESTNET_POSITION_RISK_FAILED"}])
        return 0

    if state.open_position is None and ex_side != "FLAT":
        append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "RECON_MISMATCH", "state": "FLAT", "exchange": ex_side, "qty": ex_qty, "entry_price": ex_entry}])
        if flatten_on_mismatch:
            try:
                # Flatten by market in opposite direction
                side = "SELL" if ex_side == "LONG" else "BUY"
                broker.place_market_order(symbol=symbol, side=side, quantity=ex_qty, reduce_only=True)
                append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "RECON_FLATTENED"}])
            except Exception as e:
                append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "RECON_FLATTEN_FAILED", "error": str(e)}])
        append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "LIVE_RUN_END", "bars_processed": 0, "final_equity": float(initial_capital), "reason": "RECON_MISMATCH"}])
        return 0

    if state.open_position is not None and ex_side == "FLAT":
        append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "RECON_MISMATCH", "state": state.open_position.side, "exchange": "FLAT"}])
        append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "LIVE_RUN_END", "bars_processed": 0, "final_equity": float(initial_capital), "reason": "RECON_MISMATCH"}])
        return 0

    if state.open_position is not None and ex_side != "FLAT":
        # Basic qty check
        if abs(state.open_position.qty - ex_qty) > 1e-6 or state.open_position.side != ex_side:
            append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "RECON_MISMATCH", "state": state.open_position.to_dict() if hasattr(state.open_position, 'to_dict') else {}, "exchange": {"side": ex_side, "qty": ex_qty}}])
            append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "LIVE_RUN_END", "bars_processed": 0, "final_equity": float(initial_capital), "reason": "RECON_MISMATCH"}])
            return 0

    # Smoke test: place tiny order and optionally flatten
    if smoke_test:
        append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "TESTNET_SMOKE_START", "qty": float(smoke_qty)}])
        try:
            entry_resp = broker.place_market_order(symbol=symbol, side="SELL", quantity=float(smoke_qty), reduce_only=False)
            entry_oid = _extract_order_id(entry_resp)
            fill_price = _order_avg_price(entry_resp)
            ex_side2, ex_qty2, ex_entry2, _ = fetch_exchange_position()
            append_jsonl(
                events_path,
                [{"ts": _utcnow_iso(), "type": "TESTNET_SMOKE_ENTRY", "order_id": entry_oid, "fill_price": float(fill_price), "exchange_side": ex_side2, "exchange_qty": float(ex_qty2), "exchange_entry": float(ex_entry2)}],
            )
            _append_rows(
                orders_path,
                [{"timestamp_utc": _utcnow_iso(), "due_timestamp_utc": "", "order_id": str(entry_oid or ""), "symbol": symbol, "side": "SHORT", "qty": float(smoke_qty), "order_type": "MARKET", "limit_price": "", "status": "sent", "status_detail": "smoke_entry", "reason": ""}],
                ORDERS_COLUMNS,
                "orders.csv",
            )
            _append_rows(
                fills_path,
                [{"timestamp_utc": _utcnow_iso(), "order_id": str(entry_oid or ""), "symbol": symbol, "side": "SHORT", "qty": float(smoke_qty), "fill_price": float(fill_price), "fee": 0.0, "slippage_bps": float(slippage_bps), "exec_model": "testnet_market"}],
                FILLS_COLUMNS,
                "fills.csv",
            )

            if smoke_auto_flatten and ex_side2 != "FLAT":
                flat_resp = broker.place_market_order(symbol=symbol, side="BUY", quantity=float(ex_qty2), reduce_only=True)
                flat_oid = _extract_order_id(flat_resp)
                flat_price = _order_avg_price(flat_resp)
                append_jsonl(
                    events_path,
                    [{"ts": _utcnow_iso(), "type": "TESTNET_SMOKE_FLATTEN", "order_id": flat_oid, "fill_price": float(flat_price), "qty": float(ex_qty2)}],
                )
                # Mirror the entry record: write flatten order + fill for complete audit trail.
                _append_rows(
                    orders_path,
                    [{"timestamp_utc": _utcnow_iso(), "due_timestamp_utc": "", "order_id": str(flat_oid or ""), "symbol": symbol, "side": "LONG", "qty": float(ex_qty2), "order_type": "MARKET", "limit_price": "", "status": "sent", "status_detail": "smoke_flatten", "reason": ""}],
                    ORDERS_COLUMNS,
                    "orders.csv",
                )
                _append_rows(
                    fills_path,
                    [{"timestamp_utc": _utcnow_iso(), "order_id": str(flat_oid or ""), "symbol": symbol, "side": "LONG", "qty": float(ex_qty2), "fill_price": float(flat_price), "fee": 0.0, "slippage_bps": float(slippage_bps), "exec_model": "testnet_market_reduce_only"}],
                    FILLS_COLUMNS,
                    "fills.csv",
                )
        except Exception as e:
            append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "TESTNET_SMOKE_FAILED", "error": str(e)}])

        # Ensure state.json exists for smoke runs too.
        try:
            state.open_position = None
            save_state(state_path, state)
        except Exception:
            pass

        append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "LIVE_RUN_END", "bars_processed": 0, "final_equity": float(initial_capital), "reason": "SMOKE_TEST"}])
        return 0

    # Live streaming
    src = BinanceLiveKlineSource(
        symbol=symbol,
        interval=timeframe,
        market=market,
        max_backoff_seconds=max_backoff_seconds,
    )

    last_connect_count = src.connect_count

    # Seed stale detection from last_close_ts
    try:
        last_closed_bar_at = last_close_ts.to_pydatetime()
    except Exception:
        last_closed_bar_at = datetime.now(timezone.utc)

    # Resume de-dupe from state
    last_bar_open_ts: Optional[pd.Timestamp] = None
    if state.last_bar_open_time_utc:
        try:
            last_bar_open_ts = pd.to_datetime(state.last_bar_open_time_utc, utc=True)
        except Exception:
            last_bar_open_ts = trading_start_ts
    else:
        last_bar_open_ts = trading_start_ts

    bars_processed = int(state.bars_processed or 0)
    start_wall = datetime.now(timezone.utc)

    async def heartbeat_task() -> None:
        nonlocal last_closed_bar_at
        while not stop_event.is_set():
            await asyncio.sleep(max(1, int(stale_check_interval_seconds)))
            if stop_event.is_set():
                break

            now = datetime.now(timezone.utc)

            last_msg = src.last_message_at or src.last_connect_at
            if last_msg is not None:
                since_msg = (now - last_msg).total_seconds()
                if since_msg >= heartbeat_seconds:
                    append_jsonl(
                        events_path,
                        [{"ts": _utcnow_iso(), "type": "DATA_HEARTBEAT_MISSED", "since_seconds": float(since_msg), "threshold_seconds": int(heartbeat_seconds)}],
                    )

            since_closed = (now - last_closed_bar_at).total_seconds()
            gap_bars = int(risk_limits.kill_switch.max_data_gap_bars) if risk_limits is not None else 2
            allowed = float(bar_seconds * gap_bars)
            if since_closed > allowed:
                append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "KILL_SWITCH_DATA_STALE", "since_seconds": float(since_closed), "allowed_seconds": float(allowed)}])
                stop_event.set()
                continue

    hb = asyncio.create_task(heartbeat_task())

    def record_position_snapshot() -> None:
        ex_side3, ex_qty3, ex_entry3, ex_upl3 = fetch_exchange_position()
        _append_rows(
            positions_path,
            [
                {
                    "timestamp_utc": _utcnow_iso(),
                    "symbol": symbol,
                    "side": ex_side3,
                    "qty": float(ex_qty3),
                    "entry_price": float(ex_entry3),
                    "mark_price": "",
                    "unrealized_pnl": float(ex_upl3),
                    "equity": "",
                    "margin_used": "",
                    "leverage": float(leverage),
                }
            ],
            POSITIONS_COLUMNS,
            "positions.csv",
        )

    async def poll_open_orders() -> None:
        # Best-effort polling of tp/sl orders in state.
        if state.open_position is None:
            return
        op = state.open_position
        changed = False
        for kind, oid in [("tp", op.tp_order_id), ("sl", op.sl_order_id)]:
            if oid is None:
                continue
            try:
                o = broker.get_algo_order(symbol=symbol, algo_id=int(oid))
                st = _order_status(o)
                # Algo orders: NEW/TRIGGERING/TRIGGERED/FINISHED/CANCELED/REJECTED/EXPIRED
                if st in ("FINISHED", "CANCELED", "REJECTED", "EXPIRED"):
                    append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "ORDER_UPDATE", "order_id": int(oid), "kind": kind, "status": st}])
                    # If a leg finishes, reconcile exchange position to confirm closure.
                    if st == "FINISHED":
                        ex_side4, ex_qty4, _, _ = fetch_exchange_position()
                        if ex_side4 == "FLAT" or ex_qty4 < 1e-9:
                            append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "POSITION_CLOSED", "via": kind}])
                            state.open_position = None
                            changed = True
            except Exception as e:
                append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "ORDER_POLL_FAILED", "order_id": int(oid), "error": str(e)}])
        if changed:
            save_state(state_path, state)

    def maybe_kill_on_margin_ratio() -> None:
        """Best-effort margin ratio check: totalMaintMargin / totalMarginBalance."""
        nonlocal state
        try:
            acct = broker.account()
            if not isinstance(acct, dict):
                return
            maint = _float(acct.get("totalMaintMargin"), 0.0)
            mb = _float(acct.get("totalMarginBalance"), 0.0)
            if mb <= 0:
                return
            ratio = float(maint / mb)
            if ratio >= float(max_margin_ratio):
                append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "KILL_SWITCH_MARGIN_RATIO", "ratio": float(ratio), "threshold": float(max_margin_ratio)}])
                stop_event.set()
        except Exception as e:
            append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "MARGIN_RATIO_CHECK_FAILED", "error": str(e)}])

    async def maybe_place_trade_from_signal(bar_open_time: pd.Timestamp, row: pd.Series) -> None:
        nonlocal state
        if state.open_position is not None:
            return

        signal = int(row.get("signal", 0) or 0)
        if signal == 0:
            return

        signal_type = str(row.get("signal_type", "") or "")

        # Strategy is short-only in your registered setup.
        side = "SELL" if signal < 0 else "BUY"
        pos_side = "SHORT" if side == "SELL" else "LONG"

        # Phase 4 hard controls (minimal enforcement here)
        if risk_limits is not None and bool(risk_limits.enabled):
            if float(leverage) > float(risk_limits.max_leverage):
                append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "RISK_BLOCK", "reason": "max_leverage", "leverage": float(leverage), "max_leverage": float(risk_limits.max_leverage)}])
                _append_rows(
                    orders_path,
                    [{"timestamp_utc": _utcnow_iso(), "due_timestamp_utc": "", "order_id": "", "symbol": symbol, "side": pos_side, "qty": 0.0, "order_type": "MARKET", "limit_price": "", "status": "blocked", "status_detail": "risk_max_leverage", "reason": signal_type}],
                    ORDERS_COLUMNS,
                    "orders.csv",
                )
                return
            if float(position_size) > float(risk_limits.max_position_margin_frac):
                append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "RISK_BLOCK", "reason": "max_position_margin_frac", "position_size": float(position_size), "max_position_margin_frac": float(risk_limits.max_position_margin_frac)}])
                _append_rows(
                    orders_path,
                    [{"timestamp_utc": _utcnow_iso(), "due_timestamp_utc": "", "order_id": "", "symbol": symbol, "side": pos_side, "qty": 0.0, "order_type": "MARKET", "limit_price": "", "status": "blocked", "status_detail": "risk_max_position_margin", "reason": signal_type}],
                    ORDERS_COLUMNS,
                    "orders.csv",
                )
                return

        # Estimate quantity from equity proxy (initial_capital) and position_size (margin fraction).
        # This is a testnet convenience; in production you'd use account balance.
        price = float(row.get("close") or 0.0)
        if price <= 0:
            return
        margin = float(initial_capital) * float(position_size)
        notional = margin * float(leverage)
        qty = max(0.0, notional / price)

        # Minimal sanity
        if qty <= 0:
            return

        # Entry now (market). This approximates next-bar-open, since the boundary is immediate at bar close.
        append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "ENTRY_SUBMIT", "signal_type": signal_type, "qty": float(qty), "side": pos_side}])

        try:
            entry_resp = broker.place_market_order(symbol=symbol, side=side, quantity=float(qty), reduce_only=False)
        except (TestnetAPIError, Exception) as e:
            append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "ENTRY_REJECTED", "error": str(e)}])
            # Count rejects (kill switch)
            state.order_rejects_today = int(state.order_rejects_today or 0) + 1
            if int(state.order_rejects_today) > int(max_order_rejects_per_day):
                append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "KILL_SWITCH_ORDER_REJECTS", "rejects_today": int(state.order_rejects_today), "threshold": int(max_order_rejects_per_day)}])
                stop_event.set()
            _append_rows(
                orders_path,
                [{"timestamp_utc": _utcnow_iso(), "due_timestamp_utc": "", "order_id": "", "symbol": symbol, "side": pos_side, "qty": float(qty), "order_type": "MARKET", "limit_price": "", "status": "rejected", "status_detail": "entry_rejected", "reason": str(e)}],
                ORDERS_COLUMNS,
                "orders.csv",
            )
            save_state(state_path, state)
            return

        entry_oid = _extract_order_id(entry_resp)
        entry_price = _order_avg_price(entry_resp)
        exec_qty = _order_exec_qty(entry_resp)

        _append_rows(
            orders_path,
            [{"timestamp_utc": _utcnow_iso(), "due_timestamp_utc": "", "order_id": str(entry_oid or ""), "symbol": symbol, "side": pos_side, "qty": float(exec_qty or qty), "order_type": "MARKET", "limit_price": "", "status": "sent", "status_detail": "entry_sent", "reason": signal_type}],
            ORDERS_COLUMNS,
            "orders.csv",
        )
        _append_rows(
            fills_path,
            [{"timestamp_utc": _utcnow_iso(), "order_id": str(entry_oid or ""), "symbol": symbol, "side": pos_side, "qty": float(exec_qty or qty), "fill_price": float(entry_price), "fee": 0.0, "slippage_bps": float(slippage_bps), "exec_model": "testnet_market"}],
            FILLS_COLUMNS,
            "fills.csv",
        )

        # Compute bracket prices (mirror backtest):
        #  - TP: +2% for SHORT => entry * 0.98
        #  - SL: ORB high
        orb_high = row.get("orb_high")
        if orb_high is None or (isinstance(orb_high, float) and pd.isna(orb_high)):
            # If we can't compute SL, do not proceed.
            append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "BRACKET_SKIPPED", "reason": "missing_orb_high"}])
            state.open_position = OpenPositionState(
                symbol=symbol,
                side=pos_side,
                qty=float(exec_qty or qty),
                entry_price=float(entry_price),
                entry_time_utc=bar_open_time.tz_convert("UTC").isoformat(),
                entry_order_id=int(entry_oid or 0),
            )
            save_state(state_path, state)
            return

        sl_price = float(orb_high)
        tp_price = float(entry_price) * (0.98 if pos_side == "SHORT" else 1.02)

        # Place TP and SL reduce-only orders
        exit_side = "BUY" if pos_side == "SHORT" else "SELL"

        tp_oid = None
        sl_oid = None
        try:
            tp_resp = broker.place_take_profit_market(
                symbol=symbol, side=exit_side, quantity=float(exec_qty or qty), stop_price=float(tp_price), reduce_only=True
            )
            tp_oid = _extract_order_id(tp_resp)
        except Exception as e:
            append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "TP_PLACE_FAILED", "error": str(e)}])

        try:
            sl_resp = broker.place_stop_market(
                symbol=symbol, side=exit_side, quantity=float(exec_qty or qty), stop_price=float(sl_price), reduce_only=True
            )
            sl_oid = _extract_order_id(sl_resp)
        except Exception as e:
            append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "SL_PLACE_FAILED", "error": str(e)}])

        _append_rows(
            orders_path,
            [
                {"timestamp_utc": _utcnow_iso(), "due_timestamp_utc": "", "order_id": str(tp_oid or ""), "symbol": symbol, "side": "EXIT", "qty": float(exec_qty or qty), "order_type": "TAKE_PROFIT_MARKET", "limit_price": float(tp_price), "status": "sent", "status_detail": "tp_sent", "reason": signal_type},
                {"timestamp_utc": _utcnow_iso(), "due_timestamp_utc": "", "order_id": str(sl_oid or ""), "symbol": symbol, "side": "EXIT", "qty": float(exec_qty or qty), "order_type": "STOP_MARKET", "limit_price": float(sl_price), "status": "sent", "status_detail": "sl_sent", "reason": signal_type},
            ],
            ORDERS_COLUMNS,
            "orders.csv",
        )

        state.open_position = OpenPositionState(
            symbol=symbol,
            side=pos_side,
            qty=float(exec_qty or qty),
            entry_price=float(entry_price),
            entry_time_utc=bar_open_time.tz_convert("UTC").isoformat(),
            entry_order_id=int(entry_oid or 0),
            tp_order_id=int(tp_oid) if tp_oid is not None else None,
            sl_order_id=int(sl_oid) if sl_oid is not None else None,
            tp_price=float(tp_price),
            sl_price=float(sl_price),
        )
        save_state(state_path, state)

    stop_requested = False

    try:
        async for bar in src.stream_closed(stop_event=stop_event):
            if stop_event.is_set():
                continue

            if src.connect_count != last_connect_count:
                append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "WS_CONNECTED", "connect_count": int(src.connect_count)}])
                last_connect_count = src.connect_count

            last_closed_bar_at = datetime.now(timezone.utc)

            # Dedup
            if last_bar_open_ts is not None and bar.open_time <= last_bar_open_ts:
                append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "BAR_DUPLICATE_IGNORED", "open_time": bar.open_time.isoformat()}])
                continue

            last_bar_open_ts = bar.open_time
            state.last_bar_open_time_utc = bar.open_time.tz_convert("UTC").isoformat()

            # Day reset
            day_str = bar.open_time.tz_convert("UTC").date().isoformat()
            if state.current_day_utc != day_str:
                state.current_day_utc = day_str
                state.order_rejects_today = 0
                append_jsonl(events_path, [{"ts": _utcnow_iso(), "kind": "DAY_START", "message": "New UTC day", "day": day_str}])

            append_jsonl(
                events_path,
                [
                    {
                        "ts": _utcnow_iso(),
                        "type": "BAR_CLOSED",
                        "open_time": bar.open_time.isoformat(),
                        "close_time": bar.close_time.isoformat(),
                        # Full OHLCV so Step 5 divergence report can compare to reference parquet.
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": float(bar.volume),
                    }
                ],
            )

            # Update df_raw
            df_raw.loc[bar.open_time] = bar.to_row()
            df_raw = df_raw.sort_index()

            # Signals
            valid_days = set(df_raw.index.date)
            df_sig, _ = build_signals(
                df_raw=df_raw,
                valid_days=valid_days,
                orb_start=orb_start,
                orb_end=orb_end,
                orb_cutoff=orb_cutoff,
                adx_period=adx_period,
                adx_threshold=adx_threshold,
            )
            if bar.open_time not in df_sig.index:
                continue
            row = df_sig.loc[bar.open_time]

            # Log signal row if fires
            if int(row.get("signal", 0) or 0) != 0:
                sig_df = build_signals_df(df_sig.loc[[bar.open_time]], symbol=symbol)
                sig_rows = sig_df.to_dict(orient="records")
                _append_rows(signals_path, sig_rows, SIGNALS_COLUMNS, "signals.csv")

            # Start after bootstrap candle
            if trading_start_ts is not None and bar.open_time <= trading_start_ts:
                continue

            # Poll orders/position
            await poll_open_orders()
            record_position_snapshot()
            maybe_kill_on_margin_ratio()

            # Place trades if signal
            await maybe_place_trade_from_signal(bar.open_time, row)

            bars_processed += 1
            state.bars_processed = int(bars_processed)
            save_state(state_path, state)

            # Stop conditions
            if stop_requested:
                continue
            if max_bars is not None and bars_processed >= int(max_bars):
                append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "STOP_MAX_BARS", "max_bars": int(max_bars)}])
                stop_event.set()
                stop_requested = True
                continue

            if duration_minutes is not None:
                elapsed = (datetime.now(timezone.utc) - start_wall).total_seconds() / 60.0
                if elapsed >= float(duration_minutes):
                    append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "STOP_DURATION", "duration_minutes": int(duration_minutes)}])
                    stop_event.set()
                    stop_requested = True
                    continue

            await asyncio.sleep(max(0.0, float(poll_interval_seconds)))

    finally:
        # Cancel heartbeat
        hb.cancel()
        try:
            await hb
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

        # Optional: cancel open orders on exit
        if cancel_open_orders_on_exit:
            try:
                broker.cancel_all_open_orders(symbol=symbol)
                append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "CANCEL_ALL_OPEN_ORDERS"}])
            except Exception as e:
                append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "CANCEL_ALL_OPEN_ORDERS_FAILED", "error": str(e)}])

        append_jsonl(
            events_path,
            [{"ts": _utcnow_iso(), "type": "LIVE_RUN_END", "bars_processed": int(bars_processed), "final_equity": "", "reason": "STOP" if stop_event.is_set() else "END"}],
        )

    return 0

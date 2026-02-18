from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from forward.schemas import FILLS_COLUMNS, validate_df_columns


def to_ts(x: Any) -> pd.Timestamp:
    return pd.to_datetime(x, utc=True, errors="coerce")


def _get_event_type(ev: Dict[str, Any]) -> str:
    return str(ev.get("type") or ev.get("kind") or "").strip()


def build_bar_df(events: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for ev in events:
        if _get_event_type(ev) != "BAR_CLOSED":
            continue
        open_time = to_ts(ev.get("open_time"))
        close_time = to_ts(ev.get("close_time"))
        ingest_ts = to_ts(ev.get("ts"))
        if pd.isna(open_time) or pd.isna(close_time) or pd.isna(ingest_ts):
            continue
        rows.append(
            {
                "open_time": open_time,
                "close_time": close_time,
                "ingest_ts": ingest_ts,
                "open": ev.get("open"),
                "high": ev.get("high"),
                "low": ev.get("low"),
                "close": ev.get("close"),
                "volume": ev.get("volume"),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["open_time", "close_time", "ingest_ts", "open", "high", "low", "close", "volume"]
        ).set_index("open_time")
    df = pd.DataFrame(rows).set_index("open_time").sort_index()
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def timing_divergence(bar_df: pd.DataFrame, interval_seconds: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "bars": int(len(bar_df)),
        "ingest_delay_seconds": {},
        "missed_bars": {},
    }
    if bar_df.empty:
        return out

    delay = (bar_df["ingest_ts"] - bar_df["close_time"]).dt.total_seconds().dropna()
    if len(delay):
        out["ingest_delay_seconds"] = {
            "mean": float(delay.mean()),
            "p50": float(delay.quantile(0.50)),
            "p95": float(delay.quantile(0.95)),
            "max": float(delay.max()),
            "gt_5s": int((delay > 5).sum()),
            "gt_30s": int((delay > 30).sum()),
        }

    idx = bar_df.index
    if len(idx) >= 2:
        diffs = (idx[1:] - idx[:-1]).total_seconds()
        expected = float(interval_seconds) if interval_seconds > 0 else 1800.0
        gaps = diffs / expected
        missed_est = [int(round(g - 1)) for g in gaps if g > 1.01]
        out["missed_bars"] = {
            "gaps_detected": int(sum(g > 1.01 for g in gaps)),
            "bars_missing_estimate": int(sum(missed_est)) if missed_est else 0,
            "max_gap_bars": float(max(gaps)) if len(gaps) else 1.0,
        }
    return out


def data_divergence(bar_df: pd.DataFrame, ref_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"available": False, "notes": "", "summary": {}, "examples": []}
    if ref_df is None or bar_df.empty:
        out["notes"] = "No reference data or no bars observed." if not bar_df.empty else "No BAR_CLOSED events to compare."
        return out

    common = bar_df.index.intersection(ref_df.index)
    if len(common) == 0:
        out["notes"] = "No overlapping timestamps between live bars and reference parquet."
        return out

    out["available"] = True
    a = bar_df.loc[common]
    b = ref_df.loc[common]

    fields = [c for c in ["open", "high", "low", "close"] if c in a.columns and c in b.columns]
    if not fields:
        if "close" in a.columns and "close" in b.columns:
            fields = ["close"]
        else:
            out["notes"] = "BAR_CLOSED events are missing OHLC fields; cannot compare."
            return out

    rows = []
    for f in fields:
        denom = pd.to_numeric(b[f], errors="coerce").replace(0.0, pd.NA)
        bps = (pd.to_numeric(a[f], errors="coerce") - pd.to_numeric(b[f], errors="coerce")) / denom * 10000
        rows.append(bps.rename(f"{f}_bps"))
    div = pd.concat(rows, axis=1)
    absmax = div.abs().max(axis=1)

    out["summary"] = {
        "overlap_bars": int(len(common)),
        "fields": fields,
        "abs_bps_p50": float(absmax.quantile(0.50)),
        "abs_bps_p95": float(absmax.quantile(0.95)),
        "abs_bps_max": float(absmax.max()),
        "abs_bps_gt_5": int((absmax > 5).sum()),
        "abs_bps_gt_20": int((absmax > 20).sum()),
    }

    top = div.copy()
    top["absmax_bps"] = absmax
    top = top.sort_values("absmax_bps", ascending=False).head(20)
    out["examples"] = [
        {"open_time": ts.isoformat(), **{k: float(v) for k, v in r.items() if pd.notna(v)}} for ts, r in top.iterrows()
    ]
    return out


def execution_divergence(fills_df: Optional[pd.DataFrame], ref_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"available": False, "notes": "", "summary": {}, "examples": []}
    if fills_df is None:
        out["notes"] = "No fills.csv found (no trades executed)."
        return out
    if ref_df is None:
        out["notes"] = "Reference parquet unavailable; cannot compute next-open slippage."
        return out

    f = fills_df.copy()
    validate_df_columns(f, FILLS_COLUMNS, "fills.csv")

    time_col = "timestamp_utc"
    price_col = "fill_price"

    if not time_col or not price_col:
        out["notes"] = (
            "fills.csv missing required columns. Expected a time column (fill_time or timestamp_utc) "
            "and a price column (fill_price or price)."
        )
        return out

    if f.empty:
        out["notes"] = "fills.csv is present but contains no rows (no fills during this run)."
        return out

    f["fill_time"] = pd.to_datetime(f[time_col], utc=True, errors="coerce")
    f["fill_price"] = pd.to_numeric(f[price_col], errors="coerce")

    if "fill_kind" not in f.columns:
        kind = pd.Series(["" for _ in range(len(f))])
        if "order_id" in f.columns:
            oid = f["order_id"].astype(str)
            kind = kind.mask(oid.str.contains(r"_ENTRY\b", regex=True), "ENTRY")
            kind = kind.mask(oid.str.contains(r"_EXIT\b", regex=True), "EXIT")
            kind = kind.mask(oid.str.contains(r"_FLATTEN\b", regex=True), "EXIT")
        f["fill_kind"] = kind

    if "type" not in f.columns and "side" in f.columns:
        f["type"] = f["side"].astype(str)

    f = f.dropna(subset=["fill_time", "fill_price"])
    if f.empty:
        out["notes"] = "fills.csv had rows, but fill_time/fill_price could not be parsed."
        return out

    fe = f
    if "fill_kind" in f.columns:
        fe = f[f["fill_kind"].astype(str).str.upper().eq("ENTRY")].copy()

    if fe.empty:
        out["notes"] = "No ENTRY fills found to compare."
        return out

    fe = fe[fe["fill_time"].isin(ref_df.index)].copy()
    if fe.empty:
        out["notes"] = (
            "No ENTRY fill_time timestamps overlap the reference parquet index. "
            "(This can happen if fills are timestamped at exchange execution time rather than bar open.)"
        )
        return out

    ref_col = "open" if "open" in ref_df.columns else ("close" if "close" in ref_df.columns else "")
    if not ref_col:
        out["notes"] = "Reference parquet missing open/close columns."
        return out

    ref_open = pd.to_numeric(ref_df.loc[fe["fill_time"], ref_col], errors="coerce")
    fe["ref_open"] = ref_open.values
    fe = fe.dropna(subset=["ref_open"])
    if fe.empty:
        out["notes"] = "Reference prices not available for overlapped fills."
        return out

    fe["slippage_bps"] = (fe["fill_price"] - fe["ref_open"]) / fe["ref_open"] * 10000
    fe["abs_slippage_bps"] = fe["slippage_bps"].abs()

    out["available"] = True
    out["summary"] = {
        "entry_fills_compared": int(len(fe)),
        "abs_bps_p50": float(fe["abs_slippage_bps"].quantile(0.50)),
        "abs_bps_p95": float(fe["abs_slippage_bps"].quantile(0.95)),
        "abs_bps_max": float(fe["abs_slippage_bps"].max()),
    }

    top = fe.sort_values("abs_slippage_bps", ascending=False).head(20)
    out["examples"] = [
        {
            "fill_time": r["fill_time"].isoformat(),
            "type": str(r.get("type", "")),
            "fill_price": float(r["fill_price"]),
            "ref_open": float(r["ref_open"]),
            "slippage_bps": float(r["slippage_bps"]),
            "order_id": str(r.get("order_id", "")),
        }
        for _, r in top.iterrows()
    ]
    return out


def reject_divergence(orders_df: Optional[pd.DataFrame], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    rejects: List[Dict[str, Any]] = []

    if orders_df is not None:
        o = orders_df.copy()
        if "status" in o.columns:
            rej = o[o["status"].astype(str).str.lower().isin(["rejected", "reject", "error"])].copy()
            for _, r in rej.head(50).iterrows():
                rejects.append(
                    {
                        "source": "orders.csv",
                        "order_id": str(r.get("order_id", "")),
                        "status": str(r.get("status", "")),
                        "detail": str(r.get("status_detail", "")),
                    }
                )
        if "status_detail" in o.columns:
            rej2 = o[o["status_detail"].astype(str).str.contains("reject", case=False, na=False)].copy()
            for _, r in rej2.head(50).iterrows():
                rejects.append(
                    {
                        "source": "orders.csv",
                        "order_id": str(r.get("order_id", "")),
                        "status": str(r.get("status", "")),
                        "detail": str(r.get("status_detail", "")),
                    }
                )

    for ev in events:
        t = _get_event_type(ev)
        if "REJECT" in t or "ERROR" in t:
            rejects.append(
                {
                    "source": "events.jsonl",
                    "type": t,
                    "ts": str(ev.get("ts", "")),
                    "code": ev.get("code"),
                    "msg": ev.get("msg"),
                    "detail": ev.get("detail"),
                }
            )

    return {"available": True, "summary": {"reject_events": int(len(rejects))}, "examples": rejects[:20]}


def funding_divergence(config_used: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    assumed = None
    for key_path in [
        ("futures", "funding_per_8h"),
        ("futures", "funding_rate_per_8h"),
        ("engine", "funding_per_8h"),
        ("engine", "funding_rate_per_8h"),
    ]:
        d: Any = config_used
        ok = True
        for k in key_path:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                ok = False
                break
        if ok:
            assumed = d
            break

    realized = [ev for ev in events if _get_event_type(ev) in {"FUNDING_PAYMENT", "FUNDING_FEE", "INCOME_FUNDING"}]
    total_realized = 0.0
    for ev in realized:
        for k in ["amount", "income", "funding"]:
            if k in ev:
                try:
                    total_realized += float(ev[k])
                except Exception:
                    pass
                break

    return {
        "assumed_funding_per_8h": float(assumed) if assumed is not None and str(assumed) != "" else None,
        "realized_records": int(len(realized)),
        "realized_total": float(total_realized) if realized else 0.0,
        "notes": "Realized funding is only available if the runner logs funding income events (optional).",
    }


def interval_seconds_from_timeframe(timeframe: str) -> int:
    interval_seconds = 1800
    try:
        s = str(timeframe).strip().lower()
        if s.endswith("m"):
            interval_seconds = int(s[:-1]) * 60
        elif s.endswith("h"):
            interval_seconds = int(s[:-1]) * 3600
    except Exception:
        interval_seconds = 1800
    return interval_seconds


def build_report(
    *,
    generated_at_utc: str,
    run_id: str,
    run_start: Dict[str, Any],
    symbol: str,
    timeframe: str,
    bar_df: pd.DataFrame,
    state: Dict[str, Any],
    run_meta: Dict[str, Any],
    ref_path: str,
    ref_note: str,
    interval_seconds: int,
    cfg_used: Dict[str, Any],
    events: List[Dict[str, Any]],
    ref_df: Optional[pd.DataFrame],
    fills_df: Optional[pd.DataFrame],
    orders_df: Optional[pd.DataFrame],
    input_paths: Dict[str, str],
) -> Dict[str, Any]:
    return {
        "generated_at_utc": generated_at_utc,
        "run": {
            "run_id": run_id,
            "mode": run_start.get("mode"),
            "source": run_start.get("source"),
            "symbol": symbol,
            "timeframe": timeframe,
            "market": run_start.get("market"),
            "bars_observed": int(len(bar_df)),
            "bars_processed": int(state.get("bars_processed", 0) or 0),
            "last_bar_open_time_utc": str(state.get("last_bar_open_time_utc", "")),
            "config_sha256": run_meta.get("config_sha256"),
            "dataset_sha256": run_meta.get("dataset_sha256"),
            "reference_parquet": ref_path,
            "reference_note": ref_note,
        },
        "timing_divergence": timing_divergence(bar_df, interval_seconds=interval_seconds),
        "data_divergence": data_divergence(bar_df, ref_df),
        "execution_divergence": execution_divergence(fills_df, ref_df),
        "reject_divergence": reject_divergence(orders_df, events),
        "funding_divergence": funding_divergence(cfg_used, events),
        "notes": {
            "how_to_use": "python scripts/forward_test_report.py --run-id <RUN_ID>",
            "inputs": input_paths,
        },
    }

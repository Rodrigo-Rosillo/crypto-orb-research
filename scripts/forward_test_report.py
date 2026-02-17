from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forward.schemas import FILLS_COLUMNS, validate_df_columns  # noqa: E402


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def to_ts(x: Any) -> pd.Timestamp:
    return pd.to_datetime(x, utc=True, errors="coerce")


def _get_event_type(ev: Dict[str, Any]) -> str:
    return str(ev.get("type") or ev.get("kind") or "").strip()


@dataclass
class ReportPaths:
    run_dir: Path
    events: Path
    signals: Path
    orders: Path
    fills: Path
    positions: Path
    config_used: Path
    run_meta: Path
    state: Path


def resolve_run_dir(repo_root: Path, run_id: str) -> Path:
    run_dir = repo_root / "reports" / "forward_test" / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run folder not found: {run_dir}")
    return run_dir


def get_paths(run_dir: Path) -> ReportPaths:
    return ReportPaths(
        run_dir=run_dir,
        events=run_dir / "events.jsonl",
        signals=run_dir / "signals.csv",
        orders=run_dir / "orders.csv",
        fills=run_dir / "fills.csv",
        positions=run_dir / "positions.csv",
        config_used=run_dir / "config_used.yaml",
        run_meta=run_dir / "run_metadata.json",
        state=run_dir / "state.json",
    )


def try_load_parquet(path: Path) -> Tuple[Optional[pd.DataFrame], str]:
    if not path.exists():
        return None, f"Reference parquet not found: {path}"
    try:
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            for c in ["timestamp", "open_time"]:
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
                    df = df.set_index(c)
                    break
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
        df = df.sort_index()
        return df, ""
    except Exception as e:
        return None, f"Failed to read parquet ({path.name}): {type(e).__name__}: {e}"


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
        return pd.DataFrame(columns=["open_time", "close_time", "ingest_ts", "open", "high", "low", "close", "volume"]).set_index(
            "open_time"
        )
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


def execution_divergence(fills_path: Path, ref_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"available": False, "notes": "", "summary": {}, "examples": []}
    if not fills_path.exists() or fills_path.stat().st_size == 0:
        out["notes"] = "No fills.csv found (no trades executed)."
        return out
    if ref_df is None:
        out["notes"] = "Reference parquet unavailable; cannot compute next-open slippage."
        return out

    f = pd.read_csv(fills_path)
    validate_df_columns(f, FILLS_COLUMNS, "fills.csv")

    # ---- schema normalization ----
    # Different runners may emit fills with slightly different column names.
    # We normalize to: fill_time, fill_price, fill_kind (optional), type (optional), order_id.
    time_col = "fill_time" if "fill_time" in f.columns else (
        "timestamp_utc" if "timestamp_utc" in f.columns else (
            "timestamp" if "timestamp" in f.columns else (
                "ts" if "ts" in f.columns else ""
            )
        )
    )
    price_col = "fill_price" if "fill_price" in f.columns else (
        "price" if "price" in f.columns else ""
    )

    if not time_col or not price_col:
        out["notes"] = (
            "fills.csv missing required columns. Expected a time column (fill_time or timestamp_utc) "
            "and a price column (fill_price or price)."
        )
        return out

    if f.empty:
        out["notes"] = "fills.csv is present but contains no rows (no fills during this run)."
        return out

    # Create normalized columns
    f["fill_time"] = pd.to_datetime(f[time_col], utc=True, errors="coerce")
    f["fill_price"] = pd.to_numeric(f[price_col], errors="coerce")

    # Infer fill_kind if not provided
    if "fill_kind" not in f.columns:
        kind = pd.Series(["" for _ in range(len(f))])
        if "order_id" in f.columns:
            oid = f["order_id"].astype(str)
            kind = kind.mask(oid.str.contains(r"_ENTRY\b", regex=True), "ENTRY")
            kind = kind.mask(oid.str.contains(r"_EXIT\b", regex=True), "EXIT")
            kind = kind.mask(oid.str.contains(r"_FLATTEN\b", regex=True), "EXIT")
        f["fill_kind"] = kind

    # Normalize 'type' for display (optional)
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


def reject_divergence(orders_path: Path, events: List[Dict[str, Any]]) -> Dict[str, Any]:
    rejects: List[Dict[str, Any]] = []

    if orders_path.exists() and orders_path.stat().st_size > 0:
        o = pd.read_csv(orders_path)
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


def build_html(report: Dict[str, Any]) -> str:
    def kv_table(d: Dict[str, Any]) -> str:
        rows = "".join(
            f"<tr><th>{escape(str(k))}</th><td><pre style='margin:0; white-space:pre-wrap'>{escape(str(v))}</pre></td></tr>"
            for k, v in d.items()
        )
        return f"<table>{rows or '<tr><td colspan=2>No data</td></tr>'}</table>"

    def list_table(items: List[Dict[str, Any]], limit: int = 20) -> str:
        if not items:
            return "<p class='small'>No examples.</p>"
        df = pd.DataFrame(items[:limit])
        return df.to_html(index=False, escape=True)

    run = report.get("run", {})
    timing = report.get("timing_divergence", {})
    data = report.get("data_divergence", {})
    exe = report.get("execution_divergence", {})
    rej = report.get("reject_divergence", {})
    fund = report.get("funding_divergence", {})

    return f"""<!doctype html>
<html>
<head>
  <meta charset='utf-8' />
  <title>Forward Test Report</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; color:#111; }}
    h1,h2,h3 {{ margin: 0 0 12px 0; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: start; }}
    .card {{ border: 1px solid #e5e5e5; border-radius: 12px; padding: 16px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 6px 8px; border-bottom: 1px solid #eee; vertical-align: top; }}
    th {{ text-align: left; width: 40%; color:#333; }}
    pre {{ font-size: 12px; }}
    .small {{ color:#666; font-size: 12px; }}
    .pill {{ display:inline-block; padding:2px 8px; border-radius:999px; background:#f3f4f6; font-size:12px; }}
  </style>
</head>
<body>
  <h1>Forward Test Divergence Report</h1>
  <p class='small'>Generated at {escape(str(report.get('generated_at_utc','')))} for run <span class='pill'>{escape(str(run.get('run_id','')))}</span></p>

  <div class='grid'>
    <div class='card'>
      <h2>Run Summary</h2>
      {kv_table(run)}
    </div>
    <div class='card'>
      <h2>Timing Divergence</h2>
      {kv_table(timing.get('ingest_delay_seconds', {}))}
      <h3 style='margin-top:12px;'>Missed Bars</h3>
      {kv_table(timing.get('missed_bars', {}))}
    </div>
  </div>

  <div class='card' style='margin-top:16px;'>
    <h2>Data Divergence (Live vs Reference Parquet)</h2>
    <p class='small'>{escape(str(data.get('notes','')))}</p>
    {kv_table(data.get('summary', {}))}
    <h3 style='margin-top:12px;'>Top Examples (bps)</h3>
    {list_table(data.get('examples', []))}
  </div>

  <div class='card' style='margin-top:16px;'>
    <h2>Execution Divergence (Fill vs Next-Open Reference)</h2>
    <p class='small'>{escape(str(exe.get('notes','')))}</p>
    {kv_table(exe.get('summary', {}))}
    <h3 style='margin-top:12px;'>Top Examples (bps)</h3>
    {list_table(exe.get('examples', []))}
  </div>

  <div class='card' style='margin-top:16px;'>
    <h2>Reject Divergence</h2>
    {kv_table(rej.get('summary', {}))}
    <h3 style='margin-top:12px;'>Examples</h3>
    {list_table(rej.get('examples', []))}
  </div>

  <div class='card' style='margin-top:16px;'>
    <h2>Funding Divergence</h2>
    {kv_table(fund)}
  </div>

  <div class='card' style='margin-top:16px;'>
    <h2>Raw JSON</h2>
    <p class='small'>For programmatic use, open forward_test_report.json in the same run folder.</p>
    <pre style='white-space:pre-wrap'>{escape(json.dumps(report, indent=2, ensure_ascii=False)[:8000])}</pre>
  </div>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate forward-test divergence report (HTML + JSON).")
    ap.add_argument("--run-id", required=True, help="Run folder id under reports/forward_test")
    ap.add_argument(
        "--ref-parquet",
        default="",
        help="Optional path to reference parquet (defaults to data/processed/<symbol>_<tf>.parquet)",
    )
    ap.add_argument("--interval", default="30m", help="Bar interval (default 30m)")
    args = ap.parse_args()

    repo_root = REPO_ROOT
    run_dir = resolve_run_dir(repo_root, args.run_id)
    paths = get_paths(run_dir)

    events = read_jsonl(paths.events)
    run_start = next((ev for ev in events if _get_event_type(ev) == "LIVE_RUN_START"), {})
    symbol = str(run_start.get("symbol") or "SOLUSDT")
    timeframe = str(run_start.get("timeframe") or args.interval)

    ref_path = Path(args.ref_parquet) if args.ref_parquet else (repo_root / "data" / "processed" / f"{symbol}_{timeframe}.parquet")
    ref_df, ref_note = try_load_parquet(ref_path)

    bar_df = build_bar_df(events)

    interval_seconds = 1800
    try:
        s = str(timeframe).strip().lower()
        if s.endswith("m"):
            interval_seconds = int(s[:-1]) * 60
        elif s.endswith("h"):
            interval_seconds = int(s[:-1]) * 3600
    except Exception:
        interval_seconds = 1800

    cfg_used = read_yaml(paths.config_used) if paths.config_used.exists() else {}
    run_meta = read_json(paths.run_meta) if paths.run_meta.exists() else {}
    state = read_json(paths.state) if paths.state.exists() else {}

    report: Dict[str, Any] = {
        "generated_at_utc": _utcnow().isoformat(),
        "run": {
            "run_id": args.run_id,
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
            "reference_parquet": str(ref_path),
            "reference_note": ref_note,
        },
        "timing_divergence": timing_divergence(bar_df, interval_seconds=interval_seconds),
        "data_divergence": data_divergence(bar_df, ref_df),
        "execution_divergence": execution_divergence(paths.fills, ref_df),
        "reject_divergence": reject_divergence(paths.orders, events),
        "funding_divergence": funding_divergence(cfg_used, events),
        "notes": {
            "how_to_use": "python scripts/forward_test_report.py --run-id <RUN_ID>",
            "inputs": {
                "events": str(paths.events),
                "signals": str(paths.signals),
                "orders": str(paths.orders),
                "fills": str(paths.fills),
                "positions": str(paths.positions),
            },
        },
    }

    json_out = run_dir / "forward_test_report.json"
    html_out = run_dir / "forward_test_report.html"
    json_out.write_text(stable_json(report), encoding="utf-8")
    html_out.write_text(build_html(report), encoding="utf-8")

    print(f"✅ Wrote: {json_out}")
    print(f"✅ Wrote: {html_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

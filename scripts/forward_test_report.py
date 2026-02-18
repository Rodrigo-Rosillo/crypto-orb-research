from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forward.forward_test_report_io import get_paths, read_json, read_jsonl, read_yaml, resolve_run_dir, try_load_parquet, utcnow  # noqa: E402
from forward.forward_test_report_logic import (  # noqa: E402
    _get_event_type,
    build_report,
    build_bar_df,
    interval_seconds_from_timeframe,
)
from forward.forward_test_report_render import write_report  # noqa: E402


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
    report = build_report(
        generated_at_utc=utcnow().isoformat(),
        run_id=args.run_id,
        run_start=run_start,
        symbol=symbol,
        timeframe=timeframe,
        bar_df=bar_df,
        state=(read_json(paths.state) if paths.state.exists() else {}),
        run_meta=(read_json(paths.run_meta) if paths.run_meta.exists() else {}),
        ref_path=str(ref_path),
        ref_note=ref_note,
        interval_seconds=interval_seconds_from_timeframe(timeframe),
        cfg_used=(read_yaml(paths.config_used) if paths.config_used.exists() else {}),
        events=events,
        ref_df=ref_df,
        fills_df=(pd.read_csv(paths.fills) if paths.fills.exists() and paths.fills.stat().st_size > 0 else None),
        orders_df=(pd.read_csv(paths.orders) if paths.orders.exists() and paths.orders.stat().st_size > 0 else None),
        input_paths={
            "events": str(paths.events),
            "signals": str(paths.signals),
            "orders": str(paths.orders),
            "fills": str(paths.fills),
            "positions": str(paths.positions),
        },
    )

    json_out, html_out = write_report(run_dir, report)

    print(f" Wrote: {json_out}")
    print(f" Wrote: {html_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

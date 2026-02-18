from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.utils import sha256_file, stable_json  # noqa: E402


def parse_timeframe_to_minutes(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 60 * 24
    raise ValueError(f"Unsupported timeframe: {tf}. Use like '30m', '1h', '1d'.")


def read_binance_timestamps_only(file_path: Path) -> Tuple[pd.Series, int]:
    """
    Reads only the first column (ms timestamp) from Binance kline CSV.
    Returns (timestamps_utc, nat_count).
    """
    s = pd.read_csv(file_path, header=None, skiprows=1, usecols=[0])[0]
    ts = pd.to_datetime(s, unit="ms", utc=True, errors="coerce")
    nat_count = int(ts.isna().sum())
    ts = ts.dropna()
    return ts, nat_count


def is_aligned(ts: pd.DatetimeIndex, interval_minutes: int) -> pd.Series:
    secs = ts.minute * 60 + ts.second
    aligned = (secs % (interval_minutes * 60) == 0) & (ts.microsecond == 0)
    return pd.Series(aligned, index=ts)


def summarize_gaps(ts_sorted_unique: pd.DatetimeIndex, expected_delta: pd.Timedelta) -> List[Dict[str, Any]]:
    diffs = ts_sorted_unique.to_series().diff()
    gap_mask = diffs > expected_delta
    gaps = []
    for t, d in diffs[gap_mask].items():
        prev = t - d
        missing_bars = int(d / expected_delta) - 1
        gaps.append(
            {
                "gap_start_prev_bar_utc": prev.isoformat(),
                "gap_end_next_bar_utc": t.isoformat(),
                "gap_minutes": float(d.total_seconds() / 60.0),
                "missing_bars": int(max(missing_bars, 0)),
            }
        )
    return gaps


def build_html(report: Dict[str, Any]) -> str:
    def kv_table(d: Dict[str, Any]) -> str:
        rows = []
        for k in sorted(d.keys()):
            v = d[k]
            if isinstance(v, (dict, list)):
                v_str = escape(json.dumps(v, ensure_ascii=False, indent=2))
                v_html = f"<pre style='margin:0; white-space:pre-wrap'>{v_str}</pre>"
            else:
                v_html = escape(str(v))
            rows.append(f"<tr><th>{escape(str(k))}</th><td>{v_html}</td></tr>")
        return "<table>" + "".join(rows) + "</table>"

    summary = report.get("summary", {})
    dataset = report.get("dataset", {})

    samples = {
        "gaps_sample": report.get("gaps_sample", []),
        "missing_sample": report.get("missing_sample", []),
        "misaligned_sample": report.get("misaligned_sample", []),
    }

    # Missing-by-day table
    missing_by_day = report.get("missing_by_day", [])
    if missing_by_day:
        rows = []
        for r in missing_by_day:
            rows.append(
                "<tr>"
                f"<td><code>{escape(str(r.get('date_utc','')))}</code></td>"
                f"<td>{escape(str(r.get('expected_bars','')))}</td>"
                f"<td>{escape(str(r.get('present_bars','')))}</td>"
                f"<td><b>{escape(str(r.get('missing_bars','')))}</b></td>"
                "</tr>"
            )
        missing_by_day_table = (
            "<table>"
            "<tr><th>date_utc</th><th>expected_bars</th><th>present_bars</th><th>missing_bars</th></tr>"
            + "".join(rows)
            + "</table>"
        )
    else:
        missing_by_day_table = "<p>No missing bars by day.</p>"

    files = report.get("files", [])
    files_rows = []
    for f in files:
        files_rows.append(
            "<tr>"
            f"<td><code>{escape(str(f.get('path','')))}</code></td>"
            f"<td>{escape(str(f.get('bars', '')))}</td>"
            f"<td>{escape(str(f.get('duplicates', '')))}</td>"
            f"<td>{escape(str(f.get('nat_timestamps', '')))}</td>"
            f"<td>{escape(str(f.get('start_utc', '')))}</td>"
            f"<td>{escape(str(f.get('end_utc', '')))}</td>"
            "</tr>"
        )
    files_table = (
        "<table>"
        "<tr><th>path</th><th>bars</th><th>duplicates</th><th>NaT</th><th>start_utc</th><th>end_utc</th></tr>"
        + "".join(files_rows)
        + "</table>"
        if files_rows
        else "<p>No files.</p>"
    )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Data Quality Report</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; color:#111; }}
    h1,h2 {{ margin: 0 0 12px 0; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: start; }}
    .card {{ border: 1px solid #e5e5e5; border-radius: 12px; padding: 16px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 6px 8px; border-bottom: 1px solid #eee; vertical-align: top; text-align:left; }}
    th {{ width: 38%; color:#333; }}
    pre {{ font-size: 12px; }}
    code {{ font-size: 12px; }}
    .small {{ color:#666; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>Data Quality Report</h1>
  <p class="small">Generated: {escape(report.get("generated_utc",""))}</p>

  <div class="grid">
    <div class="card">
      <h2>Dataset</h2>
      {kv_table(dataset)}
    </div>
    <div class="card">
      <h2>Summary</h2>
      {kv_table(summary)}
    </div>
  </div>

  <div class="card" style="margin-top:16px;">
    <h2>Missing Bars by Day</h2>
    {missing_by_day_table}
  </div>

  <div class="card" style="margin-top:16px;">
    <h2>Samples</h2>
    {kv_table(samples)}
  </div>

  <div class="card" style="margin-top:16px;">
    <h2>Files</h2>
    {files_table}
  </div>
</body>
</html>
"""
    return html


def main() -> int:
    ap = argparse.ArgumentParser(description="Data quality checks for Binance OHLCV dataset (UTC)")
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml (relative to repo root by default)")
    ap.add_argument("--manifest", default="data/manifest.json", help="Path to manifest.json (relative to repo root)")
    ap.add_argument("--data-dir", default="", help="Directory containing raw CSVs. If omitted, uses manifest.data_root.")
    ap.add_argument("--out-dir", default="reports/data_quality", help="Output directory")
    ap.add_argument("--max-samples", type=int, default=50, help="Max items to include in sample lists")
    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = (REPO_ROOT / manifest_path).resolve()

    cfg_text = config_path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(cfg_text) or {}
    symbol = str(cfg.get("symbol", "SOLUSDT"))
    timeframe = str(cfg.get("timeframe", "30m"))
    interval_minutes = parse_timeframe_to_minutes(timeframe)
    expected_delta = pd.Timedelta(minutes=interval_minutes)

    # Expected bars per full UTC day (e.g., 48 for 30m)
    expected_bars_per_day = int(pd.Timedelta(days=1) / expected_delta)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_root = manifest.get("data_root")
    dataset_sha256 = manifest.get("dataset_sha256")
    manifest_sha256 = sha256_file(manifest_path)

    data_dir = Path(args.data_dir).resolve() if args.data_dir else Path(str(data_root)).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    files = manifest.get("files", [])
    file_paths = [f.get("path") for f in files if isinstance(f, dict) and isinstance(f.get("path"), str)]
    file_paths = [p for p in file_paths if p.lower().endswith(".csv")]

    prefix = f"{symbol}-{timeframe}-"
    selected = [p for p in file_paths if p.startswith(prefix)]
    if not selected:
        selected = file_paths
    selected = sorted(selected, key=lambda x: x.lower())

    if not selected:
        raise RuntimeError("No CSV files found in manifest.")

    all_ts = []
    file_summaries: List[Dict[str, Any]] = []
    total_nat = 0

    for p in selected:
        fp = data_dir / p
        if not fp.exists():
            raise FileNotFoundError(f"Missing file listed in manifest: {fp}")

        ts, nat_count = read_binance_timestamps_only(fp)
        total_nat += nat_count

        ts_sorted = ts.sort_values()
        dup_count = int(ts_sorted.duplicated().sum())

        start_utc = ts_sorted.iloc[0].isoformat() if len(ts_sorted) else None
        end_utc = ts_sorted.iloc[-1].isoformat() if len(ts_sorted) else None

        file_summaries.append(
            {
                "path": p,
                "bars": int(len(ts_sorted)),
                "duplicates": dup_count,
                "nat_timestamps": nat_count,
                "start_utc": start_utc,
                "end_utc": end_utc,
            }
        )
        all_ts.append(ts_sorted)

    ts_all = pd.concat(all_ts, ignore_index=True).sort_values()
    bars_total = int(len(ts_all))
    duplicates_total = int(ts_all.duplicated().sum())

    ts_unique = ts_all.drop_duplicates()
    bars_unique = int(len(ts_unique))

    actual_index = pd.DatetimeIndex(ts_unique).sort_values()

    # Alignment
    aligned_mask = is_aligned(actual_index, interval_minutes)
    misaligned_count = int((~aligned_mask).sum())
    misaligned_sample = (
        [t.isoformat() for t in aligned_mask[~aligned_mask].index[: args.max_samples]]
        if misaligned_count
        else []
    )

    # Missing bars over full range
    start = actual_index[0]
    end = actual_index[-1]
    expected_index = pd.date_range(start=start, end=end, freq=expected_delta, tz="UTC")
    expected_bars = int(len(expected_index))

    missing_index = expected_index.difference(actual_index)
    missing_bars = int(len(missing_index))
    missing_pct = (missing_bars / expected_bars * 100.0) if expected_bars else 0.0

    missing_sample = [t.isoformat() for t in missing_index[: args.max_samples]]

    # Gaps summary
    gaps = summarize_gaps(actual_index, expected_delta)
    gaps_sorted = sorted(gaps, key=lambda g: (g["missing_bars"], g["gap_minutes"]), reverse=True)
    gaps_sample = gaps_sorted[: args.max_samples]
    largest_gap_minutes = float(gaps_sorted[0]["gap_minutes"]) if gaps_sorted else 0.0
    gap_count = int(len(gaps_sorted))
    missing_bars_from_gaps = int(sum(g["missing_bars"] for g in gaps_sorted))

    # -----------------------------
    # Missing bars by day (ALL days)
    # -----------------------------
    # Count present bars per UTC day
    present_by_day = pd.Series(1, index=actual_index).groupby(actual_index.normalize()).sum()

    # Count missing bars per UTC day
    if len(missing_index):
        missing_by_day_s = pd.Series(1, index=missing_index).groupby(missing_index.normalize()).sum()
    else:
        missing_by_day_s = pd.Series(dtype=int)

    # Build list only for days that have missing bars (>0)
    missing_by_day_list: List[Dict[str, Any]] = []
    for day_ts, miss_n in missing_by_day_s.sort_index().items():
        present_n = int(present_by_day.get(day_ts, 0))
        missing_by_day_list.append(
            {
                "date_utc": day_ts.date().isoformat(),
                "expected_bars": expected_bars_per_day,
                "present_bars": present_n,
                "missing_bars": int(miss_n),
            }
        )

    # Sort by missing desc, then date asc (nice for reading)
    missing_by_day_list = sorted(missing_by_day_list, key=lambda r: (-r["missing_bars"], r["date_utc"]))

    invalid_days_count = int(len(missing_by_day_list))

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "timeframe": timeframe,
        "interval_minutes": interval_minutes,
        "timezone": "UTC",
        "dataset": {
            "data_dir": str(data_dir),
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "dataset_sha256": dataset_sha256,
            "files_analyzed": int(len(selected)),
        },
        "summary": {
            "bars_total": bars_total,
            "bars_unique": bars_unique,
            "expected_bars": expected_bars,
            "expected_bars_per_day": expected_bars_per_day,
            "missing_bars": missing_bars,
            "missing_pct": round(missing_pct, 6),
            "missing_bars_from_gaps": missing_bars_from_gaps,
            "duplicates_total": duplicates_total,
            "misaligned_bars": misaligned_count,
            "nat_timestamps_total": total_nat,
            "start_utc": start.isoformat(),
            "end_utc": end.isoformat(),
            "gap_count": gap_count,
            "largest_gap_minutes": largest_gap_minutes,
            "invalid_days_count": invalid_days_count,
            "is_monotonic_increasing_after_sort": True,
        },
        "missing_by_day": missing_by_day_list,  # <-- NEW
        "gaps_sample": gaps_sample,
        "missing_sample": missing_sample,
        "misaligned_sample": misaligned_sample,
        "files": file_summaries,
    }

    (out_dir / "quality.json").write_text(stable_json(report), encoding="utf-8")
    (out_dir / "quality.html").write_text(build_html(report), encoding="utf-8")

    print(f"✅ Wrote: {out_dir / 'quality.json'}")
    print(f"✅ Wrote: {out_dir / 'quality.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

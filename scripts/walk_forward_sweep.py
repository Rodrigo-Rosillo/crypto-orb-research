import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_num_list(s: str, cast=float) -> List:
    s = (s or "").strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return [cast(p) for p in parts]


def fmt_token(x: float) -> str:
    """
    Stable token for folder names:
      1.25 -> "1p25"
      0.0001 -> "0p0001"
      -0.0001 -> "m0p0001"
    """
    sign = ""
    if x < 0:
        sign = "m"
        x = abs(x)
    s = f"{x:.10f}".rstrip("0").rstrip(".")
    s = s.replace(".", "p")
    return f"{sign}{s}"


def run_walk_forward(out_dir: Path, funding_per_8h: float, extra_args: List[str], skip_existing: bool) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    folds_csv = out_dir / "walk_forward_folds.csv"

    if skip_existing and folds_csv.exists():
        return folds_csv

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "walk_forward.py"),
        "--out-dir",
        str(out_dir),
        "--funding-per-8h",
        str(funding_per_8h),
        *extra_args,
    ]

    subprocess.check_call(cmd, cwd=str(REPO_ROOT))
    return folds_csv


def summarize_by_funding(df_all: pd.DataFrame) -> pd.DataFrame:
    # Numeric cols we care about (if present)
    cols = [
        "total_return_pct",
        "max_drawdown_pct",
        "daily_sharpe",
        "cagr",
        "total_trades",
        "win_rate_pct",
        "expectancy_per_trade",
        "total_fees",
        "total_funding",
        "liquidations",
    ]
    present = [c for c in cols if c in df_all.columns]

    rows = []
    for fund, g in df_all.groupby("funding_per_8h"):
        row = {"funding_per_8h": float(fund), "folds": int(len(g))}
        if "total_return_pct" in g.columns:
            row["pct_positive_folds"] = float((g["total_return_pct"] > 0).mean() * 100.0)
        for c in present:
            s = pd.to_numeric(g[c], errors="coerce").dropna()
            if s.empty:
                continue
            row[f"{c}_mean"] = float(s.mean())
            row[f"{c}_median"] = float(s.median())
            row[f"{c}_min"] = float(s.min())
            row[f"{c}_max"] = float(s.max())
        rows.append(row)

    return pd.DataFrame(rows).sort_values("funding_per_8h")


def main():
    ap = argparse.ArgumentParser(description="Run walk-forward for multiple funding settings and combine results.")
    ap.add_argument("--base-out-dir", default="reports/walk_forward_sweep")
    ap.add_argument("--funding-list", default="0,0.0001,-0.0001")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", action="store_true", default=False)

    # Everything else gets passed through to walk_forward.py (engine, train/test/step, slippage, etc.)
    args, extra_args = ap.parse_known_args()

    base_out = (REPO_ROOT / args.base_out_dir).resolve()
    base_out.mkdir(parents=True, exist_ok=True)

    fundings = parse_num_list(args.funding_list, float)
    if not fundings:
        raise SystemExit("No funding values parsed. Example: --funding-list 0,0.0001,-0.0001")

    skip_existing = True
    if args.no_skip_existing:
        skip_existing = False
    else:
        skip_existing = bool(args.skip_existing)

    all_frames = []

    for f in fundings:
        token = fmt_token(float(f))
        out_dir = base_out / f"fund_{token}"
        print(f"\n=== Walk-forward funding_per_8h={f} -> {out_dir} ===")

        folds_csv = run_walk_forward(out_dir, float(f), extra_args, skip_existing)
        df = pd.read_csv(folds_csv)

        df["funding_per_8h"] = float(f)
        df["run_dir"] = str(out_dir)
        all_frames.append(df)

    df_all = pd.concat(all_frames, ignore_index=True)
    combined_path = base_out / "walk_forward_funding_sweep.csv"
    df_all.to_csv(combined_path, index=False)

    summary = summarize_by_funding(df_all)
    summary_path = base_out / "walk_forward_funding_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"\n[OK] Wrote combined: {combined_path}")
    print(f"[OK] Wrote summary:  {summary_path}")


if __name__ == "__main__":
    main()

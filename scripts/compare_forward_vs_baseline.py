"""Compare Phase 5 forward-test (replay+shadow) outputs to the Phase 0-4 baseline.

Phase 5 / Step 2 acceptance check (replay + shadow) is essentially:
  - same trades are taken (count + entry timestamps)
  - prices match the expected execution model (next-open entries + intrabar exits)
  - optional: sizing/fees match when configs/leverage are identical

This script compares:
  baseline: reports/baseline/trades.csv (+ results.json optional)
  forward : reports/forward_test/<run_id>/fills.csv

It builds a trade table from fills.csv by grouping ENTRY/EXIT fills by trade id.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd


@dataclass
class CompareResult:
    ok: bool
    summary: str


def _parse_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def _trade_key_from_order_id(order_id: str) -> Tuple[str, str, str]:
    """Return (trade_id, signal_type, leg) where leg in {'ENTRY','EXIT'}.

    Expected formats:
      T00000_20210111T170000Z_downtrend_breakdown_ENTRY
      T00000_20210111T170000Z_downtrend_breakdown_EXIT

    NOTE: signal types may contain underscores, so we cannot take only the last segment.
    """
    if not isinstance(order_id, str):
        return ("", "", "")

    m = re.match(r"^(T\d+_\d{8}T\d{6}Z)_(.+)_(ENTRY|EXIT)$", order_id)
    if not m:
        return (order_id, "", "")

    prefix = m.group(1)
    signal_type = m.group(2)
    leg = m.group(3)
    trade_id = f"{prefix}_{signal_type}"
    return trade_id, signal_type, leg


def build_trades_from_fills(fills_csv: Path) -> pd.DataFrame:
    fills = pd.read_csv(fills_csv)
    if fills.empty:
        return pd.DataFrame()

    fills["timestamp_utc"] = _parse_dt(fills["timestamp_utc"])

    tids, sts, legs = [], [], []
    for oid in fills["order_id"].astype(str).tolist():
        tid, st, leg = _trade_key_from_order_id(oid)
        tids.append(tid)
        sts.append(st)
        legs.append(leg)
    fills["trade_id"] = tids
    fills["signal_type"] = sts
    fills["leg"] = legs

    entry = (
        fills[fills["leg"] == "ENTRY"]
        .sort_values("timestamp_utc")
        .groupby("trade_id", as_index=False)
        .first()
        .rename(
            columns={
                "timestamp_utc": "entry_time",
                "fill_price": "entry_price",
                "fee": "entry_fee",
                "qty": "qty",
            }
        )
    )
    exit_ = (
        fills[fills["leg"] == "EXIT"]
        .sort_values("timestamp_utc")
        .groupby("trade_id", as_index=False)
        .first()
        .rename(
            columns={
                "timestamp_utc": "exit_time",
                "fill_price": "exit_price",
                "fee": "exit_fee",
            }
        )
    )

    trades = entry.merge(
        exit_[["trade_id", "exit_time", "exit_price", "exit_fee"]],
        on="trade_id",
        how="left",
    )

    def _typ_from_side(s: str) -> str:
        s = str(s).lower()
        if s == "sell":
            return "SHORT"
        if s == "buy":
            return "LONG"
        return ""

    trades["type"] = trades.get("side").map(_typ_from_side)
    trades["fees_total"] = trades[["entry_fee", "exit_fee"]].fillna(0).sum(axis=1)

    keep = [
        "entry_time",
        "exit_time",
        "type",
        "signal_type",
        "entry_price",
        "exit_price",
        "qty",
        "entry_fee",
        "exit_fee",
        "fees_total",
        "trade_id",
    ]
    for c in keep:
        if c not in trades.columns:
            trades[c] = pd.NA
    return trades[keep].sort_values("entry_time").reset_index(drop=True)


def _float_close(a: float, b: float, atol: float, rtol: float) -> bool:
    if a is None or b is None or (isinstance(a, float) and math.isnan(a)) or (isinstance(b, float) and math.isnan(b)):
        return False
    return math.isclose(float(a), float(b), abs_tol=atol, rel_tol=rtol)


def compare_trades(
    baseline_csv: Path,
    forward_fills_csv: Path,
    price_atol: float,
    price_rtol: float,
    time_tol_seconds: int,
    strict: bool,
    show_mismatches: int,
) -> CompareResult:
    base = pd.read_csv(baseline_csv)
    if base.empty:
        return CompareResult(False, f"Baseline trades.csv is empty: {baseline_csv}")
    base["entry_time"] = _parse_dt(base["entry_time"])
    base["exit_time"] = _parse_dt(base["exit_time"])
    base = base.sort_values("entry_time").reset_index(drop=True)

    fwd = build_trades_from_fills(forward_fills_csv)
    if fwd.empty:
        return CompareResult(False, f"Forward fills.csv produced no trades: {forward_fills_csv}")

    base_key = base[["entry_time", "signal_type", "type", "entry_price", "exit_time", "exit_price"]].copy()
    fwd_key = fwd[["entry_time", "signal_type", "type", "entry_price", "exit_time", "exit_price", "trade_id"]].copy()

    merged = base_key.merge(
        fwd_key,
        on=["entry_time", "signal_type", "type"],
        how="outer",
        indicator=True,
        suffixes=("_base", "_fwd"),
    )

    missing_in_forward = merged[merged["_merge"] == "left_only"]
    extra_in_forward = merged[merged["_merge"] == "right_only"]
    matched = merged[merged["_merge"] == "both"].copy()

    diffs = []
    for _, r in matched.iterrows():
        issues = []
        if not _float_close(r.get("entry_price_base"), r.get("entry_price_fwd"), price_atol, price_rtol):
            issues.append("entry_price")
        if not _float_close(r.get("exit_price_base"), r.get("exit_price_fwd"), price_atol, price_rtol):
            issues.append("exit_price")
        bt = r.get("exit_time_base")
        ft = r.get("exit_time_fwd")
        if pd.isna(bt) or pd.isna(ft):
            issues.append("exit_time")
        else:
            if abs((pd.to_datetime(ft, utc=True) - pd.to_datetime(bt, utc=True)).total_seconds()) > time_tol_seconds:
                issues.append("exit_time")
        if issues:
            diffs.append(
                {
                    "entry_time": r["entry_time"],
                    "signal_type": r["signal_type"],
                    "type": r["type"],
                    "issues": ",".join(issues),
                    "entry_price_base": r.get("entry_price_base"),
                    "entry_price_fwd": r.get("entry_price_fwd"),
                    "exit_price_base": r.get("exit_price_base"),
                    "exit_price_fwd": r.get("exit_price_fwd"),
                    "exit_time_base": r.get("exit_time_base"),
                    "exit_time_fwd": r.get("exit_time_fwd"),
                    "trade_id_fwd": r.get("trade_id"),
                }
            )
    diffs_df = pd.DataFrame(diffs)

    ok_core = missing_in_forward.empty and extra_in_forward.empty and diffs_df.empty

    if strict and ok_core:
        base_full = base[["entry_time", "signal_type", "type", "qty", "fees_total"]].copy()
        fwd_full = fwd[["entry_time", "signal_type", "type", "qty", "fees_total"]].copy()
        base_full["entry_time"] = _parse_dt(base_full["entry_time"])
        fwd_full["entry_time"] = _parse_dt(fwd_full["entry_time"])
        mm = base_full.merge(fwd_full, on=["entry_time", "signal_type", "type"], suffixes=("_base", "_fwd"))
        if len(mm) != len(base_full) or len(mm) != len(fwd_full):
            ok_core = False
        else:
            for _, r in mm.iterrows():
                if not _float_close(r.get("qty_base"), r.get("qty_fwd"), 1e-9, 1e-9):
                    ok_core = False
                    break
                if not _float_close(r.get("fees_total_base"), r.get("fees_total_fwd"), 1e-6, 1e-6):
                    ok_core = False
                    break

    lines = [f"Baseline trades: {len(base)}", f"Forward trades (from fills): {len(fwd)}"]
    if not missing_in_forward.empty:
        lines.append(f"Missing in forward: {len(missing_in_forward)}")
    if not extra_in_forward.empty:
        lines.append(f"Extra in forward: {len(extra_in_forward)}")
    if not diffs_df.empty:
        lines.append(f"Matched trades with field diffs: {len(diffs_df)}")

    if show_mismatches > 0:
        if not missing_in_forward.empty:
            lines.append("\nExamples missing in forward:")
            lines.append(missing_in_forward[["entry_time", "signal_type", "type"]].head(show_mismatches).to_string(index=False))
        if not extra_in_forward.empty:
            lines.append("\nExamples extra in forward:")
            lines.append(extra_in_forward[["entry_time", "signal_type", "type", "trade_id"]].head(show_mismatches).to_string(index=False))
        if not diffs_df.empty:
            lines.append("\nExamples diffs:")
            lines.append(diffs_df.head(show_mismatches).to_string(index=False))

    status = "PASS" if ok_core else "FAIL"
    if strict:
        status = "PASS (strict)" if ok_core else "FAIL (strict)"
    return CompareResult(ok_core, status + "\n" + "\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=False, help="Forward-test run id folder name under reports/forward_test")
    ap.add_argument("--baseline-dir", default="reports/baseline")
    ap.add_argument("--forward-root", default="reports/forward_test")
    ap.add_argument("--price-atol", type=float, default=1e-9)
    ap.add_argument("--price-rtol", type=float, default=1e-9)
    ap.add_argument("--time-tol-seconds", type=int, default=0)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--show-mismatches", type=int, default=10)
    args = ap.parse_args()

    baseline_dir = Path(args.baseline_dir)
    forward_root = Path(args.forward_root)
    if not baseline_dir.exists():
        raise SystemExit(f"Baseline dir not found: {baseline_dir}")
    if not forward_root.exists():
        raise SystemExit(f"Forward root not found: {forward_root}")

    run_id = args.run_id
    if not run_id:
        runs = [p for p in forward_root.iterdir() if p.is_dir()]
        if not runs:
            raise SystemExit(f"No forward-test runs found under {forward_root}")
        run_id = sorted(runs, key=lambda p: p.name)[-1].name

    forward_dir = forward_root / run_id
    baseline_trades = baseline_dir / "trades.csv"
    forward_fills = forward_dir / "fills.csv"
    if not baseline_trades.exists():
        raise SystemExit(f"Missing baseline trades.csv: {baseline_trades}")
    if not forward_fills.exists():
        raise SystemExit(f"Missing forward fills.csv: {forward_fills}")

    res = compare_trades(
        baseline_csv=baseline_trades,
        forward_fills_csv=forward_fills,
        price_atol=args.price_atol,
        price_rtol=args.price_rtol,
        time_tol_seconds=args.time_tol_seconds,
        strict=args.strict,
        show_mismatches=args.show_mismatches,
    )
    print(res.summary)


if __name__ == "__main__":
    main()

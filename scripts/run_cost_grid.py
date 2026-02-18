import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_num_list(s: str, cast=float) -> List:
    """
    Parse comma-separated list: "1,2,3" -> [1,2,3]
    Also accepts whitespace.
    """
    s = (s or "").strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return [cast(p) for p in parts]


def fmt_token(x: float) -> str:
    """
    Stable-ish string token for folder names:
    1.25 -> "1p25"
    0.0001 -> "0p0001"
    -0.0001 -> "m0p0001"
    """
    sign = ""
    if x < 0:
        sign = "m"
        x = abs(x)
    s = f"{x:.10f}".rstrip("0").rstrip(".")  # avoid scientific notation
    s = s.replace(".", "p")
    return f"{sign}{s}"


def run_one(
    engine: str,
    out_dir: Path,
    fee_mult: float,
    slippage_bps: float,
    delay_bars: int,
    config_path: Optional[str] = None,
    leverage: Optional[float] = None,
    mmr: float = 0.005,
    funding_per_8h: Optional[float] = None,
    skip_existing: bool = True,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "results.json"
    if skip_existing and results_path.exists():
        r = json.loads(results_path.read_text(encoding="utf-8"))
    else:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_baseline.py"),
            "--engine",
            engine,
            "--out-dir",
            str(out_dir),
            "--fee-mult",
            str(fee_mult),
            "--slippage-bps",
            str(slippage_bps),
            "--delay-bars",
            str(delay_bars),
        ]

        if config_path:
            cmd += ["--config", str(config_path)]

        if engine == "futures":
            if leverage is None:
                raise ValueError("leverage must be provided for futures runs")
            if funding_per_8h is None:
                raise ValueError("funding_per_8h must be provided for futures runs")
            cmd += [
                "--leverage",
                str(leverage),
                "--mmr",
                str(mmr),
                "--funding-per-8h",
                str(funding_per_8h),
            ]

        subprocess.check_call(cmd, cwd=str(REPO_ROOT))
        r = json.loads(results_path.read_text(encoding="utf-8"))

    m = r.get("metrics", {})
    p = r.get("params", {})

    return {
        "engine": engine,
        "fee_mult": fee_mult,
        "slippage_bps": slippage_bps,
        "delay_bars": delay_bars,
        "leverage": p.get("leverage", leverage),
        "mmr": p.get("mmr", mmr if engine == "futures" else None),
        "funding_per_8h": p.get("funding_per_8h", funding_per_8h),
        "total_trades": m.get("Total Trades"),
        "win_rate_pct": m.get("Win Rate %"),
        "total_return_pct": m.get("Total Return %"),
        "max_drawdown_pct": m.get("Max Drawdown %"),
        "final_equity": m.get("Final Equity"),
        "total_fees_paid": m.get("Total Fees Paid"),
        "total_funding_paid": m.get("Total Funding Paid"),
        "liquidations": m.get("Liquidations"),
        "results_path": str(results_path),
    }


def main():
    ap = argparse.ArgumentParser(description="Run cost stress grid for spot + futures engines")
    ap.add_argument("--config", default="config.yaml", help="Config file passed through to run_baseline.py")
    ap.add_argument("--fee-mults", default="1.0,1.25,1.5")
    ap.add_argument("--slippages-bps", default="0,1,3,5")
    ap.add_argument("--delays", default="1,2")

    # Futures sweeps:
    ap.add_argument("--leverage-list", default="1")
    ap.add_argument("--funding-list", default="0.0001")  # 0.01% per 8h by default
    ap.add_argument("--mmr", type=float, default=0.005)

    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", action="store_true", default=False)

    args = ap.parse_args()

    fee_mults = parse_num_list(args.fee_mults, float)
    slippages = parse_num_list(args.slippages_bps, float)
    delays = parse_num_list(args.delays, int)

    leverages = parse_num_list(args.leverage_list, float)
    fundings = parse_num_list(args.funding_list, float)
    mmr = float(args.mmr)

    # override skip behavior if explicitly requested
    skip_existing = True
    if args.no_skip_existing:
        skip_existing = False
    else:
        skip_existing = bool(args.skip_existing)

    base = REPO_ROOT / "reports" / "scenarios"
    spot_base = base / "spot"
    fut_base = base / "futures"
    spot_base.mkdir(parents=True, exist_ok=True)
    fut_base.mkdir(parents=True, exist_ok=True)

    spot_rows = []
    fut_rows = []

    # ---- SPOT RUNS ----
    for fm in fee_mults:
        for sb in slippages:
            for d in delays:
                name = f"fee{fmt_token(fm)}_slip{fmt_token(sb)}bps_delay{d}"
                out_dir = spot_base / name
                print(f"\n=== SPOT    {name} ===")
                spot_rows.append(
                    run_one(
                        engine="spot",
                        config_path=args.config,
                        out_dir=out_dir,
                        fee_mult=fm,
                        slippage_bps=sb,
                        delay_bars=d,
                        skip_existing=skip_existing,
                    )
                )

    # ---- FUTURES RUNS (fee/slip/delay) x leverage x funding ----
    for lev in leverages:
        for fund in fundings:
            for fm in fee_mults:
                for sb in slippages:
                    for d in delays:
                        name = (
                            f"fee{fmt_token(fm)}_slip{fmt_token(sb)}bps_delay{d}"
                            f"_lev{fmt_token(lev)}_mmr{fmt_token(mmr)}_fund{fmt_token(fund)}"
                        )
                        out_dir = fut_base / name
                        print(f"\n=== FUTURES {name} ===")
                        fut_rows.append(
                            run_one(
                                engine="futures",
                                config_path=args.config,
                                out_dir=out_dir,
                                fee_mult=fm,
                                slippage_bps=sb,
                                delay_bars=d,
                                leverage=lev,
                                mmr=mmr,
                                funding_per_8h=fund,
                                skip_existing=skip_existing,
                            )
                        )

    # Write summaries
    spot_df = pd.DataFrame(spot_rows).sort_values(["fee_mult", "slippage_bps", "delay_bars"])
    fut_df = pd.DataFrame(fut_rows).sort_values(
        ["funding_per_8h", "leverage", "fee_mult", "slippage_bps", "delay_bars"]
    )

    spot_out = base / "grid_summary_spot.csv"
    fut_out = base / "grid_summary_futures.csv"
    spot_df.to_csv(spot_out, index=False)
    fut_df.to_csv(fut_out, index=False)

    print(f"\n[OK] Wrote: {spot_out}")
    print(f"[OK] Wrote: {fut_out}")

    # Convenience slices: one CSV per funding value
    for fund in sorted(set(fundings)):
        fdf = fut_df[fut_df["funding_per_8h"] == fund].copy()
        if fdf.empty:
            continue
        out = base / f"grid_summary_futures_funding_{fmt_token(fund)}.csv"
        fdf.to_csv(out, index=False)
        print(f"[OK] Wrote: {out}")


if __name__ == "__main__":
    main()

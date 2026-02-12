import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_one(out_dir: Path, fee_mult: float, slippage_bps: float, delay_bars: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_baseline.py"),
        "--out-dir",
        str(out_dir),
        "--fee-mult",
        str(fee_mult),
        "--slippage-bps",
        str(slippage_bps),
        "--delay-bars",
        str(delay_bars),
    ]
    subprocess.check_call(cmd, cwd=str(REPO_ROOT))

    results_path = out_dir / "results.json"
    r = json.loads(results_path.read_text(encoding="utf-8"))

    m = r["metrics"]
    return {
        "fee_mult": fee_mult,
        "slippage_bps": slippage_bps,
        "delay_bars": delay_bars,
        "total_trades": m["Total Trades"],
        "win_rate_pct": m["Win Rate %"],
        "total_return_pct": m["Total Return %"],
        "max_drawdown_pct": m["Max Drawdown %"],
        "final_capital": m["Final Capital"],
        "total_fees_paid": m["Total Fees Paid"],
    }


def main():
    # Required grid from the Phase 2 checklist
    fee_mults = [1.0, 1.25, 1.5]
    slippages = [0.0, 1.0, 3.0, 5.0]  # bps
    delays = [1, 2]  # bars (1 = next open, 2 = extra delay)

    base = REPO_ROOT / "reports" / "scenarios"
    rows = []

    for fm in fee_mults:
        for sb in slippages:
            for d in delays:
                name = f"fee{fm:g}_slip{sb:g}bps_delay{d}"
                out_dir = base / name
                print(f"\n=== Running {name} ===")
                rows.append(run_one(out_dir, fm, sb, d))

    df = pd.DataFrame(rows).sort_values(["fee_mult", "slippage_bps", "delay_bars"])
    (base / "grid_summary.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    print(f"\n✅ Wrote: {base / 'grid_summary.csv'}")


if __name__ == "__main__":
    main()

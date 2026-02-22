import os

os.environ["PYTHONHASHSEED"] = "0"

import argparse
import hashlib
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

random.seed(0)
np.random.seed(0)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.utils import sha256_file, stable_json  # noqa: E402


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def get_git_info() -> Dict[str, Any]:
    def run(cmd: List[str]) -> str:
        return subprocess.check_output(cmd, cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL).decode().strip()

    info: Dict[str, Any] = {}
    try:
        info["commit"] = run(["git", "rev-parse", "HEAD"])
        info["branch"] = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        porcelain = run(["git", "status", "--porcelain"])
        info["dirty"] = bool(porcelain)
    except Exception:
        info["commit"] = None
        info["branch"] = None
        info["dirty"] = None
    return info


def compute_max_drawdown_pct(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(dd.min() * 100.0)


def compute_sharpe_daily(rets: np.ndarray) -> float:
    # Annualized Sharpe using daily returns (sqrt(365)).
    if rets.size < 2:
        return 0.0
    std = float(np.std(rets, ddof=1))
    if std == 0.0 or np.isnan(std):
        return 0.0
    mean = float(np.mean(rets))
    return float((mean / std) * np.sqrt(365.0))


def compute_cagr_from_dates(initial: float, final: float, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> float:
    if initial <= 0 or final <= 0:
        return 0.0
    days = (end_dt - start_dt).total_seconds() / 86400.0
    if days <= 0:
        return 0.0
    years = days / 365.0
    return float((final / initial) ** (1.0 / years) - 1.0)


def base_metrics_from_daily_equity(daily_equity: pd.Series) -> Dict[str, Any]:
    daily_equity = daily_equity.dropna()
    if daily_equity.size < 3:
        raise ValueError("Not enough daily equity points to compute metrics.")

    initial = float(daily_equity.iloc[0])
    final = float(daily_equity.iloc[-1])

    rets = daily_equity.pct_change().dropna().to_numpy(dtype=float)
    dd = compute_max_drawdown_pct(daily_equity.to_numpy(dtype=float))
    sharpe = compute_sharpe_daily(rets)

    start_dt = pd.to_datetime(daily_equity.index[0], utc=True)
    end_dt = pd.to_datetime(daily_equity.index[-1], utc=True)
    cagr = compute_cagr_from_dates(initial, final, start_dt, end_dt)

    return {
        "initial_equity": initial,
        "final_equity": final,
        "total_return_pct": float((final / initial - 1.0) * 100.0),
        "max_drawdown_pct": float(dd),
        "daily_sharpe": float(sharpe),
        "cagr": float(cagr),
        "days": int((end_dt - start_dt).days),
        "n_daily_returns": int(rets.size),
    }


def block_bootstrap_returns(
    rets: np.ndarray,
    n_samples: int,
    block_len: int,
    seed: int,
) -> np.ndarray:
    """
    Circular block bootstrap on daily returns.

    Returns a matrix of shape (n_samples, N) where N = len(rets).
    """
    if block_len <= 0:
        raise ValueError("block_len must be >= 1")
    N = int(rets.size)
    if N < 2:
        raise ValueError("Need at least 2 daily returns")

    rng = np.random.default_rng(seed)
    k = int(np.ceil(N / block_len))  # number of blocks per sample

    out = np.empty((n_samples, N), dtype=float)

    for s in range(n_samples):
        starts = rng.integers(0, N, size=k)
        chunks = []
        for st in starts:
            en = st + block_len
            if en <= N:
                chunks.append(rets[st:en])
            else:
                # wrap (circular)
                wrap = en - N
                chunks.append(np.concatenate([rets[st:], rets[:wrap]]))
        sample = np.concatenate(chunks)[:N]
        out[s, :] = sample

    return out


def metrics_from_returns_path(
    rets: np.ndarray,
    initial_equity: float,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> Dict[str, Any]:
    """
    Given daily returns array length N, build equity path length N+1 and compute metrics.
    """
    # avoid invalid values (should never happen with normal pct changes)
    one_plus = 1.0 + rets
    if np.any(one_plus <= 0):
        # catastrophic; treat as extremely bad path
        return {
            "total_return_pct": -100.0,
            "max_drawdown_pct": -100.0,
            "daily_sharpe": 0.0,
            "cagr": -1.0,
        }

    eq = initial_equity * np.concatenate([[1.0], np.cumprod(one_plus)])
    final = float(eq[-1])

    dd = compute_max_drawdown_pct(eq)
    sharpe = compute_sharpe_daily(rets)
    cagr = compute_cagr_from_dates(initial_equity, final, start_dt, end_dt)

    return {
        "total_return_pct": float((final / initial_equity - 1.0) * 100.0),
        "max_drawdown_pct": float(dd),
        "daily_sharpe": float(sharpe),
        "cagr": float(cagr),
    }


def ci_from_samples(values: np.ndarray, alpha: float = 0.05) -> Dict[str, float]:
    lo = float(np.quantile(values, alpha / 2.0))
    mid = float(np.quantile(values, 0.5))
    hi = float(np.quantile(values, 1.0 - alpha / 2.0))
    return {"ci_low": lo, "median": mid, "ci_high": hi}


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 3: block bootstrap confidence intervals (daily equity returns)")
    ap.add_argument("--equity-csv", default="reports/baseline/equity_curve.csv")
    ap.add_argument("--results-json", default="reports/baseline/results.json", help="Optional, used for metadata only")
    ap.add_argument("--config", default="config.yaml", help="Optional, used for metadata only")
    ap.add_argument("--out-dir", default="reports/bootstrap")

    ap.add_argument("--n", type=int, default=2000, help="# bootstrap samples")
    ap.add_argument("--block-days", type=int, default=14, help="block length in days")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05, help="CI alpha (0.05 => 95%% CI)")

    args = ap.parse_args()

    equity_path = (REPO_ROOT / args.equity_csv).resolve()
    results_path = (REPO_ROOT / args.results_json).resolve()
    config_path = (REPO_ROOT / args.config).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not equity_path.exists():
        raise FileNotFoundError(f"Equity CSV not found: {equity_path}")

    equity_df = pd.read_csv(equity_path)
    if "timestamp" not in equity_df.columns or "equity" not in equity_df.columns:
        raise ValueError("equity_curve.csv must have columns: timestamp, equity")

    equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], utc=True, errors="coerce")
    equity_df["equity"] = pd.to_numeric(equity_df["equity"], errors="coerce")
    equity_df = equity_df.dropna(subset=["timestamp", "equity"]).sort_values("timestamp")

    # Daily equity series
    daily_equity = equity_df.set_index("timestamp")["equity"].resample("1D").last().dropna()
    if daily_equity.size < 30:
        print("⚠️ Warning: fewer than 30 daily points; CIs may be unreliable.")

    base = base_metrics_from_daily_equity(daily_equity)

    # Returns series for bootstrap
    daily_rets = daily_equity.pct_change().dropna().to_numpy(dtype=float)
    N = int(daily_rets.size)
    if N < 10:
        raise RuntimeError("Not enough daily returns to bootstrap (need >= 10).")

    initial_equity = float(daily_equity.iloc[0])
    start_dt = pd.to_datetime(daily_equity.index[0], utc=True)
    end_dt = pd.to_datetime(daily_equity.index[-1], utc=True)

    samples_rets = block_bootstrap_returns(
        rets=daily_rets,
        n_samples=int(args.n),
        block_len=int(args.block_days),
        seed=int(args.seed),
    )

    # Compute metrics for each sample
    rows = []
    for i in range(samples_rets.shape[0]):
        m = metrics_from_returns_path(samples_rets[i, :], initial_equity, start_dt, end_dt)
        m["sample_id"] = i
        rows.append(m)

    samples_df = pd.DataFrame(rows)
    samples_csv = out_dir / "bootstrap_samples.csv"
    samples_df.to_csv(samples_csv, index=False)

    # CI summary
    ci = {}
    for col in ["total_return_pct", "cagr", "daily_sharpe", "max_drawdown_pct"]:
        vals = pd.to_numeric(samples_df[col], errors="coerce").dropna().to_numpy(dtype=float)
        ci[col] = ci_from_samples(vals, alpha=float(args.alpha))

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "base_metrics": base,
        "bootstrap": {
            "n_samples": int(args.n),
            "block_days": int(args.block_days),
            "seed": int(args.seed),
            "alpha": float(args.alpha),
            "ci": ci,
        },
        "inputs": {
            "equity_csv": str(equity_path),
            "equity_csv_sha256": sha256_file(equity_path),
            "results_json": str(results_path) if results_path.exists() else None,
            "results_json_sha256": sha256_file(results_path) if results_path.exists() else None,
            "config_yaml": str(config_path) if config_path.exists() else None,
            "config_yaml_sha256": sha256_file(config_path) if config_path.exists() else None,
        },
        "git": get_git_info(),
        "platform": {"platform": platform.platform(), "python": sys.version},
        "outputs": {
            "bootstrap_report_json": str(out_dir / "bootstrap_report.json"),
            "bootstrap_samples_csv": str(samples_csv),
        },
    }

    report_path = out_dir / "bootstrap_report.json"
    report_path.write_text(stable_json(report), encoding="utf-8")

    print(f"[OK] Wrote: {report_path}")
    print(f"[OK] Wrote: {samples_csv}")
    print("\nCI summary:")
    for k, v in ci.items():
        print(f"  - {k}: [{v['ci_low']:.4f}, {v['ci_high']:.4f}] (median {v['median']:.4f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import os

# Determinism locks (must be set before Python does much work)
os.environ["PYTHONHASHSEED"] = "0"

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml


# Determinism: RNG seeds
random.seed(0)
np.random.seed(0)


# Ensure repo root is importable when running: python scripts/robustness_table.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategy import add_trend_indicators, generate_orb_signals, identify_orb_ranges  # noqa: E402
from backtester.futures_engine import FuturesEngineConfig, backtest_futures_orb  # noqa: E402
from backtester.risk import risk_limits_from_config  # noqa: E402


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


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


def parse_hhmm(s: str):
    hh, mm = s.strip().split(":")
    return datetime(2000, 1, 1, int(hh), int(mm)).time()


def fmt_hhmm(t) -> str:
    return f"{t.hour:02d}:{t.minute:02d}"


def add_minutes_to_time(t, minutes: int):
    base = datetime(2000, 1, 1, t.hour, t.minute)
    out = base + timedelta(minutes=int(minutes))
    return out.time()


def load_valid_days_csv(path: Path) -> set:
    """Returns a set of datetime.date objects (UTC days) that are valid."""
    if not path.exists():
        raise FileNotFoundError(f"Valid days file not found: {path}. Run: python scripts/build_parquet.py")
    vdf = pd.read_csv(path)
    if "date_utc" not in vdf.columns:
        raise ValueError(f"{path} must contain a 'date_utc' column")
    return set(pd.to_datetime(vdf["date_utc"], utc=True).dt.date)


def compute_max_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    roll_max = equity.cummax()
    dd = (equity / roll_max) - 1.0
    return float(dd.min() * 100.0)


def compute_daily_sharpe(equity_df: pd.DataFrame) -> float:
    """Annualized Sharpe using daily equity returns (sqrt(365))."""
    if equity_df.empty:
        return 0.0
    s = equity_df.set_index("timestamp")["equity"]
    daily = s.resample("1D").last().dropna()
    rets = daily.pct_change().dropna()
    if len(rets) < 2:
        return 0.0
    std = float(rets.std(ddof=1))
    if std == 0.0 or np.isnan(std):
        return 0.0
    return float((rets.mean() / std) * np.sqrt(365.0))


def compute_cagr(equity_df: pd.DataFrame, initial_capital: float) -> float:
    if equity_df.empty or initial_capital <= 0:
        return 0.0
    start_ts = pd.to_datetime(equity_df["timestamp"].iloc[0], utc=True)
    end_ts = pd.to_datetime(equity_df["timestamp"].iloc[-1], utc=True)
    days = (end_ts - start_ts).total_seconds() / 86400.0
    if days <= 0:
        return 0.0
    years = days / 365.0
    final_equity = float(equity_df["equity"].iloc[-1])
    return float((final_equity / initial_capital) ** (1.0 / years) - 1.0)


def maybe_write_equity_plot(equity_df: pd.DataFrame, out_path: Path) -> None:
    """Best-effort plot writer.

    Matplotlib is optional. Some environments have binary wheels mismatches that can emit noisy
    tracebacks on import; we suppress stderr to keep runs clean.
    """
    try:
        import io
        import contextlib

        with contextlib.redirect_stderr(io.StringIO()):
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt  # noqa: F401

        # Plot (suppress any backend noise)
        with contextlib.redirect_stderr(io.StringIO()):
            plt.figure()
            plt.plot(equity_df["timestamp"], equity_df["equity"])
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
    except Exception:
        return



    plt.figure()
    plt.plot(equity_df["timestamp"], equity_df["equity"])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def summarize_run(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    initial_capital: float,
    total_fees: float,
    total_funding: float,
    liquidations: int,
) -> Dict[str, Any]:
    total_trades = int(len(trades_df))
    pnl_col = "pnl_net" if "pnl_net" in trades_df.columns else None

    if total_trades and pnl_col:
        pnl = trades_df[pnl_col]
        wins = int((pnl > 0).sum())
        losses = int((pnl <= 0).sum())
        win_rate = (wins / total_trades) * 100.0
        avg_win = float(pnl[pnl > 0].mean()) if wins else 0.0
        avg_loss = float(pnl[pnl <= 0].mean()) if losses else 0.0
        total_pnl_net = float(pnl.sum())
        expectancy = float(total_pnl_net / total_trades)
    else:
        wins = losses = 0
        win_rate = avg_win = avg_loss = total_pnl_net = expectancy = 0.0

    final_equity = float(equity_df["equity"].iloc[-1]) if not equity_df.empty else float(initial_capital)
    total_return_pct = (final_equity / initial_capital - 1.0) * 100.0 if initial_capital else 0.0
    max_dd = compute_max_drawdown_pct(equity_df["equity"]) if not equity_df.empty else 0.0
    sharpe = compute_daily_sharpe(equity_df)
    cagr = compute_cagr(equity_df, initial_capital)

    return {
        "initial_capital": float(initial_capital),
        "final_equity": float(final_equity),
        "total_return_pct": float(total_return_pct),
        "cagr": float(cagr),
        "max_drawdown_pct": float(max_dd),
        "daily_sharpe": float(sharpe),
        "total_trades": int(total_trades),
        "winning_trades": int(wins),
        "losing_trades": int(losses),
        "win_rate_pct": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "total_pnl_net": float(total_pnl_net),
        "expectancy_per_trade": float(expectancy),
        "total_fees": float(total_fees),
        "total_funding": float(total_funding),
        "liquidations": int(liquidations),
    }


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_time_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 3: robustness table over a small parameter neighborhood (guards against overfitting)."
    )

    ap.add_argument("--config", default="config.yaml")
    ap.add_argument(
        "--data",
        default="",
        help="Processed parquet path. Default: data/processed/<symbol>_<timeframe>.parquet",
    )
    ap.add_argument("--valid-days", default="data/processed/valid_days.csv")
    ap.add_argument("--out-dir", default="reports/robustness")

    # Parameter neighborhood
    ap.add_argument("--adx-threshold-grid", default="35,38,43,48,55")
    ap.add_argument("--orb-start-grid", default="13:00,13:30,14:00")
    ap.add_argument("--orb-window-min", type=int, default=30)
    ap.add_argument("--cutoff-offset-min", type=int, default=0, help="cutoff = orb_end + offset (minutes)")

    # Engine + cost model knobs (defaults match your Phase 2/3 work)
    ap.add_argument("--engine", choices=["spot", "futures"], default="futures")
    ap.add_argument("--fee-mult", type=float, default=1.0)
    ap.add_argument("--slippage-bps", type=float, default=0.0)
    ap.add_argument("--delay-bars", type=int, default=1)

    # Futures knobs
    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--mmr", type=float, default=0.005)
    ap.add_argument("--funding-per-8h", type=float, default=0.0001)

    # Optional range filter
    ap.add_argument("--start", default="", help="Optional ISO start UTC")
    ap.add_argument("--end", default="", help="Optional ISO end UTC (exclusive)")

    # Reporting
    ap.add_argument("--objective", choices=["daily_sharpe", "total_return_pct", "cagr"], default="daily_sharpe")
    ap.add_argument("--max-scenarios", type=int, default=0, help="0 = run all")

    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()

    valid_days_path = Path(args.valid_days)
    if not valid_days_path.is_absolute():
        valid_days_path = (REPO_ROOT / valid_days_path).resolve()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios_dir = out_dir / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    cfg_text = config_path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(cfg_text) or {}

    symbol = str(cfg.get("symbol", "SOLUSDT"))
    timeframe = str(cfg.get("timeframe", "30m"))
    adx_period = int(cfg["adx"]["period"])

    initial_capital = float(cfg["risk"]["initial_capital"])
    position_size = float(cfg["risk"]["position_size"])
    taker_fee_rate = float(cfg["fees"]["taker_fee_rate"])

    # Phase 4 risk controls
    risk_limits = risk_limits_from_config(cfg)

    # Load parquet
    data_path = Path(args.data) if args.data else Path(f"data/processed/{symbol}_{timeframe}.parquet")
    if not data_path.is_absolute():
        data_path = (REPO_ROOT / data_path).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Parquet not found: {data_path}. Run: python scripts/build_parquet.py")

    df = pd.read_parquet(data_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Parquet dataset must have a DatetimeIndex")
    if df.index.tz is None:
        df = df.tz_localize("UTC")

    needed = ["open", "high", "low", "close", "volume"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing required column in parquet: {c}")
    df = df[needed].copy().sort_index()

    if args.start:
        start_ts = pd.to_datetime(args.start, utc=True)
        df = df[df.index >= start_ts]
    if args.end:
        end_ts = pd.to_datetime(args.end, utc=True)
        df = df[df.index < end_ts]

    if df.empty:
        raise RuntimeError("No candles after applying start/end filters.")

    # Valid day list (exactly 48 bars/day)
    valid_days = load_valid_days_csv(valid_days_path)
    print(f"✅ Valid days loaded: {len(valid_days)} ({valid_days_path})")

    # Precompute indicators once
    df_ind = add_trend_indicators(df, period=adx_period)

    adx_threshold_grid = parse_float_list(args.adx_threshold_grid)
    orb_start_grid = parse_time_list(args.orb_start_grid)

    # Engine config
    futures_cfg = FuturesEngineConfig(
        initial_capital=initial_capital,
        position_size=position_size,
        leverage=float(args.leverage),
        taker_fee_rate=taker_fee_rate,
        fee_mult=float(args.fee_mult),
        slippage_bps=float(args.slippage_bps),
        delay_bars=int(args.delay_bars),
        maintenance_margin_rate=float(args.mmr),
        funding_rate_per_8h=float(args.funding_per_8h),
    )

    # Run scenarios
    rows: List[Dict[str, Any]] = []
    outputs: Dict[str, str] = {}
    scenario_count = 0

    for orb_start_str in orb_start_grid:
        orb_start = parse_hhmm(orb_start_str)
        orb_end = add_minutes_to_time(orb_start, int(args.orb_window_min))
        orb_cutoff = add_minutes_to_time(orb_end, int(args.cutoff_offset_min))

        # ORB ranges depend only on orb times
        orb_ranges = identify_orb_ranges(df_ind, orb_start_time=orb_start, orb_end_time=orb_end)
        orb_ranges = orb_ranges.loc[orb_ranges.index.isin(valid_days)]

        for thr in adx_threshold_grid:
            scenario_count += 1
            if args.max_scenarios and scenario_count > int(args.max_scenarios):
                break

            sid = f"adx{thr:g}_orb{orb_start.hour:02d}{orb_start.minute:02d}".replace(".", "p")
            sdir = scenarios_dir / sid
            sdir.mkdir(parents=True, exist_ok=True)

            # Signals for this threshold + cutoff
            df_sig = generate_orb_signals(
                df_ind,
                orb_ranges=orb_ranges,
                adx_threshold=float(thr),
                orb_cutoff_time=orb_cutoff,
            )

            # Zero out signals on invalid days (keeps candles but blocks entries)
            invalid_mask = ~df_sig["date"].isin(valid_days)
            df_sig.loc[invalid_mask, "signal"] = 0
            df_sig.loc[invalid_mask, "signal_type"] = ""

            # Helpful counts
            signal_counts = df_sig.loc[df_sig["signal"] != 0, "signal_type"].value_counts(dropna=False).to_dict()
            signals_total = int((df_sig["signal"] != 0).sum())

            # Run engine
            total_fees = 0.0
            total_funding = 0.0
            liquidations = 0
            engine_stats: Dict[str, Any] = {}

            if args.engine == "spot":
                from scripts.spot_engine import backtest_orb_strategy  # type: ignore

                trades, equity_curve, final_capital, total_fees = backtest_orb_strategy(
                    df=df_sig,
                    orb_ranges=orb_ranges,
                    initial_capital=initial_capital,
                    position_size=position_size,
                    taker_fee_rate=taker_fee_rate,
                    valid_days=valid_days,
                    fee_mult=float(args.fee_mult),
                    slippage_bps=float(args.slippage_bps),
                    delay_bars=int(args.delay_bars),
                )
                engine_stats = {
                    "final_capital": float(final_capital),
                    "total_fees": float(total_fees),
                    "total_funding": 0.0,
                    "liquidations": 0,
                }
            else:
                trades, equity_curve, stats = backtest_futures_orb(
                    df=df_sig,
                    orb_ranges=orb_ranges,
                    valid_days=valid_days,
                    cfg=futures_cfg,
                    risk_limits=risk_limits,
                )
                total_fees = float(stats.get("total_fees", 0.0))
                total_funding = float(stats.get("total_funding", 0.0))
                liquidations = int(stats.get("liquidations", 0))
                engine_stats = stats

            trades_df = pd.DataFrame(trades)
            equity_df = pd.DataFrame({"timestamp": df_sig.index, "equity": equity_curve})

            metrics = summarize_run(
                trades_df=trades_df,
                equity_df=equity_df,
                initial_capital=initial_capital,
                total_fees=total_fees,
                total_funding=total_funding,
                liquidations=liquidations,
            )

            # Write scenario artifacts
            trades_path = sdir / "trades.csv"
            equity_path = sdir / "equity_curve.csv"
            results_path = sdir / "results.json"
            plot_path = sdir / "equity_curve.png"

            trades_df.to_csv(trades_path, index=False)
            equity_df.to_csv(equity_path, index=False)
            maybe_write_equity_plot(equity_df, plot_path)

            result_obj = {
                "scenario_id": sid,
                "symbol": symbol,
                "timeframe": timeframe,
                "range": {
                    "start": equity_df["timestamp"].min().isoformat(),
                    "end": equity_df["timestamp"].max().isoformat(),
                    "candles": int(len(equity_df)),
                },
                "params": {
                    "adx_period": int(adx_period),
                    "adx_threshold": float(thr),
                    "orb_start": fmt_hhmm(orb_start),
                    "orb_end": fmt_hhmm(orb_end),
                    "orb_cutoff": fmt_hhmm(orb_cutoff),
                    "orb_window_min": int(args.orb_window_min),
                    "cutoff_offset_min": int(args.cutoff_offset_min),
                    "engine": args.engine,
                    "initial_capital": float(initial_capital),
                    "position_size": float(position_size),
                    "taker_fee_rate": float(taker_fee_rate),
                    "fee_mult": float(args.fee_mult),
                    "slippage_bps": float(args.slippage_bps),
                    "delay_bars": int(args.delay_bars),
                    "leverage": float(args.leverage),
                    "mmr": float(args.mmr),
                    "funding_per_8h": float(args.funding_per_8h),
                },
                "signals": {
                    "signals_total": int(signals_total),
                    "signal_type_counts": signal_counts,
                },
                "engine_stats": engine_stats,
                "metrics": metrics,
            }
            results_path.write_text(stable_json(result_obj), encoding="utf-8")

            # Summary row
            row = {
                "scenario_id": sid,
                "adx_threshold": float(thr),
                "orb_start": fmt_hhmm(orb_start),
                "orb_end": fmt_hhmm(orb_end),
                "orb_cutoff": fmt_hhmm(orb_cutoff),
                "signals_total": int(signals_total),
                **metrics,
            }
            rows.append(row)

            outputs[f"scenarios/{sid}/results.json"] = str(results_path)
            outputs[f"scenarios/{sid}/trades.csv"] = str(trades_path)
            outputs[f"scenarios/{sid}/equity_curve.csv"] = str(equity_path)
            if plot_path.exists():
                outputs[f"scenarios/{sid}/equity_curve.png"] = str(plot_path)

            print(
                f"✅ {sid} | sharpe={metrics['daily_sharpe']:.2f} | ret={metrics['total_return_pct']:.1f}% | dd={metrics['max_drawdown_pct']:.1f}%"
            )

        if args.max_scenarios and scenario_count >= int(args.max_scenarios):
            break

    # Write robustness table
    table_df = pd.DataFrame(rows)
    if table_df.empty:
        raise RuntimeError("No scenarios ran. Check your grids / filters.")

    table_df = table_df.sort_values(["adx_threshold", "orb_start"]).reset_index(drop=True)
    table_csv = out_dir / "robustness_table.csv"
    table_df.to_csv(table_csv, index=False)
    outputs["robustness_table.csv"] = str(table_csv)

    # Summary (best/median/worst for the chosen objective)
    obj = str(args.objective)
    if obj not in table_df.columns:
        raise ValueError(f"Objective '{obj}' not in table columns")

    best_row = table_df.sort_values(obj, ascending=False).iloc[0].to_dict()
    worst_row = table_df.sort_values(obj, ascending=True).iloc[0].to_dict()
    median_row = table_df.sort_values(obj).iloc[len(table_df) // 2].to_dict()

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "objective": obj,
        "scenario_count": int(len(table_df)),
        "best": best_row,
        "median": median_row,
        "worst": worst_row,
        "notes": {
            "interpretation": "If 'best' is much better than 'median' and 'worst', you may be overfitting/cherry-picking. We want a strong median and not-too-bad worst.",
        },
    }

    summary_path = out_dir / "robustness_summary.json"
    summary_path.write_text(stable_json(summary), encoding="utf-8")
    outputs["robustness_summary.json"] = str(summary_path)

    # Repro metadata + hashes
    hashes = {k: sha256_file(Path(v)) for k, v in outputs.items() if Path(v).exists()}
    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git": get_git_info(),
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": {"platform": platform.platform()},
        "inputs": {
            "config_path": str(config_path),
            "config_sha256": sha256_bytes(cfg_text.encode("utf-8")),
            "data_path": str(data_path),
            "data_sha256": sha256_file(data_path),
            "valid_days_path": str(valid_days_path),
            "valid_days_sha256": sha256_file(valid_days_path),
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "grids": {
            "adx_threshold_grid": adx_threshold_grid,
            "orb_start_grid": orb_start_grid,
            "orb_window_min": int(args.orb_window_min),
            "cutoff_offset_min": int(args.cutoff_offset_min),
        },
        "assumptions": {
            "engine": args.engine,
            "fee_mult": float(args.fee_mult),
            "slippage_bps": float(args.slippage_bps),
            "delay_bars": int(args.delay_bars),
            "leverage": float(args.leverage),
            "mmr": float(args.mmr),
            "funding_per_8h": float(args.funding_per_8h),
        },
        "outputs": outputs,
        "output_sha256": hashes,
    }

    meta_path = out_dir / "run_metadata.json"
    meta_path.write_text(stable_json(meta), encoding="utf-8")

    print(f"\n✅ Wrote: {table_csv}")
    print(f"✅ Wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

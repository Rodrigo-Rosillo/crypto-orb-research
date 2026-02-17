import os

# Determinism locks
os.environ["PYTHONHASHSEED"] = "0"

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

# Determinism: RNG seeds
random.seed(0)
np.random.seed(0)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.utils import load_valid_days_csv, parse_hhmm, sha256_file  # noqa: E402
from strategy import add_trend_indicators, generate_orb_signals, identify_orb_ranges
from backtester.futures_engine import FuturesEngineConfig, backtest_futures_orb
from backtester.risk import risk_limits_from_config


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


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


def summarize_run(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    initial_capital: float,
    total_fees: float,
    total_funding: float,
    liquidations: int,
) -> Dict[str, Any]:
    total_trades = int(len(trades_df))
    pnl_col = None
    if "pnl_net" in trades_df.columns:
        pnl_col = "pnl_net"
    elif "pnl" in trades_df.columns:
        pnl_col = "pnl"

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
        "total_trades": total_trades,
        "winning_trades": wins,
        "losing_trades": losses,
        "win_rate_pct": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "total_pnl_net": float(total_pnl_net),
        "expectancy_per_trade": float(expectancy),
        "total_fees": float(total_fees),
        "total_funding": float(total_funding),
        "liquidations": int(liquidations),
    }


def generate_folds(
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    train_months: int,
    test_months: int,
    step_months: int,
) -> List[Dict[str, pd.Timestamp]]:
    """Generate rolling walk-forward folds with month-based windows."""
    folds: List[Dict[str, pd.Timestamp]] = []
    cursor = start_utc.normalize()

    while True:
        train_start = cursor
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)

        if test_end > end_utc:
            break

        folds.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
        cursor = cursor + pd.DateOffset(months=step_months)

    return folds


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 3: walk-forward evaluation (rolling train/test windows)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data", default="data/processed/SOLUSDT_30m.parquet", help="Processed parquet dataset")
    ap.add_argument("--valid-days", default="data/processed/valid_days.csv")
    ap.add_argument("--out-dir", default="reports/walk_forward")

    ap.add_argument("--engine", choices=["futures", "spot"], default="futures")

    # Walk-forward windowing
    ap.add_argument("--train-months", type=int, default=24)
    ap.add_argument("--test-months", type=int, default=6)
    ap.add_argument("--step-months", type=int, default=6)
    ap.add_argument("--start", default="", help="Optional ISO date/time (UTC) to start folds")
    ap.add_argument("--end", default="", help="Optional ISO date/time (UTC) to end folds")

    # Execution assumptions (Phase 2)
    ap.add_argument("--fee-mult", type=float, default=1.0)
    ap.add_argument("--slippage-bps", type=float, default=0.0)
    ap.add_argument("--delay-bars", type=int, default=1)

    # Futures-specific (locked baseline defaults)
    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--mmr", type=float, default=0.005)
    ap.add_argument("--funding-per-8h", type=float, default=0.0001)

    # Optional tiny tuning (kept off by default)
    ap.add_argument(
        "--tune-adx-threshold",
        default="",
        help="Comma list like '38,43,48'. If provided, chooses best ADX threshold on TRAIN, reports TEST.",
    )

    args = ap.parse_args()

    config_path = (REPO_ROOT / args.config).resolve()
    data_path = (REPO_ROOT / args.data).resolve()
    valid_days_path = (REPO_ROOT / args.valid_days).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    folds_dir = out_dir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    cfg_text = config_path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(cfg_text) or {}

    symbol = str(cfg.get("symbol", "SOLUSDT"))
    timeframe = str(cfg.get("timeframe", "30m"))

    orb_start = parse_hhmm(cfg["orb"]["start"])
    orb_end = parse_hhmm(cfg["orb"]["end"])
    orb_cutoff = parse_hhmm(cfg["orb"]["cutoff"])

    adx_period = int(cfg["adx"]["period"])
    base_adx_threshold = float(cfg["adx"]["threshold"])

    initial_capital = float(cfg["risk"]["initial_capital"])
    position_size = float(cfg["risk"]["position_size"])
    taker_fee_rate = float(cfg["fees"]["taker_fee_rate"])

    # Phase 4 risk controls
    risk_limits = risk_limits_from_config(cfg)

    valid_days_all = load_valid_days_csv(valid_days_path)

    # Load parquet (requires pyarrow)
    df = pd.read_parquet(data_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Parquet dataset must have a DatetimeIndex")
    if df.index.tz is None:
        df = df.tz_localize("UTC")

    needed_cols = ["open", "high", "low", "close", "volume"]
    for c in needed_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column in parquet: {c}")
    df = df[needed_cols].copy().sort_index()

    start_ts = pd.to_datetime(args.start, utc=True) if args.start else df.index.min()
    end_ts = pd.to_datetime(args.end, utc=True) if args.end else df.index.max()

    # Precompute indicators + ORB ranges once
    df_ind = add_trend_indicators(df, period=adx_period)
    orb_ranges_all = identify_orb_ranges(df_ind, orb_start_time=orb_start, orb_end_time=orb_end)

    tune_list: List[float] = []
    if args.tune_adx_threshold.strip():
        tune_list = [float(x.strip()) for x in args.tune_adx_threshold.split(",") if x.strip()]
        tune_list = sorted(set(tune_list))

    folds = generate_folds(start_ts, end_ts, args.train_months, args.test_months, args.step_months)
    if not folds:
        raise RuntimeError("No folds generated. Try smaller train/test months or adjust --start/--end.")

    spot_engine = None
    if args.engine == "spot":
        from scripts.spot_engine import backtest_orb_strategy  # type: ignore

        spot_engine = backtest_orb_strategy

    summary_rows: List[Dict[str, Any]] = []

    for i, f in enumerate(folds, start=1):
        fold_id = f"fold_{i:02d}"
        fold_out = folds_dir / fold_id
        fold_out.mkdir(parents=True, exist_ok=True)

        train_start, train_end = f["train_start"], f["train_end"]
        test_start, test_end = f["test_start"], f["test_end"]

        df_train = df_ind[(df_ind.index >= train_start) & (df_ind.index < train_end)]
        df_test = df_ind[(df_ind.index >= test_start) & (df_ind.index < test_end)]

        train_days = set(pd.Series(df_train.index.normalize().date).unique())
        test_days = set(pd.Series(df_test.index.normalize().date).unique())

        valid_train_days = valid_days_all.intersection(train_days)
        valid_test_days = valid_days_all.intersection(test_days)

        orb_train = orb_ranges_all.loc[orb_ranges_all.index.isin(valid_train_days)]
        orb_test = orb_ranges_all.loc[orb_ranges_all.index.isin(valid_test_days)]

        chosen_adx = base_adx_threshold
        tune_table: List[Dict[str, Any]] = []

        if tune_list:
            best_score = -1e18
            for thr in tune_list:
                train_sig = generate_orb_signals(
                    df_train,
                    orb_ranges=orb_train,
                    adx_threshold=float(thr),
                    orb_cutoff_time=orb_cutoff,
                )
                invalid_mask = ~train_sig["date"].isin(valid_train_days)
                train_sig.loc[invalid_mask, "signal"] = 0
                train_sig.loc[invalid_mask, "signal_type"] = ""

                if args.engine == "futures":
                    cfg_eng = FuturesEngineConfig(
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
                    trades, equity_curve, stats = backtest_futures_orb(
                        df=train_sig,
                        orb_ranges=orb_train,
                        valid_days=valid_train_days,
                        cfg=cfg_eng,
                        risk_limits=risk_limits,
                    )
                    trades_df = pd.DataFrame(trades)
                    equity_df = pd.DataFrame({"timestamp": train_sig.index, "equity": equity_curve})
                    m = summarize_run(
                        trades_df,
                        equity_df,
                        initial_capital,
                        total_fees=float(stats.get("total_fees", 0.0)),
                        total_funding=float(stats.get("total_funding", 0.0)),
                        liquidations=int(stats.get("liquidations", 0)),
                    )
                else:
                    assert spot_engine is not None
                    trades, equity_curve, _, total_fees = spot_engine(
                        df=train_sig,
                        orb_ranges=orb_train,
                        initial_capital=initial_capital,
                        position_size=position_size,
                        taker_fee_rate=taker_fee_rate,
                        valid_days=valid_train_days,
                        fee_mult=float(args.fee_mult),
                        slippage_bps=float(args.slippage_bps),
                        delay_bars=int(args.delay_bars),
                    )
                    trades_df = pd.DataFrame(trades)
                    equity_df = pd.DataFrame({"timestamp": train_sig.index, "equity": equity_curve})
                    m = summarize_run(trades_df, equity_df, initial_capital, float(total_fees), 0.0, 0)

                score = float(m["total_return_pct"])
                tune_table.append({"adx_threshold": thr, "score_train_total_return_pct": score})
                if score > best_score:
                    best_score = score
                    chosen_adx = float(thr)

            (fold_out / "tuning_train.json").write_text(
                stable_json({"tune": tune_table, "chosen_adx": chosen_adx}), encoding="utf-8"
            )

        test_sig = generate_orb_signals(
            df_test,
            orb_ranges=orb_test,
            adx_threshold=float(chosen_adx),
            orb_cutoff_time=orb_cutoff,
        )
        invalid_mask = ~test_sig["date"].isin(valid_test_days)
        test_sig.loc[invalid_mask, "signal"] = 0
        test_sig.loc[invalid_mask, "signal_type"] = ""

        total_fees = 0.0
        total_funding = 0.0
        liquidations = 0
        engine_stats: Dict[str, Any] = {}

        if args.engine == "futures":
            cfg_eng = FuturesEngineConfig(
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
            trades, equity_curve, stats = backtest_futures_orb(
                df=test_sig,
                orb_ranges=orb_test,
                valid_days=valid_test_days,
                cfg=cfg_eng,
                risk_limits=risk_limits,
            )
            total_fees = float(stats.get("total_fees", 0.0))
            total_funding = float(stats.get("total_funding", 0.0))
            liquidations = int(stats.get("liquidations", 0))
            engine_stats = stats
        else:
            assert spot_engine is not None
            trades, equity_curve, _, total_fees = spot_engine(
                df=test_sig,
                orb_ranges=orb_test,
                initial_capital=initial_capital,
                position_size=position_size,
                taker_fee_rate=taker_fee_rate,
                valid_days=valid_test_days,
                fee_mult=float(args.fee_mult),
                slippage_bps=float(args.slippage_bps),
                delay_bars=int(args.delay_bars),
            )

        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame({"timestamp": test_sig.index, "equity": equity_curve})
        metrics = summarize_run(trades_df, equity_df, initial_capital, total_fees, total_funding, liquidations)

        trades_df.to_csv(fold_out / "trades.csv", index=False)
        equity_df.to_csv(fold_out / "equity_curve.csv", index=False)
        (fold_out / "results.json").write_text(
            stable_json(
                {
                    "fold_id": fold_id,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "engine": args.engine,
                    "window": {
                        "train_start": train_start.isoformat(),
                        "train_end": train_end.isoformat(),
                        "test_start": test_start.isoformat(),
                        "test_end": test_end.isoformat(),
                    },
                    "chosen_adx_threshold": float(chosen_adx),
                    "params": {
                        "fee_mult": float(args.fee_mult),
                        "slippage_bps": float(args.slippage_bps),
                        "delay_bars": int(args.delay_bars),
                        "leverage": float(args.leverage),
                        "mmr": float(args.mmr),
                        "funding_per_8h": float(args.funding_per_8h),
                    },
                    "metrics": metrics,
                    "engine_stats": engine_stats,
                }
            ),
            encoding="utf-8",
        )

        summary_rows.append(
            {
                "fold_id": fold_id,
                "train_start": train_start.date().isoformat(),
                "train_end": (train_end - pd.Timedelta(seconds=1)).date().isoformat(),
                "test_start": test_start.date().isoformat(),
                "test_end": (test_end - pd.Timedelta(seconds=1)).date().isoformat(),
                "chosen_adx_threshold": float(chosen_adx),
                **metrics,
                "fold_dir": str(fold_out),
            }
        )

        print(
            f"✅ {fold_id} TEST {test_start.date()}→{(test_end - pd.Timedelta(days=1)).date()} | "
            f"ret={metrics['total_return_pct']:.2f}% dd={metrics['max_drawdown_pct']:.2f}% "
            f"sharpe={metrics['daily_sharpe']:.2f} trades={metrics['total_trades']}"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "walk_forward_folds.csv"
    summary_df.to_csv(summary_csv, index=False)

    agg: Dict[str, Any] = {}
    if not summary_df.empty:
        for col in [
            "total_return_pct",
            "max_drawdown_pct",
            "daily_sharpe",
            "cagr",
            "total_trades",
            "win_rate_pct",
            "expectancy_per_trade",
        ]:
            if col in summary_df.columns:
                s = pd.to_numeric(summary_df[col], errors="coerce").dropna()
                if not s.empty:
                    agg[col] = {
                        "mean": float(s.mean()),
                        "median": float(s.median()),
                        "p25": float(s.quantile(0.25)),
                        "p75": float(s.quantile(0.75)),
                        "min": float(s.min()),
                        "max": float(s.max()),
                    }

        agg["folds"] = int(len(summary_df))
        agg["pct_folds_positive_return"] = float((summary_df["total_return_pct"] > 0).mean() * 100.0)

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
        },
        "params": {
            "engine": args.engine,
            "train_months": int(args.train_months),
            "test_months": int(args.test_months),
            "step_months": int(args.step_months),
            "start": args.start,
            "end": args.end,
            "fee_mult": float(args.fee_mult),
            "slippage_bps": float(args.slippage_bps),
            "delay_bars": int(args.delay_bars),
            "leverage": float(args.leverage),
            "mmr": float(args.mmr),
            "funding_per_8h": float(args.funding_per_8h),
            "tune_adx_threshold": tune_list,
        },
        "outputs": {
            "folds_csv": str(summary_csv),
            "folds_dir": str(folds_dir),
        },
        "aggregate": agg,
    }
    (out_dir / "walk_forward_metadata.json").write_text(stable_json(meta), encoding="utf-8")

    html_lines = [
        "<html><head><meta charset='utf-8'><title>Walk Forward Report</title></head><body>",
        "<h1>Walk Forward Report</h1>",
        f"<p><b>Symbol:</b> {symbol} | <b>Timeframe:</b> {timeframe} | <b>Engine:</b> {args.engine}</p>",
        f"<p><b>Windows:</b> train={args.train_months}m test={args.test_months}m step={args.step_months}m</p>",
        f"<p><b>Folds:</b> {agg.get('folds', 0)} | <b>% positive folds:</b> {agg.get('pct_folds_positive_return', 0):.2f}%</p>",
        "<h2>Aggregate (distribution)</h2>",
        "<pre>" + stable_json(agg) + "</pre>",
        "<h2>Per-fold</h2>",
        "<p>See CSV: walk_forward_folds.csv and each fold folder under reports/walk_forward/folds/</p>",
        "</body></html>",
    ]
    (out_dir / "walk_forward_report.html").write_text("\n".join(html_lines), encoding="utf-8")

    print(f"\n✅ Walk-forward written to: {out_dir}")
    print("  - walk_forward_folds.csv")
    print("  - walk_forward_report.html")
    print("  - walk_forward_metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

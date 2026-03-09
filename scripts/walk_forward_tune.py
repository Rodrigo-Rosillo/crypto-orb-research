import os

# Determinism locks (must be set before Python does much work)
os.environ["PYTHONHASHSEED"] = "0"

import argparse
import hashlib
import platform
import random
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone, time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

# Determinism: RNG seeds
random.seed(0)
np.random.seed(0)

# Ensure repo root is importable when running: python scripts/walk_forward_tune.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.utils import load_valid_days_csv, parse_hhmm, sha256_file, stable_json  # noqa: E402
from strategy import SignalRule, add_trend_indicators, build_signals_from_ruleset, load_signal_rules_from_config  # noqa: E402
from backtester.futures_engine import FuturesEngineConfig, backtest_futures_orb  # noqa: E402
from backtester.risk import risk_limits_from_config  # noqa: E402


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def get_git_info() -> Dict[str, Any]:
    def run(cmd: List[str]) -> str:
        return subprocess.check_output(cmd, cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL).decode().strip()

    info: Dict[str, Any] = {}
    try:
        info["commit"] = run(["git", "rev-parse", "HEAD"])
        info["branch"] = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        info["dirty"] = bool(run(["git", "status", "--porcelain"]))
    except Exception:
        info["commit"] = None
        info["branch"] = None
        info["dirty"] = None
    return info


def fmt_hhmm(t: time) -> str:
    return f"{t.hour:02d}:{t.minute:02d}"


def add_minutes_to_time(t: time, minutes: int) -> time:
    base = datetime(2000, 1, 1, t.hour, t.minute, tzinfo=timezone.utc)
    out = base + timedelta(minutes=int(minutes))
    return time(out.hour, out.minute)


def single_rule_orb_ranges(rule_orb_ranges_df: pd.DataFrame) -> pd.DataFrame:
    orb_out = rule_orb_ranges_df.loc[:, ["date", "orb_high", "orb_low"]].copy()
    orb_out = orb_out.drop_duplicates(subset=["date"]).set_index("date").sort_index()
    return orb_out


def build_rule_variant(
    base_rule: SignalRule,
    *,
    adx_threshold: float,
    orb_start: time,
    orb_window_min: int,
    cutoff_offset_min: int,
) -> SignalRule:
    orb_end = add_minutes_to_time(orb_start, int(orb_window_min))
    orb_cutoff = add_minutes_to_time(orb_end, int(cutoff_offset_min))
    return replace(
        base_rule,
        adx_threshold=float(adx_threshold),
        orb_start=orb_start,
        orb_end=orb_end,
        orb_cutoff=orb_cutoff,
    )


def slice_by_date_inclusive(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Slice by UTC date inclusive. start_date/end_date are 'YYYY-MM-DD'.
    """
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_excl = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
    return df[(df.index >= start_ts) & (df.index < end_excl)].copy()


def compute_max_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    roll_max = equity.cummax()
    dd = (equity / roll_max) - 1.0
    return float(dd.min() * 100.0)


def compute_daily_sharpe(equity_df: pd.DataFrame) -> float:
    """
    Annualized Sharpe using daily equity returns (sqrt(365)).
    """
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
    pnl_col = "pnl_net" if "pnl_net" in trades_df.columns else ("pnl" if "pnl" in trades_df.columns else None)

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


def run_backtest_one(
    df_ind_slice: pd.DataFrame,
    valid_days_slice: set,
    base_rule: SignalRule,
    adx_threshold: float,
    orb_start: time,
    orb_window_min: int,
    cutoff_offset_min: int,
    engine: str,
    futures_cfg: FuturesEngineConfig,
    initial_capital: float,
    position_size: float,
    taker_fee_rate: float,
    fee_mult: float,
    slippage_bps: float,
    delay_bars: int,
    risk_limits,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Returns (trades_df, equity_df, metrics, extra)
    extra includes orb params + signals counts.
    """
    scenario_rule = build_rule_variant(
        base_rule,
        adx_threshold=float(adx_threshold),
        orb_start=orb_start,
        orb_window_min=int(orb_window_min),
        cutoff_offset_min=int(cutoff_offset_min),
    )
    df_sig, rule_orb_ranges_df, scenario_rules = build_signals_from_ruleset(df_ind_slice, [scenario_rule], valid_days_slice)
    effective_rule = scenario_rules[0]
    orb_ranges = single_rule_orb_ranges(rule_orb_ranges_df)

    signals_total = int((df_sig["signal"] != 0).sum())
    signal_counts = df_sig.loc[df_sig["signal"] != 0, "signal_type"].value_counts(dropna=False).to_dict()

    total_fees = 0.0
    total_funding = 0.0
    liquidations = 0

    if engine == "spot":
        from backtester.spot_engine import backtest_orb_strategy  # type: ignore

        trades, equity_curve, final_capital, total_fees = backtest_orb_strategy(
            df=df_sig,
            orb_ranges=orb_ranges,
            initial_capital=initial_capital,
            position_size=position_size,
            taker_fee_rate=taker_fee_rate,
            valid_days=valid_days_slice,
            fee_mult=float(fee_mult),
            slippage_bps=float(slippage_bps),
            delay_bars=int(delay_bars),
        )
        extra_engine = {
            "final_capital": float(final_capital),
            "total_fees": float(total_fees),
            "total_funding": 0.0,
            "liquidations": 0,
        }
    else:
        trades, equity_curve, stats = backtest_futures_orb(
            df=df_sig,
            orb_ranges=orb_ranges,
            valid_days=valid_days_slice,
            cfg=futures_cfg,
            risk_limits=risk_limits,
        )
        total_fees = float(stats.get("total_fees", 0.0))
        total_funding = float(stats.get("total_funding", 0.0))
        liquidations = int(stats.get("liquidations", 0))
        extra_engine = stats

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

    extra = {
        "orb_start": fmt_hhmm(effective_rule.orb_start),
        "orb_end": fmt_hhmm(effective_rule.orb_end),
        "orb_cutoff": fmt_hhmm(effective_rule.orb_cutoff),
        "signals_total": signals_total,
        "signal_type_counts": signal_counts,
        "engine_stats": extra_engine,
    }
    return trades_df, equity_df, metrics, extra


def choose_best(df_grid: pd.DataFrame, objective: str, min_trades: int) -> Dict[str, Any]:
    """
    Choose best scenario by objective with sensible tie-breakers.
    """
    if objective not in df_grid.columns:
        raise ValueError(f"Objective '{objective}' not in grid columns")

    stable = df_grid[df_grid["total_trades"] >= int(min_trades)].copy()
    if stable.empty:
        stable = df_grid.copy()

    # Note: max_drawdown_pct is negative; higher (=less negative) is better
    stable = stable.sort_values(
        by=[objective, "total_return_pct", "max_drawdown_pct", "total_trades"],
        ascending=[False, False, False, False],
    )
    return stable.iloc[0].to_dict()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Nested walk-forward tuning: pick params on TRAIN only, evaluate on TEST."
    )

    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--folds-csv", default="reports/walk_forward/walk_forward_folds.csv")
    ap.add_argument("--out-dir", default="reports/walk_forward_tune")

    ap.add_argument(
        "--data",
        default="",
        help="Processed parquet path. Default: data/processed/<symbol>_<timeframe>.parquet",
    )
    ap.add_argument("--valid-days", default="data/processed/valid_days.csv")

    # Grid
    ap.add_argument("--adx-threshold-grid", default="35,38,43,48,55")
    ap.add_argument("--orb-start-grid", default="13:00,13:30,14:00")
    ap.add_argument("--orb-window-min", type=int, default=30)
    ap.add_argument("--cutoff-offset-min", type=int, default=0)

    # Objective / selection guard
    ap.add_argument("--objective", choices=["daily_sharpe", "total_return_pct", "cagr"], default="daily_sharpe")
    ap.add_argument("--min-trades", type=int, default=20)

    # Engine + costs
    ap.add_argument("--engine", choices=["spot", "futures"], default="futures")
    ap.add_argument("--fee-mult", type=float, default=1.0)
    ap.add_argument("--slippage-bps", type=float, default=0.0)
    ap.add_argument("--delay-bars", type=int, default=1)

    # Futures knobs
    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--mmr", type=float, default=0.005)
    ap.add_argument("--funding-per-8h", type=float, default=0.0001)

    # Save train artifacts for the selected scenario
    ap.add_argument("--save-train-chosen", action="store_true")

    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()

    folds_path = Path(args.folds_csv)
    if not folds_path.is_absolute():
        folds_path = (REPO_ROOT / folds_path).resolve()
    if not folds_path.exists():
        raise FileNotFoundError(f"Folds CSV not found: {folds_path}")

    valid_days_path = Path(args.valid_days)
    if not valid_days_path.is_absolute():
        valid_days_path = (REPO_ROOT / valid_days_path).resolve()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "folds").mkdir(parents=True, exist_ok=True)

    cfg_text = config_path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(cfg_text) or {}
    rules = load_signal_rules_from_config(cfg)
    if len(rules) != 1:
        raise ValueError(
            "walk_forward_tune currently supports configs that resolve to exactly one signal rule."
        )
    base_rule = rules[0]

    symbol = str(cfg.get("symbol", "SOLUSDT"))
    timeframe = str(cfg.get("timeframe", "30m"))
    adx_period = int(cfg["adx"]["period"])

    initial_capital = float(cfg["risk"]["initial_capital"])
    position_size = float(cfg["risk"]["position_size"])
    taker_fee_rate = float(cfg["fees"]["taker_fee_rate"])

    # Phase 4 risk controls (parsed here for consistency with run config).
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

    valid_days_all = load_valid_days_csv(valid_days_path)

    # Precompute indicators once
    df_ind = add_trend_indicators(df, period=adx_period)

    adx_grid = parse_float_list(args.adx_threshold_grid)
    orb_start_grid = parse_time_list(args.orb_start_grid)

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

    folds_df = pd.read_csv(folds_path)
    required_cols = ["fold_id", "train_start", "train_end", "test_start", "test_end"]
    for c in required_cols:
        if c not in folds_df.columns:
            raise ValueError(f"folds CSV must include column: {c}")

    fold_summaries: List[Dict[str, Any]] = []
    selection_counter: Dict[str, int] = {}

    for _, row in folds_df.iterrows():
        fold_id = str(row["fold_id"])
        train_start = str(row["train_start"])
        train_end = str(row["train_end"])
        test_start = str(row["test_start"])
        test_end = str(row["test_end"])

        fold_dir = out_dir / "folds" / fold_id
        fold_dir.mkdir(parents=True, exist_ok=True)

        df_train = slice_by_date_inclusive(df_ind, train_start, train_end)
        df_test = slice_by_date_inclusive(df_ind, test_start, test_end)

        train_days = set(pd.date_range(train_start, train_end, freq="D", tz="UTC").date)
        test_days = set(pd.date_range(test_start, test_end, freq="D", tz="UTC").date)
        valid_train = valid_days_all.intersection(train_days)
        valid_test = valid_days_all.intersection(test_days)

        # TRAIN grid search
        grid_rows: List[Dict[str, Any]] = []
        for orb_start_str in orb_start_grid:
            orb_start = parse_hhmm(orb_start_str)
            for thr in adx_grid:
                scenario_id = f"adx{thr:g}_orb{orb_start.hour:02d}{orb_start.minute:02d}".replace(".", "p")

                _, _, metrics, extra = run_backtest_one(
                    df_ind_slice=df_train,
                    valid_days_slice=valid_train,
                    base_rule=base_rule,
                    adx_threshold=float(thr),
                    orb_start=orb_start,
                    orb_window_min=int(args.orb_window_min),
                    cutoff_offset_min=int(args.cutoff_offset_min),
                    engine=args.engine,
                    futures_cfg=futures_cfg,
                    initial_capital=initial_capital,
                    position_size=position_size,
                    taker_fee_rate=taker_fee_rate,
                    fee_mult=float(args.fee_mult),
                    slippage_bps=float(args.slippage_bps),
                    delay_bars=int(args.delay_bars),
                    risk_limits=risk_limits,
                )

                grid_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "adx_threshold": float(thr),
                        "orb_start": extra["orb_start"],
                        "orb_end": extra["orb_end"],
                        "orb_cutoff": extra["orb_cutoff"],
                        "signals_total": int(extra["signals_total"]),
                        **metrics,
                    }
                )

        grid_df = pd.DataFrame(grid_rows).sort_values(by=["orb_start", "adx_threshold"]).reset_index(drop=True)
        grid_df.to_csv(fold_dir / "train_grid.csv", index=False)

        chosen = choose_best(grid_df, objective=str(args.objective), min_trades=int(args.min_trades))
        chosen_id = str(chosen["scenario_id"])
        selection_counter[chosen_id] = selection_counter.get(chosen_id, 0) + 1

        (fold_dir / "selection.json").write_text(
            stable_json(
                {
                    "fold_id": fold_id,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "objective": str(args.objective),
                    "min_trades": int(args.min_trades),
                    "chosen": chosen,
                }
            ),
            encoding="utf-8",
        )

        chosen_thr = float(chosen["adx_threshold"])
        chosen_orb_start = parse_hhmm(str(chosen["orb_start"]))

        # Optional: save chosen TRAIN artifacts
        if args.save_train_chosen:
            tr_trades_df, tr_equity_df, tr_metrics, tr_extra = run_backtest_one(
                df_ind_slice=df_train,
                valid_days_slice=valid_train,
                base_rule=base_rule,
                adx_threshold=float(chosen_thr),
                orb_start=chosen_orb_start,
                orb_window_min=int(args.orb_window_min),
                cutoff_offset_min=int(args.cutoff_offset_min),
                engine=args.engine,
                futures_cfg=futures_cfg,
                initial_capital=initial_capital,
                position_size=position_size,
                taker_fee_rate=taker_fee_rate,
                fee_mult=float(args.fee_mult),
                slippage_bps=float(args.slippage_bps),
                delay_bars=int(args.delay_bars),
                risk_limits=risk_limits,
            )
            tr_trades_df.to_csv(fold_dir / "train_chosen_trades.csv", index=False)
            tr_equity_df.to_csv(fold_dir / "train_chosen_equity_curve.csv", index=False)
            (fold_dir / "train_chosen_results.json").write_text(
                stable_json({"metrics": tr_metrics, **tr_extra}),
                encoding="utf-8",
            )

        # TEST evaluation
        te_trades_df, te_equity_df, te_metrics, te_extra = run_backtest_one(
            df_ind_slice=df_test,
            valid_days_slice=valid_test,
            base_rule=base_rule,
            adx_threshold=float(chosen_thr),
            orb_start=chosen_orb_start,
            orb_window_min=int(args.orb_window_min),
            cutoff_offset_min=int(args.cutoff_offset_min),
            engine=args.engine,
            futures_cfg=futures_cfg,
            initial_capital=initial_capital,
            position_size=position_size,
            taker_fee_rate=taker_fee_rate,
            fee_mult=float(args.fee_mult),
            slippage_bps=float(args.slippage_bps),
            delay_bars=int(args.delay_bars),
            risk_limits=risk_limits,
        )

        te_trades_df.to_csv(fold_dir / "test_trades.csv", index=False)
        te_equity_df.to_csv(fold_dir / "test_equity_curve.csv", index=False)
        (fold_dir / "test_results.json").write_text(
            stable_json(
                {
                    "fold_id": fold_id,
                    "chosen_scenario_id": chosen_id,
                    "chosen_params": {
                        "adx_threshold": float(chosen_thr),
                        "orb_start": te_extra["orb_start"],
                        "orb_end": te_extra["orb_end"],
                        "orb_cutoff": te_extra["orb_cutoff"],
                        "orb_window_min": int(args.orb_window_min),
                        "cutoff_offset_min": int(args.cutoff_offset_min),
                    },
                    "signals": {
                        "signals_total": int(te_extra["signals_total"]),
                        "signal_type_counts": te_extra["signal_type_counts"],
                    },
                    "engine_stats": te_extra["engine_stats"],
                    "metrics": te_metrics,
                }
            ),
            encoding="utf-8",
        )

        fold_summaries.append(
            {
                "fold_id": fold_id,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "chosen_scenario_id": chosen_id,
                "chosen_adx_threshold": float(chosen_thr),
                "chosen_orb_start": str(te_extra["orb_start"]),
                "objective": str(args.objective),
                "train_daily_sharpe": float(chosen.get("daily_sharpe", 0.0)),
                "train_total_return_pct": float(chosen.get("total_return_pct", 0.0)),
                "train_cagr": float(chosen.get("cagr", 0.0)),
                "train_max_drawdown_pct": float(chosen.get("max_drawdown_pct", 0.0)),
                "train_total_trades": int(chosen.get("total_trades", 0)),
                "test_daily_sharpe": float(te_metrics["daily_sharpe"]),
                "test_total_return_pct": float(te_metrics["total_return_pct"]),
                "test_cagr": float(te_metrics["cagr"]),
                "test_max_drawdown_pct": float(te_metrics["max_drawdown_pct"]),
                "test_total_trades": int(te_metrics["total_trades"]),
                "test_liquidations": int(te_metrics["liquidations"]),
                "test_total_fees": float(te_metrics["total_fees"]),
                "test_total_funding": float(te_metrics["total_funding"]),
                "fold_dir": str(fold_dir),
            }
        )

        print(
            f"[OK] {fold_id} | chosen={chosen_id} | TRAIN {args.objective}={chosen[args.objective]:.3f} | "
            f"TEST sharpe={te_metrics['daily_sharpe']:.3f} ret={te_metrics['total_return_pct']:.1f}%"
        )

    summary_df = pd.DataFrame(fold_summaries)
    summary_csv = out_dir / "walk_forward_tune_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    counts_df = (
        pd.Series(selection_counter)
        .rename("selected_count")
        .reset_index()
        .rename(columns={"index": "scenario_id"})
        .sort_values("selected_count", ascending=False)
    )
    counts_csv = out_dir / "selection_counts.csv"
    counts_df.to_csv(counts_csv, index=False)

    agg = {
        "folds": int(len(summary_df)),
        "test_daily_sharpe_mean": float(summary_df["test_daily_sharpe"].mean()) if len(summary_df) else 0.0,
        "test_daily_sharpe_median": float(summary_df["test_daily_sharpe"].median()) if len(summary_df) else 0.0,
        "test_total_return_pct_mean": float(summary_df["test_total_return_pct"].mean()) if len(summary_df) else 0.0,
        "test_total_return_pct_median": float(summary_df["test_total_return_pct"].median()) if len(summary_df) else 0.0,
        "test_max_drawdown_pct_worst": float(summary_df["test_max_drawdown_pct"].min()) if len(summary_df) else 0.0,
    }
    (out_dir / "aggregate_test_stats.json").write_text(stable_json(agg), encoding="utf-8")

    outputs = {
        "walk_forward_tune_summary.csv": str(summary_csv),
        "selection_counts.csv": str(counts_csv),
        "aggregate_test_stats.json": str(out_dir / "aggregate_test_stats.json"),
    }
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
            "folds_csv": str(folds_path),
            "folds_csv_sha256": sha256_file(folds_path),
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "grid": {
            "adx_threshold_grid": adx_grid,
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
            "objective": str(args.objective),
            "min_trades": int(args.min_trades),
        },
        "outputs": outputs,
        "output_sha256": hashes,
    }
    (out_dir / "run_metadata.json").write_text(stable_json(meta), encoding="utf-8")

    print(f"\n[OK] Wrote: {summary_csv}")
    print(f"[OK] Wrote: {counts_csv}")
    print(f"[OK] Wrote: {out_dir / 'aggregate_test_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

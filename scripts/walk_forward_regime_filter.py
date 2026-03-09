import os

os.environ["PYTHONHASHSEED"] = "0"

import argparse
import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import yaml

random.seed(0)
np.random.seed(0)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.utils import sha256_file, stable_json  # noqa: E402
from strategy import add_trend_indicators, build_signals_from_ruleset, load_signal_rules_from_config  # noqa: E402
from backtester.futures_engine import FuturesEngineConfig, backtest_futures_orb  # noqa: E402
from backtester.risk import risk_limits_from_config  # noqa: E402


def get_git_info() -> Dict[str, Any]:
    def run(cmd: List[str]) -> str:
        return subprocess.check_output(cmd, cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL).decode().strip()

    try:
        return {
            "commit": run(["git", "rev-parse", "HEAD"]),
            "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": bool(run(["git", "status", "--porcelain"])),
        }
    except Exception:
        return {"commit": None, "branch": None, "dirty": None}


def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Dataset must have a DatetimeIndex.")
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    return df.sort_index()


def compute_daily_realized_vol(df_prices: pd.DataFrame) -> pd.Series:
    """
    Daily realized vol proxy from intraday log returns:
      rv_day = sqrt(sum_{intraday} logret^2)
    Index: python date
    """
    close = df_prices["close"].astype(float)
    logret = np.log(close).diff()

    tmp = pd.DataFrame({"logret": logret})
    tmp["date"] = tmp.index.date

    rv2 = tmp.groupby("date")["logret"].apply(lambda x: float(np.nansum(np.square(x.values))))
    rv = np.sqrt(rv2)
    return rv


def single_rule_orb_ranges(rule_orb_ranges_df: pd.DataFrame) -> pd.DataFrame:
    orb_out = rule_orb_ranges_df.loc[:, ["date", "orb_high", "orb_low"]].copy()
    orb_out = orb_out.drop_duplicates(subset=["date"]).set_index("date").sort_index()
    return orb_out


def adx_at_time(df_ind: pd.DataFrame, dates: List[datetime.date], t) -> pd.Series:
    """
    ADX snapshot at time t (UTC) for each day. Uses ffill to nearest prior candle.
    Index: python date
    """
    ts = pd.to_datetime(
        [datetime(d.year, d.month, d.day, t.hour, t.minute, tzinfo=timezone.utc) for d in dates]
    )
    snap = df_ind.reindex(ts, method="ffill")["adx"].astype(float)
    snap.index = pd.Index(dates, name="date")
    return snap


def compute_fold_blocked_days(
    df_fold_ind: pd.DataFrame,
    valid_days_fold: Set[datetime.date],
    train_days: Set[datetime.date],
    test_days: Set[datetime.date],
    orb_end_time,
    vol_lookback_days: int,
    low_vol_q: float,
    high_adx_q: float,
) -> Tuple[Set[datetime.date], Dict[str, float]]:
    """
    Defines blocked days in TEST as:
      vol_score(day) <= q_low(train vol_score)
      AND
      adx_at_orb_end(day) >= q_high(train adx_at_orb_end)

    vol_score = rolling mean (vol_lookback_days) of realized vol, shifted 1 day (no leakage).
    """
    fold_days_sorted = sorted(valid_days_fold)

    rv = compute_daily_realized_vol(df_fold_ind)
    rv = rv.reindex(pd.Index(fold_days_sorted, name="date"))

    vol_score = rv.rolling(vol_lookback_days, min_periods=vol_lookback_days).mean().shift(1)
    adx_snap = adx_at_time(df_fold_ind, fold_days_sorted, orb_end_time)

    # TRAIN thresholds (exclude NaNs)
    train_idx = [d for d in fold_days_sorted if d in train_days]
    vs_train = vol_score.reindex(train_idx).dropna()
    adx_train = adx_snap.reindex(train_idx).dropna()

    if vs_train.empty or adx_train.empty:
        # Not enough history; block nothing.
        return set(), {"vol_thr": float("nan"), "adx_thr": float("nan")}

    vol_thr = float(vs_train.quantile(low_vol_q))
    adx_thr = float(adx_train.quantile(high_adx_q))

    blocked = set()
    for d in fold_days_sorted:
        if d not in test_days:
            continue
        v = vol_score.get(d, np.nan)
        a = adx_snap.get(d, np.nan)
        if np.isnan(v) or np.isnan(a):
            continue
        if (v <= vol_thr) and (a >= adx_thr):
            blocked.add(d)

    return blocked, {"vol_thr": vol_thr, "adx_thr": adx_thr}


def summarize_equity(equity_df: pd.DataFrame, initial_capital: float) -> Dict[str, float]:
    if equity_df.empty:
        return {
            "final_equity": float(initial_capital),
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
        }

    eq = equity_df["equity"].astype(float)
    final_equity = float(eq.iloc[-1])
    total_return_pct = (final_equity / initial_capital - 1.0) * 100.0 if initial_capital else 0.0

    roll_max = eq.cummax()
    dd = (eq / roll_max) - 1.0
    max_dd_pct = float(dd.min() * 100.0)

    return {
        "final_equity": final_equity,
        "total_return_pct": float(total_return_pct),
        "max_drawdown_pct": float(max_dd_pct),
    }


def daily_sharpe(equity_df: pd.DataFrame) -> float:
    if equity_df.empty:
        return 0.0
    s = equity_df.set_index("timestamp")["equity"].astype(float)
    daily = s.resample("1D").last().dropna()
    rets = daily.pct_change().dropna()
    if len(rets) < 2:
        return 0.0
    std = float(rets.std(ddof=1))
    if std == 0.0 or np.isnan(std):
        return 0.0
    return float((rets.mean() / std) * np.sqrt(365.0))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Walk-forward evaluation with an experimental regime filter (no tuning, fixed params)."
    )

    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--folds-csv", default="reports/walk_forward/walk_forward_folds.csv")
    ap.add_argument("--valid-days", default="data/processed/valid_days.csv")
    ap.add_argument("--data", default="", help="Default: data/processed/<symbol>_<timeframe>.parquet")
    ap.add_argument("--out-dir", default="reports/walk_forward_regime_filter")

    # Regime filter knobs (TRAIN-derived thresholds, applied to TEST)
    ap.add_argument("--enable-filter", action="store_true")
    ap.add_argument("--vol-lookback-days", type=int, default=30)
    ap.add_argument("--low-vol-quantile", type=float, default=1 / 3)
    ap.add_argument("--high-adx-quantile", type=float, default=2 / 3)

    # Futures knobs (match your baseline)
    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--mmr", type=float, default=0.005)
    ap.add_argument("--funding-per-8h", type=float, default=0.0001)
    ap.add_argument("--fee-mult", type=float, default=1.0)
    ap.add_argument("--slippage-bps", type=float, default=0.0)
    ap.add_argument("--delay-bars", type=int, default=1)

    args = ap.parse_args()

    config_path = (REPO_ROOT / args.config).resolve()
    folds_path = (REPO_ROOT / args.folds_csv).resolve()
    valid_days_path = (REPO_ROOT / args.valid_days).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "folds").mkdir(parents=True, exist_ok=True)

    if not folds_path.exists():
        raise FileNotFoundError(f"Folds CSV not found: {folds_path}")

    cfg_text = config_path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(cfg_text) or {}
    rules = load_signal_rules_from_config(cfg)
    if len(rules) != 1:
        raise ValueError(
            "walk_forward_regime_filter currently supports configs that resolve to exactly one signal rule."
        )
    fixed_rule = rules[0]

    symbol = str(cfg.get("symbol", "SOLUSDT"))
    timeframe = str(cfg.get("timeframe", "30m"))

    orb_start = fixed_rule.orb_start
    orb_end = fixed_rule.orb_end
    orb_cutoff = fixed_rule.orb_cutoff

    adx_period = int(cfg["adx"]["period"])
    adx_threshold = float(fixed_rule.adx_threshold)

    initial_capital = float(cfg["risk"]["initial_capital"])
    position_size = float(cfg["risk"]["position_size"])
    taker_fee_rate = float(cfg["fees"]["taker_fee_rate"])

    # Phase 4 risk controls
    risk_limits = risk_limits_from_config(cfg)

    data_path = Path(args.data) if args.data else Path(f"data/processed/{symbol}_{timeframe}.parquet")
    data_path = (REPO_ROOT / data_path).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Parquet not found: {data_path}. Run: python scripts/build_parquet.py")

    # Load dataset + indicators once
    df = pd.read_parquet(data_path)
    df = ensure_utc_index(df)
    df = df[["open", "high", "low", "close", "volume"]].copy()

    df_ind = add_trend_indicators(df, period=adx_period)

    # Valid days (exactly 48 bars)
    vdf = pd.read_csv(valid_days_path)
    if "date_utc" not in vdf.columns:
        raise ValueError(f"{valid_days_path} must contain column 'date_utc'")
    valid_days_all = set(pd.to_datetime(vdf["date_utc"], utc=True).dt.date)

    folds = pd.read_csv(folds_path)
    required = ["fold_id", "train_start", "train_end", "test_start", "test_end"]
    for c in required:
        if c not in folds.columns:
            raise ValueError(f"folds csv missing column: {c}")

    # Futures engine config
    fut_cfg = FuturesEngineConfig(
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

    summary_rows = []
    blocked_rows = []

    for _, r in folds.iterrows():
        fold_id = str(r["fold_id"])
        train_start = str(r["train_start"])
        train_end = str(r["train_end"])
        test_start = str(r["test_start"])
        test_end = str(r["test_end"])

        fold_dir = out_dir / "folds" / fold_id
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Day sets
        train_days = set(pd.date_range(train_start, train_end, freq="D", tz="UTC").date)
        test_days = set(pd.date_range(test_start, test_end, freq="D", tz="UTC").date)
        fold_days = set(pd.date_range(train_start, test_end, freq="D", tz="UTC").date)

        valid_fold_days = valid_days_all.intersection(fold_days)
        valid_test_days = valid_days_all.intersection(test_days)

        # Slice data for fold ranges
        df_fold = df_ind.loc[
            (df_ind.index >= pd.Timestamp(train_start, tz="UTC"))
            & (df_ind.index < pd.Timestamp(test_end, tz="UTC") + pd.Timedelta(days=1))
        ].copy()

        # Compute blocked days (TEST only)
        blocked_days = set()
        thr_info = {"vol_thr": float("nan"), "adx_thr": float("nan")}
        if args.enable_filter:
            blocked_days, thr_info = compute_fold_blocked_days(
                df_fold_ind=df_fold,
                valid_days_fold=valid_fold_days,
                train_days=train_days,
                test_days=test_days,
                orb_end_time=orb_end,  # fixed ORB end from config
                vol_lookback_days=int(args.vol_lookback_days),
                low_vol_q=float(args.low_vol_quantile),
                high_adx_q=float(args.high_adx_quantile),
            )
            # keep only valid TEST days
            blocked_days = blocked_days.intersection(valid_test_days)

        # Now run the FIXED strategy on TEST window, blocking entries on blocked_days
        df_test = df_ind.loc[
            (df_ind.index >= pd.Timestamp(test_start, tz="UTC"))
            & (df_ind.index < pd.Timestamp(test_end, tz="UTC") + pd.Timedelta(days=1))
        ].copy()

        df_sig, rule_orb_ranges_df, _ = build_signals_from_ruleset(df_test, [fixed_rule], valid_test_days)
        orb_ranges = single_rule_orb_ranges(rule_orb_ranges_df)

        if blocked_days:
            blk = df_sig["date"].isin(blocked_days)
            df_sig.loc[blk, "signal"] = 0
            df_sig.loc[blk, "signal_type"] = ""

        signals_total = int((df_sig["signal"] != 0).sum())
        signal_counts = df_sig.loc[df_sig["signal"] != 0, "signal_type"].value_counts(dropna=False).to_dict()

        trades, equity_curve, stats = backtest_futures_orb(
            df=df_sig,
            orb_ranges=orb_ranges,
            valid_days=valid_test_days,  # still the data-quality policy
            cfg=fut_cfg,
            risk_limits=risk_limits,
        )

        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame({"timestamp": df_sig.index, "equity": equity_curve})

        # Save per-fold artifacts
        trades_df.to_csv(fold_dir / "test_trades.csv", index=False)
        equity_df.to_csv(fold_dir / "test_equity_curve.csv", index=False)
        (fold_dir / "test_engine_stats.json").write_text(stable_json(stats), encoding="utf-8")

        eq_sum = summarize_equity(equity_df, initial_capital)
        sh = daily_sharpe(equity_df)

        summary_rows.append(
            {
                "fold_id": fold_id,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "enable_filter": bool(args.enable_filter),
                "blocked_days_count": int(len(blocked_days)),
                "vol_thr_train_q33": float(thr_info["vol_thr"]),
                "adx_thr_train_q67": float(thr_info["adx_thr"]),
                "signals_total": int(signals_total),
                "signal_type_counts_json": json.dumps(signal_counts, sort_keys=True),
                "test_daily_sharpe": float(sh),
                "test_total_return_pct": float(eq_sum["total_return_pct"]),
                "test_max_drawdown_pct": float(eq_sum["max_drawdown_pct"]),
                "test_final_equity": float(eq_sum["final_equity"]),
                "test_total_trades": int(len(trades_df)),
                "test_total_fees": float(stats.get("total_fees", 0.0)),
                "test_total_funding": float(stats.get("total_funding", 0.0)),
                "test_liquidations": int(stats.get("liquidations", 0)),
            }
        )

        for d in sorted(blocked_days):
            blocked_rows.append({"fold_id": fold_id, "date_utc": str(d)})

        print(
            f"[OK] {fold_id} | blocked={len(blocked_days)} | trades={len(trades_df)} | "
            f"ret={eq_sum['total_return_pct']:.2f}% | sharpe={sh:.2f}"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "walk_forward_regime_filter_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    blocked_df = pd.DataFrame(blocked_rows)
    blocked_csv = out_dir / "blocked_days.csv"
    blocked_df.to_csv(blocked_csv, index=False)

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git": get_git_info(),
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": {"platform": platform.platform()},
        "inputs": {
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "folds_csv": str(folds_path),
            "folds_sha256": sha256_file(folds_path),
            "parquet": str(data_path),
            "parquet_sha256": sha256_file(data_path),
            "valid_days": str(valid_days_path),
            "valid_days_sha256": sha256_file(valid_days_path),
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "settings": {
            "fixed_params": {
                "adx_threshold": adx_threshold,
                "orb_start": orb_start.strftime("%H:%M"),
                "orb_end": orb_end.strftime("%H:%M"),
                "orb_cutoff": orb_cutoff.strftime("%H:%M"),
            },
            "engine": {
                "leverage": float(args.leverage),
                "mmr": float(args.mmr),
                "funding_per_8h": float(args.funding_per_8h),
                "fee_mult": float(args.fee_mult),
                "slippage_bps": float(args.slippage_bps),
                "delay_bars": int(args.delay_bars),
            },
            "regime_filter": {
                "enabled": bool(args.enable_filter),
                "vol_lookback_days": int(args.vol_lookback_days),
                "low_vol_quantile": float(args.low_vol_quantile),
                "high_adx_quantile": float(args.high_adx_quantile),
                "definition": "Block TEST days where vol_score<=q_low(TRAIN) AND adx_at_orb_end>=q_high(TRAIN). vol_score is trailing mean RV(lookback) shifted 1 day.",
            },
        },
        "outputs": {
            "walk_forward_regime_filter_summary.csv": str(summary_csv),
            "blocked_days.csv": str(blocked_csv),
        },
        "output_sha256": {
            "walk_forward_regime_filter_summary.csv": sha256_file(summary_csv),
            "blocked_days.csv": sha256_file(blocked_csv),
        },
    }
    (out_dir / "run_metadata.json").write_text(stable_json(meta), encoding="utf-8")

    print(f"\n[OK] Wrote: {summary_csv}")
    print(f"[OK] Wrote: {blocked_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

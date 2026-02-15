from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from backtester.futures_engine import FuturesEngineConfig, backtest_futures_orb
from backtester.risk import RiskLimits
from strategy import add_trend_indicators, generate_orb_signals, identify_orb_ranges


@dataclass
class ShadowRunResult:
    df_sig: pd.DataFrame
    orb_ranges: pd.DataFrame
    trades: List[Dict[str, Any]]
    equity_curve: pd.Series
    stats: Dict[str, Any]


def build_signals(
    df_raw: pd.DataFrame,
    valid_days: set,
    orb_start: time,
    orb_end: time,
    orb_cutoff: time,
    adx_period: int,
    adx_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Exact same signal pipeline used by scripts/run_baseline.py."""
    df_ind = add_trend_indicators(df_raw, period=adx_period)
    orb_ranges = identify_orb_ranges(df_ind, orb_start_time=orb_start, orb_end_time=orb_end)
    orb_ranges = orb_ranges.loc[orb_ranges.index.isin(valid_days)]

    df_sig = generate_orb_signals(
        df_ind,
        orb_ranges=orb_ranges,
        adx_threshold=adx_threshold,
        orb_cutoff_time=orb_cutoff,
    )

    invalid_mask = ~df_sig["date"].isin(valid_days)
    df_sig.loc[invalid_mask, "signal"] = 0
    df_sig.loc[invalid_mask, "signal_type"] = ""

    return df_sig, orb_ranges


def run_shadow_futures(
    df_raw: pd.DataFrame,
    valid_days: set,
    orb_start: time,
    orb_end: time,
    orb_cutoff: time,
    adx_period: int,
    adx_threshold: float,
    initial_capital: float,
    position_size: float,
    taker_fee_rate: float,
    leverage: float,
    delay_bars: int,
    slippage_bps: float,
    fee_mult: float = 1.0,
    funding_rate_per_8h: float = 0.0,
    risk_limits: Optional[RiskLimits] = None,
) -> ShadowRunResult:
    """Deterministic replay runner for shadow execution."""

    df_sig, orb_ranges = build_signals(
        df_raw=df_raw,
        valid_days=valid_days,
        orb_start=orb_start,
        orb_end=orb_end,
        orb_cutoff=orb_cutoff,
        adx_period=adx_period,
        adx_threshold=adx_threshold,
    )

    engine_cfg = FuturesEngineConfig(
        initial_capital=float(initial_capital),
        position_size=float(position_size),
        leverage=float(leverage),
        taker_fee_rate=float(taker_fee_rate),
        fee_mult=float(fee_mult),
        slippage_bps=float(slippage_bps),
        delay_bars=int(delay_bars),
        funding_rate_per_8h=float(funding_rate_per_8h),
    )

    trades, equity_curve, stats = backtest_futures_orb(
        df=df_sig,
        orb_ranges=orb_ranges,
        valid_days=set(valid_days),
        cfg=engine_cfg,
        risk_limits=risk_limits,
    )

    equity_series = pd.Series(equity_curve, index=df_sig.index, name="equity")
    return ShadowRunResult(
        df_sig=df_sig,
        orb_ranges=orb_ranges,
        trades=trades,
        equity_curve=equity_series,
        stats=stats,
    )

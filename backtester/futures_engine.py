# backtester/futures_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .futures_core import FuturesExecutionCore
from .risk import RiskLimits, expected_bar_seconds_from_index


@dataclass
class FuturesEngineConfig:
    """Isolated-margin USDT-margined futures config (single position at a time)."""

    initial_capital: float = 10_000.0
    position_size: float = 0.95  # fraction of FREE balance used as initial margin
    leverage: float = 1.0  # notional = margin * leverage
    taker_fee_rate: float = 0.0005  # base taker fee
    fee_mult: float = 1.0  # stress multiplier
    slippage_bps: float = 0.0  # adverse slippage, bps
    delay_bars: int = 1  # 1=next bar open
    maintenance_margin_rate: float = 0.005  # mmr (approx)

    # Funding (optional)
    # If funding_series is provided, index must be UTC timestamps at funding event times.
    # Else funding_rate_per_8h is applied at 00:00/08:00/16:00 UTC.
    funding_rate_per_8h: float = 0.0  # e.g., 0.0001 = 0.01%
    funding_series: Optional[pd.Series] = None


def backtest_futures_orb(
    df: pd.DataFrame,
    orb_ranges: pd.DataFrame,
    valid_days: Optional[set] = None,
    cfg: Optional[FuturesEngineConfig] = None,
    risk_limits: Optional[RiskLimits] = None,
) -> Tuple[List[Dict[str, Any]], List[float], Dict[str, Any]]:
    """Futures backtest engine (isolated margin), next-open execution with realism switches.

    Expected df columns:
      open, high, low, close (floats)
      date (python date)
      signal (int)
      signal_type (str)
    Index must be UTC timestamps.

    Expected orb_ranges:
      index is python date; columns: orb_high, orb_low

    Returns:
      trades, equity_curve (mark-to-market at bar close), stats
    """
    if cfg is None:
        cfg = FuturesEngineConfig()

    valid_days = valid_days or set()
    if risk_limits is None:
        risk_limits = RiskLimits(enabled=False)

    core = FuturesExecutionCore(
        cfg=cfg,
        risk_limits=risk_limits,
        expected_bar_seconds=expected_bar_seconds_from_index(df.index),
    )

    equity_curve: List[float] = []

    for i in range(len(df)):
        ts = df.index[i]
        bar_open = float(df["open"].iloc[i])
        bar_high = float(df["high"].iloc[i])
        bar_low = float(df["low"].iloc[i])
        bar_close = float(df["close"].iloc[i])

        current_date = df["date"].iloc[i]
        signal = int(df["signal"].iloc[i])
        signal_type = str(df["signal_type"].iloc[i])

        orb_high: Optional[float] = None
        orb_low: Optional[float] = None
        if current_date in orb_ranges.index:
            orb_high = float(orb_ranges.loc[current_date, "orb_high"])
            orb_low = float(orb_ranges.loc[current_date, "orb_low"])

        # Backtester-only bound check stays in adapter: don't schedule past final bar.
        allow_schedule = (i + int(core.delay_bars)) < len(df)

        core.on_bar(
            ts=ts,
            bar_open=bar_open,
            bar_high=bar_high,
            bar_low=bar_low,
            bar_close=bar_close,
            current_date=current_date,
            signal=signal,
            signal_type=signal_type,
            orb_high=orb_high,
            orb_low=orb_low,
            valid_days=valid_days,
            allow_schedule=allow_schedule,
        )

        # Equity mark-to-market at close
        equity_curve.append(float(core.equity(mark_price=bar_close)))

    # Close any open position at end
    if core.side is not None and core.qty > 0 and len(df):
        last_ts = df.index[-1]
        last_close = float(df["close"].iloc[-1])
        pnl_net = core.close_position(last_ts, raw_exit_price=last_close, reason="end")
        if pnl_net is not None and core.risk_mgr is not None:
            core.risk_mgr.record_trade_close(last_ts, df["date"].iloc[-1], pnl_net)

    final_equity = float(core.equity(mark_price=float(df["close"].iloc[-1]))) if len(df) else float(cfg.initial_capital)

    stats: Dict[str, Any] = {
        "final_equity": final_equity,
        "free_balance_end": float(core.free_balance),
        "total_fees": float(core.total_fees),
        "total_funding": float(core.total_funding),
        "liquidations": int(core.liquidations),
        "trades": int(len(core.trades)),
        "assumptions": {
            "leverage": float(core.leverage),
            "mmr": float(core.mmr),
            "fee_rate_effective": float(core.fee_rate),
            "slippage_bps": float(cfg.slippage_bps),
            "delay_bars": int(core.delay_bars),
            "funding_rate_per_8h": float(cfg.funding_rate_per_8h),
            "funding_series_used": cfg.funding_series is not None,
        },
    }

    if core.risk_mgr is not None:
        stats["risk"] = core.risk_mgr.snapshot()
    else:
        stats["risk"] = {"enabled": False}

    return core.trades, equity_curve, stats

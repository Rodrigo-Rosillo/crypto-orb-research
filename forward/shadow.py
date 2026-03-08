from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from backtester.futures_engine import FuturesEngineConfig, backtest_futures_orb
from backtester.risk import RiskLimits
from strategy import SignalRule, build_signals_from_config


def _parse_leverage(cfg: Dict[str, Any]) -> float:
    lev_cfg = cfg.get("leverage") or {}
    if not isinstance(lev_cfg, dict):
        raise ValueError("leverage must be a mapping when provided")

    leverage = float(lev_cfg.get("max_leverage", 1.0)) if bool(lev_cfg.get("enabled", True)) else 1.0
    if leverage != float(int(leverage)):
        raise ValueError(
            f"leverage.max_leverage must be a whole number, got {leverage}. "
            f"Binance applies int() truncation. Use {int(leverage)} or {int(leverage) + 1}."
        )
    return float(int(leverage))


@dataclass
class ShadowRunResult:
    df_sig: pd.DataFrame
    rule_orb_ranges: pd.DataFrame
    rules: list[SignalRule]
    trades: List[Dict[str, Any]]
    equity_curve: pd.Series
    stats: Dict[str, Any]


def build_signals(
    df_raw: pd.DataFrame,
    valid_days: set,
    cfg: Dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[SignalRule]]:
    return build_signals_from_config(df_raw=df_raw, cfg=cfg, valid_days=valid_days)


def run_shadow_futures(
    df_raw: pd.DataFrame,
    valid_days: set,
    cfg: Dict[str, Any],
    delay_bars: int,
    slippage_bps: float,
    fee_mult: float = 1.0,
    funding_rate_per_8h: Optional[float] = None,
    risk_limits: Optional[RiskLimits] = None,
) -> ShadowRunResult:
    """Deterministic replay runner for shadow execution."""
    df_sig, rule_orb_ranges, rules = build_signals(
        df_raw=df_raw,
        valid_days=valid_days,
        cfg=cfg,
    )

    initial_capital = float(cfg["risk"]["initial_capital"])
    position_size = float(cfg["risk"]["position_size"])
    taker_fee_rate = float(cfg["fees"]["taker_fee_rate"])
    leverage = _parse_leverage(cfg)
    funding_rate = (
        float(funding_rate_per_8h)
        if funding_rate_per_8h is not None
        else float(cfg.get("funding", {}).get("rate_per_8h", 0.0))
        if isinstance(cfg.get("funding"), dict)
        else 0.0
    )

    engine_cfg = FuturesEngineConfig(
        initial_capital=initial_capital,
        position_size=position_size,
        leverage=leverage,
        taker_fee_rate=taker_fee_rate,
        fee_mult=float(fee_mult),
        slippage_bps=float(slippage_bps),
        delay_bars=int(delay_bars),
        funding_rate_per_8h=funding_rate,
    )

    trades, equity_curve, stats = backtest_futures_orb(
        df=df_sig,
        orb_ranges=None,
        valid_days=set(valid_days),
        cfg=engine_cfg,
        risk_limits=risk_limits,
    )

    equity_series = pd.Series(equity_curve, index=df_sig.index, name="equity")
    return ShadowRunResult(
        df_sig=df_sig,
        rule_orb_ranges=rule_orb_ranges,
        rules=rules,
        trades=trades,
        equity_curve=equity_series,
        stats=stats,
    )

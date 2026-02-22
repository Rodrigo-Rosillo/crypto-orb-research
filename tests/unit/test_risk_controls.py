from __future__ import annotations

from datetime import date
from math import nextafter

import pandas as pd

from backtester.risk import RiskLimits, RiskManager


def test_daily_loss_halt_triggers_at_exact_threshold() -> None:
    limits = RiskLimits(
        enabled=True,
        max_daily_loss_pct=0.03,
        max_drawdown_pct=1.0,
    )
    rm = RiskManager(limits=limits, expected_bar_seconds=1800)
    d = date(2024, 1, 20)

    rm.on_bar(pd.Timestamp("2024-01-20 00:00:00", tz="UTC"), d, equity=1000.0)
    rm.on_bar(pd.Timestamp("2024-01-20 00:30:00", tz="UTC"), d, equity=970.0)

    assert rm.halted_today is True
    assert rm.halt_day == d
    assert rm.halted_global is False


def test_drawdown_halt_triggers_at_exact_threshold() -> None:
    limits = RiskLimits(
        enabled=True,
        max_daily_loss_pct=1.0,
        max_drawdown_pct=0.20,
    )
    rm = RiskManager(limits=limits, expected_bar_seconds=1800)
    d = date(2024, 1, 21)

    rm.on_bar(pd.Timestamp("2024-01-21 00:00:00", tz="UTC"), d, equity=1000.0)
    # Nudge one ULP below the mathematical 20% boundary to avoid float tie issues.
    threshold_equity = 1000.0 * (1.0 - limits.max_drawdown_pct)
    rm.on_bar(
        pd.Timestamp("2024-01-21 00:30:00", tz="UTC"),
        d,
        equity=nextafter(threshold_equity, 0.0),
    )

    assert rm.halted_global is True
    assert rm.halt_reason == "max_drawdown"


def test_consecutive_losses_reset_on_new_utc_day() -> None:
    limits = RiskLimits(
        enabled=True,
        max_daily_loss_pct=1.0,
        max_drawdown_pct=1.0,
        max_consecutive_losses=2,
    )
    rm = RiskManager(limits=limits, expected_bar_seconds=1800)
    day1 = date(2024, 1, 22)
    day2 = date(2024, 1, 23)

    ts0 = pd.Timestamp("2024-01-22 00:00:00", tz="UTC")
    rm.on_bar(ts0, day1, equity=1000.0)
    rm.record_trade_close(ts0 + pd.Timedelta(minutes=5), day1, pnl_net=-1.0)
    rm.record_trade_close(ts0 + pd.Timedelta(minutes=10), day1, pnl_net=-2.0)

    assert rm.consecutive_losses == 2
    assert rm.halted_today is True
    assert rm.halt_day == day1

    rm.on_bar(pd.Timestamp("2024-01-23 00:00:00", tz="UTC"), day2, equity=1000.0)

    assert rm.current_day == day2
    assert rm.consecutive_losses == 0
    assert rm.halted_today is False
    assert rm.halt_day is None

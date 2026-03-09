from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from backtester.futures_core import FuturesExecutionCore
from backtester.futures_engine import FuturesEngineConfig
from backtester.risk import KillSwitchConfig, RiskLimits

BASE_TS = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
BASE_DAY = BASE_TS.date()


@st.composite
def _bars_and_signals(
    draw,
    *,
    min_bars: int = 5,
    max_bars: int = 40,
    open_min: float = 50.0,
    open_max: float = 150.0,
) -> tuple[list[float], list[int]]:
    n = draw(st.integers(min_value=min_bars, max_value=max_bars))
    opens = draw(
        st.lists(
            st.floats(
                min_value=open_min,
                max_value=open_max,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=n,
            max_size=n,
        )
    )
    signals = draw(st.lists(st.sampled_from([0, 1, -1]), min_size=n, max_size=n))
    return opens, signals


def _cap_limits(max_frac: float) -> RiskLimits:
    return RiskLimits(
        enabled=True,
        max_position_margin_frac=float(max_frac),
        max_leverage=10.0,
        max_daily_loss_pct=2.0,
        max_drawdown_pct=2.0,
        max_consecutive_losses=10_000,
        max_exposure_bars=10_000,
        kill_switch=KillSwitchConfig(
            max_data_gap_bars=10_000,
            max_order_rejects_per_day=10_000,
            max_margin_ratio=10.0,
        ),
    )


@settings(max_examples=100, deadline=None)
@given(
    max_frac=st.floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False),
    bars=_bars_and_signals(open_min=50.0, open_max=150.0),
)
def test_property_position_size_cap(max_frac: float, bars: tuple[list[float], list[int]]) -> None:
    opens, signals = bars
    assume(any(s != 0 for s in signals[:-1]))

    cfg = FuturesEngineConfig(
        initial_capital=1000.0,
        position_size=1.0,
        leverage=2.0,
        taker_fee_rate=0.001,
        fee_mult=1.0,
        slippage_bps=0.0,
        delay_bars=1,
        funding_rate_per_8h=0.0,
    )
    limits = _cap_limits(max_frac=max_frac)
    core = FuturesExecutionCore(cfg=cfg, risk_limits=limits, expected_bar_seconds=1800)
    valid_days: set[date] = {BASE_DAY}

    for i, (bar_open, signal) in enumerate(zip(opens, signals)):
        ts = BASE_TS + pd.Timedelta(minutes=30 * i)
        eq_before = core.equity(mark_price=float(bar_open))
        signal_type = "" if int(signal) == 0 else ("uptrend_reversion" if int(signal) > 0 else "downtrend_breakdown")

        step = core.on_bar(
            ts=ts,
            bar_open=float(bar_open),
            bar_high=float(bar_open),
            bar_low=float(bar_open),
            bar_close=float(bar_open),
            current_date=BASE_DAY,
            signal=int(signal),
            signal_type=signal_type,
            orb_high=200.0,
            orb_low=1.0,
            valid_days=valid_days,
        )

        if bool(step["entered"]):
            assert core.current_initial_margin >= 0.0
            assert core.current_initial_margin <= (
                eq_before * limits.max_position_margin_frac + 1e-9
            )


@settings(max_examples=100, deadline=None)
@given(
    bars=_bars_and_signals(open_min=90.0, open_max=110.0),
)
def test_property_accounting_reconciliation(
    bars: tuple[list[float], list[int]],
) -> None:
    opens, signals = bars

    cfg = FuturesEngineConfig(
        initial_capital=1000.0,
        position_size=1.0,
        leverage=2.0,
        taker_fee_rate=0.001,
        fee_mult=1.0,
        slippage_bps=0.0,
        delay_bars=1,
        funding_rate_per_8h=0.0,
    )
    limits = _cap_limits(max_frac=0.5)
    core = FuturesExecutionCore(cfg=cfg, risk_limits=limits, expected_bar_seconds=1800)
    valid_days: set[date] = {BASE_DAY}

    for i, (bar_open, signal) in enumerate(zip(opens, signals)):
        ts = BASE_TS + pd.Timedelta(minutes=30 * i)
        mark = float(bar_open)
        signal_type = "" if int(signal) == 0 else ("uptrend_reversion" if int(signal) > 0 else "downtrend_breakdown")

        core.on_bar(
            ts=ts,
            bar_open=mark,
            bar_high=mark,
            bar_low=mark,
            bar_close=mark,
            current_date=BASE_DAY,
            signal=int(signal),
            signal_type=signal_type,
            orb_high=200.0,
            orb_low=1.0,
            valid_days=valid_days,
        )

        realized_gross = sum(float(t["pnl_gross"]) for t in core.trades)
        if core.side is None:
            unreal = 0.0
        elif core.side == "long":
            unreal = core.qty * (mark - core.entry_price)
        else:
            unreal = core.qty * (core.entry_price - mark)

        expected_equity = (
            cfg.initial_capital + realized_gross - core.total_fees - core.total_funding + unreal
        )
        actual_equity = core.equity(mark_price=mark)
        assert actual_equity == pytest.approx(expected_equity, abs=1e-7)


@settings(max_examples=100, deadline=None)
@given(
    bars=_bars_and_signals(open_min=50.0, open_max=150.0),
)
def test_property_no_lookahead_entries(
    bars: tuple[list[float], list[int]],
) -> None:
    opens, signals = bars
    assume(any(s != 0 for s in signals[:-1]))

    cfg = FuturesEngineConfig(
        initial_capital=1000.0,
        position_size=1.0,
        leverage=2.0,
        taker_fee_rate=0.001,
        fee_mult=1.0,
        slippage_bps=0.0,
        delay_bars=1,
        funding_rate_per_8h=0.0,
    )
    core = FuturesExecutionCore(cfg=cfg, risk_limits=None, expected_bar_seconds=1800)
    valid_days: set[date] = {BASE_DAY}

    last_scheduled_ts: pd.Timestamp | None = None
    saw_enter = False

    for i, (bar_open, signal) in enumerate(zip(opens, signals)):
        ts = BASE_TS + pd.Timedelta(minutes=30 * i)
        mark = float(bar_open)
        signal_type = "" if int(signal) == 0 else ("uptrend_reversion" if int(signal) > 0 else "downtrend_breakdown")

        step = core.on_bar(
            ts=ts,
            bar_open=mark,
            bar_high=mark,
            bar_low=mark,
            bar_close=mark,
            current_date=BASE_DAY,
            signal=int(signal),
            signal_type=signal_type,
            orb_high=200.0,
            orb_low=1.0,
            valid_days=valid_days,
        )

        if bool(step["entered"]):
            saw_enter = True
            assert last_scheduled_ts is not None
            assert ts > last_scheduled_ts

        if bool(step["scheduled"]):
            last_scheduled_ts = ts

    assert saw_enter

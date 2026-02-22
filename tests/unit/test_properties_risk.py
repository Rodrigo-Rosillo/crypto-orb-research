from __future__ import annotations

from datetime import timedelta

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from backtester.risk import RiskLimits, RiskManager

BASE_TS = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
BASE_DAY = BASE_TS.date()


@settings(max_examples=100, deadline=None)
@given(
    events=st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=3),
            st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        ),
        min_size=1,
        max_size=40,
    )
)
def test_property_consecutive_losses_day_semantics(
    events: list[tuple[int, float]],
) -> None:
    limits = RiskLimits(
        enabled=True,
        max_daily_loss_pct=2.0,
        max_drawdown_pct=2.0,
        max_consecutive_losses=10_000,
    )
    rm = RiskManager(limits=limits, expected_bar_seconds=1800)

    ordered_events = sorted(enumerate(events), key=lambda pair: (pair[1][0], pair[0]))
    expected_counter = 0
    current_expected_day = None

    for i, (_, (day_offset, pnl_net)) in enumerate(ordered_events):
        ts = BASE_TS + pd.Timedelta(minutes=30 * i)
        day = BASE_DAY + timedelta(days=int(day_offset))

        day_changed = current_expected_day != day
        rm.on_bar(ts, day, equity=1000.0)

        if day_changed:
            expected_counter = 0
            current_expected_day = day
            assert rm.consecutive_losses == 0

        previous = expected_counter
        rm.record_trade_close(ts, day, pnl_net=float(pnl_net))

        if pnl_net <= 0:
            expected_counter = previous + 1
            assert rm.consecutive_losses == previous + 1
        else:
            expected_counter = 0

        assert rm.consecutive_losses == expected_counter


@settings(max_examples=100, deadline=None)
@given(
    trigger_by_gap=st.booleans(),
    day_offsets=st.lists(st.integers(min_value=0, max_value=3), min_size=1, max_size=20),
)
def test_property_kill_switch_latches_once_triggered(
    trigger_by_gap: bool,
    day_offsets: list[int],
) -> None:
    limits = RiskLimits(enabled=True)
    rm = RiskManager(limits=limits, expected_bar_seconds=1800)

    if trigger_by_gap:
        rm.on_bar(BASE_TS, BASE_DAY, equity=1000.0)
        gap_seconds = int(1800 * limits.kill_switch.max_data_gap_bars + 1)
        trigger_ts = BASE_TS + pd.Timedelta(seconds=gap_seconds)
        rm.on_bar(trigger_ts, BASE_DAY, equity=1000.0)
    else:
        rm.halt(BASE_TS, reason="test", message="test")
        trigger_ts = BASE_TS

    assert rm.halted_global is True

    for i, day_offset in enumerate(day_offsets, start=1):
        ts = trigger_ts + pd.Timedelta(minutes=30 * i)
        day = BASE_DAY + timedelta(days=int(day_offset))
        rm.on_bar(ts, day, equity=1000.0)
        assert rm.halted_global is True

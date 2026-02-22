from __future__ import annotations

from datetime import date, time

import numpy as np
import pandas as pd

from strategy import calculate_adx, identify_orb_ranges


def _assert_finite_or_nan(series: pd.Series) -> None:
    values = series.to_numpy(dtype=float, na_value=np.nan)
    finite_values = values[~np.isnan(values)]
    assert np.isfinite(finite_values).all()


def test_calculate_adx_flat_market_no_inf_and_index_aligned() -> None:
    idx = pd.date_range("2024-01-01", periods=64, freq="30min", tz="UTC")
    df = pd.DataFrame(
        {
            "high": np.full(len(idx), 100.0),
            "low": np.full(len(idx), 100.0),
            "close": np.full(len(idx), 100.0),
        },
        index=idx,
    )

    adx, plus_di, minus_di = calculate_adx(df, period=14)

    assert len(adx) == len(df)
    assert len(plus_di) == len(df)
    assert len(minus_di) == len(df)
    assert adx.index.equals(df.index)
    assert plus_di.index.equals(df.index)
    assert minus_di.index.equals(df.index)

    _assert_finite_or_nan(adx)
    _assert_finite_or_nan(plus_di)
    _assert_finite_or_nan(minus_di)


def test_calculate_adx_trend_direction_sanity() -> None:
    idx = pd.date_range("2024-01-01", periods=80, freq="30min", tz="UTC")
    ramp = np.arange(len(idx), dtype=float)

    up = pd.DataFrame(
        {
            "high": 100.0 + 0.8 * ramp + 1.0,
            "low": 100.0 + 0.8 * ramp - 1.0,
            "close": 100.0 + 0.8 * ramp,
        },
        index=idx,
    )
    down = pd.DataFrame(
        {
            "high": 250.0 - 0.8 * ramp + 1.0,
            "low": 250.0 - 0.8 * ramp - 1.0,
            "close": 250.0 - 0.8 * ramp,
        },
        index=idx,
    )

    _, plus_up, minus_up = calculate_adx(up, period=14)
    _, plus_down, minus_down = calculate_adx(down, period=14)

    assert pd.notna(plus_up.iloc[-1])
    assert pd.notna(minus_up.iloc[-1])
    assert plus_up.iloc[-1] > minus_up.iloc[-1]

    assert pd.notna(plus_down.iloc[-1])
    assert pd.notna(minus_down.iloc[-1])
    assert minus_down.iloc[-1] > plus_down.iloc[-1]


def test_identify_orb_ranges_boundaries_are_inclusive() -> None:
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-03 13:29:00", tz="UTC"),
            pd.Timestamp("2024-01-03 13:30:00", tz="UTC"),
            pd.Timestamp("2024-01-03 13:45:00", tz="UTC"),
            pd.Timestamp("2024-01-03 14:00:00", tz="UTC"),
            pd.Timestamp("2024-01-03 14:01:00", tz="UTC"),
        ]
    )
    df = pd.DataFrame(
        {
            "high": [9.0, 10.0, 12.0, 11.0, 20.0],
            "low": [7.0, 6.0, 5.0, 4.0, 1.0],
        },
        index=idx,
    )

    orb_ranges = identify_orb_ranges(
        df,
        orb_start_time=time(13, 30),
        orb_end_time=time(14, 0),
    )

    d = date(2024, 1, 3)
    assert d in orb_ranges.index
    assert orb_ranges.loc[d, "orb_high"] == 12.0
    assert orb_ranges.loc[d, "orb_low"] == 4.0


def test_identify_orb_ranges_single_bar_window_uses_exactly_one_bar() -> None:
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-04 13:59:00", tz="UTC"),
            pd.Timestamp("2024-01-04 14:00:00", tz="UTC"),
            pd.Timestamp("2024-01-04 14:01:00", tz="UTC"),
        ]
    )
    df = pd.DataFrame(
        {
            "high": [15.0, 21.5, 18.0],
            "low": [11.0, 19.5, 10.0],
        },
        index=idx,
    )

    orb_ranges = identify_orb_ranges(
        df,
        orb_start_time=time(14, 0),
        orb_end_time=time(14, 0),
    )

    d = date(2024, 1, 4)
    assert d in orb_ranges.index
    assert orb_ranges.loc[d, "orb_high"] == 21.5
    assert orb_ranges.loc[d, "orb_low"] == 19.5


def test_identify_orb_ranges_skips_days_without_orb_bars() -> None:
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-05 13:30:00", tz="UTC"),
            pd.Timestamp("2024-01-05 14:00:00", tz="UTC"),
            pd.Timestamp("2024-01-06 13:00:00", tz="UTC"),
            pd.Timestamp("2024-01-06 15:00:00", tz="UTC"),
        ]
    )
    df = pd.DataFrame(
        {
            "high": [11.0, 12.0, 30.0, 31.0],
            "low": [9.0, 8.0, 28.0, 29.0],
        },
        index=idx,
    )

    orb_ranges = identify_orb_ranges(
        df,
        orb_start_time=time(13, 30),
        orb_end_time=time(14, 0),
    )

    assert date(2024, 1, 5) in orb_ranges.index
    assert date(2024, 1, 6) not in orb_ranges.index

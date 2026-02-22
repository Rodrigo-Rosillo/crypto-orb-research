from __future__ import annotations

from datetime import date, time

import numpy as np
import pandas as pd
import pytest

from strategy import generate_orb_signals


def test_generate_orb_signals_respects_cutoff_and_fires_once_per_day() -> None:
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-10 13:59:00", tz="UTC"),
            pd.Timestamp("2024-01-10 14:00:00", tz="UTC"),
            pd.Timestamp("2024-01-10 14:01:00", tz="UTC"),
            pd.Timestamp("2024-01-10 14:02:00", tz="UTC"),
        ]
    )
    df = pd.DataFrame(
        {
            "close": [99.0, 99.0, 99.0, 98.0],
            "trend": ["downtrend", "downtrend", "downtrend", "downtrend"],
            "adx": [50.0, 50.0, 50.0, 50.0],
        },
        index=idx,
    )
    orb_ranges = pd.DataFrame(
        {"orb_high": [120.0], "orb_low": [100.0]},
        index=pd.Index([date(2024, 1, 10)], name="date"),
    )

    out = generate_orb_signals(
        df,
        orb_ranges=orb_ranges,
        adx_threshold=30.0,
        orb_cutoff_time=time(14, 0),
    )

    assert out.loc[pd.Timestamp("2024-01-10 14:00:00", tz="UTC"), "signal"] == 0
    assert out.loc[pd.Timestamp("2024-01-10 14:01:00", tz="UTC"), "signal"] == -1
    assert (
        out.loc[pd.Timestamp("2024-01-10 14:01:00", tz="UTC"), "signal_type"]
        == "downtrend_breakdown"
    )
    assert out.loc[pd.Timestamp("2024-01-10 14:02:00", tz="UTC"), "signal"] == 0
    assert int((out["signal"] != 0).sum()) == 1


@pytest.mark.parametrize(
    ("close_value", "adx_value", "trend_value"),
    [
        pytest.param(100.0, 50.0, "downtrend", id="close_equal_orb_low"),
        pytest.param(99.0, np.nan, "downtrend", id="adx_nan"),
        pytest.param(99.0, 29.0, "downtrend", id="adx_below_threshold"),
        pytest.param(99.0, 50.0, "uptrend", id="trend_not_downtrend"),
    ],
)
def test_generate_orb_signals_partial_conditions_do_not_fire(
    close_value: float,
    adx_value: float,
    trend_value: str,
) -> None:
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-11 14:01:00", tz="UTC")])
    df = pd.DataFrame(
        {
            "close": [close_value],
            "trend": [trend_value],
            "adx": [adx_value],
        },
        index=idx,
    )
    orb_ranges = pd.DataFrame(
        {"orb_high": [120.0], "orb_low": [100.0]},
        index=pd.Index([date(2024, 1, 11)], name="date"),
    )

    out = generate_orb_signals(
        df,
        orb_ranges=orb_ranges,
        adx_threshold=30.0,
        orb_cutoff_time=time(14, 0),
    )

    assert int((out["signal"] != 0).sum()) == 0
    assert (out["signal"] == 0).all()
    assert (out["signal_type"] == "").all()

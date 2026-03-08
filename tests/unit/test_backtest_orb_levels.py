from __future__ import annotations

from datetime import date

import pandas as pd

from backtester.futures_engine import FuturesEngineConfig, backtest_futures_orb
from backtester.spot_engine import backtest_orb_strategy


def signal_df() -> pd.DataFrame:
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-02-01 14:30", tz="UTC"),
            pd.Timestamp("2024-02-01 15:00", tz="UTC"),
        ]
    )
    return pd.DataFrame(
        {
            "open": [100.0, 100.0],
            "high": [100.0, 125.0],
            "low": [100.0, 99.0],
            "close": [100.0, 110.0],
            "date": [date(2024, 2, 1), date(2024, 2, 1)],
            "signal": [1, 0],
            "signal_type": ["uptrend_reversion", ""],
            "orb_high": [120.0, float("nan")],
            "orb_low": [80.0, float("nan")],
        },
        index=idx,
    )


def shared_orb_ranges() -> pd.DataFrame:
    return pd.DataFrame(
        {"orb_high": [150.0], "orb_low": [50.0]},
        index=pd.Index([date(2024, 2, 1)], name="date"),
    )


def test_futures_engine_prefers_row_level_orb_values() -> None:
    trades, _, stats = backtest_futures_orb(
        df=signal_df(),
        orb_ranges=shared_orb_ranges(),
        valid_days={date(2024, 2, 1)},
        cfg=FuturesEngineConfig(
            initial_capital=1_000.0,
            position_size=1.0,
            leverage=1.0,
            taker_fee_rate=0.0,
            delay_bars=1,
        ),
    )

    assert stats["trades"] == 1
    assert trades[0]["exit_reason"] == "target"
    assert trades[0]["exit_price"] == 120.0


def test_spot_engine_prefers_pending_row_level_orb_values() -> None:
    trades, _, _, _ = backtest_orb_strategy(
        df=signal_df(),
        orb_ranges=shared_orb_ranges(),
        initial_capital=1_000.0,
        position_size=1.0,
        taker_fee_rate=0.0,
        valid_days={date(2024, 2, 1)},
        delay_bars=1,
    )

    assert len(trades) == 1
    assert trades[0]["exit_reason"] == "target"
    assert trades[0]["exit_price"] == 120.0

from __future__ import annotations

from datetime import date

import pandas as pd

from backtester.futures_core import FuturesExecutionCore
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


def continuation_signal_df() -> pd.DataFrame:
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-02-02 15:00", tz="UTC"),
            pd.Timestamp("2024-02-02 15:30", tz="UTC"),
        ]
    )
    return pd.DataFrame(
        {
            "open": [101.0, 100.0],
            "high": [101.0, 102.5],
            "low": [100.0, 99.5],
            "close": [101.0, 101.5],
            "date": [date(2024, 2, 2), date(2024, 2, 2)],
            "signal": [2, 0],
            "signal_type": ["uptrend_continuation", ""],
            "orb_high": [104.0, float("nan")],
            "orb_low": [99.0, float("nan")],
        },
        index=idx,
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


def test_spot_engine_executes_uptrend_continuation_from_signal_type_spec() -> None:
    trades, _, _, _ = backtest_orb_strategy(
        df=continuation_signal_df(),
        orb_ranges=None,
        initial_capital=1_000.0,
        position_size=1.0,
        taker_fee_rate=0.0,
        valid_days={date(2024, 2, 2)},
        delay_bars=1,
    )

    assert len(trades) == 1
    assert trades[0]["type"] == "LONG"
    assert trades[0]["signal_type"] == "uptrend_continuation"
    assert trades[0]["entry_price"] == 100.0
    assert trades[0]["target_price"] == 102.0
    assert trades[0]["stop_loss"] == 99.0
    assert trades[0]["exit_reason"] == "target"
    assert trades[0]["exit_price"] == 102.0


def test_futures_core_executes_uptrend_continuation_from_signal_type_spec() -> None:
    trade_day = date(2024, 2, 2)
    cfg = FuturesEngineConfig(
        initial_capital=1_000.0,
        position_size=1.0,
        leverage=1.0,
        taker_fee_rate=0.0,
        delay_bars=1,
    )
    core = FuturesExecutionCore(cfg=cfg, risk_limits=None, expected_bar_seconds=1800)

    ts0 = pd.Timestamp("2024-02-02 15:00", tz="UTC")
    ts1 = pd.Timestamp("2024-02-02 15:30", tz="UTC")
    ts2 = pd.Timestamp("2024-02-02 16:00", tz="UTC")

    step0 = core.on_bar(
        ts=ts0,
        bar_open=101.0,
        bar_high=101.0,
        bar_low=100.0,
        bar_close=101.0,
        current_date=trade_day,
        signal=2,
        signal_type="uptrend_continuation",
        orb_high=104.0,
        orb_low=99.0,
        valid_days={trade_day},
    )

    assert step0["scheduled"] is True
    assert step0["scheduled_side"] == "LONG"

    step1 = core.on_bar(
        ts=ts1,
        bar_open=100.0,
        bar_high=101.0,
        bar_low=99.5,
        bar_close=100.5,
        current_date=trade_day,
        signal=0,
        signal_type="",
        orb_high=104.0,
        orb_low=99.0,
        valid_days={trade_day},
    )

    assert step1["entered"] is True
    assert core.side == "long"
    assert core.entry_price == 100.0
    assert core.target_price == 102.0
    assert core.stop_loss == 99.0

    core.on_bar(
        ts=ts2,
        bar_open=100.5,
        bar_high=102.5,
        bar_low=99.5,
        bar_close=102.0,
        current_date=trade_day,
        signal=0,
        signal_type="",
        orb_high=104.0,
        orb_low=99.0,
        valid_days={trade_day},
    )

    assert len(core.trades) == 1
    assert core.trades[0]["type"] == "LONG"
    assert core.trades[0]["signal_type"] == "uptrend_continuation"
    assert core.trades[0]["exit_reason"] == "target"
    assert core.trades[0]["exit_price"] == 102.0

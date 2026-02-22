from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from backtester.futures_core import FuturesExecutionCore
from backtester.futures_engine import FuturesEngineConfig


def test_futures_execution_core_fee_and_funding_round_trip_long() -> None:
    initial_capital = 1000.0
    position_size = 1.0
    leverage = 2.0
    fee_rate = 0.001
    mark_price = 100.0
    entry_price = 100.0
    exit_price = 110.0
    funding_rate = 0.0001

    trade_day = date(2024, 1, 2)
    ts0 = pd.Timestamp("2024-01-02 00:00:00", tz="UTC")
    ts1 = pd.Timestamp("2024-01-02 00:30:00", tz="UTC")
    ts2 = pd.Timestamp("2024-01-02 08:00:00", tz="UTC")
    ts3 = pd.Timestamp("2024-01-02 16:00:00", tz="UTC")
    exit_ts = pd.Timestamp("2024-01-02 16:30:00", tz="UTC")

    funding_series = pd.Series([funding_rate, funding_rate], index=pd.DatetimeIndex([ts2, ts3]))

    cfg = FuturesEngineConfig(
        initial_capital=initial_capital,
        position_size=position_size,
        leverage=leverage,
        taker_fee_rate=fee_rate,
        fee_mult=1.0,
        slippage_bps=0.0,
        delay_bars=1,
        funding_rate_per_8h=0.0,
        funding_series=funding_series,
    )
    core = FuturesExecutionCore(cfg=cfg, risk_limits=None, expected_bar_seconds=1800)
    valid_days = {trade_day}

    core.on_bar(
        ts=ts0,
        bar_open=mark_price,
        bar_high=101.0,
        bar_low=99.0,
        bar_close=mark_price,
        current_date=trade_day,
        signal=1,
        signal_type="unit_long",
        orb_high=130.0,
        orb_low=90.0,
        valid_days=valid_days,
    )
    core.on_bar(
        ts=ts1,
        bar_open=entry_price,
        bar_high=101.0,
        bar_low=99.0,
        bar_close=mark_price,
        current_date=trade_day,
        signal=0,
        signal_type="",
        orb_high=130.0,
        orb_low=90.0,
        valid_days=valid_days,
    )

    assert core.side == "long"
    assert core.qty > 0

    core.on_bar(
        ts=ts2,
        bar_open=mark_price,
        bar_high=101.0,
        bar_low=99.0,
        bar_close=mark_price,
        current_date=trade_day,
        signal=0,
        signal_type="",
        orb_high=130.0,
        orb_low=90.0,
        valid_days=valid_days,
    )
    core.on_bar(
        ts=ts3,
        bar_open=mark_price,
        bar_high=101.0,
        bar_low=99.0,
        bar_close=mark_price,
        current_date=trade_day,
        signal=0,
        signal_type="",
        orb_high=130.0,
        orb_low=90.0,
        valid_days=valid_days,
    )

    pnl_net = core.close_position(exit_ts=exit_ts, raw_exit_price=exit_price, reason="test")
    assert pnl_net is not None

    margin_used = initial_capital * position_size
    notional_entry = margin_used * leverage
    qty = notional_entry / entry_price
    entry_fee = notional_entry * fee_rate
    exit_fee = (qty * exit_price) * fee_rate
    pnl_gross = qty * (exit_price - entry_price)
    expected_total_funding = 2 * (qty * mark_price * funding_rate)
    expected_final_free_balance = (
        initial_capital - entry_fee - exit_fee - expected_total_funding + pnl_gross
    )

    assert core.total_funding == pytest.approx(expected_total_funding, abs=1e-8)
    assert core.free_balance == pytest.approx(expected_final_free_balance, abs=1e-8)

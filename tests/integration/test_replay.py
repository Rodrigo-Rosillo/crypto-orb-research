from __future__ import annotations

from pathlib import Path
import sys


import pandas as pd
import pytest
import yaml

from backtester.futures_engine import FuturesEngineConfig
from forward.binance_live import interval_to_seconds
from forward.replay import load_processed_parquet
from forward.shadow import build_signals, run_shadow_futures
from forward.stream_engine import StreamingFuturesShadowEngine

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
with open(REPO_ROOT / "config_forward_test.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

symbol = cfg["symbol"]
timeframe = cfg["timeframe"]
initial_capital = float(cfg["risk"]["initial_capital"])
position_size = float(cfg["risk"]["position_size"])
taker_fee_rate = float(cfg["fees"]["taker_fee_rate"])
lev_cfg = cfg.get("leverage") or {}
leverage = (
    float(lev_cfg.get("max_leverage", 1.0))
    if bool(lev_cfg.get("enabled", True))
    else 1.0
)
ft_cfg = cfg.get("forward_test") or {}
exec_cfg = ft_cfg.get("execution_model") or {}
delay_bars = int(exec_cfg.get("delay_bars", 1))
slippage_bps = float(exec_cfg.get("slippage_bps", 0.0))

start_utc = pd.Timestamp("2024-01-01", tz="UTC")
end_utc = pd.Timestamp("2024-03-31 23:59:59", tz="UTC")

parquet_path = REPO_ROOT / "data" / "processed" / f"{symbol}_{timeframe}.parquet"
valid_days_path = REPO_ROOT / "data" / "processed" / "valid_days.csv"

pytestmark = [
    pytest.mark.skipif(
        not parquet_path.exists(),
        reason=f"Parquet dataset not found: {parquet_path}. Run: python scripts/build_parquet.py",
    ),
    pytest.mark.skipif(
        not valid_days_path.exists(),
        reason=f"valid_days.csv not found: {valid_days_path}. Run: python scripts/build_parquet.py",
    ),
]

if parquet_path.exists() and valid_days_path.exists():
    df_slice, _ = load_processed_parquet(
        REPO_ROOT,
        symbol=symbol,
        timeframe=timeframe,
        start_utc=start_utc,
        end_utc=end_utc,
    )
    valid_days_df = pd.read_csv(valid_days_path)
    valid_days = set(pd.to_datetime(valid_days_df["date_utc"], utc=True).dt.date)
    bar_seconds = interval_to_seconds(timeframe)
else:
    df_slice = pd.DataFrame()
    valid_days_df = pd.DataFrame()
    valid_days = set()
    bar_seconds = 0


@pytest.fixture(scope="module")
def replay_results():
    batch_result = run_shadow_futures(
        df_raw=df_slice,
        valid_days=valid_days,
        cfg=cfg,
        delay_bars=delay_bars,
        slippage_bps=slippage_bps,
        fee_mult=1.0,
        funding_rate_per_8h=0.0,
        risk_limits=None,
    )
    batch_trades = batch_result.trades
    batch_final_equity = (
        float(batch_result.equity_curve.iloc[-1])
        if len(batch_result.equity_curve) > 0
        else initial_capital
    )

    df_sig, _, _ = build_signals(
        df_raw=df_slice,
        valid_days=valid_days,
        cfg=cfg,
    )

    cfg_obj = FuturesEngineConfig(
        initial_capital=initial_capital,
        position_size=position_size,
        leverage=leverage,
        taker_fee_rate=taker_fee_rate,
        fee_mult=1.0,
        slippage_bps=slippage_bps,
        delay_bars=delay_bars,
        funding_rate_per_8h=0.0,
    )
    engine = StreamingFuturesShadowEngine(
        cfg=cfg_obj,
        risk_limits=None,
        expected_bar_seconds=bar_seconds,
    )

    last_close = None
    for ts, row in df_sig.iterrows():
        engine.on_bar(
            ts=ts,
            bar_open=float(row["open"]),
            bar_high=float(row["high"]),
            bar_low=float(row["low"]),
            bar_close=float(row["close"]),
            current_date=row.get("date"),
            signal=int(row.get("signal", 0) or 0),
            signal_type=str(row.get("signal_type", "") or ""),
            orb_high=None if pd.isna(row.get("orb_high")) else float(row.get("orb_high")),
            orb_low=None if pd.isna(row.get("orb_low")) else float(row.get("orb_low")),
            valid_days=valid_days,
        )
        last_close = float(row["close"])

    streaming_trades = list(engine.core.trades)
    streaming_final_equity = (
        float(engine.equity(last_close)) if last_close is not None else initial_capital
    )

    return {
        "batch": batch_trades,
        "streaming": streaming_trades,
        "batch_final_equity": batch_final_equity,
        "streaming_final_equity": streaming_final_equity,
    }


def test_replay_trade_count(replay_results):
    batch_trades = replay_results["batch"]
    streaming_trades = replay_results["streaming"]

    assert len(batch_trades) == len(
        streaming_trades
    ), f"Trade count mismatch: batch={len(batch_trades)} streaming={len(streaming_trades)}"
    assert len(batch_trades) > 0, "Expected at least one trade in replay slice"


def test_replay_entry_times(replay_results):
    batch_trades = replay_results["batch"]
    streaming_trades = replay_results["streaming"]
    if len(batch_trades) != len(streaming_trades):
        pytest.skip("Trade counts differ — see test_replay_trade_count")

    for i, (batch_trade, streaming_trade) in enumerate(zip(batch_trades, streaming_trades)):
        assert (
            batch_trade["entry_time"] == streaming_trade["entry_time"]
        ), (
            f"Entry time mismatch at trade {i}: "
            f"batch={batch_trade['entry_time']} streaming={streaming_trade['entry_time']}"
        )


def test_replay_exit_times(replay_results):
    batch_trades = replay_results["batch"]
    streaming_trades = replay_results["streaming"]
    if len(batch_trades) != len(streaming_trades):
        pytest.skip("Trade counts differ — see test_replay_trade_count")

    for i, (batch_trade, streaming_trade) in enumerate(zip(batch_trades, streaming_trades)):
        assert (
            batch_trade["exit_time"] == streaming_trade["exit_time"]
        ), (
            f"Exit time mismatch at trade {i}: "
            f"batch={batch_trade['exit_time']} streaming={streaming_trade['exit_time']}"
        )


def test_replay_pnl_per_trade(replay_results):
    batch_trades = replay_results["batch"]
    streaming_trades = replay_results["streaming"]
    if len(batch_trades) != len(streaming_trades):
        pytest.skip("Trade counts differ — see test_replay_trade_count")

    for i, (batch_trade, streaming_trade) in enumerate(zip(batch_trades, streaming_trades)):
        batch_pnl = float(batch_trade["pnl_net"])
        streaming_pnl = float(streaming_trade["pnl_net"])
        diff = abs(batch_pnl - streaming_pnl)
        assert (
            diff < 1e-8
        ), f"PnL mismatch at trade {i}: batch={batch_pnl} streaming={streaming_pnl} diff={diff}"


def test_replay_final_equity(replay_results):
    batch_trades = replay_results["batch"]
    streaming_trades = replay_results["streaming"]
    if len(batch_trades) != len(streaming_trades):
        pytest.skip("Trade counts differ — see test_replay_trade_count")

    batch_final_equity = float(replay_results["batch_final_equity"])
    streaming_final_equity = float(replay_results["streaming_final_equity"])
    diff = abs(batch_final_equity - streaming_final_equity)
    assert (
        diff < 1e-8
    ), f"Final equity mismatch: batch={batch_final_equity} streaming={streaming_final_equity} diff={diff}"

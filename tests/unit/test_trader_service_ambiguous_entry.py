from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd

from forward.state_store_sqlite import SQLiteStateStore
from forward.testnet_broker import AmbiguousOrderError
from tests.integration.mocks import FakeBinanceClient, build_trader_service


def _build_subject(tmp_path: Path, *, broker: FakeBinanceClient) -> tuple[SQLiteStateStore, object]:
    db_path = tmp_path / "state.db"
    store = SQLiteStateStore(db_path=db_path)
    store.open()
    state = store.load_state()
    trader = build_trader_service(
        broker=broker,
        store=store,
        state=state,
        work_dir=tmp_path,
        leverage=1.0,
        position_size=0.1,
        initial_capital=1000.0,
        max_order_rejects_per_day=10,
    )
    return store, trader


def test_recover_ambiguous_entry_persists_provisional_state_with_bar_time(tmp_path: Path) -> None:
    broker = FakeBinanceClient(fill_price=100.0)
    broker._position_amt = -1.0
    broker._entry_price = 100.0

    store, trader = _build_subject(tmp_path, broker=broker)
    try:
        bar_open_time = pd.Timestamp("2024-01-01 00:30:00", tz="UTC")

        recovered = asyncio.run(
            trader._recover_ambiguous_entry(
                error=AmbiguousOrderError(
                    "ambiguous",
                    client_order_id="entry-test",
                    context={"symbol": "SOLUSDT", "side": "SELL"},
                ),
                bar_open_time=bar_open_time,
                signal_type="unit",
                pos_side="SHORT",
                qty_sent=1.0,
            )
        )

        assert recovered is not None
        loaded = store.load_state()
        assert loaded.open_position is not None
        assert loaded.open_position.side == "SHORT"
        assert loaded.open_position.qty == 1.0
        assert loaded.open_position.entry_price == 100.0
        assert loaded.open_position.entry_time_utc == "2024-01-01T00:30:00+00:00"
        assert loaded.open_position.entry_order_id is None
        assert loaded.open_position.tp_order_id is None
        assert loaded.open_position.sl_order_id is None
    finally:
        store.close()


def test_recover_ambiguous_entry_clears_state_only_after_confirmed_flatten(tmp_path: Path) -> None:
    broker = FakeBinanceClient(fill_price=100.0)
    broker._position_amt = -1.0
    broker._entry_price = 0.0

    store, trader = _build_subject(tmp_path, broker=broker)
    try:
        bar_open_time = pd.Timestamp("2024-01-01 00:30:00", tz="UTC")

        recovered = asyncio.run(
            trader._recover_ambiguous_entry(
                error=AmbiguousOrderError(
                    "ambiguous",
                    client_order_id="entry-test",
                    context={"symbol": "SOLUSDT", "side": "SELL"},
                ),
                bar_open_time=bar_open_time,
                signal_type="unit",
                pos_side="SHORT",
                qty_sent=1.0,
            )
        )

        assert recovered is None
        loaded = store.load_state()
        assert loaded.open_position is None
        assert trader.stop_event.is_set() is True
        assert broker._position_amt == 0.0
    finally:
        store.close()


def test_recover_ambiguous_entry_preserves_known_position_when_flatten_fails(tmp_path: Path) -> None:
    broker = FakeBinanceClient(reject_flatten=True, fill_price=100.0)
    broker._position_amt = -1.0
    broker._entry_price = 0.0

    store, trader = _build_subject(tmp_path, broker=broker)
    try:
        bar_open_time = pd.Timestamp("2024-01-01 00:30:00", tz="UTC")

        recovered = asyncio.run(
            trader._recover_ambiguous_entry(
                error=AmbiguousOrderError(
                    "ambiguous",
                    client_order_id="entry-test",
                    context={"symbol": "SOLUSDT", "side": "SELL"},
                ),
                bar_open_time=bar_open_time,
                signal_type="unit",
                pos_side="SHORT",
                qty_sent=1.0,
            )
        )

        assert recovered is None
        loaded = store.load_state()
        assert loaded.open_position is not None
        assert loaded.open_position.side == "SHORT"
        assert loaded.open_position.qty == 1.0
        assert loaded.open_position.entry_price == 0.0
        assert loaded.open_position.entry_time_utc == "2024-01-01T00:30:00+00:00"
        assert trader.stop_event.is_set() is True
        assert trader.skip_cancel_open_orders_on_exit_runtime is True
        assert broker._position_amt == -1.0
    finally:
        store.close()

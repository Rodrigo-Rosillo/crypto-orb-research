from __future__ import annotations

from pathlib import Path

import pytest

from forward.state_store_sqlite import OpenPositionState, SQLiteStateStore
from tests.integration.mocks import FakeBinanceClient, build_trader_service


def _seed_open_position(*, side: str = "LONG", qty: float = 1.0) -> OpenPositionState:
    return OpenPositionState(
        symbol="SOLUSDT",
        side=str(side),
        qty=float(qty),
        entry_price=100.0,
        entry_time_utc="2026-03-05T00:00:00+00:00",
        entry_order_id=123,
        tp_order_id=456,
        sl_order_id=789,
        tp_price=102.0,
        sl_price=98.0,
    )


def _build_subject(tmp_path: Path) -> tuple[SQLiteStateStore, object, FakeBinanceClient]:
    db_path = tmp_path / "state.db"
    store = SQLiteStateStore(db_path=db_path)
    store.open()
    state = store.load_state()
    broker = FakeBinanceClient(fill_price=100.0)
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
    return store, trader, broker


def test_reconciliation_matches_flat_state_and_exchange(tmp_path: Path) -> None:
    store, trader, _ = _build_subject(tmp_path)
    try:
        result = trader.classify_exchange_position_reconciliation()

        assert result["status"] == "match"
        snapshot = result["snapshot"]
        assert snapshot["side"] == "FLAT"
        assert snapshot["qty"] == pytest.approx(0.0, abs=1e-12)
    finally:
        store.close()


def test_reconciliation_matches_open_position_within_qty_tolerance(tmp_path: Path) -> None:
    store, trader, broker = _build_subject(tmp_path)
    try:
        trader.state.open_position = _seed_open_position(side="LONG", qty=1.0)
        broker._position_amt = 1.0 + 5e-7

        result = trader.classify_exchange_position_reconciliation()

        assert result["status"] == "match"
        assert result["snapshot"]["side"] == "LONG"
        assert result["snapshot"]["qty"] == pytest.approx(1.0 + 5e-7, abs=1e-12)
    finally:
        store.close()


def test_reconciliation_flags_exchange_open_when_local_state_is_flat(tmp_path: Path) -> None:
    store, trader, broker = _build_subject(tmp_path)
    try:
        broker._position_amt = 1.0

        result = trader.classify_exchange_position_reconciliation()

        assert result["status"] == "mismatch"
        assert result["flatten_on_mismatch"] is True
        assert result["payload"] == {
            "state": "FLAT",
            "exchange": "LONG",
            "qty": pytest.approx(1.0, abs=1e-12),
            "entry_price": pytest.approx(100.0, abs=1e-12),
        }
    finally:
        store.close()


def test_reconciliation_flags_exchange_flat_when_local_state_is_open(tmp_path: Path) -> None:
    store, trader, _ = _build_subject(tmp_path)
    try:
        trader.state.open_position = _seed_open_position(side="LONG", qty=1.0)

        result = trader.classify_exchange_position_reconciliation()

        assert result["status"] == "mismatch"
        assert result["flatten_on_mismatch"] is False
        assert result["payload"] == {
            "state": "LONG",
            "exchange": "FLAT",
        }
    finally:
        store.close()


def test_reconciliation_flags_side_or_qty_drift(tmp_path: Path) -> None:
    store, trader, broker = _build_subject(tmp_path)
    try:
        trader.state.open_position = _seed_open_position(side="LONG", qty=1.0)
        broker._position_amt = -1.5

        result = trader.classify_exchange_position_reconciliation()

        assert result["status"] == "mismatch"
        assert result["flatten_on_mismatch"] is False
        assert result["payload"] == {
            "state": trader.state.open_position.to_dict(),
            "exchange": {
                "side": "SHORT",
                "qty": pytest.approx(1.5, abs=1e-12),
            },
        }
    finally:
        store.close()

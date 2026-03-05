from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from forward.state_store_sqlite import OpenPositionState, RunnerState, SQLiteStateStore
from forward.trader_service import TraderService


class FakeBroker:
    def __init__(
        self,
        *,
        position_amt: float,
        entry_price: float,
        algo_order_by_id: dict[int, dict[str, Any]] | None = None,
        flatten_avg_price: float = 100.0,
        flatten_exec_qty: float = 1.0,
    ) -> None:
        self.position_amt = float(position_amt)
        self.entry_price = float(entry_price)
        self.algo_order_by_id = dict(algo_order_by_id or {})
        self.flatten_avg_price = float(flatten_avg_price)
        self.flatten_exec_qty = float(flatten_exec_qty)
        self._next_oid = 1000

    def position_risk(self, *, symbol: str) -> Any:
        return {
            "symbol": str(symbol),
            "positionAmt": f"{self.position_amt}",
            "entryPrice": f"{self.entry_price}",
            "unRealizedProfit": "0",
        }

    def get_algo_order(self, *, symbol: str, algo_id: int) -> Any:
        _ = symbol
        return dict(self.algo_order_by_id.get(int(algo_id), {"algoId": int(algo_id), "status": "NEW"}))

    def place_market_order(self, *, symbol: str, side: str, quantity: float, reduce_only: bool = False) -> Any:
        _ = symbol
        self._next_oid += 1
        qty = float(quantity)
        side_u = str(side).upper()

        if side_u == "BUY":
            self.position_amt += qty
        elif side_u == "SELL":
            self.position_amt -= qty

        if abs(self.position_amt) < 1e-9:
            self.position_amt = 0.0

        if bool(reduce_only):
            self.position_amt = 0.0

        return {
            "orderId": int(self._next_oid),
            "status": "FILLED",
            "avgPrice": f"{self.flatten_avg_price}",
            "executedQty": f"{self.flatten_exec_qty}",
            "origQty": f"{qty}",
        }


def _seed_open_position(
    state: RunnerState,
    *,
    side: str,
    qty: float,
    entry_price: float,
    tp_order_id: int | None,
    sl_order_id: int | None,
) -> None:
    state.open_position = OpenPositionState(
        symbol="SOLUSDT",
        side=str(side),
        qty=float(qty),
        entry_price=float(entry_price),
        entry_time_utc="2026-01-01T00:00:00+00:00",
        entry_order_id=123,
        tp_order_id=tp_order_id,
        sl_order_id=sl_order_id,
        tp_price=105.0,
        sl_price=95.0,
    )


def _build_trader(
    *,
    tmp_path: Path,
    broker: FakeBroker,
    store: SQLiteStateStore,
    state: RunnerState,
    emitted: list[dict[str, Any]],
) -> TraderService:
    def _append_rows(path: Path, rows: list[dict[str, Any]], columns: list[str], name: str) -> None:
        _ = (path, rows, columns, name)

    def _emit_event(rows: list[dict[str, Any]]) -> None:
        emitted.extend(rows)

    return TraderService(
        broker=broker,
        store=store,
        state=state,
        symbol="SOLUSDT",
        leverage=1.0,
        position_size=0.2,
        initial_capital=1000.0,
        slippage_bps=0.0,
        taker_fee_rate=0.0005,
        state_path=tmp_path / "state.json",
        events_path=tmp_path / "events.jsonl",
        run_id="unit-test",
        stop_event=asyncio.Event(),
        risk_limits=None,
        max_order_rejects_per_day=10,
        margin_ratio_threshold=0.85,
        orders_path=tmp_path / "orders.csv",
        fills_path=tmp_path / "fills.csv",
        positions_path=tmp_path / "positions.csv",
        append_rows=_append_rows,
        emit_event=_emit_event,
    )


def _latest_exit_row(db_path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """
            SELECT id, event_type, symbol, side, qty, price, realized_pnl, fee, funding_applied, reason
            FROM trade_log
            WHERE event_type = 'EXIT'
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
        assert row is not None
        return dict(row)
    finally:
        conn.close()


def test_exit_row_enriched_long_tp(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    emitted: list[dict[str, Any]] = []
    with SQLiteStateStore(db_path=db_path) as store:
        state = store.load_state()
        _seed_open_position(state, side="LONG", qty=1.0, entry_price=100.0, tp_order_id=2001, sl_order_id=None)
        store.save_state(state)

        broker = FakeBroker(
            position_amt=0.0,
            entry_price=100.0,
            algo_order_by_id={
                2001: {
                    "algoId": 2001,
                    "status": "FINISHED",
                    "avgPrice": "105.0",
                    "executedQty": "1.0",
                }
            },
        )
        trader = _build_trader(tmp_path=tmp_path, broker=broker, store=store, state=state, emitted=emitted)

        asyncio.run(trader.poll_open_orders())

    exit_row = _latest_exit_row(db_path)
    assert float(exit_row["price"]) == pytest.approx(105.0)
    assert float(exit_row["fee"]) == pytest.approx(0.1025)
    assert float(exit_row["realized_pnl"]) == pytest.approx(4.8975)


def test_exit_row_enriched_short_sl(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    emitted: list[dict[str, Any]] = []
    with SQLiteStateStore(db_path=db_path) as store:
        state = store.load_state()
        _seed_open_position(state, side="SHORT", qty=1.0, entry_price=100.0, tp_order_id=2002, sl_order_id=None)
        store.save_state(state)

        broker = FakeBroker(
            position_amt=0.0,
            entry_price=100.0,
            algo_order_by_id={
                2002: {
                    "algoId": 2002,
                    "status": "FINISHED",
                    "avgPrice": "98.0",
                    "executedQty": "1.0",
                }
            },
        )
        trader = _build_trader(tmp_path=tmp_path, broker=broker, store=store, state=state, emitted=emitted)

        asyncio.run(trader.poll_open_orders())

    exit_row = _latest_exit_row(db_path)
    fee = (100.0 + 98.0) * 0.0005
    expected = 2.0 - fee
    assert float(exit_row["realized_pnl"]) > 0
    assert float(exit_row["realized_pnl"]) == pytest.approx(expected)


def test_exit_row_fallback_when_avg_price_zero(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    emitted: list[dict[str, Any]] = []
    with SQLiteStateStore(db_path=db_path) as store:
        state = store.load_state()
        _seed_open_position(state, side="LONG", qty=1.0, entry_price=100.0, tp_order_id=2003, sl_order_id=None)
        store.save_state(state)

        broker = FakeBroker(
            position_amt=0.0,
            entry_price=100.0,
            algo_order_by_id={
                2003: {
                    "algoId": 2003,
                    "status": "FINISHED",
                    "avgPrice": "0",
                    "executedQty": "1.0",
                }
            },
        )
        trader = _build_trader(tmp_path=tmp_path, broker=broker, store=store, state=state, emitted=emitted)

        asyncio.run(trader.poll_open_orders())

    fallback_events = [e for e in emitted if str(e.get("type") or "") == "EXIT_ENRICH_FALLBACK"]
    failed_events = [e for e in emitted if str(e.get("type") or "") == "EXIT_ENRICH_FAILED"]
    assert len(fallback_events) >= 1
    assert len(failed_events) == 0

    exit_row = _latest_exit_row(db_path)
    assert float(exit_row["price"]) == pytest.approx(100.0)
    assert float(exit_row["realized_pnl"]) == pytest.approx(-0.1)


def test_exit_row_emergency_flatten_enriched(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    emitted: list[dict[str, Any]] = []
    with SQLiteStateStore(db_path=db_path) as store:
        state = store.load_state()
        _seed_open_position(state, side="LONG", qty=1.0, entry_price=100.0, tp_order_id=2004, sl_order_id=2005)
        store.save_state(state)

        broker = FakeBroker(
            position_amt=1.0,
            entry_price=100.0,
            algo_order_by_id={},
            flatten_avg_price=102.0,
            flatten_exec_qty=1.0,
        )
        trader = _build_trader(tmp_path=tmp_path, broker=broker, store=store, state=state, emitted=emitted)

        flatten_ok, detail = trader._emergency_flatten(reason="SHUTDOWN_GUARD")

    assert flatten_ok is True
    assert detail == "flattened"

    exit_row = _latest_exit_row(db_path)
    fee = (100.0 + 102.0) * 0.0005
    expected = 2.0 - fee
    assert float(exit_row["fee"]) == pytest.approx(fee)
    assert float(exit_row["realized_pnl"]) == pytest.approx(expected)

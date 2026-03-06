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
        position_amts: list[float] | None = None,
        algo_order_error_by_id: dict[int, Exception] | None = None,
        cancel_algo_error_by_id: dict[int, Exception] | None = None,
    ) -> None:
        self.position_amt = float(position_amt)
        self.entry_price = float(entry_price)
        self.algo_order_by_id = dict(algo_order_by_id or {})
        self.flatten_avg_price = float(flatten_avg_price)
        self.flatten_exec_qty = float(flatten_exec_qty)
        self.position_amts = [float(v) for v in (position_amts or [])]
        self.algo_order_error_by_id = dict(algo_order_error_by_id or {})
        self.cancel_algo_error_by_id = dict(cancel_algo_error_by_id or {})
        self.position_risk_call_count = 0
        self.cancel_algo_calls: list[int] = []
        self.cancelled_algo_ids: list[int] = []
        self._next_oid = 1000

    def position_risk(self, *, symbol: str) -> Any:
        idx = min(self.position_risk_call_count, len(self.position_amts) - 1) if self.position_amts else -1
        amt = self.position_amts[idx] if idx >= 0 else self.position_amt
        self.position_risk_call_count += 1
        return {
            "symbol": str(symbol),
            "positionAmt": f"{amt}",
            "entryPrice": f"{self.entry_price if abs(float(amt)) > 1e-9 else 0.0}",
            "unRealizedProfit": "0",
        }

    def get_algo_order(self, *, symbol: str, algo_id: int) -> Any:
        _ = symbol
        oid = int(algo_id)
        error = self.algo_order_error_by_id.get(oid)
        if error is not None:
            raise error
        return dict(self.algo_order_by_id.get(oid, {"algoId": oid, "status": "NEW"}))

    def cancel_algo_order(self, *, algo_id: int, symbol: str | None = None) -> Any:
        _ = symbol
        oid = int(algo_id)
        self.cancel_algo_calls.append(oid)
        error = self.cancel_algo_error_by_id.get(oid)
        if error is not None:
            raise error
        self.cancelled_algo_ids.append(oid)
        if oid in self.algo_order_by_id:
            row = dict(self.algo_order_by_id[oid])
            row["status"] = "CANCELED"
            self.algo_order_by_id[oid] = row
        return {"algoId": oid, "status": "CANCELED"}

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


def _trade_log_rows(db_path: Path, event_type: str) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT id, event_type, symbol, side, qty, price, realized_pnl, fee, funding_applied, reason
            FROM trade_log
            WHERE event_type = ?
            ORDER BY id ASC
            """,
            (str(event_type),),
        ).fetchall()
        return [dict(row) for row in rows]
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


def test_poll_open_orders_tp_finished_cancels_open_sl_and_persists(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    emitted: list[dict[str, Any]] = []
    with SQLiteStateStore(db_path=db_path) as store:
        state = store.load_state()
        _seed_open_position(state, side="LONG", qty=1.0, entry_price=100.0, tp_order_id=2101, sl_order_id=2102)
        store.save_state(state)

        broker = FakeBroker(
            position_amt=0.0,
            entry_price=100.0,
            algo_order_by_id={
                2101: {"algoId": 2101, "status": "FINISHED", "avgPrice": "105.0", "executedQty": "1.0"},
                2102: {"algoId": 2102, "status": "NEW"},
            },
        )
        trader = _build_trader(tmp_path=tmp_path, broker=broker, store=store, state=state, emitted=emitted)

        asyncio.run(trader.poll_open_orders())

        reloaded = store.load_state()

    assert trader.state.open_position is None
    assert reloaded.open_position is None
    assert broker.cancelled_algo_ids == [2102]
    assert trader.skip_cancel_open_orders_on_exit_runtime is False
    assert trader.stop_event.is_set() is False
    assert _latest_exit_row(db_path)["reason"] == "tp"


def test_poll_open_orders_sl_finished_cancels_open_tp_and_persists(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    emitted: list[dict[str, Any]] = []
    with SQLiteStateStore(db_path=db_path) as store:
        state = store.load_state()
        _seed_open_position(state, side="SHORT", qty=1.0, entry_price=100.0, tp_order_id=2201, sl_order_id=2202)
        store.save_state(state)

        broker = FakeBroker(
            position_amt=0.0,
            entry_price=100.0,
            algo_order_by_id={
                2201: {"algoId": 2201, "status": "NEW"},
                2202: {"algoId": 2202, "status": "FINISHED", "avgPrice": "103.0", "executedQty": "1.0"},
            },
        )
        trader = _build_trader(tmp_path=tmp_path, broker=broker, store=store, state=state, emitted=emitted)

        asyncio.run(trader.poll_open_orders())

        reloaded = store.load_state()

    assert trader.state.open_position is None
    assert reloaded.open_position is None
    assert broker.cancelled_algo_ids == [2201]
    assert trader.skip_cancel_open_orders_on_exit_runtime is False
    assert trader.stop_event.is_set() is False
    assert _latest_exit_row(db_path)["reason"] == "sl"


def test_poll_open_orders_both_finished_prefers_tp_once(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    emitted: list[dict[str, Any]] = []
    with SQLiteStateStore(db_path=db_path) as store:
        state = store.load_state()
        _seed_open_position(state, side="LONG", qty=1.0, entry_price=100.0, tp_order_id=2301, sl_order_id=2302)
        store.save_state(state)

        broker = FakeBroker(
            position_amt=0.0,
            entry_price=100.0,
            algo_order_by_id={
                2301: {"algoId": 2301, "status": "FINISHED", "avgPrice": "105.0", "executedQty": "1.0"},
                2302: {"algoId": 2302, "status": "FINISHED", "avgPrice": "95.0", "executedQty": "1.0"},
            },
        )
        trader = _build_trader(tmp_path=tmp_path, broker=broker, store=store, state=state, emitted=emitted)

        asyncio.run(trader.poll_open_orders())

    anomaly_events = [e for e in emitted if str(e.get("type") or "") == "PROTECTION_RUNTIME_ANOMALY"]
    assert broker.cancel_algo_calls == []
    assert len(_trade_log_rows(db_path, "EXIT")) == 1
    assert _latest_exit_row(db_path)["reason"] == "tp"
    assert any(str(e.get("reason") or "") == "dual_finished" for e in anomaly_events)


def test_poll_open_orders_terminal_flat_race_treated_as_normal_close(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    emitted: list[dict[str, Any]] = []
    with SQLiteStateStore(db_path=db_path) as store:
        state = store.load_state()
        _seed_open_position(state, side="LONG", qty=1.0, entry_price=100.0, tp_order_id=2401, sl_order_id=2402)
        store.save_state(state)

        broker = FakeBroker(
            position_amt=0.0,
            entry_price=100.0,
            algo_order_by_id={
                2401: {"algoId": 2401, "status": "CANCELED"},
                2402: {"algoId": 2402, "status": "NEW"},
            },
        )
        trader = _build_trader(tmp_path=tmp_path, broker=broker, store=store, state=state, emitted=emitted)

        asyncio.run(trader.poll_open_orders())

        reloaded = store.load_state()

    anomaly_events = [e for e in emitted if str(e.get("type") or "") == "PROTECTION_RUNTIME_ANOMALY"]
    assert trader.state.open_position is None
    assert reloaded.open_position is None
    assert trader.stop_event.is_set() is False
    assert trader.skip_cancel_open_orders_on_exit_runtime is False
    assert broker.cancelled_algo_ids == [2402]
    assert _latest_exit_row(db_path)["reason"] == "tp"
    assert any(str(e.get("reason") or "") == "terminal_flat_race" for e in anomaly_events)


def test_poll_open_orders_terminal_leg_with_exchange_open_kills(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    emitted: list[dict[str, Any]] = []
    with SQLiteStateStore(db_path=db_path) as store:
        state = store.load_state()
        _seed_open_position(state, side="LONG", qty=1.0, entry_price=100.0, tp_order_id=2501, sl_order_id=2502)
        store.save_state(state)

        broker = FakeBroker(
            position_amt=1.0,
            entry_price=100.0,
            algo_order_by_id={
                2501: {"algoId": 2501, "status": "EXPIRED"},
                2502: {"algoId": 2502, "status": "NEW"},
            },
        )
        trader = _build_trader(tmp_path=tmp_path, broker=broker, store=store, state=state, emitted=emitted)

        asyncio.run(trader.poll_open_orders())

        reloaded = store.load_state()

    kill_events = [e for e in emitted if str(e.get("type") or "") == "KILL_SWITCH_PROTECTION_RUNTIME"]
    assert trader.state.open_position is not None
    assert reloaded.open_position is not None
    assert trader.stop_event.is_set() is True
    assert trader.skip_cancel_open_orders_on_exit_runtime is True
    assert broker.cancel_algo_calls == []
    assert len(_trade_log_rows(db_path, "EXIT")) == 0
    assert len(kill_events) == 1


def test_poll_open_orders_query_exception_kills(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    emitted: list[dict[str, Any]] = []
    with SQLiteStateStore(db_path=db_path) as store:
        state = store.load_state()
        _seed_open_position(state, side="LONG", qty=1.0, entry_price=100.0, tp_order_id=2601, sl_order_id=2602)
        store.save_state(state)

        broker = FakeBroker(
            position_amt=1.0,
            entry_price=100.0,
            algo_order_by_id={2602: {"algoId": 2602, "status": "NEW"}},
            algo_order_error_by_id={2601: RuntimeError("poll boom")},
        )
        trader = _build_trader(tmp_path=tmp_path, broker=broker, store=store, state=state, emitted=emitted)

        asyncio.run(trader.poll_open_orders())

        reloaded = store.load_state()

    kill_events = [e for e in emitted if str(e.get("type") or "") == "KILL_SWITCH_PROTECTION_RUNTIME"]
    assert trader.state.open_position is not None
    assert reloaded.open_position is not None
    assert trader.stop_event.is_set() is True
    assert trader.skip_cancel_open_orders_on_exit_runtime is True
    assert len(_trade_log_rows(db_path, "EXIT")) == 0
    assert len(kill_events) == 1


def test_poll_open_orders_finished_leg_with_exchange_open_kills(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    emitted: list[dict[str, Any]] = []
    with SQLiteStateStore(db_path=db_path) as store:
        state = store.load_state()
        _seed_open_position(state, side="LONG", qty=1.0, entry_price=100.0, tp_order_id=2701, sl_order_id=2702)
        store.save_state(state)

        broker = FakeBroker(
            position_amt=1.0,
            entry_price=100.0,
            algo_order_by_id={
                2701: {"algoId": 2701, "status": "FINISHED", "avgPrice": "105.0", "executedQty": "1.0"},
                2702: {"algoId": 2702, "status": "NEW"},
            },
        )
        trader = _build_trader(tmp_path=tmp_path, broker=broker, store=store, state=state, emitted=emitted)

        asyncio.run(trader.poll_open_orders())

        reloaded = store.load_state()

    kill_events = [e for e in emitted if str(e.get("type") or "") == "KILL_SWITCH_PROTECTION_RUNTIME"]
    assert trader.state.open_position is not None
    assert reloaded.open_position is not None
    assert trader.stop_event.is_set() is True
    assert trader.skip_cancel_open_orders_on_exit_runtime is True
    assert len(_trade_log_rows(db_path, "EXIT")) == 0
    assert len(kill_events) == 1


def test_poll_open_orders_sibling_cancel_failure_after_flat_close_kills_after_persist(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    emitted: list[dict[str, Any]] = []
    with SQLiteStateStore(db_path=db_path) as store:
        state = store.load_state()
        _seed_open_position(state, side="LONG", qty=1.0, entry_price=100.0, tp_order_id=2801, sl_order_id=2802)
        store.save_state(state)

        broker = FakeBroker(
            position_amt=0.0,
            entry_price=100.0,
            algo_order_by_id={
                2801: {"algoId": 2801, "status": "FINISHED", "avgPrice": "105.0", "executedQty": "1.0"},
                2802: {"algoId": 2802, "status": "NEW"},
            },
            cancel_algo_error_by_id={2802: RuntimeError("cancel failed")},
        )
        trader = _build_trader(tmp_path=tmp_path, broker=broker, store=store, state=state, emitted=emitted)

        asyncio.run(trader.poll_open_orders())

        reloaded = store.load_state()

    kill_events = [e for e in emitted if str(e.get("type") or "") == "KILL_SWITCH_PROTECTION_RUNTIME"]
    assert trader.state.open_position is None
    assert reloaded.open_position is None
    assert trader.stop_event.is_set() is True
    assert trader.skip_cancel_open_orders_on_exit_runtime is False
    assert broker.cancel_algo_calls == [2802]
    assert len(_trade_log_rows(db_path, "EXIT")) == 1
    assert len(kill_events) == 1

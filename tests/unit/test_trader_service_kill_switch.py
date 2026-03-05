from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from forward.state_store_sqlite import OpenPositionState, RunnerState, SQLiteStateStore
from forward.trader_service import TraderService


class FakeBroker:
    def __init__(
        self,
        *,
        margin_maint: float,
        margin_balance: float,
        position_amt: float,
        fail_flatten_submit: bool = False,
    ) -> None:
        self.margin_maint = float(margin_maint)
        self.margin_balance = float(margin_balance)
        self.position_amt = float(position_amt)
        self.entry_price = 100.0
        self.fail_flatten_submit = bool(fail_flatten_submit)
        self._next_oid = 1000

    def account(self) -> Any:
        return {
            "totalMaintMargin": f"{self.margin_maint}",
            "totalMarginBalance": f"{self.margin_balance}",
        }

    def position_risk(self, *, symbol: str) -> Any:
        return {
            "symbol": str(symbol),
            "positionAmt": f"{self.position_amt}",
            "entryPrice": f"{self.entry_price}",
            "unRealizedProfit": "0",
        }

    def place_market_order(self, *, symbol: str, side: str, quantity: float, reduce_only: bool = False) -> Any:
        _ = symbol
        self._next_oid += 1

        if bool(reduce_only) and self.fail_flatten_submit:
            raise RuntimeError("simulated_flatten_submit_error")

        qty = float(quantity)
        side_u = str(side).upper()
        if side_u == "BUY":
            self.position_amt += qty
        elif side_u == "SELL":
            self.position_amt -= qty

        if abs(self.position_amt) < 1e-9:
            self.position_amt = 0.0

        return {
            "orderId": int(self._next_oid),
            "status": "FILLED",
            "avgPrice": f"{self.entry_price}",
            "executedQty": f"{qty}",
            "origQty": f"{qty}",
        }


def _seed_open_position(state: RunnerState) -> None:
    state.open_position = OpenPositionState(
        symbol="SOLUSDT",
        side="LONG",
        qty=0.2,
        entry_price=100.0,
        entry_time_utc="2024-01-01T00:00:00+00:00",
        entry_order_id=123,
        tp_order_id=456,
        sl_order_id=789,
        tp_price=102.0,
        sl_price=98.0,
    )


def _build_trader_service(*, tmp_path: Path, broker: FakeBroker, store: SQLiteStateStore, state: RunnerState) -> TraderService:
    def _append_rows(path: Path, rows: list[dict[str, Any]], columns: list[str], name: str) -> None:
        _ = (path, rows, columns, name)

    def _emit_event(rows: list[dict[str, Any]]) -> None:
        _ = rows

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


def _simulate_shutdown_guard(trader: TraderService) -> None:
    if trader.state.open_position is None:
        return
    try:
        flatten_ok, _ = trader._emergency_flatten(reason="SHUTDOWN_GUARD")
        if flatten_ok:
            trader.state.open_position = None
            trader.skip_cancel_open_orders_on_exit_runtime = False
            try:
                trader.persist_state()
            except Exception:
                pass
        else:
            trader.skip_cancel_open_orders_on_exit_runtime = True
    except Exception:
        trader.skip_cancel_open_orders_on_exit_runtime = True


def test_margin_ratio_kill_switch_flattens_open_position(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteStateStore(db_path=db_path) as store:
        state = store.load_state()
        _seed_open_position(state)

        broker = FakeBroker(margin_maint=90.0, margin_balance=100.0, position_amt=0.2)
        trader = _build_trader_service(tmp_path=tmp_path, broker=broker, store=store, state=state)

        flatten_calls: list[str] = []

        def _fake_flatten(*, reason: str, known_qty: float | None = None, known_side: str | None = None) -> tuple[bool, str]:
            _ = (known_qty, known_side)
            flatten_calls.append(str(reason))
            return True, "flattened"

        trader._emergency_flatten = _fake_flatten  # type: ignore[method-assign]
        trader.maybe_kill_on_margin_ratio()

        assert flatten_calls == ["KILL_SWITCH_MARGIN_RATIO"]
        assert trader.stop_event.is_set() is True


def test_margin_ratio_kill_switch_sets_skip_flag_when_flatten_fails(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteStateStore(db_path=db_path) as store:
        state = store.load_state()
        _seed_open_position(state)

        broker = FakeBroker(
            margin_maint=90.0,
            margin_balance=100.0,
            position_amt=0.2,
            fail_flatten_submit=True,
        )
        trader = _build_trader_service(tmp_path=tmp_path, broker=broker, store=store, state=state)

        trader.maybe_kill_on_margin_ratio()

        assert trader.skip_cancel_open_orders_on_exit_runtime is True
        assert trader.stop_event.is_set() is True


def test_shutdown_guard_flattens_when_position_open(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteStateStore(db_path=db_path) as store:
        state = store.load_state()
        _seed_open_position(state)
        store.save_state(state)

        broker = FakeBroker(margin_maint=0.0, margin_balance=100.0, position_amt=0.2)
        trader = _build_trader_service(tmp_path=tmp_path, broker=broker, store=store, state=state)

        _simulate_shutdown_guard(trader)

        assert trader.state.open_position is None
        assert trader.skip_cancel_open_orders_on_exit_runtime is False
        reloaded = store.load_state()
        assert reloaded.open_position is None


def test_shutdown_guard_resets_skip_flag_after_prior_failure(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteStateStore(db_path=db_path) as store:
        state = store.load_state()
        _seed_open_position(state)

        broker = FakeBroker(margin_maint=0.0, margin_balance=100.0, position_amt=0.2)
        trader = _build_trader_service(tmp_path=tmp_path, broker=broker, store=store, state=state)
        trader.skip_cancel_open_orders_on_exit_runtime = True

        _simulate_shutdown_guard(trader)

        assert trader.skip_cancel_open_orders_on_exit_runtime is False


def test_shutdown_guard_sets_skip_flag_when_flatten_fails(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteStateStore(db_path=db_path) as store:
        state = store.load_state()
        _seed_open_position(state)

        broker = FakeBroker(
            margin_maint=0.0,
            margin_balance=100.0,
            position_amt=0.2,
            fail_flatten_submit=True,
        )
        trader = _build_trader_service(tmp_path=tmp_path, broker=broker, store=store, state=state)

        _simulate_shutdown_guard(trader)

        assert trader.skip_cancel_open_orders_on_exit_runtime is True


def test_margin_ratio_kill_switch_clears_and_persists_open_position_on_flatten_success(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteStateStore(db_path=db_path) as store:
        state = store.load_state()
        _seed_open_position(state)
        store.save_state(state)

        broker = FakeBroker(margin_maint=90.0, margin_balance=100.0, position_amt=0.2)
        trader = _build_trader_service(tmp_path=tmp_path, broker=broker, store=store, state=state)
        trader.skip_cancel_open_orders_on_exit_runtime = True

        trader.maybe_kill_on_margin_ratio()

        assert trader.state.open_position is None
        reloaded = store.load_state()
        assert reloaded.open_position is None
        assert trader.skip_cancel_open_orders_on_exit_runtime is False

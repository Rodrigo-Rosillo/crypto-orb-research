from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from forward.schemas import ORDERS_COLUMNS
from forward.state_store_sqlite import SQLiteStateStore
from tests.integration.mocks import FakeBinanceClient, build_trader_service


class LiveSizingFakeBinanceClient(FakeBinanceClient):
    def __init__(self, *, account_payload: Any = None, account_error: Exception | None = None, fill_price: float = 100.0) -> None:
        super().__init__(fill_price=float(fill_price))
        self._account_payload = account_payload
        self._account_error = account_error
        self.entry_market_order_calls: list[dict[str, Any]] = []

    def account(self) -> Any:
        if self._account_error is not None:
            raise self._account_error
        if self._account_payload is not None:
            return self._account_payload
        return {
            "availableBalance": "1000",
            "totalMarginBalance": "1000",
            "totalMaintMargin": "0",
        }

    def place_market_order(self, *, symbol: str, side: str, quantity: float, reduce_only: bool = False) -> Any:
        if not bool(reduce_only):
            self.entry_market_order_calls.append(
                {
                    "symbol": str(symbol),
                    "side": str(side),
                    "quantity": float(quantity),
                    "reduce_only": bool(reduce_only),
                }
            )
        return super().place_market_order(symbol=symbol, side=side, quantity=quantity, reduce_only=reduce_only)


def _entry_row() -> pd.Series:
    return pd.Series({"signal": 1, "signal_type": "unit", "close": 100.0, "orb_high": 200.0})


def test_entry_sizing_uses_live_available_balance_capped_by_config(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteStateStore(db_path=db_path) as store:
        state = store.load_state()
        broker = LiveSizingFakeBinanceClient(
            account_payload={
                "availableBalance": "200.0",
                "totalMarginBalance": "300.0",
                "totalMaintMargin": "0",
            }
        )
        trader = build_trader_service(
            broker=broker,
            store=store,
            state=state,
            work_dir=tmp_path,
            leverage=1.0,
            position_size=0.2,
            initial_capital=1000.0,
            max_order_rejects_per_day=10,
        )

        emitted: list[dict[str, Any]] = []

        def _capture_events(rows: list[dict[str, Any]]) -> None:
            emitted.extend(rows)

        appended: list[tuple[Path, list[dict[str, Any]], list[str], str]] = []

        def _capture_append_rows(path: Path, rows: list[dict[str, Any]], cols: list[str], name: str) -> None:
            appended.append((path, rows, cols, name))

        trader.emit_event = _capture_events
        trader.append_rows = _capture_append_rows

        row = _entry_row()
        bar_open_time = pd.Timestamp("2026-01-01T00:00:00Z")
        asyncio.run(trader.maybe_place_trade_from_signal(bar_open_time, row))

        assert len(broker.entry_market_order_calls) == 1
        assert broker.entry_market_order_calls[0]["quantity"] == pytest.approx(0.4, abs=1e-12)

        sizing_events = [e for e in emitted if str(e.get("type") or "") == "SIZING_SNAPSHOT"]
        assert len(sizing_events) == 1
        sizing = sizing_events[0]
        assert float(sizing["effective_capital_used"]) == pytest.approx(200.0, abs=1e-12)
        assert float(sizing["live_availableBalance"]) == pytest.approx(200.0, abs=1e-12)
        assert float(sizing["live_totalMarginBalance"]) == pytest.approx(300.0, abs=1e-12)

        assert any(name == "orders.csv" and cols == ORDERS_COLUMNS for _, _, cols, name in appended)


def test_entry_skips_trade_when_balance_fetch_raises(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteStateStore(db_path=db_path) as store:
        state = store.load_state()
        broker = LiveSizingFakeBinanceClient(account_error=RuntimeError("boom"))
        trader = build_trader_service(
            broker=broker,
            store=store,
            state=state,
            work_dir=tmp_path,
            leverage=1.0,
            position_size=0.2,
            initial_capital=1000.0,
            max_order_rejects_per_day=10,
        )

        emitted: list[dict[str, Any]] = []

        def _capture_events(rows: list[dict[str, Any]]) -> None:
            emitted.extend(rows)

        appended: list[tuple[Path, list[dict[str, Any]], list[str], str]] = []

        def _capture_append_rows(path: Path, rows: list[dict[str, Any]], cols: list[str], name: str) -> None:
            appended.append((path, rows, cols, name))

        trader.emit_event = _capture_events
        trader.append_rows = _capture_append_rows

        row = _entry_row()
        bar_open_time = pd.Timestamp("2026-01-01T00:00:00Z")
        asyncio.run(trader.maybe_place_trade_from_signal(bar_open_time, row))

        assert len(broker.entry_market_order_calls) == 0
        assert trader.state.open_position is None

        skipped_events = [e for e in emitted if str(e.get("type") or "") == "ENTRY_SKIPPED"]
        assert len(skipped_events) == 1
        skipped = skipped_events[0]
        assert skipped.get("reason") == "balance_fetch_failed"
        details = skipped.get("details")
        assert isinstance(details, dict)
        assert details.get("exception") == "boom"

        blocked_rows = [
            row_item
            for _, rows, cols, name in appended
            if name == "orders.csv" and cols == ORDERS_COLUMNS
            for row_item in rows
            if row_item.get("status") == "blocked"
            and row_item.get("status_detail") == "balance_fetch_failed"
            and float(row_item.get("qty") or 0.0) == 0.0
        ]
        assert len(blocked_rows) == 1

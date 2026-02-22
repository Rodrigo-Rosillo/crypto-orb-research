from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional

from backtester.risk import RiskLimits
from forward.state_store_sqlite import RunnerState, SQLiteStateStore
from forward.testnet_broker import TestnetAPIError
from forward.trader_service import TraderService


class FakeBinanceClient:
    """Minimal fake broker surface used by forward.trader_service.TraderService."""

    def __init__(
        self,
        *,
        reject_entry: bool = False,
        simulate_partial_fill_poll: bool = False,
        fill_price: float = 100.0,
    ) -> None:
        self.reject_entry = bool(reject_entry)
        self.simulate_partial_fill_poll = bool(simulate_partial_fill_poll)
        self.fill_price = float(fill_price)

        self._next_id = 1000
        self._position_amt = 0.0
        self._entry_price = 0.0
        self._algo_status: dict[int, str] = {}
        self._entry_status_script: dict[int, list[dict[str, Any]]] = {}
        self._entry_status_idx: dict[int, int] = {}
        self.last_entry_order_id: Optional[int] = None

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def place_market_order(self, *, symbol: str, side: str, quantity: float, reduce_only: bool = False) -> Any:
        if self.reject_entry and not bool(reduce_only):
            raise TestnetAPIError("simulated_rejection", status_code=400, payload={"code": -2010})

        oid = self._new_id()
        qty = float(quantity)
        s = str(side).upper()
        if not bool(reduce_only):
            self.last_entry_order_id = int(oid)

        if s == "BUY":
            self._position_amt += qty
        elif s == "SELL":
            self._position_amt -= qty

        if abs(self._position_amt) < 1e-12:
            self._position_amt = 0.0
            self._entry_price = 0.0
        elif not bool(reduce_only):
            self._entry_price = float(self.fill_price)

        if self.simulate_partial_fill_poll and not bool(reduce_only):
            half = qty / 2.0
            self._entry_status_script[int(oid)] = [
                {
                    "orderId": int(oid),
                    "status": "PARTIALLY_FILLED",
                    "executedQty": f"{half}",
                    "avgPrice": f"{self.fill_price}",
                },
                {
                    "orderId": int(oid),
                    "status": "FILLED",
                    "executedQty": f"{qty}",
                    "avgPrice": f"{self.fill_price}",
                },
            ]
            self._entry_status_idx[int(oid)] = 0

        return {
            "orderId": int(oid),
            "status": "FILLED",
            "avgPrice": f"{self.fill_price}",
            "executedQty": f"{qty}",
            "origQty": f"{qty}",
        }

    def poll_entry_order(self, order_id: int) -> dict[str, Any]:
        oid = int(order_id)
        script = self._entry_status_script.get(oid)
        if not script:
            return {
                "orderId": oid,
                "status": "FILLED",
                "executedQty": "0",
                "avgPrice": f"{self.fill_price}",
            }
        idx = self._entry_status_idx.get(oid, 0)
        if idx >= len(script):
            return dict(script[-1])
        out = dict(script[idx])
        self._entry_status_idx[oid] = idx + 1
        return out

    def place_take_profit_market(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        reduce_only: bool = True,
    ) -> Any:
        algo_id = self._new_id()
        self._algo_status[int(algo_id)] = "NEW"
        return {"algoId": int(algo_id), "status": "NEW"}

    def place_stop_market(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        reduce_only: bool = True,
    ) -> Any:
        algo_id = self._new_id()
        self._algo_status[int(algo_id)] = "NEW"
        return {"algoId": int(algo_id), "status": "NEW"}

    def get_algo_order(self, *, symbol: str, algo_id: int) -> Any:
        return {"algoId": int(algo_id), "status": self._algo_status.get(int(algo_id), "NEW")}

    def position_risk(self, *, symbol: str) -> Any:
        return {
            "symbol": str(symbol),
            "positionAmt": f"{self._position_amt}",
            "entryPrice": f"{self._entry_price}",
            "unRealizedProfit": "0",
        }

    def account(self) -> Any:
        return {
            "totalMaintMargin": "0",
            "totalMarginBalance": "1000",
        }


def build_trader_service(
    *,
    broker: FakeBinanceClient,
    store: SQLiteStateStore,
    state: RunnerState,
    work_dir: Path,
    risk_limits: Optional[RiskLimits] = None,
    max_order_rejects_per_day: int = 5,
    leverage: float = 1.0,
    position_size: float = 0.1,
    initial_capital: float = 1000.0,
) -> TraderService:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    def _append_rows(path: Path, rows: list[dict[str, Any]], cols: list[str], name: str) -> None:
        return None

    def _emit_event(rows: list[dict[str, Any]]) -> None:
        return None

    return TraderService(
        broker=broker,
        store=store,
        state=state,
        symbol="SOLUSDT",
        leverage=float(leverage),
        position_size=float(position_size),
        initial_capital=float(initial_capital),
        slippage_bps=0.0,
        state_path=work_dir / "state.json",
        events_path=work_dir / "events.jsonl",
        run_id="integration-test",
        stop_event=asyncio.Event(),
        risk_limits=risk_limits,
        max_order_rejects_per_day=int(max_order_rejects_per_day),
        margin_ratio_threshold=0.85,
        orders_path=work_dir / "orders.csv",
        fills_path=work_dir / "fills.csv",
        positions_path=work_dir / "positions.csv",
        append_rows=_append_rows,
        emit_event=_emit_event,
    )

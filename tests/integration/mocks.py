from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional

from backtester.risk import RiskLimits
from forward.state_store_sqlite import RunnerState, SQLiteStateStore
from forward.testnet_broker import AmbiguousOrderError, TestnetAPIError
from forward.trader_service import TraderService


class FakeBinanceClient:
    """Minimal fake broker surface used by forward.trader_service.TraderService."""

    def __init__(
        self,
        *,
        reject_entry: bool = False,
        simulate_partial_fill_poll: bool = False,
        fill_price: float = 100.0,
        reject_tp: bool = False,
        reject_sl: bool = False,
        reject_flatten: bool = False,
        tp_raise_but_land: bool = False,
        sl_raise_but_land: bool = False,
        entry_raise_ambiguous: bool = False,
        ambiguous_entry_lands: bool = False,
        ambiguous_entry_entry_price: Optional[float] = None,
        open_algo_orders_override: Optional[Any] = None,
        fail_get_algo_open_orders: bool = False,
        server_time_payload: Optional[dict[str, Any]] = None,
        fail_server_time: bool = False,
    ) -> None:
        self.reject_entry = bool(reject_entry)
        self.simulate_partial_fill_poll = bool(simulate_partial_fill_poll)
        self.fill_price = float(fill_price)
        self.reject_tp = bool(reject_tp)
        self.reject_sl = bool(reject_sl)
        self.reject_flatten = bool(reject_flatten)
        self.tp_raise_but_land = bool(tp_raise_but_land)
        self.sl_raise_but_land = bool(sl_raise_but_land)
        self.entry_raise_ambiguous = bool(entry_raise_ambiguous)
        self.ambiguous_entry_lands = bool(ambiguous_entry_lands)
        self.ambiguous_entry_entry_price = ambiguous_entry_entry_price
        self.open_algo_orders_override = open_algo_orders_override
        self.fail_get_algo_open_orders = bool(fail_get_algo_open_orders)
        self.server_time_payload = server_time_payload
        self.fail_server_time = bool(fail_server_time)

        self.cancel_all_called = False
        self.cancel_all_call_count = 0
        self.last_cancel_all_symbol: Optional[str] = None
        self.position_risk_call_count = 0

        self._next_id = 1000
        self._now_ms = 1_700_000_000_000
        self._position_amt = 0.0
        self._entry_price = 0.0
        self._algo_status: dict[int, str] = {}
        self._algo_rows: dict[int, dict[str, Any]] = {}
        self._entry_status_script: dict[int, list[dict[str, Any]]] = {}
        self._entry_status_idx: dict[int, int] = {}
        self.last_entry_order_id: Optional[int] = None
        self.last_entry_client_order_id: Optional[str] = None
        self.last_tp_client_algo_id: Optional[str] = None
        self.last_sl_client_algo_id: Optional[str] = None

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _next_ts_ms(self) -> int:
        self._now_ms += 1000
        return int(self._now_ms)

    def quantize_price(self, *, symbol: str, price: Any, field_name: str = "price") -> tuple[str, dict[str, Any]]:
        sent = str(price)
        return sent, {
            "symbol": str(symbol).upper(),
            "field": str(field_name),
            "raw": sent,
            "sent": sent,
            "tickSize": "0.1",
        }

    def place_market_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        reduce_only: bool = False,
        reference_price: float | None = None,
        client_order_id: str | None = None,
    ) -> Any:
        _ = reference_price
        if self.reject_entry and not bool(reduce_only):
            raise TestnetAPIError("simulated_rejection", status_code=400, payload={"code": -2010})
        if self.reject_flatten and bool(reduce_only):
            raise TestnetAPIError("simulated_flatten_rejection", status_code=400, payload={"code": -2010})

        oid = self._new_id()
        qty = float(quantity)
        s = str(side).upper()
        if not bool(reduce_only):
            self.last_entry_order_id = int(oid)
            self.last_entry_client_order_id = str(client_order_id or f"entry_{int(oid)}")

        if s == "BUY":
            self._position_amt += qty
        elif s == "SELL":
            self._position_amt -= qty

        if abs(self._position_amt) < 1e-12:
            self._position_amt = 0.0
            self._entry_price = 0.0
        elif not bool(reduce_only):
            if self.ambiguous_entry_entry_price is not None and self.entry_raise_ambiguous:
                self._entry_price = float(self.ambiguous_entry_entry_price)
            else:
                self._entry_price = float(self.fill_price)

        if self.entry_raise_ambiguous and not bool(reduce_only):
            if not self.ambiguous_entry_lands:
                if s == "BUY":
                    self._position_amt -= qty
                elif s == "SELL":
                    self._position_amt += qty
                self._entry_price = 0.0
            raise AmbiguousOrderError(
                "simulated_entry_ambiguous",
                client_order_id=str(self.last_entry_client_order_id or client_order_id or f"entry_{int(oid)}"),
                context={"source": "FakeBinanceClient"},
            )

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
        client_algo_id: str | None = None,
    ) -> Any:
        self.last_tp_client_algo_id = str(client_algo_id) if client_algo_id is not None else None
        algo_id = self._new_id()
        row = {
            "algoId": int(algo_id),
            "symbol": str(symbol).upper(),
            "type": "TAKE_PROFIT_MARKET",
            "side": str(side).upper(),
            "status": "NEW",
            "triggerPrice": f"{float(stop_price)}",
            "time": int(self._next_ts_ms()),
        }
        self._algo_status[int(algo_id)] = "NEW"
        self._algo_rows[int(algo_id)] = dict(row)
        if self.tp_raise_but_land:
            raise TestnetAPIError("simulated_tp_raise_but_land", status_code=500, payload={"code": -1000})
        if self.reject_tp:
            self._algo_rows.pop(int(algo_id), None)
            self._algo_status.pop(int(algo_id), None)
            raise TestnetAPIError("simulated_tp_rejection", status_code=400, payload={"code": -2010})
        return {"algoId": int(algo_id), "status": "NEW"}

    def place_stop_market(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        reduce_only: bool = True,
        client_algo_id: str | None = None,
    ) -> Any:
        self.last_sl_client_algo_id = str(client_algo_id) if client_algo_id is not None else None
        algo_id = self._new_id()
        row = {
            "algoId": int(algo_id),
            "symbol": str(symbol).upper(),
            "type": "STOP_MARKET",
            "side": str(side).upper(),
            "status": "NEW",
            "triggerPrice": f"{float(stop_price)}",
            "time": int(self._next_ts_ms()),
        }
        self._algo_status[int(algo_id)] = "NEW"
        self._algo_rows[int(algo_id)] = dict(row)
        if self.sl_raise_but_land:
            raise TestnetAPIError("simulated_sl_raise_but_land", status_code=500, payload={"code": -1000})
        if self.reject_sl:
            self._algo_rows.pop(int(algo_id), None)
            self._algo_status.pop(int(algo_id), None)
            raise TestnetAPIError("simulated_sl_rejection", status_code=400, payload={"code": -2010})
        return {"algoId": int(algo_id), "status": "NEW"}

    def get_algo_order(self, *, symbol: str, algo_id: int) -> Any:
        row = dict(self._algo_rows.get(int(algo_id), {}))
        if not row:
            row = {
                "algoId": int(algo_id),
                "symbol": str(symbol).upper(),
                "status": self._algo_status.get(int(algo_id), "NEW"),
            }
        row["status"] = self._algo_status.get(int(algo_id), str(row.get("status") or "NEW"))
        return row

    def get_algo_open_orders(self, *, symbol: str) -> Any:
        if self.fail_get_algo_open_orders:
            raise TestnetAPIError("simulated_get_algo_open_orders_failure", status_code=500, payload={"code": -1000})
        if self.open_algo_orders_override is not None:
            return self.open_algo_orders_override
        symbol_u = str(symbol).upper()
        out: list[dict[str, Any]] = []
        for oid, row in self._algo_rows.items():
            status = str(self._algo_status.get(int(oid), row.get("status") or "NEW")).upper()
            if status in ("FILLED", "FINISHED", "CANCELED", "CANCELLED", "REJECTED", "EXPIRED"):
                continue
            if str(row.get("symbol") or "").upper() != symbol_u:
                continue
            merged = dict(row)
            merged["status"] = status
            out.append(merged)
        return out

    def server_time(self) -> Any:
        if self.fail_server_time:
            raise TestnetAPIError("simulated_server_time_failure", status_code=500, payload={"code": -1000})
        if self.server_time_payload is not None:
            return dict(self.server_time_payload)
        return {"serverTime": int(self._next_ts_ms())}

    def position_risk(self, *, symbol: str) -> Any:
        self.position_risk_call_count += 1
        return {
            "symbol": str(symbol),
            "positionAmt": f"{self._position_amt}",
            "entryPrice": f"{self._entry_price}",
            "unRealizedProfit": "0",
        }

    def cancel_all_open_orders(self, *, symbol: str) -> Any:
        self.cancel_all_called = True
        self.cancel_all_call_count += 1
        self.last_cancel_all_symbol = str(symbol)
        return {"symbol": str(symbol), "status": "ok"}

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
    taker_fee_rate: float = 0.0005,
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
        taker_fee_rate=float(taker_fee_rate),
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


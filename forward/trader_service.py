from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import pandas as pd

from backtester.risk import RiskLimits
from forward.risk_engine import RiskDecision, check_margin_ratio, check_order_rejects
from forward.schemas import FILLS_COLUMNS, ORDERS_COLUMNS, POSITIONS_COLUMNS
from forward.state_store_sqlite import OpenPositionState, RunnerState, SQLiteStateStore
from forward.testnet_broker import BinanceFuturesTestnetBroker, TestnetAPIError


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _pos_side_from_amt(amt: float) -> str:
    if amt > 0:
        return "LONG"
    if amt < 0:
        return "SHORT"
    return "FLAT"


def _extract_order_id(resp: Any) -> Optional[int]:
    if isinstance(resp, dict):
        oid = resp.get("orderId") or resp.get("orderID") or resp.get("algoId") or resp.get("algoID")
        if oid is not None:
            try:
                return int(oid)
            except Exception:
                return None
    return None


def _order_status(resp: Any) -> str:
    if isinstance(resp, dict):
        return str(resp.get("status") or resp.get("algoStatus") or "")
    return ""


def _order_avg_price(resp: Any) -> float:
    if isinstance(resp, dict):
        if resp.get("avgPrice") not in (None, ""):
            return _float(resp.get("avgPrice"), 0.0)
        if resp.get("price") not in (None, ""):
            return _float(resp.get("price"), 0.0)
    return 0.0


def _order_exec_qty(resp: Any) -> float:
    if isinstance(resp, dict):
        if resp.get("executedQty") not in (None, ""):
            return _float(resp.get("executedQty"), 0.0)
        if resp.get("origQty") not in (None, ""):
            return _float(resp.get("origQty"), 0.0)
    return 0.0


class TraderService:
    def __init__(
        self,
        broker: BinanceFuturesTestnetBroker,
        store: SQLiteStateStore,
        state: RunnerState,
        symbol: str,
        leverage: float,
        position_size: float,
        initial_capital: float,
        slippage_bps: float,
        state_path: Path,
        events_path: Path,
        run_id: str,
        stop_event: asyncio.Event,
        risk_limits: Optional[RiskLimits],
        max_order_rejects_per_day: int,
        margin_ratio_threshold: float,
        orders_path: Path,
        fills_path: Path,
        positions_path: Path,
        append_rows: Callable[[Path, list[dict], list[str], str], None],
        emit_event: Callable[[list[dict]], None],
    ) -> None:
        self.broker = broker
        self.store = store
        self.state = state
        self.symbol = symbol
        self.leverage = float(leverage)
        self.position_size = float(position_size)
        self.initial_capital = float(initial_capital)
        self.slippage_bps = float(slippage_bps)
        self.state_path = state_path
        self.events_path = events_path
        self.run_id = run_id
        self.stop_event = stop_event
        self.risk_limits = risk_limits
        self.max_order_rejects_per_day = int(max_order_rejects_per_day)
        self.margin_ratio_threshold = float(margin_ratio_threshold)
        self.orders_path = orders_path
        self.fills_path = fills_path
        self.positions_path = positions_path
        self.append_rows = append_rows
        self.emit_event = emit_event

    def persist_state(self) -> None:
        self.store.save_state(self.state)
        self.store.export_state_json_snapshot(self.state_path, self.state)

    def fetch_exchange_position(self) -> Tuple[str, float, float, float]:
        pr = self.broker.position_risk(symbol=self.symbol)
        amt = _float(pr.get("positionAmt"), 0.0)
        entry = _float(pr.get("entryPrice"), 0.0)
        upl = _float(pr.get("unRealizedProfit"), 0.0)
        side = _pos_side_from_amt(amt)
        return side, float(abs(amt)), float(entry), float(upl)

    def record_position_snapshot(self) -> None:
        ex_side, ex_qty, ex_entry, ex_upl = self.fetch_exchange_position()
        self.append_rows(
            self.positions_path,
            [
                {
                    "timestamp_utc": _utcnow_iso(),
                    "symbol": self.symbol,
                    "side": ex_side,
                    "qty": float(ex_qty),
                    "entry_price": float(ex_entry),
                    "mark_price": "",
                    "unrealized_pnl": float(ex_upl),
                    "equity": "",
                    "margin_used": "",
                    "leverage": float(self.leverage),
                }
            ],
            POSITIONS_COLUMNS,
            "positions.csv",
        )

    async def poll_open_orders(self) -> None:
        if self.state.open_position is None:
            return
        op = self.state.open_position
        changed = False
        for kind, oid in [("tp", op.tp_order_id), ("sl", op.sl_order_id)]:
            if oid is None:
                continue
            try:
                o = self.broker.get_algo_order(symbol=self.symbol, algo_id=int(oid))
                st = _order_status(o)
                if st in ("FINISHED", "CANCELED", "REJECTED", "EXPIRED"):
                    self.emit_event([{"ts": _utcnow_iso(), "type": "ORDER_UPDATE", "order_id": int(oid), "kind": kind, "status": st}])
                    if st == "FINISHED":
                        ex_side, ex_qty, _, _ = self.fetch_exchange_position()
                        if ex_side == "FLAT" or ex_qty < 1e-9:
                            self.emit_event([{"ts": _utcnow_iso(), "type": "POSITION_CLOSED", "via": kind}])
                            closed_side = op.side
                            closed_qty = float(op.qty)
                            closed_symbol = op.symbol
                            self.state.open_position = None
                            self.store.append_trade_log(
                                event_type="EXIT",
                                symbol=closed_symbol,
                                side=closed_side,
                                qty=closed_qty,
                                price=None,
                                realized_pnl=None,
                                fee=None,
                                funding_applied=None,
                                reason=kind,
                                bar_time_utc=_utcnow_iso(),
                            )
                            changed = True
            except Exception as e:
                self.emit_event([{"ts": _utcnow_iso(), "type": "ORDER_POLL_FAILED", "order_id": int(oid), "error": str(e)}])
        if changed:
            self.persist_state()

    def maybe_kill_on_margin_ratio(self) -> None:
        try:
            acct = self.broker.account()
            if not isinstance(acct, dict):
                return
            maint = _float(acct.get("totalMaintMargin"), 0.0)
            balance = _float(acct.get("totalMarginBalance"), 0.0)
            result = check_margin_ratio(maint=maint, balance=balance, threshold=float(self.margin_ratio_threshold))
            if result.decision == RiskDecision.KILL_SWITCH:
                ratio = float(maint / balance) if balance > 0 else 0.0
                self.emit_event(
                    [
                        {
                            "ts": _utcnow_iso(),
                            "type": "KILL_SWITCH_MARGIN_RATIO",
                            "ratio": float(ratio),
                            "threshold": float(self.margin_ratio_threshold),
                        }
                    ]
                )
                self.store.append_trade_log(
                    event_type="KILL_SWITCH",
                    symbol=self.symbol,
                    side=None,
                    qty=None,
                    price=None,
                    realized_pnl=None,
                    fee=None,
                    funding_applied=None,
                    reason="KILL_SWITCH_MARGIN_RATIO",
                    bar_time_utc=_utcnow_iso(),
                )
                self.stop_event.set()
        except Exception as e:
            self.emit_event([{"ts": _utcnow_iso(), "type": "MARGIN_RATIO_CHECK_FAILED", "error": str(e)}])

    async def maybe_place_trade_from_signal(self, bar_open_time: pd.Timestamp, row: pd.Series) -> None:
        if self.state.open_position is not None:
            return

        signal = int(row.get("signal", 0) or 0)
        if signal == 0:
            return

        signal_type = str(row.get("signal_type", "") or "")
        side = "SELL" if signal < 0 else "BUY"
        pos_side = "SHORT" if side == "SELL" else "LONG"

        if self.risk_limits is not None and bool(self.risk_limits.enabled):
            if float(self.leverage) > float(self.risk_limits.max_leverage):
                self.emit_event(
                    [
                        {
                            "ts": _utcnow_iso(),
                            "type": "RISK_BLOCK",
                            "reason": "max_leverage",
                            "leverage": float(self.leverage),
                            "max_leverage": float(self.risk_limits.max_leverage),
                        }
                    ]
                )
                self.append_rows(
                    self.orders_path,
                    [
                        {
                            "timestamp_utc": _utcnow_iso(),
                            "due_timestamp_utc": "",
                            "order_id": "",
                            "symbol": self.symbol,
                            "side": pos_side,
                            "qty": 0.0,
                            "order_type": "MARKET",
                            "limit_price": "",
                            "status": "blocked",
                            "status_detail": "risk_max_leverage",
                            "reason": signal_type,
                        }
                    ],
                    ORDERS_COLUMNS,
                    "orders.csv",
                )
                return
            if float(self.position_size) > float(self.risk_limits.max_position_margin_frac):
                self.emit_event(
                    [
                        {
                            "ts": _utcnow_iso(),
                            "type": "RISK_BLOCK",
                            "reason": "max_position_margin_frac",
                            "position_size": float(self.position_size),
                            "max_position_margin_frac": float(self.risk_limits.max_position_margin_frac),
                        }
                    ]
                )
                self.append_rows(
                    self.orders_path,
                    [
                        {
                            "timestamp_utc": _utcnow_iso(),
                            "due_timestamp_utc": "",
                            "order_id": "",
                            "symbol": self.symbol,
                            "side": pos_side,
                            "qty": 0.0,
                            "order_type": "MARKET",
                            "limit_price": "",
                            "status": "blocked",
                            "status_detail": "risk_max_position_margin",
                            "reason": signal_type,
                        }
                    ],
                    ORDERS_COLUMNS,
                    "orders.csv",
                )
                return

        price = float(row.get("close") or 0.0)
        if price <= 0:
            return
        margin = float(self.initial_capital) * float(self.position_size)
        notional = margin * float(self.leverage)
        qty = max(0.0, notional / price)
        if qty <= 0:
            return

        self.emit_event([{"ts": _utcnow_iso(), "type": "ENTRY_SUBMIT", "signal_type": signal_type, "qty": float(qty), "side": pos_side}])

        try:
            entry_resp = self.broker.place_market_order(symbol=self.symbol, side=side, quantity=float(qty), reduce_only=False)
        except (TestnetAPIError, Exception) as e:
            self.emit_event([{"ts": _utcnow_iso(), "type": "ENTRY_REJECTED", "error": str(e)}])
            self.store.append_trade_log(
                event_type="REJECT",
                symbol=self.symbol,
                side=pos_side,
                qty=None,
                price=None,
                realized_pnl=None,
                fee=None,
                funding_applied=None,
                reason=str(e),
                bar_time_utc=bar_open_time.tz_convert("UTC").isoformat(),
            )
            self.state.order_rejects_today = int(self.state.order_rejects_today or 0) + 1
            reject_result = check_order_rejects(
                rejects_today=int(self.state.order_rejects_today),
                max_rejects=int(self.max_order_rejects_per_day),
            )
            if reject_result.decision == RiskDecision.KILL_SWITCH:
                self.emit_event(
                    [
                        {
                            "ts": _utcnow_iso(),
                            "type": "KILL_SWITCH_ORDER_REJECTS",
                            "rejects_today": int(self.state.order_rejects_today),
                            "threshold": int(self.max_order_rejects_per_day),
                        }
                    ]
                )
                self.store.append_trade_log(
                    event_type="KILL_SWITCH",
                    symbol=self.symbol,
                    side=None,
                    qty=None,
                    price=None,
                    realized_pnl=None,
                    fee=None,
                    funding_applied=None,
                    reason="KILL_SWITCH_ORDER_REJECTS",
                    bar_time_utc=_utcnow_iso(),
                )
                self.stop_event.set()
            self.append_rows(
                self.orders_path,
                [
                    {
                        "timestamp_utc": _utcnow_iso(),
                        "due_timestamp_utc": "",
                        "order_id": "",
                        "symbol": self.symbol,
                        "side": pos_side,
                        "qty": float(qty),
                        "order_type": "MARKET",
                        "limit_price": "",
                        "status": "rejected",
                        "status_detail": "entry_rejected",
                        "reason": str(e),
                    }
                ],
                ORDERS_COLUMNS,
                "orders.csv",
            )
            self.persist_state()
            return

        entry_oid = _extract_order_id(entry_resp)
        entry_price = _order_avg_price(entry_resp)
        exec_qty = _order_exec_qty(entry_resp)

        self.append_rows(
            self.orders_path,
            [
                {
                    "timestamp_utc": _utcnow_iso(),
                    "due_timestamp_utc": "",
                    "order_id": str(entry_oid or ""),
                    "symbol": self.symbol,
                    "side": pos_side,
                    "qty": float(exec_qty or qty),
                    "order_type": "MARKET",
                    "limit_price": "",
                    "status": "sent",
                    "status_detail": "entry_sent",
                    "reason": signal_type,
                }
            ],
            ORDERS_COLUMNS,
            "orders.csv",
        )
        self.append_rows(
            self.fills_path,
            [
                {
                    "timestamp_utc": _utcnow_iso(),
                    "order_id": str(entry_oid or ""),
                    "symbol": self.symbol,
                    "side": pos_side,
                    "qty": float(exec_qty or qty),
                    "fill_price": float(entry_price),
                    "fee": 0.0,
                    "slippage_bps": float(self.slippage_bps),
                    "exec_model": "testnet_market",
                }
            ],
            FILLS_COLUMNS,
            "fills.csv",
        )
        self.store.append_trade_log(
            event_type="ENTRY",
            symbol=self.symbol,
            side=pos_side,
            qty=float(exec_qty or qty),
            price=float(entry_price),
            realized_pnl=None,
            fee=None,
            funding_applied=None,
            reason=None,
            bar_time_utc=bar_open_time.tz_convert("UTC").isoformat(),
        )

        orb_high = row.get("orb_high")
        if orb_high is None or (isinstance(orb_high, float) and pd.isna(orb_high)):
            self.emit_event([{"ts": _utcnow_iso(), "type": "BRACKET_SKIPPED", "reason": "missing_orb_high"}])
            self.state.open_position = OpenPositionState(
                symbol=self.symbol,
                side=pos_side,
                qty=float(exec_qty or qty),
                entry_price=float(entry_price),
                entry_time_utc=bar_open_time.tz_convert("UTC").isoformat(),
                entry_order_id=int(entry_oid or 0),
            )
            self.persist_state()
            return

        sl_price = float(orb_high)
        tp_price = float(entry_price) * (0.98 if pos_side == "SHORT" else 1.02)
        exit_side = "BUY" if pos_side == "SHORT" else "SELL"

        tp_oid = None
        sl_oid = None
        try:
            tp_resp = self.broker.place_take_profit_market(
                symbol=self.symbol,
                side=exit_side,
                quantity=float(exec_qty or qty),
                stop_price=float(tp_price),
                reduce_only=True,
            )
            tp_oid = _extract_order_id(tp_resp)
        except Exception as e:
            self.emit_event([{"ts": _utcnow_iso(), "type": "TP_PLACE_FAILED", "error": str(e)}])

        try:
            sl_resp = self.broker.place_stop_market(
                symbol=self.symbol,
                side=exit_side,
                quantity=float(exec_qty or qty),
                stop_price=float(sl_price),
                reduce_only=True,
            )
            sl_oid = _extract_order_id(sl_resp)
        except Exception as e:
            self.emit_event([{"ts": _utcnow_iso(), "type": "SL_PLACE_FAILED", "error": str(e)}])

        self.append_rows(
            self.orders_path,
            [
                {
                    "timestamp_utc": _utcnow_iso(),
                    "due_timestamp_utc": "",
                    "order_id": str(tp_oid or ""),
                    "symbol": self.symbol,
                    "side": "EXIT",
                    "qty": float(exec_qty or qty),
                    "order_type": "TAKE_PROFIT_MARKET",
                    "limit_price": float(tp_price),
                    "status": "sent",
                    "status_detail": "tp_sent",
                    "reason": signal_type,
                },
                {
                    "timestamp_utc": _utcnow_iso(),
                    "due_timestamp_utc": "",
                    "order_id": str(sl_oid or ""),
                    "symbol": self.symbol,
                    "side": "EXIT",
                    "qty": float(exec_qty or qty),
                    "order_type": "STOP_MARKET",
                    "limit_price": float(sl_price),
                    "status": "sent",
                    "status_detail": "sl_sent",
                    "reason": signal_type,
                },
            ],
            ORDERS_COLUMNS,
            "orders.csv",
        )

        self.state.open_position = OpenPositionState(
            symbol=self.symbol,
            side=pos_side,
            qty=float(exec_qty or qty),
            entry_price=float(entry_price),
            entry_time_utc=bar_open_time.tz_convert("UTC").isoformat(),
            entry_order_id=int(entry_oid or 0),
            tp_order_id=int(tp_oid) if tp_oid is not None else None,
            sl_order_id=int(sl_oid) if sl_oid is not None else None,
            tp_price=float(tp_price),
            sl_price=float(sl_price),
        )
        self.persist_state()


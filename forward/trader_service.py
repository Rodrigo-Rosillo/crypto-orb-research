from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import pandas as pd

from backtester.risk import RiskLimits
from forward.risk_engine import RiskDecision, check_margin_ratio, check_order_rejects
from forward.schemas import FILLS_COLUMNS, ORDERS_COLUMNS, POSITIONS_COLUMNS
from forward.state_store_sqlite import OpenPositionState, RunnerState, SQLiteStateStore
from forward.testnet_broker import BinanceFuturesTestnetBroker, OrderValidationError, TestnetAPIError


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


PROTECTION_RECENCY_WINDOW_SECONDS = 180
PROTECTION_AMBIGUITY_RETRY_WINDOW_SECONDS = 30
PROTECTION_PRICE_TOL_TICKS = 1

def _compact_json(data: Any) -> str:
    try:
        return json.dumps(data, sort_keys=True, separators=(",", ":"))
    except Exception:
        return str(data)


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
        self.skip_cancel_open_orders_on_exit_runtime = False

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
                if self.state.open_position is not None:
                    flatten_ok, _ = self._emergency_flatten(reason="KILL_SWITCH_MARGIN_RATIO")
                    if flatten_ok:
                        self.state.open_position = None
                        self.skip_cancel_open_orders_on_exit_runtime = False
                        try:
                            self.persist_state()
                        except Exception:
                            pass
                    else:
                        self.skip_cancel_open_orders_on_exit_runtime = True
                self.stop_event.set()
        except Exception as e:
            self.emit_event([{"ts": _utcnow_iso(), "type": "MARGIN_RATIO_CHECK_FAILED", "error": str(e)}])

    def _get_effective_capital_for_entry(self) -> tuple[float | None, dict[str, Any]]:
        account_method = getattr(self.broker, "account", None)
        if not callable(account_method):
            return None, {"error": "account_method_missing"}

        try:
            acct = account_method()
        except Exception as e:
            return None, {"error": "account_fetch_exception", "exception": str(e)}

        if not isinstance(acct, dict):
            return None, {"error": "account_payload_not_dict", "payload_type": type(acct).__name__}

        live_available = _float(acct.get("availableBalance"), 0.0)
        live_margin_balance = _float(acct.get("totalMarginBalance"), 0.0)
        if live_available > 0:
            live_base = live_available
        elif live_margin_balance > 0:
            live_base = live_margin_balance
        else:
            return None, {
                "error": "account_balance_missing_or_nonpositive",
                "availableBalance": float(live_available),
                "totalMarginBalance": float(live_margin_balance),
            }

        cap = float(self.initial_capital)
        effective = min(cap, live_base) if cap > 0 else live_base
        return float(effective), {
            "availableBalance": float(live_available),
            "totalMarginBalance": float(live_margin_balance),
            "live_base": float(live_base),
            "cap": float(cap),
            "effective": float(effective),
        }

    def _protection_baseline_epoch_seconds(self) -> tuple[float, str]:
        if hasattr(self.broker, "server_time"):
            try:
                resp = self.broker.server_time()
                return float(int(resp["serverTime"]) / 1000.0), "exchange"
            except Exception:
                pass
        # Local time fallback is acceptable here because recency matching allows drift.
        return float(time.time()), "local_fallback"

    @staticmethod
    def _is_terminal_algo_status(status: str) -> bool:
        st = str(status or "").upper()
        return st in ("FILLED", "FINISHED", "CANCELED", "CANCELLED", "REJECTED", "EXPIRED")

    @staticmethod
    def _safe_epoch_seconds(value: Any) -> Optional[float]:
        try:
            v = float(value)
        except Exception:
            return None
        if v != v or v in (float("inf"), float("-inf")) or v <= 0:
            return None
        if v > 1e11:
            v = v / 1000.0
        return float(v)

    def _algo_rows_from_payload(self, payload: Any) -> tuple[list[dict[str, Any]], bool]:
        if isinstance(payload, list):
            return [r for r in payload if isinstance(r, dict)], True
        if isinstance(payload, dict):
            for key in ("data", "rows", "orders", "openOrders", "list"):
                v = payload.get(key)
                if isinstance(v, list):
                    return [r for r in v if isinstance(r, dict)], True
            if any(k in payload for k in ("algoId", "algoID", "orderId", "orderID")):
                return [payload], True
        return [], False

    def _algo_timestamp_seconds(self, row: dict[str, Any]) -> Optional[float]:
        for key in ("time", "updateTime", "createTime", "workingTime", "timestamp"):
            if row.get(key) not in (None, ""):
                ts = self._safe_epoch_seconds(row.get(key))
                if ts is not None:
                    return ts
        return None

    @staticmethod
    def _algo_trigger_price(row: dict[str, Any]) -> Optional[float]:
        for key in ("triggerPrice", "stopPrice", "price"):
            if row.get(key) not in (None, ""):
                return _float(row.get(key), 0.0)
        return None

    @staticmethod
    def _algo_type(row: dict[str, Any]) -> str:
        return str(row.get("type") or row.get("orderType") or row.get("algoType") or "").upper()

    @staticmethod
    def _algo_side(row: dict[str, Any]) -> str:
        return str(row.get("side") or "").upper()

    @staticmethod
    def _tick_size_from_meta(meta: dict[str, Any]) -> float:
        if not isinstance(meta, dict):
            return 0.0
        return max(0.0, _float(meta.get("tickSize"), 0.0))

    @staticmethod
    def _trigger_price_matches(expected: float, actual: float, tick_size: float) -> bool:
        tol = float(tick_size) * float(PROTECTION_PRICE_TOL_TICKS) if float(tick_size) > 0 else 0.0
        return abs(float(expected) - float(actual)) <= max(0.0, tol)
    
    def _find_matching_leg_from_open_orders(
        self,
        *,
        rows: list[dict[str, Any]],
        expected_type: str,
        expected_side: str,
        expected_price: float,
        tick_size: float,
        baseline_epoch_s: float,
    ) -> dict[str, Any]:
        symbol_u = str(self.symbol).upper()
        structural: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_symbol = str(row.get("symbol") or "").upper()
            if row_symbol and row_symbol != symbol_u:
                continue
            if self._algo_type(row) != str(expected_type).upper():
                continue
            if self._algo_side(row) != str(expected_side).upper():
                continue
            status = str(row.get("status") or row.get("algoStatus") or "")
            if self._is_terminal_algo_status(status):
                continue
            trigger = self._algo_trigger_price(row)
            if trigger is None:
                continue
            if not self._trigger_price_matches(float(expected_price), float(trigger), float(tick_size)):
                continue
            structural.append(
                {
                    "row": row,
                    "ts": self._algo_timestamp_seconds(row),
                    "algo_id": _extract_order_id(row),
                }
            )

        if not structural:
            return {"status": "missing", "reason": "no_structural_candidates", "candidates": 0}

        with_ts = [c for c in structural if c.get("ts") is not None]
        if not with_ts:
            return {
                "status": "unknown",
                "reason": "timestamps_unavailable",
                "candidates": int(len(structural)),
            }

        recent = [
            c
            for c in with_ts
            if float(c.get("ts") or 0.0) >= float(baseline_epoch_s - float(PROTECTION_RECENCY_WINDOW_SECONDS))
        ]
        if not recent:
            return {
                "status": "missing",
                "reason": "no_recent_candidates",
                "candidates": int(len(with_ts)),
            }
        if len(recent) == 1:
            oid = recent[0].get("algo_id")
            if oid is None:
                return {"status": "unknown", "reason": "algo_id_missing", "candidates": 1}
            return {"status": "recovered", "order_id": int(oid), "reason": "single_recent_candidate", "candidates": 1}

        tight = [
            c
            for c in recent
            if float(c.get("ts") or 0.0) >= float(baseline_epoch_s - float(PROTECTION_AMBIGUITY_RETRY_WINDOW_SECONDS))
        ]
        if len(tight) == 1:
            oid = tight[0].get("algo_id")
            if oid is None:
                return {"status": "unknown", "reason": "algo_id_missing", "candidates": 1}
            return {"status": "recovered", "order_id": int(oid), "reason": "single_tight_candidate", "candidates": 1}

        pool = tight if len(tight) > 1 else recent
        try:
            max_ts = max(float(c.get("ts") or 0.0) for c in pool)
        except Exception:
            return {
                "status": "unknown",
                "reason": "timestamps_not_comparable",
                "candidates": int(len(pool)),
            }

        most_recent = [c for c in pool if float(c.get("ts") or 0.0) == float(max_ts)]
        with_algo_id = [c for c in most_recent if c.get("algo_id") is not None]
        if with_algo_id:
            chosen = max(with_algo_id, key=lambda c: int(c.get("algo_id") or 0))
            return {
                "status": "recovered",
                "order_id": int(chosen.get("algo_id")),
                "reason": "resolved_by_recent_ts_then_algo_id",
                "candidates": int(len(pool)),
            }

        chosen = most_recent[-1]
        if chosen.get("algo_id") is None:
            return {
                "status": "unknown",
                "reason": "algo_id_missing_after_tiebreak",
                "candidates": int(len(pool)),
            }
        return {
            "status": "recovered",
            "order_id": int(chosen.get("algo_id")),
            "reason": "resolved_by_recent_ts",
            "candidates": int(len(pool)),
        }

    def _resolve_protection_orders(
        self,
        *,
        tp_oid: Optional[int],
        sl_oid: Optional[int],
        exit_side: str,
        tp_sent_price: float,
        sl_sent_price: float,
        tp_tick_size: float,
        sl_tick_size: float,
        baseline_epoch_s: float,
    ) -> dict[str, Any]:
        tp_final = int(tp_oid) if tp_oid is not None else None
        sl_final = int(sl_oid) if sl_oid is not None else None
        details: dict[str, Any] = {}
        recovered_from_fallback = False

        if tp_final is not None and sl_final is not None:
            return {
                "status": "protected",
                "tp_order_id": tp_final,
                "sl_order_id": sl_final,
                "reason": "ids_from_place_calls",
                "details": details,
                "recovered_from_fallback": False,
            }

        if not hasattr(self.broker, "get_algo_open_orders"):
            return {
                "status": "unknown",
                "tp_order_id": tp_final,
                "sl_order_id": sl_final,
                "reason": "get_algo_open_orders_not_supported",
                "details": details,
                "recovered_from_fallback": False,
            }

        try:
            open_payload = self.broker.get_algo_open_orders(symbol=self.symbol)
        except Exception as e:
            return {
                "status": "unknown",
                "tp_order_id": tp_final,
                "sl_order_id": sl_final,
                "reason": f"get_algo_open_orders_failed:{e}",
                "details": details,
                "recovered_from_fallback": False,
            }

        rows, parse_ok = self._algo_rows_from_payload(open_payload)
        if not parse_ok:
            return {
                "status": "unknown",
                "tp_order_id": tp_final,
                "sl_order_id": sl_final,
                "reason": "get_algo_open_orders_unparseable",
                "details": {"raw": str(open_payload)},
                "recovered_from_fallback": False,
            }

        if tp_final is None:
            tp_match = self._find_matching_leg_from_open_orders(
                rows=rows,
                expected_type="TAKE_PROFIT_MARKET",
                expected_side=str(exit_side),
                expected_price=float(tp_sent_price),
                tick_size=float(tp_tick_size),
                baseline_epoch_s=float(baseline_epoch_s),
            )
            details["tp"] = tp_match
            if tp_match.get("status") == "recovered":
                tp_final = int(tp_match.get("order_id"))
                recovered_from_fallback = True

        if sl_final is None:
            sl_match = self._find_matching_leg_from_open_orders(
                rows=rows,
                expected_type="STOP_MARKET",
                expected_side=str(exit_side),
                expected_price=float(sl_sent_price),
                tick_size=float(sl_tick_size),
                baseline_epoch_s=float(baseline_epoch_s),
            )
            details["sl"] = sl_match
            if sl_match.get("status") == "recovered":
                sl_final = int(sl_match.get("order_id"))
                recovered_from_fallback = True

        if any(str(details.get(k, {}).get("status", "")) == "unknown" for k in ("tp", "sl") if k in details):
            return {
                "status": "unknown",
                "tp_order_id": tp_final,
                "sl_order_id": sl_final,
                "reason": "fallback_unknown",
                "details": details,
                "recovered_from_fallback": recovered_from_fallback,
            }

        if tp_final is None or sl_final is None:
            return {
                "status": "missing",
                "tp_order_id": tp_final,
                "sl_order_id": sl_final,
                "reason": "one_or_more_legs_missing",
                "details": details,
                "recovered_from_fallback": recovered_from_fallback,
            }

        return {
            "status": "protected",
            "tp_order_id": tp_final,
            "sl_order_id": sl_final,
            "reason": "resolved",
            "details": details,
            "recovered_from_fallback": recovered_from_fallback,
        }

    def _emergency_flatten(
        self,
        *,
        reason: str,
        known_qty: Optional[float] = None,
        known_side: Optional[str] = None,
    ) -> tuple[bool, str]:
        close_qty = 0.0
        close_side = ""
        source = "exchange_position"
        known_side_u = str(known_side or "").upper()

        if known_qty is not None and float(known_qty) > 1e-9 and known_side_u in ("LONG", "SHORT"):
            close_qty = float(known_qty)
            close_side = "SELL" if known_side_u == "LONG" else "BUY"
            source = "known_values"
        else:
            try:
                ex_side, ex_qty, _, _ = self.fetch_exchange_position()
            except Exception as e:
                self.emit_event(
                    [
                        {
                            "ts": _utcnow_iso(),
                            "type": "EMERGENCY_FLATTEN_FAILED",
                            "reason": str(reason),
                            "error": f"position_fetch_failed:{e}",
                        }
                    ]
                )
                return False, "position_fetch_failed"

            if ex_side == "FLAT" or ex_qty < 1e-9:
                self.emit_event(
                    [
                        {
                            "ts": _utcnow_iso(),
                            "type": "EMERGENCY_FLATTEN_SUCCESS",
                            "reason": str(reason),
                            "status": "already_flat",
                        }
                    ]
                )
                return True, "already_flat"
            close_qty = float(ex_qty)
            close_side = "SELL" if ex_side == "LONG" else "BUY"
            known_side_u = str(ex_side)

        self.emit_event(
            [
                {
                    "ts": _utcnow_iso(),
                    "type": "EMERGENCY_FLATTEN_SUBMIT",
                    "reason": str(reason),
                    "side": str(close_side),
                    "qty": float(close_qty),
                    "source": str(source),
                }
            ]
        )

        flatten_oid = None
        try:
            flatten_resp = self.broker.place_market_order(
                symbol=self.symbol,
                side=str(close_side),
                quantity=float(close_qty),
                reduce_only=True,
            )
            flatten_oid = _extract_order_id(flatten_resp)
        except Exception as e:
            self.emit_event(
                [
                    {
                        "ts": _utcnow_iso(),
                        "type": "EMERGENCY_FLATTEN_FAILED",
                        "reason": str(reason),
                        "error": str(e),
                    }
                ]
            )
            return False, "flatten_submit_failed"

        try:
            post_side, post_qty, _, _ = self.fetch_exchange_position()
        except Exception as e:
            self.emit_event(
                [
                    {
                        "ts": _utcnow_iso(),
                        "type": "EMERGENCY_FLATTEN_FAILED",
                        "reason": str(reason),
                        "order_id": int(flatten_oid or 0),
                        "error": f"post_flatten_position_fetch_failed:{e}",
                    }
                ]
            )
            return False, "post_flatten_position_fetch_failed"

        if post_side == "FLAT" or post_qty < 1e-9:
            self.emit_event(
                [
                    {
                        "ts": _utcnow_iso(),
                        "type": "EMERGENCY_FLATTEN_SUCCESS",
                        "reason": str(reason),
                        "order_id": int(flatten_oid or 0),
                        "side": str(close_side),
                        "qty": float(close_qty),
                    }
                ]
            )
            try:
                self.store.append_trade_log(
                    event_type="EXIT",
                    symbol=self.symbol,
                    side=known_side_u if known_side_u in ("LONG", "SHORT") else None,
                    qty=float(close_qty),
                    price=None,
                    realized_pnl=None,
                    fee=None,
                    funding_applied=None,
                    reason=f"EMERGENCY_FLATTEN:{reason}",
                    bar_time_utc=_utcnow_iso(),
                )
            except Exception:
                pass
            return True, "flattened"

        self.emit_event(
            [
                {
                    "ts": _utcnow_iso(),
                    "type": "EMERGENCY_FLATTEN_FAILED",
                    "reason": str(reason),
                    "order_id": int(flatten_oid or 0),
                    "exchange_side": str(post_side),
                    "exchange_qty": float(post_qty),
                    "error": "position_not_flat_after_flatten",
                }
            ]
        )
        return False, "position_not_flat_after_flatten"

    def _trigger_protection_kill_switch(self, reason: str, detail: str) -> None:
        self.emit_event(
            [
                {
                    "ts": _utcnow_iso(),
                    "type": str(reason),
                    "detail": str(detail),
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
            reason=f"{reason}:{detail}",
            bar_time_utc=_utcnow_iso(),
        )
        self.stop_event.set()
    
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

        effective_capital, details = self._get_effective_capital_for_entry()
        if effective_capital is None:
            self.emit_event(
                [
                    {
                        "ts": _utcnow_iso(),
                        "type": "ENTRY_SKIPPED",
                        "reason": "balance_fetch_failed",
                        "details": details,
                        "symbol": self.symbol,
                        "signal_type": signal_type,
                        "leverage": float(self.leverage),
                        "position_size": float(self.position_size),
                        "price": float(price),
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
                        "status_detail": "balance_fetch_failed",
                        "reason": signal_type,
                    }
                ],
                ORDERS_COLUMNS,
                "orders.csv",
            )
            return

        self.emit_event(
            [
                {
                    "ts": _utcnow_iso(),
                    "type": "SIZING_SNAPSHOT",
                    "config_initial_capital": float(self.initial_capital),
                    "live_availableBalance": float(details["availableBalance"]),
                    "live_totalMarginBalance": float(details["totalMarginBalance"]),
                    "effective_capital_used": float(effective_capital),
                    "position_size": float(self.position_size),
                    "leverage": float(self.leverage),
                    "price": float(price),
                }
            ]
        )
        margin = float(effective_capital) * float(self.position_size)
        notional = margin * float(self.leverage)
        qty = max(0.0, notional / price)
        if qty <= 0:
            return

        qty_sent = float(qty)
        qty_quant_meta: dict[str, Any] = {}
        try:
            if hasattr(self.broker, "quantize_qty"):
                qty_sent_str, qty_quant_meta = self.broker.quantize_qty(
                    symbol=self.symbol,
                    qty=float(qty),
                    is_market=True,
                    reference_price=float(price),
                    enforce_min_notional=True,
                )
                qty_sent = _float(qty_sent_str, qty_sent)

            entry_submit_event: dict[str, Any] = {
                "ts": _utcnow_iso(),
                "type": "ENTRY_SUBMIT",
                "signal_type": signal_type,
                "qty": float(qty),
                "qty_raw": float(qty),
                "qty_sent": float(qty_sent),
                "step_size": str(qty_quant_meta.get("stepSize", "")),
                "side": pos_side,
            }
            self.emit_event([entry_submit_event])

            if isinstance(self.broker, BinanceFuturesTestnetBroker):
                entry_resp = self.broker.place_market_order(
                    symbol=self.symbol,
                    side=side,
                    quantity=qty_sent,
                    reduce_only=False,
                    reference_price=float(price),
                )
            else:
                entry_resp = self.broker.place_market_order(symbol=self.symbol, side=side, quantity=qty_sent, reduce_only=False)
        except (OrderValidationError, TestnetAPIError, Exception) as e:
            reject_quantization = {}
            if isinstance(e, OrderValidationError):
                reject_quantization = {"symbol": self.symbol, "orderType": "MARKET", "fields": {"quantity": dict(e.meta or {})}}
            if isinstance(self.broker, BinanceFuturesTestnetBroker):
                last_q = self.broker.get_last_quantization()
                if isinstance(last_q, dict) and last_q and not reject_quantization:
                    reject_quantization = last_q
            if not reject_quantization and qty_quant_meta:
                reject_quantization = {"symbol": self.symbol, "orderType": "MARKET", "fields": {"quantity": qty_quant_meta}}

            reject_event: dict[str, Any] = {"ts": _utcnow_iso(), "type": "ENTRY_REJECTED", "error": str(e)}
            if reject_quantization:
                reject_event["quantization"] = reject_quantization
            self.emit_event([reject_event])

            reject_reason = str(e)
            if reject_quantization:
                reject_reason = f"{reject_reason}; quantization={_compact_json(reject_quantization)}"
            self.store.append_trade_log(
                event_type="REJECT",
                symbol=self.symbol,
                side=pos_side,
                qty=None,
                price=None,
                realized_pnl=None,
                fee=None,
                funding_applied=None,
                reason=reject_reason,
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
                        "qty": float(qty_sent),
                        "order_type": "MARKET",
                        "limit_price": "",
                        "status": "rejected",
                        "status_detail": "entry_rejected_local" if isinstance(e, OrderValidationError) else "entry_rejected",
                        "reason": reject_reason,
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
        entry_qty = float(exec_qty or qty_sent)

        self.append_rows(
            self.orders_path,
            [
                {
                    "timestamp_utc": _utcnow_iso(),
                    "due_timestamp_utc": "",
                    "order_id": str(entry_oid or ""),
                    "symbol": self.symbol,
                    "side": pos_side,
                    "qty": float(entry_qty),
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
                    "qty": float(entry_qty),
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
            qty=float(entry_qty),
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
            flatten_ok, flatten_detail = self._emergency_flatten(
                reason="missing_orb_high",
                known_qty=float(entry_qty),
                known_side=pos_side,
            )
            self.state.open_position = None
            self.persist_state()
            if not flatten_ok:
                self.skip_cancel_open_orders_on_exit_runtime = True
                self._trigger_protection_kill_switch(
                    "KILL_SWITCH_UNPROTECTED_POSITION",
                    f"missing_orb_high:{flatten_detail}",
                )
            return

        sl_price = float(orb_high)
        tp_price = float(entry_price) * (0.98 if pos_side == "SHORT" else 1.02)
        exit_side = "BUY" if pos_side == "SHORT" else "SELL"
        tp_sent_price = float(tp_price)
        sl_sent_price = float(sl_price)
        tp_price_meta: dict[str, Any] = {}
        sl_price_meta: dict[str, Any] = {}
        if hasattr(self.broker, "quantize_price"):
            try:
                tp_price_sent_str, tp_price_meta = self.broker.quantize_price(symbol=self.symbol, price=float(tp_price), field_name="tp_trigger_price")
                tp_sent_price = _float(tp_price_sent_str, tp_sent_price)
            except Exception:
                tp_price_meta = {}
            try:
                sl_price_sent_str, sl_price_meta = self.broker.quantize_price(symbol=self.symbol, price=float(sl_price), field_name="sl_trigger_price")
                sl_sent_price = _float(sl_price_sent_str, sl_sent_price)
            except Exception:
                sl_price_meta = {}

        protection_baseline_epoch_s, baseline_source = self._protection_baseline_epoch_seconds()
        if baseline_source != "exchange":
            self.emit_event(
                [
                    {
                        "ts": _utcnow_iso(),
                        "type": "PROTECTION_BASELINE_LOCAL_FALLBACK",
                        "window_seconds": int(PROTECTION_RECENCY_WINDOW_SECONDS),
                    }
                ]
            )

        self.emit_event(
            [
                {
                    "ts": _utcnow_iso(),
                    "type": "TP_SUBMIT",
                    "price_raw": float(tp_price),
                    "price_sent": float(tp_sent_price),
                    "tick_size": str(tp_price_meta.get("tickSize", "")),
                },
                {
                    "ts": _utcnow_iso(),
                    "type": "SL_SUBMIT",
                    "price_raw": float(sl_price),
                    "price_sent": float(sl_sent_price),
                    "tick_size": str(sl_price_meta.get("tickSize", "")),
                },
            ]
        )

        tp_oid = None
        sl_oid = None
        try:
            if isinstance(self.broker, BinanceFuturesTestnetBroker):
                tp_resp = self.broker.place_take_profit_market(
                    symbol=self.symbol,
                    side=exit_side,
                    quantity=float(entry_qty),
                    stop_price=float(tp_price),
                    reduce_only=True,
                    reference_price=float(tp_price),
                )
            else:
                tp_resp = self.broker.place_take_profit_market(
                    symbol=self.symbol,
                    side=exit_side,
                    quantity=float(entry_qty),
                    stop_price=float(tp_price),
                    reduce_only=True,
                )
            tp_oid = _extract_order_id(tp_resp)
        except Exception as e:
            tp_reject_event: dict[str, Any] = {"ts": _utcnow_iso(), "type": "TP_PLACE_FAILED", "error": str(e)}
            if isinstance(self.broker, BinanceFuturesTestnetBroker):
                tp_q = self.broker.get_last_quantization()
                if tp_q:
                    tp_reject_event["quantization"] = tp_q
            self.emit_event([tp_reject_event])

        try:
            if isinstance(self.broker, BinanceFuturesTestnetBroker):
                sl_resp = self.broker.place_stop_market(
                    symbol=self.symbol,
                    side=exit_side,
                    quantity=float(entry_qty),
                    stop_price=float(sl_price),
                    reduce_only=True,
                    reference_price=float(sl_price),
                )
            else:
                sl_resp = self.broker.place_stop_market(
                    symbol=self.symbol,
                    side=exit_side,
                    quantity=float(entry_qty),
                    stop_price=float(sl_price),
                    reduce_only=True,
                )
            sl_oid = _extract_order_id(sl_resp)
        except Exception as e:
            sl_reject_event: dict[str, Any] = {"ts": _utcnow_iso(), "type": "SL_PLACE_FAILED", "error": str(e)}
            if isinstance(self.broker, BinanceFuturesTestnetBroker):
                sl_q = self.broker.get_last_quantization()
                if sl_q:
                    sl_reject_event["quantization"] = sl_q
            self.emit_event([sl_reject_event])

        self.append_rows(
            self.orders_path,
            [
                {
                    "timestamp_utc": _utcnow_iso(),
                    "due_timestamp_utc": "",
                    "order_id": str(tp_oid or ""),
                    "symbol": self.symbol,
                    "side": "EXIT",
                    "qty": float(entry_qty),
                    "order_type": "TAKE_PROFIT_MARKET",
                    "limit_price": float(tp_sent_price),
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
                    "qty": float(entry_qty),
                    "order_type": "STOP_MARKET",
                    "limit_price": float(sl_sent_price),
                    "status": "sent",
                    "status_detail": "sl_sent",
                    "reason": signal_type,
                },
            ],
            ORDERS_COLUMNS,
            "orders.csv",
        )

        protection = self._resolve_protection_orders(
            tp_oid=int(tp_oid) if tp_oid is not None else None,
            sl_oid=int(sl_oid) if sl_oid is not None else None,
            exit_side=str(exit_side),
            tp_sent_price=float(tp_sent_price),
            sl_sent_price=float(sl_sent_price),
            tp_tick_size=float(self._tick_size_from_meta(tp_price_meta)),
            sl_tick_size=float(self._tick_size_from_meta(sl_price_meta)),
            baseline_epoch_s=float(protection_baseline_epoch_s),
        )

        p_status = str(protection.get("status") or "")
        p_details = protection.get("details") if isinstance(protection.get("details"), dict) else {}

        if p_status == "protected":
            if bool(protection.get("recovered_from_fallback")):
                self.emit_event(
                    [
                        {
                            "ts": _utcnow_iso(),
                            "type": "PROTECTION_VERIFY_RECOVERED",
                            "reason": str(protection.get("reason") or ""),
                            "details": p_details,
                        }
                    ]
                )

            self.state.open_position = OpenPositionState(
                symbol=self.symbol,
                side=pos_side,
                qty=float(entry_qty),
                entry_price=float(entry_price),
                entry_time_utc=bar_open_time.tz_convert("UTC").isoformat(),
                entry_order_id=int(entry_oid or 0),
                tp_order_id=int(protection.get("tp_order_id")) if protection.get("tp_order_id") is not None else None,
                sl_order_id=int(protection.get("sl_order_id")) if protection.get("sl_order_id") is not None else None,
                tp_price=float(tp_sent_price),
                sl_price=float(sl_sent_price),
            )
            self.persist_state()
            return

        if p_status == "missing":
            self.emit_event(
                [
                    {
                        "ts": _utcnow_iso(),
                        "type": "PROTECTION_VERIFY_MISSING",
                        "reason": str(protection.get("reason") or ""),
                        "details": p_details,
                    }
                ]
            )
            flatten_ok, flatten_detail = self._emergency_flatten(
                reason="protection_missing",
                known_qty=float(entry_qty),
                known_side=pos_side,
            )
            self.state.open_position = None
            self.persist_state()
            if not flatten_ok:
                self.skip_cancel_open_orders_on_exit_runtime = True
                self._trigger_protection_kill_switch(
                    "KILL_SWITCH_UNPROTECTED_POSITION",
                    f"protection_missing:{flatten_detail}",
                )
            return

        self.emit_event(
            [
                {
                    "ts": _utcnow_iso(),
                    "type": "PROTECTION_VERIFY_UNKNOWN",
                    "reason": str(protection.get("reason") or ""),
                    "details": p_details,
                }
            ]
        )
        flatten_ok, flatten_detail = self._emergency_flatten(
            reason="protection_unknown",
            known_qty=float(entry_qty),
            known_side=pos_side,
        )
        self.state.open_position = None
        self.persist_state()
        if not flatten_ok:
            self.skip_cancel_open_orders_on_exit_runtime = True
        # Flatten success does not re-establish trust in broker verification APIs.
        self._trigger_protection_kill_switch(
            "KILL_SWITCH_PROTECTION_UNKNOWN",
            f"{str(protection.get('reason') or 'unknown')}:{flatten_detail}",
        )
        return


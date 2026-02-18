from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from backtester.futures_core import FuturesExecutionCore
from backtester.futures_engine import FuturesEngineConfig
from backtester.risk import RiskLimits


@dataclass
class StreamStepResult:
    orders: List[Dict[str, Any]]
    fills: List[Dict[str, Any]]
    positions: List[Dict[str, Any]]
    events: List[Dict[str, Any]]


class StreamingFuturesShadowEngine:
    """Thin streaming adapter over shared futures execution core."""

    def __init__(
        self,
        cfg: FuturesEngineConfig,
        risk_limits: Optional[RiskLimits],
        expected_bar_seconds: int,
    ) -> None:
        self.core = FuturesExecutionCore(
            cfg=cfg,
            risk_limits=risk_limits,
            expected_bar_seconds=expected_bar_seconds,
        )
        self.cfg = cfg

    def __getattr__(self, name: str) -> Any:
        return getattr(self.core, name)

    def equity(self, mark_price: float) -> float:
        return self.core.equity(mark_price)

    def snapshot_position(self, ts: pd.Timestamp, mark_price: float) -> Dict[str, Any]:
        if self.core.side is None:
            return {
                "timestamp_utc": ts.tz_convert("UTC").isoformat(),
                "symbol": "",
                "side": "FLAT",
                "qty": 0.0,
                "entry_price": "",
                "mark_price": float(mark_price),
                "unrealized_pnl": 0.0,
                "equity": float(self.core.equity(mark_price)),
                "margin_used": 0.0,
                "leverage": float(self.core.leverage),
            }

        unreal = (
            self.core.qty * (mark_price - self.core.entry_price)
            if self.core.side == "long"
            else self.core.qty * (self.core.entry_price - mark_price)
        )
        return {
            "timestamp_utc": ts.tz_convert("UTC").isoformat(),
            "symbol": "",
            "side": "LONG" if self.core.side == "long" else "SHORT",
            "qty": float(self.core.qty),
            "entry_price": float(self.core.entry_price),
            "mark_price": float(mark_price),
            "unrealized_pnl": float(unreal),
            "equity": float(self.core.equity(mark_price)),
            "margin_used": float(self.core.position_margin),
            "leverage": float(self.core.leverage),
        }

    def on_bar(
        self,
        ts: pd.Timestamp,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        current_date: Any,
        signal: int,
        signal_type: str,
        orb_high: Optional[float],
        orb_low: Optional[float],
        valid_days: Optional[set] = None,
    ) -> StreamStepResult:
        """Consume one bar and return artifact rows/events emitted during this step."""
        step = self.core.on_bar(
            ts=ts,
            bar_open=bar_open,
            bar_high=bar_high,
            bar_low=bar_low,
            bar_close=bar_close,
            current_date=current_date,
            signal=signal,
            signal_type=signal_type,
            orb_high=orb_high,
            orb_low=orb_low,
            valid_days=valid_days,
            allow_schedule=True,
        )
        events = self.core.risk_mgr.pop_events() if self.core.risk_mgr is not None else []
        step["events"] = events

        orders: List[Dict[str, Any]] = []
        fills: List[Dict[str, Any]] = []
        positions: List[Dict[str, Any]] = []

        if bool(step.get("scheduled")):
            orders.append(
                {
                    "timestamp_utc": ts.tz_convert("UTC").isoformat(),
                    "due_timestamp_utc": "",
                    "order_id": "",  # filled by runner
                    "symbol": "",
                    "side": str(step.get("scheduled_side", "")),
                    "qty": "",
                    "order_type": "market",
                    "limit_price": "",
                    "status": "scheduled",
                    "status_detail": "",
                    "reason": str(step.get("scheduled_reason", "")),
                }
            )

        if bool(step.get("entered")):
            fills.append(
                {
                    "timestamp_utc": ts.tz_convert("UTC").isoformat(),
                    "order_id": "",  # filled by runner
                    "symbol": "",
                    "side": str(step.get("entry_order_side", "")),
                    "qty": float(step.get("entry_qty", 0.0)),
                    "fill_price": float(step.get("entry_price", 0.0)),
                    "fee": float(step.get("entry_fee", 0.0)),
                    "slippage_bps": float(self.cfg.slippage_bps),
                    "exec_model": "live_shadow",
                }
            )

        positions.append(self.snapshot_position(ts, mark_price=bar_close))
        events = list(step.get("events", []))

        return StreamStepResult(orders=orders, fills=fills, positions=positions, events=events)

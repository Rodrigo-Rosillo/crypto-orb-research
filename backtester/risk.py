# backtester/risk.py
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class KillSwitchConfig:
    """Configuration for kill-switch style controls."""

    # If the gap between consecutive bars exceeds (expected_bar_seconds * max_data_gap_bars), halt.
    max_data_gap_bars: int = 2

    # Backtest approximation: count of blocked/failed entry attempts per UTC day.
    max_order_rejects_per_day: int = 3

    # If maintenance_margin / max(margin_balance, eps) exceeds this, halt + flatten.
    # margin_balance = position_margin + unrealized_pnl
    max_margin_ratio: float = 0.85


@dataclass
class RiskLimits:
    """Phase 4 risk limits.

    Notes:
      - "position" controls are expressed in terms of INITIAL MARGIN as a fraction of equity.
      - Drawdown and daily loss are percent thresholds (0.20 == 20%).
    """

    enabled: bool = False

    # Hard controls
    max_position_margin_frac: float = 0.25
    max_leverage: float = 2.0
    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.20
    max_consecutive_losses: int = 4
    max_exposure_bars: int = 48

    kill_switch: KillSwitchConfig = field(default_factory=KillSwitchConfig)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def risk_limits_from_config(cfg: Dict[str, Any]) -> RiskLimits:
    rc = (cfg or {}).get("risk_controls", {}) or {}
    ks = rc.get("kill_switch", {}) or {}

    limits = RiskLimits(
        enabled=bool(rc.get("enabled", False)),
        max_position_margin_frac=float(rc.get("max_position_margin_frac", 0.25)),
        max_leverage=float(rc.get("max_leverage", 2.0)),
        max_daily_loss_pct=float(rc.get("max_daily_loss_pct", 0.03)),
        max_drawdown_pct=float(rc.get("max_drawdown_pct", 0.20)),
        max_consecutive_losses=int(rc.get("max_consecutive_losses", 4)),
        max_exposure_bars=int(rc.get("max_exposure_bars", 48)),
        kill_switch=KillSwitchConfig(
            max_data_gap_bars=int(ks.get("max_data_gap_bars", 2)),
            max_order_rejects_per_day=int(ks.get("max_order_rejects_per_day", 3)),
            max_margin_ratio=float(ks.get("max_margin_ratio", 0.85)),
        ),
    )

    # basic clamps
    limits.max_position_margin_frac = float(min(max(limits.max_position_margin_frac, 0.0), 1.0))
    limits.max_leverage = float(max(limits.max_leverage, 1.0))
    limits.max_daily_loss_pct = float(min(max(limits.max_daily_loss_pct, 0.0), 1.0))
    limits.max_drawdown_pct = float(min(max(limits.max_drawdown_pct, 0.0), 1.0))
    limits.max_consecutive_losses = int(max(limits.max_consecutive_losses, 0))
    limits.max_exposure_bars = int(max(limits.max_exposure_bars, 0))
    limits.kill_switch.max_data_gap_bars = int(max(limits.kill_switch.max_data_gap_bars, 1))
    limits.kill_switch.max_order_rejects_per_day = int(max(limits.kill_switch.max_order_rejects_per_day, 0))
    limits.kill_switch.max_margin_ratio = float(min(max(limits.kill_switch.max_margin_ratio, 0.0), 10.0))

    return limits


def expected_bar_seconds_from_index(index: pd.DatetimeIndex, fallback_seconds: int = 1800) -> int:
    """Estimate bar spacing from the index.

    Uses the median difference of up to the first 200 deltas.
    """
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return int(fallback_seconds)

    deltas = index.to_series().diff().dropna().dt.total_seconds()
    if deltas.empty:
        return int(fallback_seconds)

    sample = deltas.iloc[:200]
    med = float(sample.median())
    if not (med > 0):
        return int(fallback_seconds)

    # Round to nearest 60s
    return int(round(med / 60.0) * 60)


class RiskManager:
    """Stateful risk governance.

    This is intentionally engine-agnostic: the engine provides equity, position state,
    and the manager returns whether to halt and/or force-flatten.
    """

    def __init__(self, limits: RiskLimits, expected_bar_seconds: int):
        self.limits = limits
        self.expected_bar_seconds = int(max(expected_bar_seconds, 60))

        self.prev_ts: Optional[pd.Timestamp] = None

        # Global circuit breaker
        self.halted_global: bool = False
        self.halt_reason: str = ""

        # Day halt
        self.halted_today: bool = False
        self.halt_day: Optional[Any] = None  # python date

        # Metrics
        self.peak_equity: Optional[float] = None
        self.current_day: Optional[Any] = None
        self.day_start_equity: Optional[float] = None
        self.consecutive_losses: int = 0
        self.order_rejects_today: int = 0

        # Position tracking
        self.position_entry_i: Optional[int] = None

        self.events: List[Dict[str, Any]] = []

    def _event(self, ts: pd.Timestamp, kind: str, message: str, **info: Any) -> None:
        self.events.append(
            {
                "ts": ts.isoformat(),
                "kind": kind,
                "message": message,
                **info,
            }
        )

    def _halt_global(self, ts: pd.Timestamp, reason: str, message: str, **info: Any) -> None:
        if not self.halted_global:
            self.halted_global = True
            self.halt_reason = reason
            self._event(ts, "HALT_GLOBAL", message, reason=reason, **info)

    def _halt_today(self, ts: pd.Timestamp, day: Any, reason: str, message: str, **info: Any) -> None:
        # Can be called multiple times; keep the first reason.
        if not self.halted_today or self.halt_day != day:
            self.halted_today = True
            self.halt_day = day
            self._event(ts, "HALT_DAY", message, reason=reason, day=str(day), **info)

    def emit_event(self, ts: pd.Timestamp, kind: str, message: str, **info: Any) -> None:
        self._event(ts, kind, message, **info)

    def halt(self, ts: pd.Timestamp, reason: str, message: str, **info: Any) -> None:
        self._halt_global(ts, reason=reason, message=message, **info)

    def is_halted(self) -> bool:
        return bool(self.halted_global)

    def pop_events(self) -> List[Dict[str, Any]]:
        out = list(self.events)
        self.events = []
        return out

    def can_enter(self, day: Any) -> bool:
        if not self.limits.enabled:
            return True
        if self.halted_global:
            return False
        if self.halted_today and self.halt_day == day:
            return False
        return True

    def on_bar(self, ts: pd.Timestamp, day: Any, equity: float) -> None:
        """Update day state + drawdown/daily checks. Call once per bar."""
        if not self.limits.enabled:
            self.prev_ts = ts
            return

        # Day rollover
        if self.current_day != day:
            self.current_day = day
            self.day_start_equity = float(equity)
            self.halted_today = False
            self.halt_day = None
            self.order_rejects_today = 0
            self.consecutive_losses = 0
            self.position_entry_i = None
            self._event(ts, "DAY_START", "New UTC day", day=str(day), day_start_equity=float(equity))

        # Data staleness kill switch
        if self.prev_ts is not None:
            gap = float((ts - self.prev_ts).total_seconds())
            allowed = float(self.expected_bar_seconds * self.limits.kill_switch.max_data_gap_bars)
            if gap > allowed:
                self._halt_global(
                    ts,
                    reason="data_feed_stale",
                    message="Data gap exceeded threshold",
                    gap_seconds=gap,
                    allowed_seconds=allowed,
                )

        self.prev_ts = ts

        # Peak equity / drawdown
        if self.peak_equity is None:
            self.peak_equity = float(equity)
        else:
            self.peak_equity = float(max(self.peak_equity, float(equity)))

        if self.peak_equity > 0 and self.limits.max_drawdown_pct > 0:
            dd = 1.0 - (float(equity) / float(self.peak_equity))
            if dd >= self.limits.max_drawdown_pct:
                self._halt_global(
                    ts,
                    reason="max_drawdown",
                    message="Max drawdown circuit breaker triggered",
                    drawdown_pct=float(dd),
                    threshold=float(self.limits.max_drawdown_pct),
                )

        # Daily loss
        if self.day_start_equity and self.day_start_equity > 0 and self.limits.max_daily_loss_pct > 0:
            loss = 1.0 - (float(equity) / float(self.day_start_equity))
            if loss >= self.limits.max_daily_loss_pct:
                self._halt_today(
                    ts,
                    day=day,
                    reason="max_daily_loss",
                    message="Max daily loss reached; stop trading for day",
                    daily_loss_pct=float(loss),
                    threshold=float(self.limits.max_daily_loss_pct),
                )

    def record_order_reject(self, ts: pd.Timestamp, day: Any, reason: str) -> None:
        if not self.limits.enabled:
            return
        if self.current_day != day:
            # if called before on_bar rollover
            self.current_day = day
            self.day_start_equity = None
            self.order_rejects_today = 0

        self.order_rejects_today += 1
        self._event(ts, "ORDER_REJECT", "Entry rejected/blocked", day=str(day), reason=reason, count=self.order_rejects_today)

        if self.order_rejects_today > self.limits.kill_switch.max_order_rejects_per_day:
            self._halt_global(
                ts,
                reason="order_rejects",
                message="Order rejects exceeded threshold",
                day=str(day),
                rejects=self.order_rejects_today,
                threshold=int(self.limits.kill_switch.max_order_rejects_per_day),
            )

    def record_trade_close(self, ts: pd.Timestamp, day: Any, pnl_net: float) -> None:
        if not self.limits.enabled:
            return

        pnl_net = float(pnl_net)
        if pnl_net <= 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        self._event(
            ts,
            "TRADE_CLOSE",
            "Trade closed",
            day=str(day),
            pnl_net=pnl_net,
            consecutive_losses=int(self.consecutive_losses),
        )

        if self.limits.max_consecutive_losses > 0 and self.consecutive_losses >= self.limits.max_consecutive_losses:
            self._halt_today(
                ts,
                day=day,
                reason="max_consecutive_losses",
                message="Max consecutive losses reached; stop trading for day",
                consecutive_losses=int(self.consecutive_losses),
                threshold=int(self.limits.max_consecutive_losses),
            )

    def mark_position_entry(self, entry_i: int) -> None:
        if not self.limits.enabled:
            return
        self.position_entry_i = int(entry_i)

    def should_force_exit_exposure(self, i: int) -> bool:
        if not self.limits.enabled:
            return False
        if self.position_entry_i is None:
            return False
        if self.limits.max_exposure_bars <= 0:
            return False
        return (int(i) - int(self.position_entry_i)) >= int(self.limits.max_exposure_bars)

    def check_margin_ratio(
        self,
        ts: pd.Timestamp,
        day: Any,
        side: Optional[str],
        qty: float,
        entry_price: float,
        position_margin: float,
        mark_price: float,
        mmr: float,
    ) -> bool:
        """Returns True if margin ratio kill switch triggers."""
        if not self.limits.enabled:
            return False
        if side is None or qty <= 0:
            return False

        qty = float(qty)
        mark_price = float(mark_price)

        unreal = qty * (mark_price - entry_price) if side == "long" else qty * (entry_price - mark_price)
        margin_balance = float(position_margin) + float(unreal)
        notional = float(qty) * float(mark_price)
        maintenance = float(mmr) * float(notional)

        denom = max(margin_balance, 1e-9)
        ratio = float(maintenance / denom)

        if ratio >= float(self.limits.kill_switch.max_margin_ratio):
            self._halt_global(
                ts,
                reason="margin_ratio_spike",
                message="Margin ratio kill switch triggered",
                day=str(day),
                margin_ratio=ratio,
                threshold=float(self.limits.kill_switch.max_margin_ratio),
            )
            return True

        return False

    def snapshot(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.limits.enabled),
            "halted_global": bool(self.halted_global),
            "halt_reason": self.halt_reason,
            "halted_today": bool(self.halted_today),
            "halt_day": str(self.halt_day) if self.halt_day is not None else None,
            "limits": self.limits.to_dict(),
            "events": list(self.events),
        }

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from execution_specs import get_execution_spec, required_orb_fields, resolve_execution_plan

from .risk import RiskLimits, RiskManager


def _slip_price(raw_price: float, side: str, slip_frac: float) -> float:
    """Apply adverse slippage: buy -> pay more, sell -> receive less."""
    if slip_frac <= 0:
        return float(raw_price)
    if side == "buy":
        return float(raw_price) * (1.0 + slip_frac)
    if side == "sell":
        return float(raw_price) * (1.0 - slip_frac)
    raise ValueError("side must be 'buy' or 'sell'")


def _is_funding_bar(ts: pd.Timestamp) -> bool:
    # 00:00, 08:00, 16:00 UTC
    return ts.minute == 0 and ts.second == 0 and ts.microsecond == 0 and ts.hour in (0, 8, 16)


def _liq_price(side: str, entry_price: float, qty: float, margin: float, mmr: float) -> float:
    """Approx liquidation threshold.

    liquidation when (margin + unrealized) <= maintenance_margin
    maintenance_margin = mmr * notional_at_price

    LONG:  price <= (entry_price - margin/qty) / (1 - mmr)
    SHORT: price >= (entry_price + margin/qty) / (1 + mmr)

    qty is positive.
    """
    if qty <= 0:
        return float("nan")

    if side == "long":
        denom = max(1.0 - mmr, 1e-9)
        return (entry_price - (margin / qty)) / denom
    if side == "short":
        denom = max(1.0 + mmr, 1e-9)
        return (entry_price + (margin / qty)) / denom

    raise ValueError("side must be 'long' or 'short'")


class FuturesExecutionCore:
    """Shared futures execution semantics used by backtest and streaming adapters."""

    def __init__(
        self,
        cfg: Any,
        risk_limits: Optional[RiskLimits],
        expected_bar_seconds: int,
    ) -> None:
        self.cfg = cfg
        self.fee_rate = float(cfg.taker_fee_rate) * float(cfg.fee_mult)
        self.slip_frac = float(cfg.slippage_bps) / 10000.0
        self.delay_bars = max(int(cfg.delay_bars), 1)
        self.leverage = float(cfg.leverage)
        self.mmr = float(cfg.maintenance_margin_rate)

        self.risk_limits = risk_limits or RiskLimits(enabled=False)
        self.risk_mgr: Optional[RiskManager] = None
        self._leverage_cap_requested: Optional[float] = None
        self._leverage_cap_emitted = False
        if self.risk_limits.enabled:
            self.risk_mgr = RiskManager(self.risk_limits, expected_bar_seconds=int(max(expected_bar_seconds, 60)))
            if self.leverage > float(self.risk_limits.max_leverage):
                self._leverage_cap_requested = float(self.leverage)
                self.leverage = float(self.risk_limits.max_leverage)

        # Account state
        self.free_balance = float(cfg.initial_capital)
        self.position_margin = 0.0
        self.side: Optional[str] = None  # "long" or "short"
        self.qty = 0.0  # positive
        self.entry_price = 0.0
        self.entry_time: Optional[pd.Timestamp] = None
        self.entry_signal_type = ""
        self.stop_loss = 0.0
        self.target_price = 0.0

        # Per-trade bookkeeping
        self.current_initial_margin = 0.0
        self.current_entry_fee = 0.0

        self.total_fees = 0.0
        self.total_funding = 0.0
        self.liquidations = 0

        self.trades: List[Dict[str, Any]] = []

        # Pending order
        self.pending_signal = 0
        self.pending_signal_type = ""
        self.pending_date: Optional[Any] = None
        self.pending_due_i: Optional[int] = None
        self.pending_orb_high: Optional[float] = None
        self.pending_orb_low: Optional[float] = None

        self.i = -1

    def equity(self, mark_price: float) -> float:
        if self.side is None:
            return float(self.free_balance)
        unreal = (
            self.qty * (mark_price - self.entry_price)
            if self.side == "long"
            else self.qty * (self.entry_price - mark_price)
        )
        return float(self.free_balance + self.position_margin + unreal)

    def _pay_fee(self, amount: float) -> None:
        amount = float(max(amount, 0.0))
        if amount <= 0:
            return
        self.total_fees += amount
        if self.free_balance >= amount:
            self.free_balance -= amount
        else:
            rem = amount - self.free_balance
            self.free_balance = 0.0
            self.position_margin = max(0.0, self.position_margin - rem)

    def _apply_funding(self, ts: pd.Timestamp, mark_price: float) -> None:
        if self.side is None:
            return

        if self.cfg.funding_series is not None:
            if ts not in self.cfg.funding_series.index:
                return
            rate = float(self.cfg.funding_series.loc[ts])
        else:
            if not _is_funding_bar(ts):
                return
            rate = float(self.cfg.funding_rate_per_8h)

        if rate == 0.0:
            return

        notional = self.qty * float(mark_price)
        pay = notional * rate * (1.0 if self.side == "long" else -1.0)
        self.total_funding += pay

        if pay > 0:
            if self.position_margin >= pay:
                self.position_margin -= pay
            else:
                rem = pay - self.position_margin
                self.position_margin = 0.0
                self.free_balance = max(0.0, self.free_balance - rem)
        else:
            self.position_margin += (-pay)

    def close_position(self, exit_ts: pd.Timestamp, raw_exit_price: float, reason: str) -> Optional[float]:
        """Close at given raw price (with slippage) and return pnl_net."""
        if self.side is None or self.qty <= 0:
            return None

        exit_side = "sell" if self.side == "long" else "buy"
        exit_price = _slip_price(raw_exit_price, exit_side, self.slip_frac)

        exit_notional = self.qty * exit_price
        exit_fee = exit_notional * self.fee_rate
        self._pay_fee(exit_fee)

        pnl_gross = (
            self.qty * (exit_price - self.entry_price)
            if self.side == "long"
            else self.qty * (self.entry_price - exit_price)
        )
        pnl_net = pnl_gross - self.current_entry_fee - exit_fee

        # realize into isolated margin then release margin
        self.position_margin += pnl_gross
        if self.position_margin < 0:
            self.position_margin = 0.0
        self.free_balance += self.position_margin

        self.trades.append(
            {
                "entry_time": self.entry_time,
                "exit_time": exit_ts,
                "type": "LONG" if self.side == "long" else "SHORT",
                "signal_type": self.entry_signal_type,
                "entry_price": float(self.entry_price),
                "exit_price": float(exit_price),
                "qty": float(self.qty),
                "leverage": float(self.leverage),
                "initial_margin_used": float(self.current_initial_margin),
                "entry_fee": float(self.current_entry_fee),
                "exit_fee": float(exit_fee),
                "fees_total": float(self.current_entry_fee + exit_fee),
                "pnl_gross": float(pnl_gross),
                "pnl_net": float(pnl_net),
                "exit_reason": reason,
            }
        )

        # reset
        self.side = None
        self.qty = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.entry_signal_type = ""
        self.position_margin = 0.0
        self.stop_loss = 0.0
        self.target_price = 0.0
        self.current_initial_margin = 0.0
        self.current_entry_fee = 0.0

        return float(pnl_net)

    def liquidate(self, exit_ts: pd.Timestamp, liq_price_raw: float) -> Optional[float]:
        """Conservative liquidation: remaining margin is wiped."""
        if self.side is None or self.qty <= 0:
            return None

        self.liquidations += 1

        exit_side = "sell" if self.side == "long" else "buy"
        exit_price = _slip_price(liq_price_raw, exit_side, self.slip_frac)

        exit_notional = self.qty * exit_price
        exit_fee = exit_notional * self.fee_rate
        self._pay_fee(exit_fee)

        margin_wiped = self.position_margin
        self.position_margin = 0.0

        pnl_net = -self.current_initial_margin - self.current_entry_fee - exit_fee

        self.trades.append(
            {
                "entry_time": self.entry_time,
                "exit_time": exit_ts,
                "type": "LONG" if self.side == "long" else "SHORT",
                "signal_type": self.entry_signal_type,
                "entry_price": float(self.entry_price),
                "exit_price": float(exit_price),
                "qty": float(self.qty),
                "leverage": float(self.leverage),
                "initial_margin_used": float(self.current_initial_margin),
                "entry_fee": float(self.current_entry_fee),
                "exit_fee": float(exit_fee),
                "fees_total": float(self.current_entry_fee + exit_fee),
                "pnl_gross": float(-margin_wiped),
                "pnl_net": float(pnl_net),
                "exit_reason": "liquidation",
            }
        )

        # reset
        self.side = None
        self.qty = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.entry_signal_type = ""
        self.current_initial_margin = 0.0
        self.current_entry_fee = 0.0

        return float(pnl_net)

    def on_bar(
        self,
        *,
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
        allow_schedule: bool = True,
    ) -> Dict[str, Any]:
        self.i += 1
        i = self.i
        valid_days = valid_days or set()

        step: Dict[str, Any] = {
            "scheduled": False,
            "scheduled_side": "",
            "scheduled_reason": "",
            "entered": False,
            "entry_order_side": "",
            "entry_qty": 0.0,
            "entry_price": 0.0,
            "entry_fee": 0.0,
            "events": [],
        }

        if (
            self.risk_mgr is not None
            and (self._leverage_cap_requested is not None)
            and not self._leverage_cap_emitted
        ):
            self.risk_mgr.emit_event(
                ts,
                "LEVERAGE_CAPPED",
                "Leverage capped by risk policy",
                requested=float(self._leverage_cap_requested),
                applied=float(self.leverage),
            )
            self._leverage_cap_emitted = True

        # Funding at bar timestamp using open as mark proxy
        self._apply_funding(ts, mark_price=bar_open)

        # Risk checks
        if self.risk_mgr is not None:
            eq_open = float(self.equity(mark_price=bar_open))
            self.risk_mgr.on_bar(ts, current_date, eq_open)

            # Unexpected position detection
            if self.side is not None and self.qty <= 0:
                self.risk_mgr.halt(ts, reason="unexpected_position", message="Position side set but qty <= 0")

            # Exposure duration
            if self.side is not None and self.risk_mgr.should_force_exit_exposure(i):
                pnl_net = self.close_position(ts, raw_exit_price=bar_open, reason="max_exposure")
                if pnl_net is not None:
                    self.risk_mgr.record_trade_close(ts, current_date, pnl_net)

            # Margin ratio spike kill switch
            if self.side is not None and self.qty > 0:
                if self.risk_mgr.check_margin_ratio(
                    ts,
                    current_date,
                    side=self.side,
                    qty=self.qty,
                    entry_price=self.entry_price,
                    position_margin=self.position_margin,
                    mark_price=bar_open,
                    mmr=self.mmr,
                ):
                    pnl_net = self.close_position(ts, raw_exit_price=bar_open, reason="kill_margin_ratio")
                    if pnl_net is not None:
                        self.risk_mgr.record_trade_close(ts, current_date, pnl_net)

        # Execute pending entry
        if self.side is None and self.pending_signal != 0 and self.pending_due_i == i:
            did_enter = False
            reject_reason = ""
            has_required_orbs = False

            if self.risk_mgr is not None and not self.risk_mgr.can_enter(current_date):
                reject_reason = "risk_halt"
            else:
                required_fields = required_orb_fields(self.pending_signal_type)
                has_required_orbs = all(
                    self.pending_orb_high is not None if field_name == "orb_high" else self.pending_orb_low is not None
                    for field_name in required_fields
                )

            if (
                reject_reason == ""
                and (self.pending_date in valid_days)
                and (current_date in valid_days)
                and (self.free_balance > 0)
                and has_required_orbs
            ):
                exec_orb_high = float(self.pending_orb_high) if self.pending_orb_high is not None else 0.0
                exec_orb_low = float(self.pending_orb_low) if self.pending_orb_low is not None else 0.0

                planned_margin = self.free_balance * float(self.cfg.position_size)
                margin_used = planned_margin

                if self.risk_mgr is not None and self.risk_mgr.limits.enabled:
                    eq_now = float(self.equity(mark_price=bar_open))
                    cap = float(eq_now) * float(self.risk_mgr.limits.max_position_margin_frac)
                    if margin_used > cap:
                        margin_used = cap
                        self.risk_mgr.emit_event(
                            ts,
                            "POSITION_SIZE_CAPPED",
                            "Initial margin capped by risk policy",
                            planned=float(planned_margin),
                            capped=float(margin_used),
                            cap=float(cap),
                        )

                margin_used = float(min(margin_used, self.free_balance))
                if margin_used <= 0:
                    reject_reason = "no_margin"
                else:
                    self.free_balance -= margin_used
                    self.position_margin = margin_used
                    self.current_initial_margin = margin_used

                    execution_spec = get_execution_spec(self.pending_signal_type)
                    self.side = execution_spec.side
                    if execution_spec.side == "long":
                        fill_px = _slip_price(bar_open, "buy", self.slip_frac)
                        order_side = "BUY"
                    else:
                        fill_px = _slip_price(bar_open, "sell", self.slip_frac)
                        order_side = "SELL"

                    self.entry_price = float(fill_px)
                    self.entry_time = ts
                    self.entry_signal_type = self.pending_signal_type

                    notional = margin_used * self.leverage
                    self.qty = float(notional / self.entry_price)

                    self.current_entry_fee = float(notional * self.fee_rate)
                    self._pay_fee(self.current_entry_fee)

                    if self.risk_mgr is not None:
                        self.risk_mgr.mark_position_entry(i)

                    execution_plan = resolve_execution_plan(
                        signal_type=self.pending_signal_type,
                        entry_price=self.entry_price,
                        orb_high=exec_orb_high,
                        orb_low=exec_orb_low,
                    )
                    self.target_price = float(execution_plan.target_price)
                    self.stop_loss = float(execution_plan.stop_loss)

                    did_enter = True
                    step["entered"] = True
                    step["entry_order_side"] = order_side
                    step["entry_qty"] = float(self.qty)
                    step["entry_price"] = float(self.entry_price)
                    step["entry_fee"] = float(self.current_entry_fee)
            else:
                reject_reason = "invalid_day_or_missing_orb"

            if self.risk_mgr is not None and self.risk_mgr.limits.enabled and not did_enter:
                self.risk_mgr.record_order_reject(ts, current_date, reason=reject_reason or "unknown")

            # clear pending
            self.pending_signal = 0
            self.pending_signal_type = ""
            self.pending_date = None
            self.pending_due_i = None
            self.pending_orb_high = None
            self.pending_orb_low = None

        # Manage open position
        if self.side is not None and self.qty > 0:
            liq_px = _liq_price(self.side, self.entry_price, self.qty, self.position_margin, self.mmr)

            if self.side == "long":
                if pd.notna(liq_px) and bar_low <= liq_px:
                    pnl_net = self.liquidate(ts, liq_price_raw=float(liq_px))
                    if pnl_net is not None and self.risk_mgr is not None:
                        self.risk_mgr.record_trade_close(ts, current_date, pnl_net)
                else:
                    if bar_low <= self.stop_loss:
                        pnl_net = self.close_position(ts, raw_exit_price=float(self.stop_loss), reason="stop_loss")
                        if pnl_net is not None and self.risk_mgr is not None:
                            self.risk_mgr.record_trade_close(ts, current_date, pnl_net)
                    elif bar_high >= self.target_price:
                        pnl_net = self.close_position(ts, raw_exit_price=float(self.target_price), reason="target")
                        if pnl_net is not None and self.risk_mgr is not None:
                            self.risk_mgr.record_trade_close(ts, current_date, pnl_net)
            else:
                if pd.notna(liq_px) and bar_high >= liq_px:
                    pnl_net = self.liquidate(ts, liq_price_raw=float(liq_px))
                    if pnl_net is not None and self.risk_mgr is not None:
                        self.risk_mgr.record_trade_close(ts, current_date, pnl_net)
                else:
                    if bar_high >= self.stop_loss:
                        pnl_net = self.close_position(ts, raw_exit_price=float(self.stop_loss), reason="stop_loss")
                        if pnl_net is not None and self.risk_mgr is not None:
                            self.risk_mgr.record_trade_close(ts, current_date, pnl_net)
                    elif bar_low <= self.target_price:
                        pnl_net = self.close_position(ts, raw_exit_price=float(self.target_price), reason="target")
                        if pnl_net is not None and self.risk_mgr is not None:
                            self.risk_mgr.record_trade_close(ts, current_date, pnl_net)

        # Schedule pending entry
        if (
            self.side is None
            and int(signal) != 0
            and self.free_balance > 0
            and current_date in valid_days
            and allow_schedule
        ):
            if self.risk_mgr is None or self.risk_mgr.can_enter(current_date):
                self.pending_signal = int(signal)
                self.pending_signal_type = str(signal_type)
                self.pending_date = current_date
                self.pending_due_i = i + int(self.delay_bars)
                self.pending_orb_high = None if orb_high is None else float(orb_high)
                self.pending_orb_low = None if orb_low is None else float(orb_low)
                step["scheduled"] = True
                execution_spec = get_execution_spec(str(signal_type))
                step["scheduled_side"] = "LONG" if execution_spec.side == "long" else "SHORT"
                step["scheduled_reason"] = str(signal_type)


        return step

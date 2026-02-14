# backtester/futures_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .risk import RiskLimits, RiskManager, expected_bar_seconds_from_index


@dataclass
class FuturesEngineConfig:
    """Isolated-margin USDT-margined futures config (single position at a time)."""

    initial_capital: float = 10_000.0
    position_size: float = 0.95  # fraction of FREE balance used as initial margin
    leverage: float = 1.0  # notional = margin * leverage
    taker_fee_rate: float = 0.0005  # base taker fee
    fee_mult: float = 1.0  # stress multiplier
    slippage_bps: float = 0.0  # adverse slippage, bps
    delay_bars: int = 1  # 1=next bar open
    maintenance_margin_rate: float = 0.005  # mmr (approx)

    # Funding (optional)
    # If funding_series is provided, index must be UTC timestamps at funding event times.
    # Else funding_rate_per_8h is applied at 00:00/08:00/16:00 UTC.
    funding_rate_per_8h: float = 0.0  # e.g., 0.0001 = 0.01%
    funding_series: Optional[pd.Series] = None


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


def backtest_futures_orb(
    df: pd.DataFrame,
    orb_ranges: pd.DataFrame,
    valid_days: Optional[set] = None,
    cfg: Optional[FuturesEngineConfig] = None,
    risk_limits: Optional[RiskLimits] = None,
) -> Tuple[List[Dict[str, Any]], List[float], Dict[str, Any]]:
    """Futures backtest engine (isolated margin), next-open execution with realism switches.

    Expected df columns:
      open, high, low, close (floats)
      date (python date)
      signal (int)
      signal_type (str)
    Index must be UTC timestamps.

    Expected orb_ranges:
      index is python date; columns: orb_high, orb_low

    Returns:
      trades, equity_curve (mark-to-market at bar close), stats
    """
    if cfg is None:
        cfg = FuturesEngineConfig()

    valid_days = valid_days or set()

    fee_rate = float(cfg.taker_fee_rate) * float(cfg.fee_mult)
    slip_frac = float(cfg.slippage_bps) / 10000.0
    delay_bars = max(int(cfg.delay_bars), 1)
    leverage = float(cfg.leverage)
    mmr = float(cfg.maintenance_margin_rate)

    # Risk manager (Phase 4)
    if risk_limits is None:
        risk_limits = RiskLimits(enabled=False)
    risk_mgr: Optional[RiskManager] = None
    if risk_limits.enabled:
        risk_mgr = RiskManager(risk_limits, expected_bar_seconds_from_index(df.index))

        # Hard control: cap leverage
        if leverage > float(risk_limits.max_leverage):
            if len(df):
                risk_mgr._event(df.index[0], "LEVERAGE_CAPPED", "Leverage capped by risk policy", requested=leverage, applied=float(risk_limits.max_leverage))
            leverage = float(risk_limits.max_leverage)

    # Account state
    free_balance = float(cfg.initial_capital)
    position_margin = 0.0
    side: Optional[str] = None  # "long" or "short"
    qty = 0.0  # positive
    entry_price = 0.0
    entry_time = None
    entry_signal_type = ""
    stop_loss = 0.0
    target_price = 0.0

    # Per-trade bookkeeping
    current_initial_margin = 0.0
    current_entry_fee = 0.0

    total_fees = 0.0
    total_funding = 0.0
    liquidations = 0

    trades: List[Dict[str, Any]] = []
    equity_curve: List[float] = []

    # Pending order
    pending_signal = 0
    pending_signal_type = ""
    pending_date = None
    pending_due_i: Optional[int] = None

    def equity(mark_price: float) -> float:
        if side is None:
            return free_balance
        unreal = qty * (mark_price - entry_price) if side == "long" else qty * (entry_price - mark_price)
        return free_balance + position_margin + unreal

    def pay_fee(amount: float) -> None:
        nonlocal free_balance, position_margin, total_fees
        amount = float(max(amount, 0.0))
        if amount <= 0:
            return
        total_fees += amount
        if free_balance >= amount:
            free_balance -= amount
        else:
            rem = amount - free_balance
            free_balance = 0.0
            position_margin = max(0.0, position_margin - rem)

    def apply_funding(ts: pd.Timestamp, mark_price: float) -> None:
        """Positive rate => longs pay, shorts receive. Applied at funding timestamps."""
        nonlocal position_margin, free_balance, total_funding

        if side is None:
            return

        if cfg.funding_series is not None:
            if ts not in cfg.funding_series.index:
                return
            rate = float(cfg.funding_series.loc[ts])
        else:
            if not _is_funding_bar(ts):
                return
            rate = float(cfg.funding_rate_per_8h)

        if rate == 0.0:
            return

        notional = qty * mark_price
        pay = notional * rate * (1.0 if side == "long" else -1.0)  # + means pay, - means receive
        total_funding += pay

        if pay > 0:
            if position_margin >= pay:
                position_margin -= pay
            else:
                rem = pay - position_margin
                position_margin = 0.0
                free_balance = max(0.0, free_balance - rem)
        else:
            position_margin += (-pay)

    def close_position(exit_ts: pd.Timestamp, raw_exit_price: float, reason: str) -> Optional[float]:
        """Close at given raw price (with slippage) and return pnl_net."""
        nonlocal side, qty, entry_price, entry_time, entry_signal_type
        nonlocal free_balance, position_margin, stop_loss, target_price
        nonlocal current_initial_margin, current_entry_fee

        if side is None or qty <= 0:
            return None

        exit_side = "sell" if side == "long" else "buy"
        exit_price = _slip_price(raw_exit_price, exit_side, slip_frac)

        exit_notional = qty * exit_price
        exit_fee = exit_notional * fee_rate
        pay_fee(exit_fee)

        pnl_gross = qty * (exit_price - entry_price) if side == "long" else qty * (entry_price - exit_price)
        pnl_net = pnl_gross - current_entry_fee - exit_fee

        # realize into isolated margin then release margin
        position_margin += pnl_gross
        if position_margin < 0:
            position_margin = 0.0
        free_balance += position_margin

        trades.append(
            {
                "entry_time": entry_time,
                "exit_time": exit_ts,
                "type": "LONG" if side == "long" else "SHORT",
                "signal_type": entry_signal_type,
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "qty": float(qty),
                "leverage": float(leverage),
                "initial_margin_used": float(current_initial_margin),
                "entry_fee": float(current_entry_fee),
                "exit_fee": float(exit_fee),
                "fees_total": float(current_entry_fee + exit_fee),
                "pnl_gross": float(pnl_gross),
                "pnl_net": float(pnl_net),
                "exit_reason": reason,
            }
        )

        # reset
        side = None
        qty = 0.0
        entry_price = 0.0
        entry_time = None
        entry_signal_type = ""
        position_margin = 0.0
        stop_loss = 0.0
        target_price = 0.0
        current_initial_margin = 0.0
        current_entry_fee = 0.0

        return float(pnl_net)

    def liquidate(exit_ts: pd.Timestamp, liq_price_raw: float) -> Optional[float]:
        """Conservative liquidation: remaining margin is wiped."""
        nonlocal side, qty, entry_price, entry_time, entry_signal_type
        nonlocal position_margin, liquidations
        nonlocal current_initial_margin, current_entry_fee

        if side is None or qty <= 0:
            return None

        liquidations += 1

        exit_side = "sell" if side == "long" else "buy"
        exit_price = _slip_price(liq_price_raw, exit_side, slip_frac)

        exit_notional = qty * exit_price
        exit_fee = exit_notional * fee_rate
        pay_fee(exit_fee)

        margin_wiped = position_margin
        position_margin = 0.0

        pnl_net = -current_initial_margin - current_entry_fee - exit_fee

        trades.append(
            {
                "entry_time": entry_time,
                "exit_time": exit_ts,
                "type": "LONG" if side == "long" else "SHORT",
                "signal_type": entry_signal_type,
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "qty": float(qty),
                "leverage": float(leverage),
                "initial_margin_used": float(current_initial_margin),
                "entry_fee": float(current_entry_fee),
                "exit_fee": float(exit_fee),
                "fees_total": float(current_entry_fee + exit_fee),
                "pnl_gross": float(-margin_wiped),
                "pnl_net": float(pnl_net),
                "exit_reason": "liquidation",
            }
        )

        # reset
        side = None
        qty = 0.0
        entry_price = 0.0
        entry_time = None
        entry_signal_type = ""
        current_initial_margin = 0.0
        current_entry_fee = 0.0

        return float(pnl_net)

    for i in range(len(df)):
        ts = df.index[i]
        bar_open = float(df["open"].iloc[i])
        bar_high = float(df["high"].iloc[i])
        bar_low = float(df["low"].iloc[i])
        bar_close = float(df["close"].iloc[i])

        current_date = df["date"].iloc[i]
        signal = int(df["signal"].iloc[i])
        signal_type = str(df["signal_type"].iloc[i])

        # Funding at bar timestamp using open as mark proxy
        apply_funding(ts, mark_price=bar_open)

        # Phase 4: per-bar checks
        if risk_mgr is not None:
            eq_open = float(equity(mark_price=bar_open))
            risk_mgr.on_bar(ts, current_date, eq_open)

            # Unexpected position detection
            if side is not None and qty <= 0:
                risk_mgr._halt_global(ts, reason="unexpected_position", message="Position side set but qty <= 0")

            # Exposure duration
            if side is not None and risk_mgr.should_force_exit_exposure(i):
                pnl_net = close_position(ts, raw_exit_price=bar_open, reason="max_exposure")
                if pnl_net is not None:
                    risk_mgr.record_trade_close(ts, current_date, pnl_net)

            # Margin ratio spike kill switch
            if side is not None and qty > 0:
                if risk_mgr.check_margin_ratio(
                    ts,
                    current_date,
                    side=side,
                    qty=qty,
                    entry_price=entry_price,
                    position_margin=position_margin,
                    mark_price=bar_open,
                    mmr=mmr,
                ):
                    # Flatten immediately at market
                    pnl_net = close_position(ts, raw_exit_price=bar_open, reason="kill_margin_ratio")
                    if pnl_net is not None:
                        risk_mgr.record_trade_close(ts, current_date, pnl_net)

        # Execute pending entry
        if side is None and pending_signal != 0 and pending_due_i == i:
            did_enter = False
            reject_reason = ""

            if risk_mgr is not None and not risk_mgr.can_enter(current_date):
                reject_reason = "risk_halt"
            elif (
                pending_date in orb_ranges.index
                and pending_date in valid_days
                and current_date in valid_days
                and free_balance > 0
            ):
                orb_high = float(orb_ranges.loc[pending_date, "orb_high"])
                orb_low = float(orb_ranges.loc[pending_date, "orb_low"])

                # Determine margin used (Phase 4 caps position size)
                planned_margin = free_balance * float(cfg.position_size)
                margin_used = planned_margin

                if risk_mgr is not None and risk_mgr.limits.enabled:
                    eq_now = float(equity(mark_price=bar_open))
                    cap = float(eq_now) * float(risk_mgr.limits.max_position_margin_frac)
                    if margin_used > cap:
                        margin_used = cap
                        risk_mgr._event(
                            ts,
                            "POSITION_SIZE_CAPPED",
                            "Initial margin capped by risk policy",
                            planned=float(planned_margin),
                            capped=float(margin_used),
                            cap=float(cap),
                        )

                margin_used = float(min(margin_used, free_balance))

                if margin_used <= 0:
                    reject_reason = "no_margin"
                else:
                    free_balance -= margin_used
                    position_margin = margin_used
                    current_initial_margin = margin_used

                    # choose side + apply slippage at entry
                    if pending_signal == 1:
                        side = "long"
                        fill = _slip_price(bar_open, "buy", slip_frac)
                    else:
                        side = "short"
                        fill = _slip_price(bar_open, "sell", slip_frac)

                    entry_price = float(fill)
                    entry_time = ts
                    entry_signal_type = pending_signal_type

                    # notional/qty
                    notional = margin_used * leverage
                    qty = float(notional / entry_price)

                    # entry fee charged on notional
                    current_entry_fee = float(notional * fee_rate)
                    pay_fee(current_entry_fee)

                    # Track exposure start
                    if risk_mgr is not None:
                        risk_mgr.mark_position_entry(i)

                    # TP/SL same as your prior logic
                    if pending_signal == 1:
                        target_price = orb_high
                        pct_to_target = (target_price - entry_price) / entry_price
                        stop_loss = entry_price * (1 - pct_to_target)
                    elif pending_signal == -1:
                        target_price = entry_price * 0.98
                        stop_loss = orb_high
                    elif pending_signal == -2:
                        target_price = orb_low
                        pct_to_target = (entry_price - target_price) / entry_price
                        stop_loss = entry_price * (1 + pct_to_target)

                    did_enter = True
            else:
                reject_reason = "invalid_day_or_missing_orb"

            # If not entered, count reject (backtest approximation)
            if risk_mgr is not None and risk_mgr.limits.enabled and not did_enter:
                risk_mgr.record_order_reject(ts, current_date, reason=reject_reason or "unknown")

            # clear pending
            pending_signal = 0
            pending_signal_type = ""
            pending_date = None
            pending_due_i = None

        # Manage open position: liquidation check first
        if side is not None and qty > 0:
            liq_px = _liq_price(side, entry_price, qty, position_margin, mmr)

            if side == "long":
                if pd.notna(liq_px) and bar_low <= liq_px:
                    pnl_net = liquidate(ts, liq_price_raw=float(liq_px))
                    if pnl_net is not None and risk_mgr is not None:
                        risk_mgr.record_trade_close(ts, current_date, pnl_net)
                else:
                    if bar_low <= stop_loss:
                        pnl_net = close_position(ts, raw_exit_price=float(stop_loss), reason="stop_loss")
                        if pnl_net is not None and risk_mgr is not None:
                            risk_mgr.record_trade_close(ts, current_date, pnl_net)
                    elif bar_high >= target_price:
                        pnl_net = close_position(ts, raw_exit_price=float(target_price), reason="target")
                        if pnl_net is not None and risk_mgr is not None:
                            risk_mgr.record_trade_close(ts, current_date, pnl_net)
            else:
                if pd.notna(liq_px) and bar_high >= liq_px:
                    pnl_net = liquidate(ts, liq_price_raw=float(liq_px))
                    if pnl_net is not None and risk_mgr is not None:
                        risk_mgr.record_trade_close(ts, current_date, pnl_net)
                else:
                    if bar_high >= stop_loss:
                        pnl_net = close_position(ts, raw_exit_price=float(stop_loss), reason="stop_loss")
                        if pnl_net is not None and risk_mgr is not None:
                            risk_mgr.record_trade_close(ts, current_date, pnl_net)
                    elif bar_low <= target_price:
                        pnl_net = close_position(ts, raw_exit_price=float(target_price), reason="target")
                        if pnl_net is not None and risk_mgr is not None:
                            risk_mgr.record_trade_close(ts, current_date, pnl_net)

        # Schedule pending entry (valid signal day only)
        if side is None and signal != 0 and free_balance > 0 and current_date in valid_days:
            if risk_mgr is None or risk_mgr.can_enter(current_date):
                due = i + delay_bars
                if due < len(df):
                    pending_signal = signal
                    pending_signal_type = signal_type
                    pending_date = current_date
                    pending_due_i = due

        # Equity mark-to-market at close
        equity_curve.append(float(equity(mark_price=bar_close)))

    # Close any open position at end
    if side is not None and qty > 0:
        last_ts = df.index[-1]
        last_close = float(df["close"].iloc[-1])
        pnl_net = close_position(last_ts, raw_exit_price=last_close, reason="end")
        if pnl_net is not None and risk_mgr is not None:
            risk_mgr.record_trade_close(last_ts, df["date"].iloc[-1], pnl_net)

    final_equity = float(equity(mark_price=float(df["close"].iloc[-1]))) if len(df) else float(cfg.initial_capital)

    stats: Dict[str, Any] = {
        "final_equity": final_equity,
        "free_balance_end": float(free_balance),
        "total_fees": float(total_fees),
        "total_funding": float(total_funding),
        "liquidations": int(liquidations),
        "trades": int(len(trades)),
        "assumptions": {
            "leverage": float(leverage),
            "mmr": float(mmr),
            "fee_rate_effective": float(fee_rate),
            "slippage_bps": float(cfg.slippage_bps),
            "delay_bars": int(delay_bars),
            "funding_rate_per_8h": float(cfg.funding_rate_per_8h),
            "funding_series_used": cfg.funding_series is not None,
        },
    }

    if risk_mgr is not None:
        stats["risk"] = risk_mgr.snapshot()
    else:
        stats["risk"] = {"enabled": False}

    return trades, equity_curve, stats

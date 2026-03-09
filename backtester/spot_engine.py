import os

# Determinism locks (must be set before Python does much work)
os.environ["PYTHONHASHSEED"] = "0"

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from execution_specs import get_execution_spec, required_orb_fields, resolve_execution_plan


def backtest_orb_strategy(
    df: pd.DataFrame,
    orb_ranges: Optional[pd.DataFrame] = None,
    initial_capital: float = 10000,
    position_size: float = 0.95,
    taker_fee_rate: float = 0.0005,
    valid_days: Optional[set] = None,
    fee_mult: float = 1.0,
    slippage_bps: float = 0.0,
    delay_bars: int = 1,
) -> Tuple[List[Dict[str, Any]], List[float], float, float]:
    """
    Next-open execution with realism switches:
      - delay_bars: bars between signal and entry (1 = next bar open)
      - slippage_bps: applied to entries and exits (conservative adverse slip)
      - fee_mult: multiplies taker_fee_rate (stress test fee tiers)
    Valid day policy enforced:
      - schedule only on valid signal days
      - execute only if BOTH signal day and execution day are valid

    TP/SL evaluated using candle high/low after entry (stop checked first).
    """
    valid_days = valid_days or set()
    delay_bars = max(int(delay_bars), 1)
    fee_rate = float(taker_fee_rate) * float(fee_mult)
    slip = float(slippage_bps) / 10000.0

    def slip_price(raw_price: float, side: str) -> float:
        # side = "buy" or "sell" (adverse slippage)
        if side == "buy":
            return raw_price * (1.0 + slip)
        if side == "sell":
            return raw_price * (1.0 - slip)
        raise ValueError("side must be 'buy' or 'sell'")

    capital = float(initial_capital)
    position = 0.0
    entry_price = 0.0
    stop_loss = 0.0
    target_price = 0.0
    trades: List[Dict[str, Any]] = []
    equity_curve: List[float] = []
    total_fees_paid = 0.0

    # Pending order state
    pending_signal: int = 0
    pending_signal_type: str = ""
    pending_date = None         # signal day (python date)
    pending_due_i: Optional[int] = None  # bar index when it should execute
    pending_orb_high: Optional[float] = None
    pending_orb_low: Optional[float] = None

    # Trade state vars
    entry_time = None
    entry_signal_type = ""
    entry_fee = 0.0

    for i in range(len(df)):
        bar_open = float(df["open"].iloc[i])
        bar_close = float(df["close"].iloc[i])
        bar_high = float(df["high"].iloc[i])
        bar_low = float(df["low"].iloc[i])

        current_date = df["date"].iloc[i]  # python date
        signal = int(df["signal"].iloc[i])
        signal_type = str(df["signal_type"].iloc[i])

        # 1) Execute pending entry at this bar open (if due)
        if position == 0.0 and pending_signal != 0 and pending_due_i == i:
            exec_orb_high = pending_orb_high
            exec_orb_low = pending_orb_low
            if (exec_orb_high is None or exec_orb_low is None) and orb_ranges is not None and not orb_ranges.empty:
                if pending_date in orb_ranges.index:
                    if exec_orb_high is None:
                        exec_orb_high = float(orb_ranges.loc[pending_date, "orb_high"])
                    if exec_orb_low is None:
                        exec_orb_low = float(orb_ranges.loc[pending_date, "orb_low"])

            required_fields = required_orb_fields(pending_signal_type)
            has_required_orbs = all(
                exec_orb_high is not None if field_name == "orb_high" else exec_orb_low is not None
                for field_name in required_fields
            )

            if (
                has_required_orbs
                and capital > 0
                and pending_date in valid_days
                and current_date in valid_days
            ):
                notional_value = capital * position_size
                entry_fee = notional_value * fee_rate
                total_fees_paid += entry_fee

                execution_spec = get_execution_spec(pending_signal_type)
                fill = slip_price(bar_open, "buy" if execution_spec.side == "long" else "sell")

                entry_price = float(fill)
                entry_time = df.index[i]
                entry_signal_type = pending_signal_type

                execution_plan = resolve_execution_plan(
                    signal_type=pending_signal_type,
                    entry_price=entry_price,
                    orb_high=float(exec_orb_high if exec_orb_high is not None else 0.0),
                    orb_low=float(exec_orb_low if exec_orb_low is not None else 0.0),
                )

                if execution_plan.side == "long":
                    position = (notional_value - entry_fee) / entry_price
                    capital -= notional_value
                else:
                    position = -((notional_value - entry_fee) / entry_price)
                    capital -= notional_value

                target_price = float(execution_plan.target_price)
                stop_loss = float(execution_plan.stop_loss)

            # Clear pending either way
            pending_signal = 0
            pending_signal_type = ""
            pending_date = None
            pending_due_i = None
            pending_orb_high = None
            pending_orb_low = None

        # 2) Manage open position (SL first, then TP). Apply adverse slippage on exit.
        if position != 0.0:
            # LONG
            if position > 0.0:
                if bar_low <= stop_loss:
                    # stop exit is a sell
                    raw_exit = float(stop_loss)
                    exit_price = float(slip_price(raw_exit, "sell"))
                    exit_fee = exit_price * position * fee_rate
                    total_fees_paid += exit_fee

                    gross_pnl = (exit_price - entry_price) * position
                    net_pnl = gross_pnl - exit_fee
                    capital += exit_price * position - exit_fee

                    trades.append(
                        {
                            "entry_time": entry_time,
                            "exit_time": df.index[i],
                            "type": "LONG",
                            "signal_type": entry_signal_type,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "target_price": target_price,
                            "stop_loss": stop_loss,
                            "position": float(position),
                            "pnl": float(net_pnl),
                            "gross_pnl": float(gross_pnl),
                            "entry_fee": float(entry_fee),
                            "exit_fee": float(exit_fee),
                            "total_fees": float(entry_fee + exit_fee),
                            "return": float(net_pnl / (entry_price * position) * 100.0),
                            "exit_reason": "stop_loss",
                        }
                    )
                    position = 0.0

                elif bar_high >= target_price:
                    # target exit is a sell
                    raw_exit = float(target_price)
                    exit_price = float(slip_price(raw_exit, "sell"))
                    exit_fee = exit_price * position * fee_rate
                    total_fees_paid += exit_fee

                    gross_pnl = (exit_price - entry_price) * position
                    net_pnl = gross_pnl - exit_fee
                    capital += exit_price * position - exit_fee

                    trades.append(
                        {
                            "entry_time": entry_time,
                            "exit_time": df.index[i],
                            "type": "LONG",
                            "signal_type": entry_signal_type,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "target_price": target_price,
                            "stop_loss": stop_loss,
                            "position": float(position),
                            "pnl": float(net_pnl),
                            "gross_pnl": float(gross_pnl),
                            "entry_fee": float(entry_fee),
                            "exit_fee": float(exit_fee),
                            "total_fees": float(entry_fee + exit_fee),
                            "return": float(net_pnl / (entry_price * position) * 100.0),
                            "exit_reason": "target",
                        }
                    )
                    position = 0.0

            # SHORT
            else:
                if bar_high >= stop_loss:
                    # stop exit is a buy (cover)
                    raw_exit = float(stop_loss)
                    exit_price = float(slip_price(raw_exit, "buy"))
                    exit_fee = exit_price * abs(position) * fee_rate
                    total_fees_paid += exit_fee

                    gross_pnl = (entry_price - exit_price) * abs(position)
                    net_pnl = gross_pnl - exit_fee
                    capital += net_pnl + (abs(position) * entry_price)

                    trades.append(
                        {
                            "entry_time": entry_time,
                            "exit_time": df.index[i],
                            "type": "SHORT",
                            "signal_type": entry_signal_type,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "target_price": target_price,
                            "stop_loss": stop_loss,
                            "position": float(abs(position)),
                            "pnl": float(net_pnl),
                            "gross_pnl": float(gross_pnl),
                            "entry_fee": float(entry_fee),
                            "exit_fee": float(exit_fee),
                            "total_fees": float(entry_fee + exit_fee),
                            "return": float(net_pnl / (entry_price * abs(position)) * 100.0),
                            "exit_reason": "stop_loss",
                        }
                    )
                    position = 0.0

                elif bar_low <= target_price:
                    # target exit is a buy (cover)
                    raw_exit = float(target_price)
                    exit_price = float(slip_price(raw_exit, "buy"))
                    exit_fee = exit_price * abs(position) * fee_rate
                    total_fees_paid += exit_fee

                    gross_pnl = (entry_price - exit_price) * abs(position)
                    net_pnl = gross_pnl - exit_fee
                    capital += net_pnl + (abs(position) * entry_price)

                    trades.append(
                        {
                            "entry_time": entry_time,
                            "exit_time": df.index[i],
                            "type": "SHORT",
                            "signal_type": entry_signal_type,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "target_price": target_price,
                            "stop_loss": stop_loss,
                            "position": float(abs(position)),
                            "pnl": float(net_pnl),
                            "gross_pnl": float(gross_pnl),
                            "entry_fee": float(entry_fee),
                            "exit_fee": float(exit_fee),
                            "total_fees": float(entry_fee + exit_fee),
                            "return": float(net_pnl / (entry_price * abs(position)) * 100.0),
                            "exit_reason": "target",
                        }
                    )
                    position = 0.0

        # 3) Schedule pending order from this bar’s signal (valid signal days only)
        if position == 0.0 and signal != 0 and capital > 0 and current_date in valid_days:
            due = i + delay_bars
            if due < len(df):
                pending_signal = signal
                pending_signal_type = signal_type
                pending_date = current_date
                pending_due_i = due
                if "orb_high" in df.columns and not pd.isna(df["orb_high"].iloc[i]):
                    pending_orb_high = float(df["orb_high"].iloc[i])
                else:
                    pending_orb_high = None
                if "orb_low" in df.columns and not pd.isna(df["orb_low"].iloc[i]):
                    pending_orb_low = float(df["orb_low"].iloc[i])
                else:
                    pending_orb_low = None

        # 4) Mark-to-market equity on bar close (deterministic)
        if position > 0:
            current_equity = capital + (position * bar_close)
        elif position < 0:
            collateral = abs(position) * entry_price
            unrealized_pnl = (entry_price - bar_close) * abs(position)
            current_equity = capital + collateral + unrealized_pnl
        else:
            current_equity = capital

        equity_curve.append(float(current_equity))

    # Close at end (last bar close) — treat as market exit with slippage
    if position != 0.0:
        last_close = float(df["close"].iloc[-1])
        if position > 0.0:
            exit_price = float(slip_price(last_close, "sell"))
            exit_fee = exit_price * position * fee_rate
            total_fees_paid += exit_fee

            gross_pnl = (exit_price - entry_price) * position
            net_pnl = gross_pnl - exit_fee
            capital += exit_price * position - exit_fee
        else:
            exit_price = float(slip_price(last_close, "buy"))
            exit_fee = exit_price * abs(position) * fee_rate
            total_fees_paid += exit_fee

            gross_pnl = (entry_price - exit_price) * abs(position)
            net_pnl = gross_pnl - exit_fee
            capital += net_pnl + (abs(position) * entry_price)

        trades.append(
            {
                "entry_time": entry_time,
                "exit_time": df.index[-1],
                "type": "LONG" if position > 0 else "SHORT",
                "signal_type": entry_signal_type,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "target_price": target_price,
                "stop_loss": stop_loss,
                "position": float(abs(position)),
                "pnl": float(net_pnl),
                "gross_pnl": float(gross_pnl),
                "entry_fee": float(entry_fee),
                "exit_fee": float(exit_fee),
                "total_fees": float(entry_fee + exit_fee),
                "return": float(net_pnl / (entry_price * abs(position)) * 100.0),
                "exit_reason": "end",
            }
        )

    return trades, equity_curve, float(capital), float(total_fees_paid)

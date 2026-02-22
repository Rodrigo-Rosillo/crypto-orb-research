import os

# Determinism locks
os.environ["PYTHONHASHSEED"] = "0"

import argparse
import hashlib
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

random.seed(0)
np.random.seed(0)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.utils import sha256_file, stable_json  # noqa: E402
# Reuse exact same primitives as futures engine (fees/slippage/liquidation/funding time)
from backtester.futures_engine import (  # noqa: E402
    FuturesEngineConfig,
    _is_funding_bar,
    _liq_price,
    _slip_price,
)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def get_git_info() -> Dict[str, Any]:
    def run(cmd: List[str]) -> str:
        return subprocess.check_output(cmd, cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL).decode().strip()

    info: Dict[str, Any] = {}
    try:
        info["commit"] = run(["git", "rev-parse", "HEAD"])
        info["branch"] = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        porcelain = run(["git", "status", "--porcelain"])
        info["dirty"] = bool(porcelain)
    except Exception:
        info["commit"] = None
        info["branch"] = None
        info["dirty"] = None
    return info


def parse_timeframe_to_minutes(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 60 * 24
    raise ValueError(f"Unsupported timeframe: {tf}. Use like '30m', '1h', '1d'.")


def compute_max_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    roll_max = equity.cummax()
    dd = (equity / roll_max) - 1.0
    return float(dd.min() * 100.0)


def compute_daily_sharpe(equity_df: pd.DataFrame) -> float:
    """Annualized Sharpe using daily equity returns (sqrt(365))."""
    if equity_df.empty:
        return 0.0
    s = equity_df.set_index("timestamp")["equity"]
    daily = s.resample("1D").last().dropna()
    rets = daily.pct_change().dropna()
    if len(rets) < 2:
        return 0.0
    std = float(rets.std(ddof=1))
    if std == 0.0 or np.isnan(std):
        return 0.0
    return float((rets.mean() / std) * np.sqrt(365.0))


def compute_cagr(equity_df: pd.DataFrame, initial_capital: float) -> float:
    if equity_df.empty or initial_capital <= 0:
        return 0.0
    start_ts = pd.to_datetime(equity_df["timestamp"].iloc[0], utc=True)
    end_ts = pd.to_datetime(equity_df["timestamp"].iloc[-1], utc=True)
    days = (end_ts - start_ts).total_seconds() / 86400.0
    if days <= 0:
        return 0.0
    years = days / 365.0
    final_equity = float(equity_df["equity"].iloc[-1])
    return float((final_equity / initial_capital) ** (1.0 / years) - 1.0)


def summarize_run(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    initial_capital: float,
    total_fees: float,
    total_funding: float,
    liquidations: int,
) -> Dict[str, Any]:
    total_trades = int(len(trades_df))
    pnl_col = "pnl_net" if "pnl_net" in trades_df.columns else None

    if total_trades and pnl_col:
        pnl = trades_df[pnl_col]
        wins = int((pnl > 0).sum())
        losses = int((pnl <= 0).sum())
        win_rate = (wins / total_trades) * 100.0
        avg_win = float(pnl[pnl > 0].mean()) if wins else 0.0
        avg_loss = float(pnl[pnl <= 0].mean()) if losses else 0.0
        total_pnl_net = float(pnl.sum())
        expectancy = float(total_pnl_net / total_trades)
    else:
        wins = losses = 0
        win_rate = avg_win = avg_loss = total_pnl_net = expectancy = 0.0

    final_equity = float(equity_df["equity"].iloc[-1]) if not equity_df.empty else float(initial_capital)
    total_return_pct = (final_equity / initial_capital - 1.0) * 100.0 if initial_capital else 0.0
    max_dd = compute_max_drawdown_pct(equity_df["equity"]) if not equity_df.empty else 0.0
    sharpe = compute_daily_sharpe(equity_df)
    cagr = compute_cagr(equity_df, initial_capital)

    return {
        "initial_capital": float(initial_capital),
        "final_equity": float(final_equity),
        "total_return_pct": float(total_return_pct),
        "cagr": float(cagr),
        "max_drawdown_pct": float(max_dd),
        "daily_sharpe": float(sharpe),
        "total_trades": int(total_trades),
        "winning_trades": int(wins),
        "losing_trades": int(losses),
        "win_rate_pct": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "total_pnl_net": float(total_pnl_net),
        "expectancy_per_trade": float(expectancy),
        "total_fees": float(total_fees),
        "total_funding": float(total_funding),
        "liquidations": int(liquidations),
    }


def maybe_write_equity_plot(equity_df: pd.DataFrame, out_path: Path) -> None:
    """Best-effort plot writer.

    Matplotlib is optional. Some environments have binary wheels mismatches that can emit noisy
    tracebacks on import; we suppress stderr to keep runs clean.
    """
    try:
        import io
        import contextlib

        with contextlib.redirect_stderr(io.StringIO()):
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt  # noqa: F401

        # Plot (suppress any backend noise)
        with contextlib.redirect_stderr(io.StringIO()):
            plt.figure()
            plt.plot(equity_df["timestamp"], equity_df["equity"])
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
    except Exception:
        return




class FuturesBenchmarkSim:
    """
    Minimal futures simulator for benchmarks (NOT ORB TP/SL).
    Uses the same cost assumptions as FuturesEngineConfig:
      - isolated margin
      - fees on notional (entry+exit)
      - adverse slippage
      - optional constant funding at 00/08/16 UTC
      - liquidation approximation via _liq_price
    """

    def __init__(self, cfg: FuturesEngineConfig):
        self.cfg = cfg
        self.fee_rate = float(cfg.taker_fee_rate) * float(cfg.fee_mult)
        self.slip_frac = float(cfg.slippage_bps) / 10000.0
        self.leverage = float(cfg.leverage)
        self.mmr = float(cfg.maintenance_margin_rate)

        self.free_balance = float(cfg.initial_capital)
        self.position_margin = 0.0

        self.side: Optional[str] = None  # "long" or "short"
        self.qty = 0.0
        self.entry_price = 0.0
        self.entry_time = None

        self.current_initial_margin = 0.0
        self.current_entry_fee = 0.0

        self.total_fees = 0.0
        self.total_funding = 0.0
        self.liquidations = 0

        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[float] = []

    def equity(self, mark_price: float) -> float:
        if self.side is None:
            return float(self.free_balance)
        unreal = self.qty * (mark_price - self.entry_price) if self.side == "long" else self.qty * (self.entry_price - mark_price)
        return float(self.free_balance + self.position_margin + unreal)

    def pay_fee(self, amount: float) -> None:
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

    def apply_funding(self, ts: pd.Timestamp, mark_price: float) -> None:
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

        notional = self.qty * mark_price
        pay = notional * rate * (1.0 if self.side == "long" else -1.0)  # + means pay, - means receive
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

    def open_position(self, ts: pd.Timestamp, raw_entry_price: float, side: str, signal_type: str) -> None:
        if self.side is not None:
            return
        if self.free_balance <= 0:
            return

        margin_used = self.free_balance * float(self.cfg.position_size)
        if margin_used <= 0:
            return

        self.free_balance -= margin_used
        self.position_margin = margin_used

        self.current_initial_margin = margin_used

        if side == "long":
            fill = _slip_price(raw_entry_price, "buy", self.slip_frac)
        elif side == "short":
            fill = _slip_price(raw_entry_price, "sell", self.slip_frac)
        else:
            raise ValueError("side must be 'long' or 'short'")

        self.side = side
        self.entry_price = float(fill)
        self.entry_time = ts

        notional = margin_used * self.leverage
        self.qty = float(notional / self.entry_price)

        self.current_entry_fee = float(notional * self.fee_rate)
        self.pay_fee(self.current_entry_fee)

        self._entry_signal_type = str(signal_type)

    def close_position(self, ts: pd.Timestamp, raw_exit_price: float, reason: str) -> None:
        if self.side is None or self.qty <= 0:
            return

        exit_side = "sell" if self.side == "long" else "buy"
        exit_price = _slip_price(raw_exit_price, exit_side, self.slip_frac)

        exit_notional = self.qty * exit_price
        exit_fee = float(exit_notional * self.fee_rate)
        self.pay_fee(exit_fee)

        pnl_gross = self.qty * (exit_price - self.entry_price) if self.side == "long" else self.qty * (self.entry_price - exit_price)
        pnl_net = pnl_gross - self.current_entry_fee - exit_fee

        self.position_margin += pnl_gross
        if self.position_margin < 0:
            self.position_margin = 0.0
        self.free_balance += self.position_margin

        self.trades.append(
            {
                "entry_time": self.entry_time,
                "exit_time": ts,
                "type": "LONG" if self.side == "long" else "SHORT",
                "signal_type": getattr(self, "_entry_signal_type", ""),
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
        self.position_margin = 0.0
        self.current_initial_margin = 0.0
        self.current_entry_fee = 0.0
        self._entry_signal_type = ""

    def liquidate(self, ts: pd.Timestamp, liq_price_raw: float) -> None:
        if self.side is None or self.qty <= 0:
            return

        self.liquidations += 1

        exit_side = "sell" if self.side == "long" else "buy"
        exit_price = _slip_price(liq_price_raw, exit_side, self.slip_frac)

        exit_notional = self.qty * exit_price
        exit_fee = float(exit_notional * self.fee_rate)
        self.pay_fee(exit_fee)

        margin_wiped = self.position_margin
        self.position_margin = 0.0

        pnl_net = -self.current_initial_margin - self.current_entry_fee - exit_fee

        self.trades.append(
            {
                "entry_time": self.entry_time,
                "exit_time": ts,
                "type": "LONG" if self.side == "long" else "SHORT",
                "signal_type": getattr(self, "_entry_signal_type", ""),
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
        self.current_initial_margin = 0.0
        self.current_entry_fee = 0.0
        self._entry_signal_type = ""


def run_cash(df: pd.DataFrame, initial_capital: float) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    equity = np.full(shape=(len(df),), fill_value=float(initial_capital), dtype=float)
    equity_df = pd.DataFrame({"timestamp": df.index, "equity": equity})
    trades_df = pd.DataFrame([])
    stats = {"total_fees": 0.0, "total_funding": 0.0, "liquidations": 0}
    return equity_df, trades_df, stats


def run_always_side(df: pd.DataFrame, cfg: FuturesEngineConfig, side: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    sim = FuturesBenchmarkSim(cfg)

    for i in range(len(df)):
        ts = df.index[i]
        o = float(df["open"].iloc[i])
        h = float(df["high"].iloc[i])
        l = float(df["low"].iloc[i])
        c = float(df["close"].iloc[i])

        # funding at open
        sim.apply_funding(ts, mark_price=o)

        # if flat, maintain exposure
        if sim.side is None and sim.free_balance > 0:
            sim.open_position(ts, raw_entry_price=o, side=side, signal_type=f"always_{side}")

        # liquidation check
        if sim.side is not None and sim.qty > 0:
            liq_px = _liq_price(sim.side, sim.entry_price, sim.qty, sim.position_margin, sim.mmr)
            if sim.side == "long" and pd.notna(liq_px) and l <= liq_px:
                sim.liquidate(ts, liq_price_raw=float(liq_px))
            elif sim.side == "short" and pd.notna(liq_px) and h >= liq_px:
                sim.liquidate(ts, liq_price_raw=float(liq_px))

        sim.equity_curve.append(float(sim.equity(mark_price=c)))

    # close at end (if still open)
    if sim.side is not None and sim.qty > 0:
        last_ts = df.index[-1]
        last_c = float(df["close"].iloc[-1])
        sim.close_position(last_ts, raw_exit_price=last_c, reason="end")
        # refresh last equity point to reflect close
        sim.equity_curve[-1] = float(sim.equity(mark_price=last_c))

    equity_df = pd.DataFrame({"timestamp": df.index, "equity": sim.equity_curve})
    trades_df = pd.DataFrame(sim.trades)
    stats = {"total_fees": float(sim.total_fees), "total_funding": float(sim.total_funding), "liquidations": int(sim.liquidations)}
    return equity_df, trades_df, stats


def run_ma_trend(
    df: pd.DataFrame,
    cfg: FuturesEngineConfig,
    timeframe_minutes: int,
    fast_days: int,
    slow_days: int,
    delay_bars: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Trend baseline:
      - compute fast/slow SMA on close using day-based windows
      - desired: long if fast > slow, short if fast < slow, else flat
      - execute switches at next-bar-open (delay_bars)
    """
    bars_per_day = int((24 * 60) / timeframe_minutes)
    fast_w = max(1, int(fast_days * bars_per_day))
    slow_w = max(2, int(slow_days * bars_per_day))
    delay_bars = max(1, int(delay_bars))

    close = pd.to_numeric(df["close"], errors="coerce")
    fast = close.rolling(fast_w).mean()
    slow = close.rolling(slow_w).mean()

    desired = pd.Series(0, index=df.index, dtype=int)
    desired[(fast > slow)] = 1
    desired[(fast < slow)] = -1
    # if either is NaN, stay flat (0)
    desired[fast.isna() | slow.isna()] = 0

    sim = FuturesBenchmarkSim(cfg)

    pending_due_i: Optional[int] = None
    pending_target: int = 0  # -1 short, 0 flat, +1 long

    for i in range(len(df)):
        ts = df.index[i]
        o = float(df["open"].iloc[i])
        h = float(df["high"].iloc[i])
        l = float(df["low"].iloc[i])
        c = float(df["close"].iloc[i])

        sim.apply_funding(ts, mark_price=o)

        # Execute pending switch at open
        if pending_due_i == i:
            # close if open
            if sim.side is not None and sim.qty > 0:
                sim.close_position(ts, raw_exit_price=o, reason="signal_switch")

            # open if target is not flat
            if pending_target != 0 and sim.free_balance > 0:
                tgt_side = "long" if pending_target == 1 else "short"
                sim.open_position(ts, raw_entry_price=o, side=tgt_side, signal_type=f"ma_{fast_days}_{slow_days}")

            pending_due_i = None
            pending_target = 0

        # Liquidation check
        if sim.side is not None and sim.qty > 0:
            liq_px = _liq_price(sim.side, sim.entry_price, sim.qty, sim.position_margin, sim.mmr)
            if sim.side == "long" and pd.notna(liq_px) and l <= liq_px:
                sim.liquidate(ts, liq_price_raw=float(liq_px))
            elif sim.side == "short" and pd.notna(liq_px) and h >= liq_px:
                sim.liquidate(ts, liq_price_raw=float(liq_px))

        # Decide desired side at close, schedule for next open
        want = int(desired.iloc[i])

        cur = 0
        if sim.side == "long":
            cur = 1
        elif sim.side == "short":
            cur = -1

        if want != cur:
            due = i + delay_bars
            if due < len(df):
                pending_due_i = due
                pending_target = want

        sim.equity_curve.append(float(sim.equity(mark_price=c)))

    # Close at end
    if sim.side is not None and sim.qty > 0:
        last_ts = df.index[-1]
        last_c = float(df["close"].iloc[-1])
        sim.close_position(last_ts, raw_exit_price=last_c, reason="end")
        sim.equity_curve[-1] = float(sim.equity(mark_price=last_c))

    equity_df = pd.DataFrame({"timestamp": df.index, "equity": sim.equity_curve})
    trades_df = pd.DataFrame(sim.trades)
    stats = {"total_fees": float(sim.total_fees), "total_funding": float(sim.total_funding), "liquidations": int(sim.liquidations)}
    stats["ma_fast_days"] = int(fast_days)
    stats["ma_slow_days"] = int(slow_days)
    stats["ma_fast_window_bars"] = int(fast_w)
    stats["ma_slow_window_bars"] = int(slow_w)
    return equity_df, trades_df, stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 3: run sanity benchmark baselines (futures cost model).")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data", default="", help="Processed parquet path. Default: data/processed/<symbol>_<timeframe>.parquet")
    ap.add_argument("--out-dir", default="reports/benchmarks")
    ap.add_argument("--start", default="", help="Optional ISO start UTC, e.g. 2021-01-01T00:00:00Z")
    ap.add_argument("--end", default="", help="Optional ISO end UTC (exclusive)")

    ap.add_argument("--benchmarks", default="cash,always_long,always_short,ma_trend")

    # Cost model knobs (match your futures runs)
    ap.add_argument("--fee-mult", type=float, default=1.0)
    ap.add_argument("--slippage-bps", type=float, default=0.0)
    ap.add_argument("--delay-bars", type=int, default=1)

    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--mmr", type=float, default=0.005)
    ap.add_argument("--funding-per-8h", type=float, default=0.0001)

    # MA baseline knobs
    ap.add_argument("--ma-fast-days", type=int, default=50)
    ap.add_argument("--ma-slow-days", type=int, default=200)

    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()

    cfg_text = config_path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(cfg_text) or {}

    symbol = str(cfg.get("symbol", "SOLUSDT"))
    timeframe = str(cfg.get("timeframe", "30m"))
    tf_min = parse_timeframe_to_minutes(timeframe)

    initial_capital = float(cfg["risk"]["initial_capital"])
    position_size = float(cfg["risk"]["position_size"])
    taker_fee_rate = float(cfg["fees"]["taker_fee_rate"])

    data_path = Path(args.data) if args.data else Path(f"data/processed/{symbol}_{timeframe}.parquet")
    if not data_path.is_absolute():
        data_path = (REPO_ROOT / data_path).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Parquet not found: {data_path}. Run: python scripts/build_parquet.py")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Parquet dataset must have a DatetimeIndex")
    if df.index.tz is None:
        df = df.tz_localize("UTC")

    needed = ["open", "high", "low", "close", "volume"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing required column in parquet: {c}")
    df = df[needed].copy().sort_index()

    if args.start:
        start_ts = pd.to_datetime(args.start, utc=True)
        df = df[df.index >= start_ts]
    if args.end:
        end_ts = pd.to_datetime(args.end, utc=True)
        df = df[df.index < end_ts]

    if df.empty:
        raise RuntimeError("No candles after applying start/end filters.")

    engine_cfg = FuturesEngineConfig(
        initial_capital=initial_capital,
        position_size=position_size,
        leverage=float(args.leverage),
        taker_fee_rate=taker_fee_rate,
        fee_mult=float(args.fee_mult),
        slippage_bps=float(args.slippage_bps),
        delay_bars=int(args.delay_bars),
        maintenance_margin_rate=float(args.mmr),
        funding_rate_per_8h=float(args.funding_per_8h),
    )

    bench_list = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    allowed = {"cash", "always_long", "always_short", "ma_trend"}
    unknown = [b for b in bench_list if b not in allowed]
    if unknown:
        raise ValueError(f"Unknown benchmarks: {unknown}. Allowed: {sorted(allowed)}")

    summary_rows: List[Dict[str, Any]] = []
    outputs_map: Dict[str, Dict[str, str]] = {}

    for bench in bench_list:
        bench_dir = out_dir / bench
        bench_dir.mkdir(parents=True, exist_ok=True)

        if bench == "cash":
            equity_df, trades_df, stats = run_cash(df, initial_capital=initial_capital)
        elif bench == "always_long":
            equity_df, trades_df, stats = run_always_side(df, cfg=engine_cfg, side="long")
        elif bench == "always_short":
            equity_df, trades_df, stats = run_always_side(df, cfg=engine_cfg, side="short")
        elif bench == "ma_trend":
            equity_df, trades_df, stats = run_ma_trend(
                df,
                cfg=engine_cfg,
                timeframe_minutes=tf_min,
                fast_days=int(args.ma_fast_days),
                slow_days=int(args.ma_slow_days),
                delay_bars=int(args.delay_bars),
            )
        else:
            raise RuntimeError("unreachable")

        trades_path = bench_dir / "trades.csv"
        equity_path = bench_dir / "equity_curve.csv"
        results_path = bench_dir / "results.json"
        plot_path = bench_dir / "equity_curve.png"

        trades_df.to_csv(trades_path, index=False)
        equity_df.to_csv(equity_path, index=False)
        maybe_write_equity_plot(equity_df, plot_path)

        total_fees = float(stats.get("total_fees", 0.0))
        total_funding = float(stats.get("total_funding", 0.0))
        liquidations = int(stats.get("liquidations", 0))

        metrics = summarize_run(
            trades_df=trades_df,
            equity_df=equity_df,
            initial_capital=initial_capital,
            total_fees=total_fees,
            total_funding=total_funding,
            liquidations=liquidations,
        )

        result_obj = {
            "benchmark": bench,
            "symbol": symbol,
            "timeframe": timeframe,
            "range": {
                "start": equity_df["timestamp"].min().isoformat(),
                "end": equity_df["timestamp"].max().isoformat(),
                "candles": int(len(equity_df)),
            },
            "params": {
                "initial_capital": float(initial_capital),
                "position_size": float(position_size),
                "taker_fee_rate": float(taker_fee_rate),
                "fee_mult": float(args.fee_mult),
                "slippage_bps": float(args.slippage_bps),
                "delay_bars": int(args.delay_bars),
                "leverage": float(args.leverage),
                "mmr": float(args.mmr),
                "funding_per_8h": float(args.funding_per_8h),
                "ma_fast_days": int(args.ma_fast_days) if bench == "ma_trend" else None,
                "ma_slow_days": int(args.ma_slow_days) if bench == "ma_trend" else None,
            },
            "engine_stats": stats,
            "metrics": metrics,
        }
        results_path.write_text(stable_json(result_obj), encoding="utf-8")

        row = {"benchmark": bench, **metrics}
        summary_rows.append(row)

        outputs_map[bench] = {
            "results.json": str(results_path),
            "trades.csv": str(trades_path),
            "equity_curve.csv": str(equity_path),
        }
        if plot_path.exists():
            outputs_map[bench]["equity_curve.png"] = str(plot_path)

        print(f"✅ {bench}: wrote -> {bench_dir}")

    summary_df = pd.DataFrame(summary_rows).sort_values("benchmark")
    summary_csv = out_dir / "benchmarks_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    # Metadata/hashes for reproducibility
    meta_path = out_dir / "run_metadata.json"
    hashes_path = out_dir / "hashes.json"
    script_path = Path(__file__).resolve()

    outputs_flat = {"benchmarks_summary.csv": str(summary_csv), "run_metadata.json": str(meta_path), "hashes.json": str(hashes_path)}
    for b, om in outputs_map.items():
        for k, v in om.items():
            outputs_flat[f"{b}/{k}"] = v

    hashes = {k: sha256_file(Path(v)) for k, v in outputs_flat.items() if Path(v).exists()}

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git": get_git_info(),
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": {"platform": platform.platform()},
        "inputs": {
            "config_path": str(config_path),
            "config_sha256": sha256_bytes(cfg_text.encode("utf-8")),
            "data_path": str(data_path),
            "data_sha256": sha256_file(data_path),
            "script_path": str(script_path),
            "script_sha256": sha256_file(script_path),
        },
        "outputs": outputs_flat,
        "output_sha256": hashes,
    }

    meta_path.write_text(stable_json(meta), encoding="utf-8")
    hashes_path.write_text(stable_json(hashes), encoding="utf-8")

    print(f"\n[OK] Wrote summary: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import os

# Determinism locks (best-effort; for full effect also run with PYTHONHASHSEED=0 in your shell)
os.environ.setdefault("PYTHONHASHSEED", "0")

import random
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, time as dtime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

random.seed(0)
np.random.seed(0)

# Ensure repo root is importable when running `python scripts/run_baseline.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from strategy import add_trend_indicators, generate_orb_signals, identify_orb_ranges  # noqa: E402


# ----------------------------
# Outputs (kept from your stub)
# ----------------------------
OUT_DIR = REPO_ROOT / "reports" / "baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

results_path = OUT_DIR / "results.json"
plot_path = OUT_DIR / "equity_curve.png"  # only created if --plot is passed


# ----------------------------
# Helpers
# ----------------------------
def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_hhmm(s: str) -> dtime:
    hh, mm = s.split(":")
    return dtime(int(hh), int(mm))


def get_git_commit_hash(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def get_pip_freeze() -> Optional[str]:
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL)
        return out.decode()
    except Exception:
        return None


# ----------------------------
# Data loading
# ----------------------------
def read_binance_csv(file_path: Path) -> pd.DataFrame:
    """
    Reads Binance kline CSV where first column is ms timestamp.
    Assumes the standard 12-column format.
    """
    column_names = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]

    df = pd.read_csv(file_path, names=column_names, header=None, skiprows=1)

    # Treat ms epoch as UTC; keep tz-aware for correct .index.time in UTC
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]
    for col in numeric_cols:
        df[col] = df[col].astype(float)

    df["number_of_trades"] = df["number_of_trades"].astype(int)
    df = df.drop(columns=["ignore"])
    df = df.set_index("timestamp")
    return df


def load_dataset_from_manifest(
    data_dir: Path,
    manifest: Dict[str, Any],
    symbol: str,
    timeframe: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Loads only files matching {symbol}-{timeframe}-*.csv if present, else all manifest files.
    Returns (df, used_paths).
    """
    files = manifest.get("files", [])
    file_paths = [f.get("path") for f in files if isinstance(f, dict) and isinstance(f.get("path"), str)]
    file_paths = [p for p in file_paths if p.lower().endswith(".csv")]

    prefix = f"{symbol}-{timeframe}-"
    selected = [p for p in file_paths if p.startswith(prefix)]
    if not selected:
        selected = file_paths

    selected = sorted(selected, key=lambda x: x.lower())
    dfs = [read_binance_csv(data_dir / p) for p in selected]

    df = pd.concat(dfs).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df, selected


def load_dataset_fallback_all_csvs(data_dir: Path) -> Tuple[pd.DataFrame, List[str]]:
    csvs = sorted([p for p in data_dir.glob("*.csv")])
    if not csvs:
        raise FileNotFoundError(f"No .csv files found in {data_dir}")
    dfs = [read_binance_csv(p) for p in csvs]
    df = pd.concat(dfs).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df, [p.name for p in csvs]


# ----------------------------
# Backtest + metrics (Phase 0 allowed inside runner)
# ----------------------------
def backtest_orb_strategy(
    df: pd.DataFrame,
    orb_ranges: pd.DataFrame,
    initial_capital: float = 10000,
    position_size: float = 0.95,
    taker_fee_rate: float = 0.0005,
) -> Tuple[List[Dict[str, Any]], List[float], float, float]:
    """
    Next-candle-open execution:
    - Signal is observed at candle close (bar i).
    - Entry executes at next candle open (bar i+1).
    - Pending entries ARE allowed to execute on the next UTC day (00:00 or later).
    - When executing a pending entry, ORB levels are taken from the signal day (pending_date).
    - TP/SL logic is evaluated using candle high/low after entry.
    - If both TP and SL occur in the same candle, SL is checked first (conservative).
    """
    capital = float(initial_capital)
    position = 0.0
    entry_price = 0.0
    stop_loss = 0.0
    target_price = 0.0
    trades: List[Dict[str, Any]] = []
    equity_curve: List[float] = []
    total_fees_paid = 0.0

    # Pending order state (created at bar close, executed next bar open)
    pending_signal: int = 0
    pending_signal_type: str = ""
    pending_date = None  # date of the bar where the signal occurred

    # Trade state variables (set when position opens)
    entry_time = None
    entry_signal_type = ""
    entry_fee = 0.0

    for i in range(len(df)):
        bar_open = float(df["open"].iloc[i])
        bar_close = float(df["close"].iloc[i])
        bar_high = float(df["high"].iloc[i])
        bar_low = float(df["low"].iloc[i])

        current_date = df["date"].iloc[i]
        signal = int(df["signal"].iloc[i])
        signal_type = str(df["signal_type"].iloc[i])

        # ---------------------------------------------------------
        # 1) Execute pending entry at *this* candle open (if any)
        # ---------------------------------------------------------
        if position == 0.0 and pending_signal != 0:
            # Allow execution even if current_date != pending_date (00:00+ next day allowed)
            # Still use ORB levels from the signal day (pending_date)
            if pending_date in orb_ranges.index and capital > 0:
                orb_high = float(orb_ranges.loc[pending_date, "orb_high"])
                orb_low = float(orb_ranges.loc[pending_date, "orb_low"])

                notional_value = capital * position_size
                entry_fee = notional_value * taker_fee_rate
                total_fees_paid += entry_fee

                entry_price = bar_open
                entry_time = df.index[i]
                entry_signal_type = pending_signal_type

                if pending_signal == 1:
                    # LONG (uptrend_reversion) - may be disabled in your rule set
                    position = (notional_value - entry_fee) / entry_price
                    capital -= notional_value

                    target_price = orb_high
                    pct_to_target = (target_price - entry_price) / entry_price
                    stop_loss = entry_price * (1 - pct_to_target)

                elif pending_signal == -1:
                    # SHORT (downtrend_breakdown)
                    position = -((notional_value - entry_fee) / entry_price)
                    capital -= notional_value

                    target_price = entry_price * 0.98
                    stop_loss = orb_high

                elif pending_signal == -2:
                    # SHORT (downtrend_reversion) - may be disabled in your rule set
                    position = -((notional_value - entry_fee) / entry_price)
                    capital -= notional_value

                    target_price = orb_low
                    pct_to_target = (entry_price - target_price) / entry_price
                    stop_loss = entry_price * (1 + pct_to_target)

            # Clear pending either way (never keep pending beyond the execution bar)
            pending_signal = 0
            pending_signal_type = ""
            pending_date = None

        # ---------------------------------------------------------
        # 2) Manage open position using this candle high/low
        # ---------------------------------------------------------
        if position != 0.0:
            # LONG
            if position > 0.0:
                if bar_low <= stop_loss:
                    exit_price = float(stop_loss)
                    exit_fee = exit_price * position * taker_fee_rate
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
                            "position": position,
                            "pnl": net_pnl,
                            "gross_pnl": gross_pnl,
                            "entry_fee": entry_fee,
                            "exit_fee": exit_fee,
                            "total_fees": entry_fee + exit_fee,
                            "return": net_pnl / (entry_price * position) * 100,
                            "exit_reason": "stop_loss",
                        }
                    )
                    position = 0.0

                elif bar_high >= target_price:
                    exit_price = float(target_price)
                    exit_fee = exit_price * position * taker_fee_rate
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
                            "position": position,
                            "pnl": net_pnl,
                            "gross_pnl": gross_pnl,
                            "entry_fee": entry_fee,
                            "exit_fee": exit_fee,
                            "total_fees": entry_fee + exit_fee,
                            "return": net_pnl / (entry_price * position) * 100,
                            "exit_reason": "target",
                        }
                    )
                    position = 0.0

            # SHORT
            else:
                if bar_high >= stop_loss:
                    exit_price = float(stop_loss)
                    exit_fee = exit_price * abs(position) * taker_fee_rate
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
                            "position": abs(position),
                            "pnl": net_pnl,
                            "gross_pnl": gross_pnl,
                            "entry_fee": entry_fee,
                            "exit_fee": exit_fee,
                            "total_fees": entry_fee + exit_fee,
                            "return": net_pnl / (entry_price * abs(position)) * 100,
                            "exit_reason": "stop_loss",
                        }
                    )
                    position = 0.0

                elif bar_low <= target_price:
                    exit_price = float(target_price)
                    exit_fee = exit_price * abs(position) * taker_fee_rate
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
                            "position": abs(position),
                            "pnl": net_pnl,
                            "gross_pnl": gross_pnl,
                            "entry_fee": entry_fee,
                            "exit_fee": exit_fee,
                            "total_fees": entry_fee + exit_fee,
                            "return": net_pnl / (entry_price * abs(position)) * 100,
                            "exit_reason": "target",
                        }
                    )
                    position = 0.0

        # ---------------------------------------------------------
        # 3) If flat, schedule a pending order from this bar’s signal
        # ---------------------------------------------------------
        if position == 0.0 and signal != 0 and capital > 0:
            # schedule entry for next bar open (i+1)
            if i + 1 < len(df):
                # Allow pending entry even if next candle is next UTC day
                if current_date in orb_ranges.index:
                    pending_signal = signal
                    pending_signal_type = signal_type
                    pending_date = current_date

        # ---------------------------------------------------------
        # 4) Mark-to-market equity on bar close (deterministic)
        # ---------------------------------------------------------
        if position > 0:
            current_equity = capital + (position * bar_close)
        elif position < 0:
            collateral = abs(position) * entry_price
            unrealized_pnl = (entry_price - bar_close) * abs(position)
            current_equity = capital + collateral + unrealized_pnl
        else:
            current_equity = capital

        equity_curve.append(float(current_equity))

    # Close at end (last bar close)
    if position != 0.0:
        last_close = float(df["close"].iloc[-1])
        if position > 0:
            exit_fee = last_close * position * taker_fee_rate
            total_fees_paid += exit_fee

            gross_pnl = (last_close - entry_price) * position
            net_pnl = gross_pnl - exit_fee
            capital += last_close * position - exit_fee

            trades.append(
                {
                    "entry_time": entry_time,
                    "exit_time": df.index[-1],
                    "type": "LONG",
                    "signal_type": entry_signal_type,
                    "entry_price": entry_price,
                    "exit_price": last_close,
                    "target_price": target_price,
                    "stop_loss": stop_loss,
                    "position": position,
                    "pnl": net_pnl,
                    "gross_pnl": gross_pnl,
                    "entry_fee": entry_fee,
                    "exit_fee": exit_fee,
                    "total_fees": entry_fee + exit_fee,
                    "return": net_pnl / (entry_price * position) * 100,
                    "exit_reason": "end",
                }
            )
        else:
            exit_fee = last_close * abs(position) * taker_fee_rate
            total_fees_paid += exit_fee

            gross_pnl = (entry_price - last_close) * abs(position)
            net_pnl = gross_pnl - exit_fee
            capital += net_pnl + (abs(position) * entry_price)

            trades.append(
                {
                    "entry_time": entry_time,
                    "exit_time": df.index[-1],
                    "type": "SHORT",
                    "signal_type": entry_signal_type,
                    "entry_price": entry_price,
                    "exit_price": last_close,
                    "target_price": target_price,
                    "stop_loss": stop_loss,
                    "position": abs(position),
                    "pnl": net_pnl,
                    "gross_pnl": gross_pnl,
                    "entry_fee": entry_fee,
                    "exit_fee": exit_fee,
                    "total_fees": entry_fee + exit_fee,
                    "return": net_pnl / (entry_price * abs(position)) * 100,
                    "exit_reason": "end",
                }
            )

    return trades, equity_curve, float(capital), float(total_fees_paid)


def calculate_metrics(
    trades: List[Dict[str, Any]],
    equity_curve: List[float],
    initial_capital: float,
    final_capital: float,
    total_fees_paid: float,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.Series]:
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    total_trades = int(len(trades_df))
    winning_trades = int((trades_df["pnl"] > 0).sum()) if total_trades else 0
    losing_trades = total_trades - winning_trades
    win_rate = (winning_trades / total_trades * 100) if total_trades else 0.0

    total_return = ((final_capital - initial_capital) / initial_capital) * 100.0
    total_pnl = float(trades_df["pnl"].sum()) if total_trades else 0.0
    total_gross_pnl = float(trades_df["gross_pnl"].sum()) if total_trades else 0.0
    fee_impact_on_pnl = total_gross_pnl - total_pnl

    avg_return = float(trades_df["return"].mean()) if total_trades else 0.0
    avg_win = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].mean()) if winning_trades else 0.0
    avg_loss = float(trades_df.loc[trades_df["pnl"] <= 0, "pnl"].mean()) if losing_trades else 0.0

    avg_fees_per_trade = float(trades_df["total_fees"].mean()) if total_trades else 0.0
    fees_as_pct_of_capital = (total_fees_paid / initial_capital) * 100.0

    # Drawdown
    equity_series = pd.Series(equity_curve, dtype=float)
    rolling_max = equity_series.expanding().max()
    drawdown = (equity_series - rolling_max) / rolling_max * 100.0
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    signal_types = trades_df["signal_type"].value_counts() if total_trades else pd.Series(dtype=int)

    metrics = {
        "Initial Capital": float(initial_capital),
        "Final Capital": float(final_capital),
        "Total Return %": float(total_return),
        "Total P&L Net": float(total_pnl),
        "Total P&L Gross": float(total_gross_pnl),
        "Total Fees Paid": float(total_fees_paid),
        "Avg Fees/Trade": float(avg_fees_per_trade),
        "Fees % of Capital": float(fees_as_pct_of_capital),
        "Fee Impact on P&L": float(fee_impact_on_pnl),
        "Total Trades": int(total_trades),
        "Winning Trades": int(winning_trades),
        "Losing Trades": int(losing_trades),
        "Win Rate %": float(win_rate),
        "Average Return %": float(avg_return),
        "Average Win": float(avg_win),
        "Average Loss": float(avg_loss),
        "Max Drawdown %": float(max_drawdown),
    }
    return metrics, trades_df, signal_types


# ----------------------------
# Run metadata
# ----------------------------
@dataclass(frozen=True)
class RunMetadata:
    created_at_utc: str
    git_commit: Optional[str]
    config_sha256: str
    manifest_sha256: Optional[str]
    dataset_sha256: Optional[str]
    data_dir: str
    python_version: str
    platform: str
    packages_freeze: Optional[str]
    argv: List[str]
    used_files: List[str]


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Frozen baseline runner")
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml (relative to repo root by default)")
    ap.add_argument("--manifest", default="data/manifest.json", help="Path to manifest.json (relative to repo root)")
    ap.add_argument(
        "--data-dir",
        default=os.getenv("DATA_DIR", ""),
        help="Directory with Binance CSVs (overrides manifest.data_root). Or set DATA_DIR.",
    )
    ap.add_argument("--include-packages", action="store_true", help="Include pip freeze in run_metadata.json")
    ap.add_argument("--plot", action="store_true", help="Also write equity_curve.png (can vary across matplotlib versions)")
    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = (REPO_ROOT / manifest_path).resolve()

    raw_cfg_text = config_path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(raw_cfg_text) or {}
    config_hash = sha256_bytes(raw_cfg_text.encode("utf-8"))

    symbol = str(cfg.get("symbol", "SOLUSDT"))
    timeframe = str(cfg.get("timeframe", "30m"))

    orb_cfg = cfg.get("orb", {}) or {}
    orb_start = parse_hhmm(str(orb_cfg.get("start", "13:30")))
    orb_end = parse_hhmm(str(orb_cfg.get("end", "14:00")))
    orb_cutoff = parse_hhmm(str(orb_cfg.get("cutoff", "14:00")))

    adx_cfg = cfg.get("adx", {}) or {}
    adx_period = int(adx_cfg.get("period", 14))
    adx_threshold = float(adx_cfg.get("threshold", 43))

    risk_cfg = cfg.get("risk", {}) or {}
    initial_capital = float(risk_cfg.get("initial_capital", 10000))
    position_size = float(risk_cfg.get("position_size", 0.95))

    fees_cfg = cfg.get("fees", {}) or {}
    taker_fee_rate = float(fees_cfg.get("taker_fee_rate", 0.0005))

    # Determine data_dir: arg/env overrides manifest
    manifest_obj: Optional[Dict[str, Any]] = None
    dataset_sha256: Optional[str] = None
    manifest_sha256: Optional[str] = None

    if manifest_path.exists():
        manifest_obj = json.loads(manifest_path.read_text(encoding="utf-8"))
        dataset_sha256 = manifest_obj.get("dataset_sha256")
        manifest_sha256 = sha256_file(manifest_path)

    data_dir = Path(args.data_dir).resolve() if args.data_dir else None
    if data_dir is None and manifest_obj and manifest_obj.get("data_root"):
        data_dir = Path(str(manifest_obj["data_root"])).expanduser().resolve()

    if data_dir is None:
        raise RuntimeError("No data directory found. Provide --data-dir or set DATA_DIR or set manifest.data_root.")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Load data
    if manifest_obj:
        df, used_files = load_dataset_from_manifest(data_dir, manifest_obj, symbol, timeframe)
    else:
        df, used_files = load_dataset_fallback_all_csvs(data_dir)

    # Strategy pipeline
    df = add_trend_indicators(df, period=adx_period)
    orb_ranges = identify_orb_ranges(df, orb_start_time=orb_start, orb_end_time=orb_end)
    df = generate_orb_signals(df, orb_ranges, adx_threshold=adx_threshold, orb_cutoff_time=orb_cutoff)

    # Ensure date exists for backtest loop (generate_orb_signals should add it)
    if "date" not in df.columns:
        df = df.copy()
        df["date"] = df.index.date

    trades, equity_curve, final_capital, total_fees_paid = backtest_orb_strategy(
        df,
        orb_ranges,
        initial_capital=initial_capital,
        position_size=position_size,
        taker_fee_rate=taker_fee_rate,
    )

    metrics, trades_df, signal_types = calculate_metrics(
        trades,
        equity_curve,
        initial_capital=initial_capital,
        final_capital=final_capital,
        total_fees_paid=total_fees_paid,
    )

    # Write baseline artifacts
    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "params": {
            "orb_start": orb_start.strftime("%H:%M"),
            "orb_end": orb_end.strftime("%H:%M"),
            "orb_cutoff": orb_cutoff.strftime("%H:%M"),
            "adx_period": adx_period,
            "adx_threshold": adx_threshold,
            "initial_capital": initial_capital,
            "position_size": position_size,
            "taker_fee_rate": taker_fee_rate,
        },
        "dataset": {
            "data_dir": str(data_dir),
            "dataset_sha256": dataset_sha256,
            "manifest_path": str(manifest_path) if manifest_obj else None,
        },
        "range": {
            "candles": int(len(df)),
            "start": df.index.min().isoformat() if len(df) else None,
            "end": df.index.max().isoformat() if len(df) else None,
        },
        "metrics": metrics,
        "signal_type_counts": signal_types.to_dict() if hasattr(signal_types, "to_dict") else {},
    }
    results_path.write_text(stable_json(payload), encoding="utf-8")

    trades_out = OUT_DIR / "trades.csv"
    equity_out = OUT_DIR / "equity_curve.csv"
    orb_out = OUT_DIR / "orb_ranges.csv"
    meta_out = OUT_DIR / "run_metadata.json"

    if len(trades_df):
        trades_df.to_csv(trades_out, index=False)
    else:
        trades_out.write_text("", encoding="utf-8")

    pd.DataFrame({"timestamp": df.index[: len(equity_curve)], "equity": equity_curve}).to_csv(equity_out, index=False)
    orb_ranges.reset_index().to_csv(orb_out, index=False)

    meta = RunMetadata(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=get_git_commit_hash(REPO_ROOT),
        config_sha256=config_hash,
        manifest_sha256=manifest_sha256,
        dataset_sha256=dataset_sha256,
        data_dir=str(data_dir),
        python_version=sys.version.replace("\n", " "),
        platform=f"{platform.system()} {platform.release()} ({platform.platform()})",
        packages_freeze=get_pip_freeze() if args.include_packages else None,
        argv=sys.argv,
        used_files=used_files,
    )
    meta_out.write_text(stable_json(asdict(meta)), encoding="utf-8")

    # Optional plot (not part of “strict identical outputs”)
    if args.plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 5))
        plt.plot(df.index[: len(equity_curve)], equity_curve)
        plt.title("Equity Curve")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

    print(f"✅ Baseline written to: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

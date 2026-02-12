import os

# Determinism locks (must be set before Python does much work)
os.environ["PYTHONHASHSEED"] = "0"

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml

# Determinism: RNG seeds
random.seed(0)
np.random.seed(0)

# Ensure repo root is importable when running: python scripts/run_baseline.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategy import add_trend_indicators, generate_orb_signals, identify_orb_ranges  # noqa: E402


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_hhmm(s: str):
    hh, mm = s.strip().split(":")
    return datetime(2000, 1, 1, int(hh), int(mm)).time()


def parse_timeframe_to_minutes(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 60 * 24
    raise ValueError(f"Unsupported timeframe: {tf}. Use like '30m', '1h', '1d'.")


def load_valid_days_csv(path: Path) -> set:
    """
    Returns a set of datetime.date objects (UTC days) that are valid.
    valid_days.csv must include a 'date_utc' column like '2025-01-01'.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Valid days file not found: {path}. Run: python scripts/build_parquet.py"
        )
    vdf = pd.read_csv(path)
    if "date_utc" not in vdf.columns:
        raise ValueError(f"{path} must contain a 'date_utc' column")
    return set(pd.to_datetime(vdf["date_utc"], utc=True).dt.date)


def read_binance_ohlcv(file_path: Path) -> pd.DataFrame:
    """
    Reads Binance kline CSV, using only the first 6 columns:
      open_time(ms), open, high, low, close, volume
    Assumes there is a header row to skip (as in your raw exports).
    Returns a DF indexed by UTC timestamp with columns: open, high, low, close, volume.
    """
    df = pd.read_csv(
        file_path,
        header=None,
        skiprows=1,
        usecols=[0, 1, 2, 3, 4, 5],
        names=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[~df.index.duplicated(keep="first")]
    return df


def load_raw_dataset_from_manifest(
    manifest_path: Path,
    data_dir_override: Optional[Path],
    symbol: str,
    timeframe: str,
) -> Tuple[pd.DataFrame, Dict[str, Any], Path, List[str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_root = manifest.get("data_root", "")

    if data_dir_override is not None:
        data_dir = data_dir_override
    else:
        # If manifest has relative path, interpret relative to repo root
        data_dir = Path(str(data_root))
        if not data_dir.is_absolute():
            data_dir = (REPO_ROOT / data_dir).resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {data_dir}")

    files = manifest.get("files", [])
    file_paths = [
        f.get("path")
        for f in files
        if isinstance(f, dict)
        and isinstance(f.get("path"), str)
        and f.get("path", "").lower().endswith(".csv")
    ]

    prefix = f"{symbol}-{timeframe}-"
    selected = [p for p in file_paths if p.startswith(prefix)]
    if not selected:
        selected = file_paths

    selected = sorted(selected, key=lambda x: x.lower())
    if not selected:
        raise RuntimeError("No CSV files found in manifest.")

    dfs = []
    for rel in selected:
        fp = data_dir / rel
        if not fp.exists():
            raise FileNotFoundError(f"Missing file listed in manifest: {fp}")
        dfs.append(read_binance_ohlcv(fp))

    df = pd.concat(dfs).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df, manifest, data_dir, selected


def compute_max_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    roll_max = equity.cummax()
    dd = (equity / roll_max) - 1.0
    return float(dd.min() * 100.0)


def backtest_orb_strategy(
    df: pd.DataFrame,
    orb_ranges: pd.DataFrame,
    initial_capital: float = 10000,
    position_size: float = 0.95,
    taker_fee_rate: float = 0.0005,
    valid_days: Optional[set] = None,
) -> Tuple[List[Dict[str, Any]], List[float], float, float]:
    """
    Next-candle-open execution:
    - Signal is observed at candle close (bar i).
    - Entry executes at next candle open (bar i+1).
    - Pending entries ARE allowed to execute on the next UTC day (00:00 or later).
    - When executing a pending entry, ORB levels are taken from the signal day (pending_date).
    - Enforces valid day policy:
        * scheduling only on valid signal days
        * execution only if both signal day and execution day are valid
    - TP/SL evaluated using candle high/low after entry (stop checked first).
    """
    valid_days = valid_days or set()

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
    pending_date = None  # UTC day (python date) where the signal occurred

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

        # 1) Execute pending entry at this bar open (if any)
        if position == 0.0 and pending_signal != 0:
            if (
                pending_date in orb_ranges.index
                and capital > 0
                and pending_date in valid_days
                and current_date in valid_days
            ):
                orb_high = float(orb_ranges.loc[pending_date, "orb_high"])
                orb_low = float(orb_ranges.loc[pending_date, "orb_low"])

                notional_value = capital * position_size
                entry_fee = notional_value * taker_fee_rate
                total_fees_paid += entry_fee

                entry_price = bar_open
                entry_time = df.index[i]
                entry_signal_type = pending_signal_type

                if pending_signal == 1:
                    # LONG (uptrend_reversion)
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
                    # SHORT (downtrend_reversion)
                    position = -((notional_value - entry_fee) / entry_price)
                    capital -= notional_value

                    target_price = orb_low
                    pct_to_target = (entry_price - target_price) / entry_price
                    stop_loss = entry_price * (1 + pct_to_target)

            # Clear pending either way
            pending_signal = 0
            pending_signal_type = ""
            pending_date = None

        # 2) Manage open position
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

        # 3) If flat, schedule pending order from this bar’s signal
        if position == 0.0 and signal != 0 and capital > 0 and current_date in valid_days:
            if i + 1 < len(df):
                pending_signal = signal
                pending_signal_type = signal_type
                pending_date = current_date

        # 4) Mark-to-market equity on bar close
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
        if position > 0.0:
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
                    "position": float(position),
                    "pnl": float(net_pnl),
                    "gross_pnl": float(gross_pnl),
                    "entry_fee": float(entry_fee),
                    "exit_fee": float(exit_fee),
                    "total_fees": float(entry_fee + exit_fee),
                    "return": float(net_pnl / (entry_price * position) * 100.0),
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


def maybe_write_equity_plot(equity_df: pd.DataFrame, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    plt.figure()
    plt.plot(equity_df["timestamp"], equity_df["equity"])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Run baseline ORB backtest (deterministic)")
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml (relative to repo root by default)")
    ap.add_argument("--manifest", default="data/manifest.json", help="Path to raw manifest.json")
    ap.add_argument("--data-dir", default="", help="Override raw data directory (otherwise uses manifest.data_root)")
    ap.add_argument("--valid-days", default="data/processed/valid_days.csv", help="Path to valid_days.csv")
    ap.add_argument("--out-dir", default="reports/baseline", help="Output directory")
    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = (REPO_ROOT / manifest_path).resolve()

    data_dir_override = Path(args.data_dir).resolve() if args.data_dir else None

    valid_days_path = Path(args.valid_days)
    if not valid_days_path.is_absolute():
        valid_days_path = (REPO_ROOT / valid_days_path).resolve()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    cfg_text = config_path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(cfg_text) or {}

    symbol = str(cfg.get("symbol", "SOLUSDT"))
    timeframe = str(cfg.get("timeframe", "30m"))

    orb_start = parse_hhmm(cfg["orb"]["start"])
    orb_end = parse_hhmm(cfg["orb"]["end"])
    orb_cutoff = parse_hhmm(cfg["orb"]["cutoff"])

    adx_period = int(cfg["adx"]["period"])
    adx_threshold = float(cfg["adx"]["threshold"])

    initial_capital = float(cfg["risk"]["initial_capital"])
    position_size = float(cfg["risk"]["position_size"])
    taker_fee_rate = float(cfg["fees"]["taker_fee_rate"])

    # Valid days (48 bars/day)
    valid_days = load_valid_days_csv(valid_days_path)
    print(f"✅ Valid days loaded: {len(valid_days)} ({valid_days_path})")

    # Load data
    df_raw, raw_manifest, data_dir, used_files = load_raw_dataset_from_manifest(
        manifest_path=manifest_path,
        data_dir_override=data_dir_override,
        symbol=symbol,
        timeframe=timeframe,
    )
    print(f"✅ Loaded candles: {len(df_raw)}  ({symbol} {timeframe})")

    # Strategy pipeline
    df_ind = add_trend_indicators(df_raw, period=adx_period)
    orb_ranges = identify_orb_ranges(df_ind, orb_start_time=orb_start, orb_end_time=orb_end)
    orb_ranges = orb_ranges.loc[orb_ranges.index.isin(valid_days)]  # enforce valid day definition

    df_sig = generate_orb_signals(
        df_ind,
        orb_ranges=orb_ranges,
        adx_threshold=adx_threshold,
        orb_cutoff_time=orb_cutoff,
    )

    # Zero out signals on invalid days (keeps candles but blocks entries)
    invalid_mask = ~df_sig["date"].isin(valid_days)
    df_sig.loc[invalid_mask, "signal"] = 0
    df_sig.loc[invalid_mask, "signal_type"] = ""

    # Backtest
    trades, equity_curve, final_capital, total_fees_paid = backtest_orb_strategy(
        df=df_sig,
        orb_ranges=orb_ranges,
        initial_capital=initial_capital,
        position_size=position_size,
        taker_fee_rate=taker_fee_rate,
        valid_days=valid_days,
    )

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame({"timestamp": df_sig.index, "equity": equity_curve})

    # Metrics
    total_trades = int(len(trades_df))
    if total_trades:
        winning_trades = int((trades_df["pnl"] > 0).sum())
        losing_trades = int((trades_df["pnl"] <= 0).sum())
        win_rate = (winning_trades / total_trades) * 100.0
        avg_win = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].mean()) if winning_trades else 0.0
        avg_loss = float(trades_df.loc[trades_df["pnl"] <= 0, "pnl"].mean()) if losing_trades else 0.0
        avg_ret = float(trades_df["return"].mean())
        total_pnl_gross = float(trades_df["gross_pnl"].sum())
        total_pnl_net = float(trades_df["pnl"].sum())
        signal_type_counts = trades_df["signal_type"].value_counts().to_dict()
    else:
        winning_trades = 0
        losing_trades = 0
        win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        avg_ret = 0.0
        total_pnl_gross = 0.0
        total_pnl_net = 0.0
        signal_type_counts = {}

    fee_impact = float(total_pnl_gross - total_pnl_net)
    total_return_pct = (final_capital / initial_capital - 1.0) * 100.0 if initial_capital else 0.0
    max_dd_pct = compute_max_drawdown_pct(equity_df["equity"])
    fees_pct_capital = (total_fees_paid / initial_capital * 100.0) if initial_capital else 0.0
    avg_fees_trade = (total_fees_paid / total_trades) if total_trades else 0.0

    results: Dict[str, Any] = {
        "symbol": symbol,
        "timeframe": timeframe,
        "range": {
            "start": df_sig.index.min().isoformat(),
            "end": df_sig.index.max().isoformat(),
            "candles": int(len(df_sig)),
        },
        "dataset": {
            "data_dir": str(data_dir),
            "manifest_path": str(manifest_path),
            "dataset_sha256": raw_manifest.get("dataset_sha256"),
        },
        "params": {
            "orb_start": cfg["orb"]["start"],
            "orb_end": cfg["orb"]["end"],
            "orb_cutoff": cfg["orb"]["cutoff"],
            "adx_period": adx_period,
            "adx_threshold": adx_threshold,
            "initial_capital": initial_capital,
            "position_size": position_size,
            "taker_fee_rate": taker_fee_rate,
        },
        "signal_type_counts": signal_type_counts,
        "metrics": {
            "Initial Capital": float(initial_capital),
            "Final Capital": float(final_capital),
            "Total Trades": total_trades,
            "Winning Trades": winning_trades,
            "Losing Trades": losing_trades,
            "Win Rate %": float(win_rate),
            "Total P&L Net": float(total_pnl_net),
            "Total P&L Gross": float(total_pnl_gross),
            "Total Return %": float(total_return_pct),
            "Average Win": float(avg_win),
            "Average Loss": float(avg_loss),
            "Average Return %": float(avg_ret),
            "Max Drawdown %": float(max_dd_pct),
            "Total Fees Paid": float(total_fees_paid),
            "Fees % of Capital": float(fees_pct_capital),
            "Avg Fees/Trade": float(avg_fees_trade),
            "Fee Impact on P&L": float(fee_impact),
        },
    }

    # Outputs
    results_path = out_dir / "results.json"
    trades_path = out_dir / "trades.csv"
    equity_path = out_dir / "equity_curve.csv"
    orb_path = out_dir / "orb_ranges.csv"
    meta_path = out_dir / "run_metadata.json"
    hashes_path = out_dir / "hashes.json"
    plot_path = out_dir / "equity_curve.png"

    results_path.write_text(stable_json(results), encoding="utf-8")
    trades_df.to_csv(trades_path, index=False)
    equity_df.to_csv(equity_path, index=False)

    orb_out = orb_ranges.copy()
    orb_out.index = orb_out.index.astype(str)
    orb_out.to_csv(orb_path, index_label="date_utc")

    maybe_write_equity_plot(equity_df, plot_path)

    # Metadata + hashes
    script_path = Path(__file__).resolve()
    outputs = {
        "results.json": str(results_path),
        "trades.csv": str(trades_path),
        "equity_curve.csv": str(equity_path),
        "orb_ranges.csv": str(orb_path),
        "run_metadata.json": str(meta_path),
        "hashes.json": str(hashes_path),
    }
    if plot_path.exists():
        outputs["equity_curve.png"] = str(plot_path)

    hashes = {name: sha256_file(Path(p)) for name, p in outputs.items() if Path(p).exists()}

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git": get_git_info(),
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": {"platform": platform.platform()},
        "inputs": {
            "config_path": str(config_path),
            "config_sha256": sha256_bytes(cfg_text.encode("utf-8")),
            "raw_manifest_path": str(manifest_path),
            "raw_manifest_sha256": sha256_file(manifest_path),
            "raw_dataset_sha256": raw_manifest.get("dataset_sha256"),
            "raw_data_dir": str(data_dir),
            "valid_days_path": str(valid_days_path),
            "valid_days_sha256": sha256_file(valid_days_path),
            "used_files": used_files,
            "script_path": str(script_path),
            "script_sha256": sha256_file(script_path),
        },
        "outputs": outputs,
        "output_sha256": hashes,
    }

    meta_path.write_text(stable_json(meta), encoding="utf-8")
    hashes_path.write_text(stable_json(hashes), encoding="utf-8")

    print(f"\n✅ Wrote outputs to: {out_dir}")
    for k in outputs.keys():
        print(f"  - {k}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

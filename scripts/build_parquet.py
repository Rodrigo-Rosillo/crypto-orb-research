from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.utils import sha256_file  # noqa: E402


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def parse_timeframe_to_minutes(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 60 * 24
    raise ValueError(f"Unsupported timeframe: {tf}. Use like '30m', '1h', '1d'.")


def read_binance_csv(file_path: Path) -> pd.DataFrame:
    """
    Reads Binance kline CSV where first column is ms timestamp.
    Assumes standard 12-column format.
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

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # numeric columns
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
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["number_of_trades"] = pd.to_numeric(df["number_of_trades"], errors="coerce").astype("Int64")

    # drop unused
    df = df.drop(columns=["ignore"], errors="ignore")

    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    return df


def load_dataset_from_manifest(
    data_dir: Path,
    manifest: Dict[str, Any],
    symbol: str,
    timeframe: str,
) -> Tuple[pd.DataFrame, List[str]]:
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


def compute_valid_invalid_days(index_utc: pd.DatetimeIndex, expected_bars_per_day: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Valid day definition: exactly expected_bars_per_day bars present (48 for 30m).
    """
    # counts per UTC day
    counts = pd.Series(1, index=index_utc).groupby(index_utc.normalize()).sum().astype(int)

    df = counts.rename("present_bars").to_frame()
    df["expected_bars"] = int(expected_bars_per_day)
    df["missing_bars"] = df["expected_bars"] - df["present_bars"]
    df["date_utc"] = df.index.date.astype(str)

    # reorder columns
    df = df[["date_utc", "expected_bars", "present_bars", "missing_bars"]]

    valid = df[df["present_bars"] == expected_bars_per_day].copy()
    invalid = df[df["present_bars"] != expected_bars_per_day].copy()

    # sort
    valid = valid.sort_values("date_utc")
    invalid = invalid.sort_values(["missing_bars", "date_utc"], ascending=[False, True])

    return valid.reset_index(drop=True), invalid.reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build processed parquet dataset + valid day lists")
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml (relative to repo root by default)")
    ap.add_argument("--raw-manifest", default="data/manifest.json", help="Raw manifest.json path")
    ap.add_argument("--raw-data-dir", default="", help="Raw CSV directory override. If omitted, uses raw manifest data_root.")
    ap.add_argument("--out-dir", default="data/processed", help="Output directory for processed files")
    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()

    raw_manifest_path = Path(args.raw_manifest)
    if not raw_manifest_path.is_absolute():
        raw_manifest_path = (REPO_ROOT / raw_manifest_path).resolve()

    cfg_text = config_path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(cfg_text) or {}
    symbol = str(cfg.get("symbol", "SOLUSDT"))
    timeframe = str(cfg.get("timeframe", "30m"))

    interval_minutes = parse_timeframe_to_minutes(timeframe)
    expected_delta = pd.Timedelta(minutes=interval_minutes)
    expected_bars_per_day = int(pd.Timedelta(days=1) / expected_delta)

    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    raw_data_root = raw_manifest.get("data_root")
    raw_dataset_sha256 = raw_manifest.get("dataset_sha256")
    raw_manifest_sha256 = sha256_file(raw_manifest_path)

    raw_data_dir = Path(args.raw_data_dir).resolve() if args.raw_data_dir else Path(str(raw_data_root)).expanduser().resolve()
    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_data_dir}")

    # Load raw dataset
    df, used_files = load_dataset_from_manifest(raw_data_dir, raw_manifest, symbol, timeframe)

    # Minimal sanity cleaning for processed dataset
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Add date column for convenience (UTC day)
    df = df.copy()
    df["date_utc"] = df.index.normalize().date.astype(str)

    # Compute valid/invalid days
    valid_days_df, invalid_days_df = compute_valid_invalid_days(df.index, expected_bars_per_day)

    # Output paths
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_name = f"{symbol}_{timeframe}.parquet".replace("/", "-")
    parquet_path = out_dir / parquet_name

    valid_days_path = out_dir / "valid_days.csv"
    invalid_days_path = out_dir / "invalid_days.csv"
    processed_manifest_path = out_dir / "manifest.json"

    # Write parquet
    # Note: requires pyarrow or fastparquet installed; if not, install pyarrow.
    df.to_parquet(parquet_path, index=True)

    # Write day lists
    valid_days_df.to_csv(valid_days_path, index=False)
    invalid_days_df.to_csv(invalid_days_path, index=False)

    # Hash outputs
    parquet_sha256 = sha256_file(parquet_path)
    valid_days_sha256 = sha256_file(valid_days_path)
    invalid_days_sha256 = sha256_file(invalid_days_path)

    # Also hash config content to bind the processed build to settings
    config_sha256 = sha256_bytes(cfg_text.encode("utf-8"))

    processed_manifest: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "timeframe": timeframe,
        "timezone": "UTC",
        "interval_minutes": interval_minutes,
        "expected_bars_per_day": expected_bars_per_day,
        "raw": {
            "raw_data_dir": str(raw_data_dir),
            "raw_manifest_path": str(raw_manifest_path),
            "raw_manifest_sha256": raw_manifest_sha256,
            "raw_dataset_sha256": raw_dataset_sha256,
            "used_files": used_files,
        },
        "config": {
            "config_path": str(config_path),
            "config_sha256": config_sha256,
        },
        "outputs": {
            "parquet_path": str(parquet_path),
            "parquet_sha256": parquet_sha256,
            "valid_days_path": str(valid_days_path),
            "valid_days_sha256": valid_days_sha256,
            "invalid_days_path": str(invalid_days_path),
            "invalid_days_sha256": invalid_days_sha256,
        },
        "stats": {
            "rows": int(len(df)),
            "start_utc": df.index.min().isoformat() if len(df) else None,
            "end_utc": df.index.max().isoformat() if len(df) else None,
            "valid_days": int(len(valid_days_df)),
            "invalid_days": int(len(invalid_days_df)),
        },
    }

    processed_manifest_path.write_text(stable_json(processed_manifest), encoding="utf-8")

    print(f"✅ Wrote parquet: {parquet_path}")
    print(f"✅ Wrote: {valid_days_path}")
    print(f"✅ Wrote: {invalid_days_path}")
    print(f"✅ Wrote: {processed_manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

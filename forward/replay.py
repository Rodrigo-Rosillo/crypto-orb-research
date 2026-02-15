from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


def load_processed_parquet(
    repo_root: Path,
    symbol: str,
    timeframe: str,
    start_utc: Optional[pd.Timestamp] = None,
    end_utc: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, Path]:
    """Load the processed parquet dataset and optionally filter by UTC timestamps."""
    p = repo_root / "data" / "processed" / f"{symbol}_{timeframe}.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"Processed parquet not found: {p}. Run: python scripts/build_parquet.py"
        )

    df = pd.read_parquet(p)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Expected parquet to load with a DatetimeIndex")

    # Normalize to UTC
    if getattr(df.index, "tz", None) is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    if start_utc is not None:
        df = df.loc[df.index >= start_utc]
    if end_utc is not None:
        df = df.loc[df.index <= end_utc]

    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[~df.index.duplicated(keep="first")].sort_index()

    return df, p

from __future__ import annotations

from datetime import time
from typing import Tuple

import numpy as np
import pandas as pd


def calculate_adx(
    df: pd.DataFrame,
    period: int = 14,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate ADX (Average Directional Index) and Directional Indicators (+DI, -DI).

    Pure: depends only on df inputs; no IO; no side effects.

    Returns:
        (adx, plus_di, minus_di) as pandas Series aligned to df.index
    """
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]

    # +DM / -DM
    high_diff = high.diff()
    low_diff = -low.diff()

    plus_dm = high_diff.copy()
    plus_dm[(high_diff < 0) | (high_diff < low_diff)] = 0.0

    minus_dm = low_diff.copy()
    minus_dm[(low_diff < 0) | (low_diff < high_diff)] = 0.0

    # True Range
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # Wilder smoothing (EMA with alpha=1/period)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()

    # Avoid divide-by-zero; keeps output deterministic
    atr_safe = atr.replace(0, np.nan)

    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_safe)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_safe)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    return adx, plus_di, minus_di


def identify_orb_ranges(
    df: pd.DataFrame,
    orb_start_time: time = time(13, 30),
    orb_end_time: time = time(14, 0),
    high_col: str = "high",
    low_col: str = "low",
) -> pd.DataFrame:
    """
    Identify Opening Range Breakout (ORB) high/low for each day.

    Assumes df.index is a DatetimeIndex.
    Returns a DataFrame indexed by date with columns: orb_high, orb_low.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("identify_orb_ranges requires df.index to be a pandas DatetimeIndex")

    dates = df.index.date
    times = df.index.time

    in_orb = (times >= orb_start_time) & (times <= orb_end_time)
    orb = df.loc[in_orb, [high_col, low_col]].copy()
    orb = orb.assign(date=dates[in_orb])

    orb_ranges = (
        orb.groupby("date")
        .agg({high_col: "max", low_col: "min"})
        .rename(columns={high_col: "orb_high", low_col: "orb_low"})
    )
    return orb_ranges


def add_trend_indicators(
    df: pd.DataFrame,
    period: int = 14,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.DataFrame:
    """
    Add ADX, +DI, -DI, and a simple trend label (uptrend/downtrend/sideways).

    Returns a copy of df with new columns:
        adx, plus_di, minus_di, trend
    """
    out = df.copy()

    adx, plus_di, minus_di = calculate_adx(
        out, period=period, high_col=high_col, low_col=low_col, close_col=close_col
    )
    out["adx"] = adx
    out["plus_di"] = plus_di
    out["minus_di"] = minus_di

    out["trend"] = "sideways"
    out.loc[out["plus_di"] > out["minus_di"], "trend"] = "uptrend"
    out.loc[out["minus_di"] > out["plus_di"], "trend"] = "downtrend"

    return out


def generate_orb_signals(
    df: pd.DataFrame,
    orb_ranges: pd.DataFrame,
    adx_threshold: float,
    orb_cutoff_time: time = time(14, 0),
    close_col: str = "close",
) -> pd.DataFrame:
    """
    Generate trading signals based on trend direction and ORB levels.

    Rules (same as your script):
      1) UPTREND + Close BELOW ORB low  -> signal = 1  (uptrend_reversion)
      2) DOWNTREND + Close BELOW ORB low -> signal = -1 (downtrend_breakdown)
      3) DOWNTREND + Close ABOVE ORB high -> signal = -2 (downtrend_reversion)

    Returns a copy of df with new columns:
      date, signal, signal_type, orb_high, orb_low
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("generate_orb_signals requires df.index to be a pandas DatetimeIndex")

    out = df.copy()

    # Ensure date column exists (kept deterministic, no IO)
    out["date"] = out.index.date

    # Prepare outputs
    out["signal"] = 0
    out["signal_type"] = ""
    out["orb_high"] = np.nan
    out["orb_low"] = np.nan

    # Fast merge ORB levels by date
    # orb_ranges is indexed by date with orb_high/orb_low
    orb_map = orb_ranges[["orb_high", "orb_low"]].copy()
    out = out.join(orb_map, on="date", rsuffix="_joined")
    out["orb_high"] = out["orb_high_joined"]
    out["orb_low"] = out["orb_low_joined"]
    out = out.drop(columns=["orb_high_joined", "orb_low_joined"])

    # One signal per day: first qualifying signal after cutoff
    for day in pd.unique(out["date"]):
        day_mask = out["date"] == day
        day_df = out.loc[day_mask]

        # Need ORB levels for that day
        if day not in orb_ranges.index:
            continue

        after_orb = day_df[day_df.index.time > orb_cutoff_time]
        if after_orb.empty:
            continue

        orb_high = orb_ranges.loc[day, "orb_high"]
        orb_low = orb_ranges.loc[day, "orb_low"]

        for idx in after_orb.index:
            close = out.at[idx, close_col]
            trend = out.at[idx, "trend"]
            adx = out.at[idx, "adx"]

            if pd.isna(adx) or adx < adx_threshold:
                continue

            # --- DISABLED RULE #1: uptrend_reversion ---
            # if trend == "uptrend" and close < orb_low:
            #     out.at[idx, "signal"] = 1
            #     out.at[idx, "signal_type"] = "uptrend_reversion"
            #     break

            # --- KEEP ONLY THIS RULE: downtrend_breakdown ---
            if trend == "downtrend" and close < orb_low:
                out.at[idx, "signal"] = -1
                out.at[idx, "signal_type"] = "downtrend_breakdown"
                break

            # --- DISABLED RULE #3: downtrend_reversion ---
            # if trend == "downtrend" and close > orb_high:
            #     out.at[idx, "signal"] = -2
            #     out.at[idx, "signal_type"] = "downtrend_reversion"
            #     break

    return out

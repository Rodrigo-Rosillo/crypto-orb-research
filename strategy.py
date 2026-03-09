from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
import math
from typing import Any, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from execution_specs import get_execution_spec

ALLOWED_TRENDS = {"uptrend", "downtrend"}
ALLOWED_TRIGGERS = {"close_below_orb_low", "close_above_orb_high"}
LEGACY_SIGNAL_TYPE = "downtrend_breakdown"
LEGACY_SIGNAL = -1
LEGACY_TREND = "downtrend"
LEGACY_TRIGGER = "close_below_orb_low"


@dataclass(frozen=True)
class SignalRule:
    signal_type: str
    signal: int
    trend: str
    trigger: str
    adx_threshold: float
    orb_start: time
    orb_end: time
    orb_cutoff: time


def _format_hhmm(value: time) -> str:
    return value.strftime("%H:%M")


def _parse_hhmm(value: Any, field_name: str) -> time:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be an HH:MM string")
    try:
        return datetime.strptime(value.strip(), "%H:%M").time()
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an HH:MM string, got {value!r}") from exc


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _parse_numeric(value: Any, field_name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be finite")
    return numeric


def _parse_int(value: Any, field_name: str) -> int:
    numeric = _parse_numeric(value, field_name)
    integer = int(numeric)
    if not math.isclose(numeric, integer):
        raise ValueError(f"{field_name} must be an integer")
    return integer


def _validate_rule(rule: SignalRule) -> None:
    if rule.trend not in ALLOWED_TRENDS:
        raise ValueError(
            f"signal rule {rule.signal_type!r} has invalid trend {rule.trend!r}; "
            f"expected one of {sorted(ALLOWED_TRENDS)}"
        )
    if rule.trigger not in ALLOWED_TRIGGERS:
        raise ValueError(
            f"signal rule {rule.signal_type!r} has invalid trigger {rule.trigger!r}; "
            f"expected one of {sorted(ALLOWED_TRIGGERS)}"
        )
    if rule.orb_start > rule.orb_end or rule.orb_end > rule.orb_cutoff:
        raise ValueError(
            f"signal rule {rule.signal_type!r} must satisfy orb_start <= orb_end <= orb_cutoff"
        )
    _parse_numeric(rule.adx_threshold, f"signal rule {rule.signal_type!r} adx_threshold")


def _validate_rule_set(rules: Sequence[SignalRule]) -> None:
    if not rules:
        raise ValueError("At least one signal rule is required")

    signal_types = [rule.signal_type for rule in rules]
    if len(set(signal_types)) != len(signal_types):
        raise ValueError("signal_type values must be unique across signal rules")

    signals = [rule.signal for rule in rules]
    if len(set(signals)) != len(signals):
        raise ValueError("signal values must be unique across signal rules")

    for rule in rules:
        _validate_rule(rule)
        get_execution_spec(rule.signal_type)


def _load_rule_from_mapping(raw_rule: Any, index: int) -> SignalRule:
    rule_cfg = _require_mapping(raw_rule, f"signals.rules[{index}]")
    orb_cfg = _require_mapping(rule_cfg.get("orb"), f"signals.rules[{index}].orb")

    signal_type = str(rule_cfg.get("signal_type", "")).strip()
    if not signal_type:
        raise ValueError(f"signals.rules[{index}].signal_type must be a non-empty string")

    return SignalRule(
        signal_type=signal_type,
        signal=_parse_int(rule_cfg.get("signal"), f"signals.rules[{index}].signal"),
        trend=str(rule_cfg.get("trend", "")).strip(),
        trigger=str(rule_cfg.get("trigger", "")).strip(),
        adx_threshold=_parse_numeric(
            rule_cfg.get("adx_threshold"),
            f"signals.rules[{index}].adx_threshold",
        ),
        orb_start=_parse_hhmm(orb_cfg.get("start"), f"signals.rules[{index}].orb.start"),
        orb_end=_parse_hhmm(orb_cfg.get("end"), f"signals.rules[{index}].orb.end"),
        orb_cutoff=_parse_hhmm(orb_cfg.get("cutoff"), f"signals.rules[{index}].orb.cutoff"),
    )


def serialize_signal_rule(rule: SignalRule) -> dict[str, Any]:
    return {
        "signal_type": rule.signal_type,
        "signal": rule.signal,
        "trend": rule.trend,
        "trigger": rule.trigger,
        "adx_threshold": rule.adx_threshold,
        "orb": {
            "start": _format_hhmm(rule.orb_start),
            "end": _format_hhmm(rule.orb_end),
            "cutoff": _format_hhmm(rule.orb_cutoff),
        },
    }


def serialize_signal_rules(rules: Sequence[SignalRule]) -> list[dict[str, Any]]:
    return [serialize_signal_rule(rule) for rule in rules]


def load_signal_rules_from_config(cfg: Mapping[str, Any]) -> list[SignalRule]:
    """Load signal rules from config.

    Preferred format:
      signals.rules: [...]

    Legacy fallback:
      orb.start / orb.end / orb.cutoff
      adx.threshold

    The fallback preserves the repo's prior single-rule behavior:
      downtrend_breakdown, signal=-1, trend=downtrend, close_below_orb_low
    """
    cfg_map = _require_mapping(cfg, "config")
    signals_cfg = cfg_map.get("signals") or {}

    rules: list[SignalRule]
    if isinstance(signals_cfg, Mapping) and "rules" in signals_cfg:
        rules_raw = signals_cfg.get("rules")
        if not isinstance(rules_raw, Sequence) or isinstance(rules_raw, (str, bytes)):
            raise ValueError("signals.rules must be a list of rule mappings")
        if not rules_raw:
            raise ValueError("signals.rules must not be empty")
        rules = [_load_rule_from_mapping(raw_rule, idx) for idx, raw_rule in enumerate(rules_raw)]
    else:
        orb_cfg = _require_mapping(cfg_map.get("orb"), "orb")
        adx_cfg = _require_mapping(cfg_map.get("adx"), "adx")
        rules = [
            SignalRule(
                signal_type=LEGACY_SIGNAL_TYPE,
                signal=LEGACY_SIGNAL,
                trend=LEGACY_TREND,
                trigger=LEGACY_TRIGGER,
                adx_threshold=_parse_numeric(adx_cfg.get("threshold"), "adx.threshold"),
                orb_start=_parse_hhmm(orb_cfg.get("start"), "orb.start"),
                orb_end=_parse_hhmm(orb_cfg.get("end"), "orb.end"),
                orb_cutoff=_parse_hhmm(orb_cfg.get("cutoff"), "orb.cutoff"),
            )
        ]

    _validate_rule_set(rules)
    return rules


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


def build_rule_orb_ranges(
    df: pd.DataFrame,
    rules: Sequence[SignalRule],
    high_col: str = "high",
    low_col: str = "low",
) -> dict[str, pd.DataFrame]:
    rule_orb_ranges: dict[str, pd.DataFrame] = {}
    for rule in rules:
        rule_orb_ranges[rule.signal_type] = identify_orb_ranges(
            df,
            orb_start_time=rule.orb_start,
            orb_end_time=rule.orb_end,
            high_col=high_col,
            low_col=low_col,
        )
    return rule_orb_ranges


def flatten_rule_orb_ranges(
    rule_orb_ranges: Mapping[str, pd.DataFrame],
    rules: Sequence[SignalRule],
) -> pd.DataFrame:
    columns = [
        "date",
        "signal_type",
        "orb_high",
        "orb_low",
        "orb_start",
        "orb_end",
        "orb_cutoff",
        "adx_threshold",
        "trend",
        "trigger",
    ]
    frames: list[pd.DataFrame] = []

    for rule in rules:
        orb_df = rule_orb_ranges.get(rule.signal_type)
        if orb_df is None or orb_df.empty:
            continue

        flat = orb_df.copy()
        flat.index = pd.Index(flat.index, name="date")
        flat = flat.reset_index()
        flat["signal_type"] = rule.signal_type
        flat["orb_start"] = _format_hhmm(rule.orb_start)
        flat["orb_end"] = _format_hhmm(rule.orb_end)
        flat["orb_cutoff"] = _format_hhmm(rule.orb_cutoff)
        flat["adx_threshold"] = rule.adx_threshold
        flat["trend"] = rule.trend
        flat["trigger"] = rule.trigger
        frames.append(flat[columns])

    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def _rule_matches_close(
    *,
    trigger: str,
    close_value: float,
    orb_high: float,
    orb_low: float,
) -> bool:
    if trigger == "close_below_orb_low":
        return close_value < orb_low
    if trigger == "close_above_orb_high":
        return close_value > orb_high
    raise ValueError(f"Unsupported trigger: {trigger}")


def generate_signals_from_rules(
    df: pd.DataFrame,
    rules: Sequence[SignalRule],
    rule_orb_ranges: Mapping[str, pd.DataFrame],
    close_col: str = "close",
) -> pd.DataFrame:
    """Generate at most one signal per day from a config-driven rule set."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("generate_signals_from_rules requires df.index to be a pandas DatetimeIndex")

    missing_orb = [rule.signal_type for rule in rules if rule.signal_type not in rule_orb_ranges]
    if missing_orb:
        raise ValueError(f"Missing ORB ranges for rule(s): {', '.join(sorted(missing_orb))}")

    out = df.copy()
    out["date"] = out.index.date
    out["signal"] = 0
    out["signal_type"] = ""
    out["orb_high"] = np.nan
    out["orb_low"] = np.nan

    for day in pd.unique(out["date"]):
        day_df = out.loc[out["date"] == day].sort_index()
        fired = False

        for ts, row in day_df.iterrows():
            bar_time = ts.time()
            close_value = row.get(close_col)
            trend = str(row.get("trend", ""))
            adx_value = row.get("adx")

            for rule in rules:
                if bar_time <= rule.orb_cutoff:
                    continue

                orb_df = rule_orb_ranges[rule.signal_type]
                if day not in orb_df.index:
                    continue

                if trend != rule.trend:
                    continue

                if pd.isna(adx_value):
                    continue
                adx_float = float(adx_value)
                if not math.isfinite(adx_float) or adx_float < rule.adx_threshold:
                    continue

                orb_high = orb_df.at[day, "orb_high"]
                orb_low = orb_df.at[day, "orb_low"]
                if pd.isna(close_value) or pd.isna(orb_high) or pd.isna(orb_low):
                    continue

                if not _rule_matches_close(
                    trigger=rule.trigger,
                    close_value=float(close_value),
                    orb_high=float(orb_high),
                    orb_low=float(orb_low),
                ):
                    continue

                out.at[ts, "signal"] = rule.signal
                out.at[ts, "signal_type"] = rule.signal_type
                out.at[ts, "orb_high"] = float(orb_high)
                out.at[ts, "orb_low"] = float(orb_low)
                fired = True
                break

            if fired:
                break

    return out


def build_signals_from_config(
    df_raw: pd.DataFrame,
    cfg: Mapping[str, Any],
    valid_days: Sequence[Any] | set[Any] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[SignalRule]]:
    adx_cfg = _require_mapping(cfg.get("adx"), "adx")
    adx_period = _parse_int(adx_cfg.get("period"), "adx.period")
    if adx_period <= 0:
        raise ValueError("adx.period must be positive")

    rules = load_signal_rules_from_config(cfg)
    df_ind = add_trend_indicators(df_raw, period=adx_period)
    return build_signals_from_ruleset(df_ind, rules, valid_days)


def build_signals_from_ruleset(
    df_ind: pd.DataFrame,
    rules: Sequence[SignalRule],
    valid_days: Sequence[Any] | set[Any] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[SignalRule]]:
    rules_list = list(rules)
    _validate_rule_set(rules_list)

    rule_orb_ranges = build_rule_orb_ranges(df_ind, rules_list)
    valid_day_set = set(valid_days) if valid_days is not None else None
    if valid_day_set is not None:
        for signal_type, orb_df in list(rule_orb_ranges.items()):
            rule_orb_ranges[signal_type] = orb_df.loc[orb_df.index.isin(valid_day_set)]

    rule_orb_ranges_df = flatten_rule_orb_ranges(rule_orb_ranges, rules_list)
    df_sig = generate_signals_from_rules(df_ind, rules_list, rule_orb_ranges)

    if valid_day_set is not None:
        invalid_mask = ~df_sig["date"].isin(valid_day_set)
        df_sig.loc[invalid_mask, "signal"] = 0
        df_sig.loc[invalid_mask, "signal_type"] = ""
        df_sig.loc[invalid_mask, "orb_high"] = np.nan
        df_sig.loc[invalid_mask, "orb_low"] = np.nan

    return df_sig, rule_orb_ranges_df, rules_list


def generate_orb_signals(
    df: pd.DataFrame,
    orb_ranges: pd.DataFrame,
    adx_threshold: float,
    orb_cutoff_time: time = time(14, 0),
    close_col: str = "close",
) -> pd.DataFrame:
    """Legacy single-rule wrapper preserved for compatibility."""
    rule = SignalRule(
        signal_type=LEGACY_SIGNAL_TYPE,
        signal=LEGACY_SIGNAL,
        trend=LEGACY_TREND,
        trigger=LEGACY_TRIGGER,
        adx_threshold=_parse_numeric(adx_threshold, "adx_threshold"),
        orb_start=orb_cutoff_time,
        orb_end=orb_cutoff_time,
        orb_cutoff=orb_cutoff_time,
    )
    return generate_signals_from_rules(
        df=df,
        rules=[rule],
        rule_orb_ranges={rule.signal_type: orb_ranges},
        close_col=close_col,
    )

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class BootstrapResult:
    events_to_log: List[dict]
    fatal_reason: Optional[str]
    last_close_ts: Optional[pd.Timestamp]
    trading_start_ts: Optional[pd.Timestamp]


def validate_bootstrap(
    df_raw: pd.DataFrame,
    rest_meta: dict,
    bar_seconds: int,
    bootstrap_max_age_bars: int,
    symbol: str,
    timeframe: str,
    market: str,
    initial_capital: float,
    clock_skew_ms: Optional[int],
    clock_meta: dict,
    skew_warn_ms: int = 5_000,
    skew_fatal_ms: int = 30_000,
) -> BootstrapResult:
    _ = initial_capital

    events_to_log: List[Dict[str, Any]] = []

    bootstrap_max_age_bars = max(1, int(bootstrap_max_age_bars))
    allowed_bootstrap_age_s = float(bar_seconds * bootstrap_max_age_bars)

    last_close_iso = str((rest_meta or {}).get("last_close_time", "") or "")
    last_close_ts: Optional[pd.Timestamp] = None
    if last_close_iso:
        try:
            last_close_ts = pd.to_datetime(last_close_iso, utc=True)
        except Exception:
            last_close_ts = None
    if last_close_ts is None and len(df_raw.index):
        last_close_ts = df_raw.index[-1] + pd.Timedelta(seconds=int(bar_seconds))

    now_utc = datetime.now(timezone.utc)
    bootstrap_age_s = None
    if last_close_ts is not None:
        bootstrap_age_s = float((now_utc - last_close_ts.to_pydatetime()).total_seconds())

    trading_start_ts: Optional[pd.Timestamp] = None
    if len(df_raw.index):
        trading_start_ts = df_raw.index[-1]

    clock_check_error = str((clock_meta or {}).get("clock_skew_check_failed_error", "") or "")
    if clock_skew_ms is None and clock_check_error:
        events_to_log.append({"ts": _utcnow_iso(), "type": "CLOCK_SKEW_CHECK_FAILED", "error": clock_check_error})

    events_to_log.append(
        {
            "ts": _utcnow_iso(),
            "type": "LIVE_RUN_START",
            "mode": "shadow",
            "source": "live",
            "symbol": symbol,
            "timeframe": timeframe,
            "market": market,
            "bootstrap": rest_meta,
            "trading_start_ts": trading_start_ts.isoformat() if trading_start_ts is not None else "",
        }
    )

    if clock_skew_ms is not None:
        if abs(clock_skew_ms) > skew_fatal_ms:
            events_to_log.append(
                {
                    "ts": _utcnow_iso(),
                    "type": "CLOCK_SKEW_FATAL",
                    "clock_skew_ms": int(clock_skew_ms),
                    "warn_ms": int(skew_warn_ms),
                    "fatal_ms": int(skew_fatal_ms),
                    "server_time": clock_meta,
                }
            )
            return BootstrapResult(
                events_to_log=events_to_log,
                fatal_reason="CLOCK_SKEW_FATAL",
                last_close_ts=last_close_ts,
                trading_start_ts=trading_start_ts,
            )
        if abs(clock_skew_ms) > skew_warn_ms:
            events_to_log.append(
                {
                    "ts": _utcnow_iso(),
                    "type": "CLOCK_SKEW_WARN",
                    "clock_skew_ms": int(clock_skew_ms),
                    "warn_ms": int(skew_warn_ms),
                    "fatal_ms": int(skew_fatal_ms),
                    "server_time": clock_meta,
                }
            )

    if last_close_ts is None:
        events_to_log.append(
            {
                "ts": _utcnow_iso(),
                "type": "BOOTSTRAP_EMPTY_OR_INVALID",
                "bootstrap_rows": int(len(df_raw.index)),
            }
        )
        return BootstrapResult(
            events_to_log=events_to_log,
            fatal_reason="BOOTSTRAP_EMPTY_OR_INVALID",
            last_close_ts=last_close_ts,
            trading_start_ts=trading_start_ts,
        )

    events_to_log.append(
        {
            "ts": _utcnow_iso(),
            "type": "BOOTSTRAP_VALIDATION",
            "last_close_time": last_close_ts.isoformat(),
            "age_seconds": float(bootstrap_age_s) if bootstrap_age_s is not None else None,
            "allowed_seconds": float(allowed_bootstrap_age_s),
            "bootstrap_max_age_bars": int(bootstrap_max_age_bars),
            "bar_seconds": int(bar_seconds),
        }
    )

    if bootstrap_age_s is not None and bootstrap_age_s > allowed_bootstrap_age_s:
        events_to_log.append(
            {
                "ts": _utcnow_iso(),
                "type": "BOOTSTRAP_STALE",
                "age_seconds": float(bootstrap_age_s),
                "allowed_seconds": float(allowed_bootstrap_age_s),
                "bootstrap_max_age_bars": int(bootstrap_max_age_bars),
            }
        )
        return BootstrapResult(
            events_to_log=events_to_log,
            fatal_reason="BOOTSTRAP_STALE",
            last_close_ts=last_close_ts,
            trading_start_ts=trading_start_ts,
        )

    events_to_log.append(
        {
            "ts": _utcnow_iso(),
            "type": "BOOTSTRAP_OK",
            "age_seconds": float(bootstrap_age_s) if bootstrap_age_s is not None else None,
        }
    )
    return BootstrapResult(
        events_to_log=events_to_log,
        fatal_reason=None,
        last_close_ts=last_close_ts,
        trading_start_ts=trading_start_ts,
    )

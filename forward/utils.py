from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from core.utils import load_valid_days_csv, parse_hhmm, stable_json

__all__ = [
    "parse_utc_ts",
    "ensure_repo_path",
    "utc_run_id",
    "maybe_get_forward_cfg",
    "load_valid_days_csv",
    "parse_hhmm",
    "stable_json",
]


def parse_utc_ts(s: Optional[str]) -> Optional[pd.Timestamp]:
    """Parse an ISO-like timestamp/date string to a UTC pandas Timestamp."""
    if not s:
        return None
    ts = pd.to_datetime(s, utc=True, errors="raise")
    if getattr(ts, "tz", None) is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def ensure_repo_path(repo_root: Path, p: str) -> Path:
    out = Path(p)
    if not out.is_absolute():
        out = (repo_root / out).resolve()
    return out


def utc_run_id(now: Optional[datetime] = None) -> str:
    now = now or datetime.now(timezone.utc)
    return now.strftime("%Y%m%dT%H%M%SZ")


def maybe_get_forward_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ft = cfg.get("forward_test")
    return ft if isinstance(ft, dict) else {}

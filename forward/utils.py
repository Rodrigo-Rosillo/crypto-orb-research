from __future__ import annotations

from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set

import pandas as pd


def parse_hhmm(s: str) -> time:
    hh, mm = s.strip().split(":")
    return datetime(2000, 1, 1, int(hh), int(mm)).time()


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


def load_valid_days_csv(path: Path) -> Set:
    """Return set of datetime.date objects (UTC days) marked valid."""
    if not path.exists():
        raise FileNotFoundError(
            f"Valid days file not found: {path}. Run: python scripts/build_parquet.py"
        )
    vdf = pd.read_csv(path)
    if "date_utc" not in vdf.columns:
        raise ValueError(f"{path} must contain a 'date_utc' column")
    return set(pd.to_datetime(vdf["date_utc"], utc=True).dt.date)


def ensure_repo_path(repo_root: Path, p: str) -> Path:
    out = Path(p)
    if not out.is_absolute():
        out = (repo_root / out).resolve()
    return out


def stable_json(obj: Any) -> str:
    import json

    return json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2, default=str)


def utc_run_id(now: Optional[datetime] = None) -> str:
    now = now or datetime.now(timezone.utc)
    return now.strftime("%Y%m%dT%H%M%SZ")


def maybe_get_forward_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ft = cfg.get("forward_test")
    return ft if isinstance(ft, dict) else {}

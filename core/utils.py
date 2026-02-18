from __future__ import annotations

import hashlib
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Set
import json

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_hhmm(s: str) -> time:
    hh, mm = s.strip().split(":")
    return datetime(2000, 1, 1, int(hh), int(mm)).time()


def load_valid_days_csv(path: Path) -> Set[date]:
    if not path.exists():
        raise FileNotFoundError(
            f"Valid days file not found: {path}. Run: python scripts/build_parquet.py"
        )
    import pandas as pd

    vdf = pd.read_csv(path)
    if "date_utc" not in vdf.columns:
        raise ValueError(f"{path} must contain a 'date_utc' column")
    return set(pd.to_datetime(vdf["date_utc"], utc=True).dt.date)


def stable_json(obj: Any, indent: int | None = None) -> str:
    """Serialize obj to a stable JSON string with sorted keys.

    Args:
        obj: The object to serialize.
        indent: If None (default), output is compact. Pass an integer (e.g. 2)
                for pretty-printed output.
    """
    separators = (",", ":") if indent is None else None
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=indent, separators=separators)

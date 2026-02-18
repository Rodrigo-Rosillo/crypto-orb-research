from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def try_load_parquet(path: Path) -> Tuple[Optional[pd.DataFrame], str]:
    if not path.exists():
        return None, f"Reference parquet not found: {path}"
    try:
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            for c in ["timestamp", "open_time"]:
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
                    df = df.set_index(c)
                    break
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df = df.sort_index()
        return df, ""
    except Exception as e:
        return None, f"Failed to read parquet ({path.name}): {type(e).__name__}: {e}"


@dataclass
class ReportPaths:
    run_dir: Path
    events: Path
    signals: Path
    orders: Path
    fills: Path
    positions: Path
    config_used: Path
    run_meta: Path
    state: Path


def resolve_run_dir(repo_root: Path, run_id: str) -> Path:
    run_dir = repo_root / "reports" / "forward_test" / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run folder not found: {run_dir}")
    return run_dir


def get_paths(run_dir: Path) -> ReportPaths:
    return ReportPaths(
        run_dir=run_dir,
        events=run_dir / "events.jsonl",
        signals=run_dir / "signals.csv",
        orders=run_dir / "orders.csv",
        fills=run_dir / "fills.csv",
        positions=run_dir / "positions.csv",
        config_used=run_dir / "config_used.yaml",
        run_meta=run_dir / "run_metadata.json",
        state=run_dir / "state.json",
    )

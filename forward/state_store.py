from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class OpenPositionState:
    symbol: str
    side: str  # "SHORT" or "LONG"
    qty: float
    entry_price: float
    entry_time_utc: str
    entry_order_id: Optional[int]
    tp_order_id: Optional[int] = None
    sl_order_id: Optional[int] = None
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunnerState:
    last_bar_open_time_utc: Optional[str] = None
    bars_processed: int = 0
    current_day_utc: Optional[str] = None
    order_rejects_today: int = 0
    open_position: Optional[OpenPositionState] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RunnerState":
        op = d.get("open_position")
        open_pos = None
        if isinstance(op, dict):
            open_pos = OpenPositionState(**op)
        return RunnerState(
            last_bar_open_time_utc=d.get("last_bar_open_time_utc"),
            bars_processed=int(d.get("bars_processed", 0) or 0),
            current_day_utc=d.get("current_day_utc"),
            order_rejects_today=int(d.get("order_rejects_today", 0) or 0),
            open_position=open_pos,
        )


def load_state(path: Path) -> RunnerState:
    if not path.exists():
        return RunnerState()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return RunnerState()
    return RunnerState.from_dict(data)


def save_state(path: Path, state: RunnerState) -> None:
    path.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Optional

from forward.binance_live import BinanceLiveKlineSource
from forward.risk_engine import RiskDecision, check_data_staleness


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DataService:
    def __init__(
        self,
        symbol: str,
        interval: str,
        market: str,
        stale_allowed_seconds: float,
        max_backoff_seconds: int,
        stale_check_interval_seconds: int,
        heartbeat_seconds: int,
        emit_event: Callable[[list[dict]], None],
        on_kill_switch: Optional[Callable[[str, str], None]] = None,
    ):
        self.symbol = symbol
        self.stale_allowed_seconds = float(stale_allowed_seconds)
        self.stale_check_interval_seconds = int(stale_check_interval_seconds)
        self.heartbeat_seconds = int(heartbeat_seconds)
        self.emit_event = emit_event
        self.on_kill_switch = on_kill_switch
        self.last_closed_bar_at: Optional[datetime] = None
        self._src = BinanceLiveKlineSource(
            symbol=symbol,
            interval=interval,
            market=market,
            max_backoff_seconds=max_backoff_seconds,
        )

    @property
    def connect_count(self) -> int:
        return int(self._src.connect_count)

    async def stream_closed(self, stop_event: asyncio.Event) -> AsyncIterator[Any]:
        async for bar in self._src.stream_closed(stop_event=stop_event):
            yield bar

    async def heartbeat_task(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            await asyncio.sleep(max(1, int(self.stale_check_interval_seconds)))
            if stop_event.is_set():
                break

            now = datetime.now(timezone.utc)
            last_msg = self._src.last_message_at or self._src.last_connect_at
            if last_msg is not None:
                since_msg = (now - last_msg).total_seconds()
                if since_msg >= self.heartbeat_seconds:
                    self.emit_event(
                        [
                            {
                                "ts": _utcnow_iso(),
                                "type": "DATA_HEARTBEAT_MISSED",
                                "since_seconds": float(since_msg),
                                "threshold_seconds": int(self.heartbeat_seconds),
                            }
                        ]
                    )

            since_closed = (
                float((now - self.last_closed_bar_at).total_seconds())
                if self.last_closed_bar_at is not None
                else float(self.stale_allowed_seconds + 1.0)
            )
            stale_result = check_data_staleness(
                since_seconds=since_closed,
                allowed_seconds=float(self.stale_allowed_seconds),
            )
            if stale_result.decision == RiskDecision.KILL_SWITCH:
                bar_time_utc = _utcnow_iso()
                self.emit_event(
                    [
                        {
                            "ts": _utcnow_iso(),
                            "type": "KILL_SWITCH_DATA_STALE",
                            "since_seconds": float(since_closed),
                            "allowed_seconds": float(self.stale_allowed_seconds),
                        }
                    ]
                )
                if self.on_kill_switch is not None:
                    self.on_kill_switch("KILL_SWITCH_DATA_STALE", bar_time_utc)
                stop_event.set()
                continue

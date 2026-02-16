from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Optional, Tuple

import pandas as pd
import requests
import websockets


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def interval_to_seconds(interval: str) -> int:
    """Convert Binance interval string to seconds.

    Supports m/h/d/w formats.
    """
    s = interval.strip().lower()
    if s.endswith("m"):
        return int(s[:-1]) * 60
    if s.endswith("h"):
        return int(s[:-1]) * 3600
    if s.endswith("d"):
        return int(s[:-1]) * 86400
    if s.endswith("w"):
        return int(s[:-1]) * 7 * 86400
    raise ValueError(f"Unsupported interval: {interval}")


@dataclass(frozen=True)
class LiveBar:
    symbol: str
    interval: str
    open_time: pd.Timestamp  # candle open (UTC)
    close_time: pd.Timestamp  # candle close (UTC)
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_row(self) -> Dict[str, Any]:
        return {
            "open": float(self.open),
            "high": float(self.high),
            "low": float(self.low),
            "close": float(self.close),
            "volume": float(self.volume),
        }


class BinanceLiveKlineSource:
    """Stream CLOSED klines from Binance (public market data).

    This uses WebSocket klines and only yields bars where kline['x'] == True.
    It includes reconnect logic with exponential backoff.

    Notes:
      - For futures perpetuals (SOLUSDT perp), use market="futures".
      - For spot, use market="spot".
    """

    def __init__(
        self,
        symbol: str,
        interval: str = "30m",
        market: str = "futures",
        ping_interval: int = 20,
        ping_timeout: int = 20,
        max_backoff_seconds: int = 60,
    ) -> None:
        self.symbol = symbol.upper().strip()
        self.interval = interval.strip().lower()
        self.market = market.strip().lower()
        self.ping_interval = int(ping_interval)
        self.ping_timeout = int(ping_timeout)
        self.max_backoff_seconds = int(max_backoff_seconds)

        self._last_emitted_open: Optional[pd.Timestamp] = None

        # Heartbeat support
        self.last_message_at: Optional[datetime] = None
        self.last_connect_at: Optional[datetime] = None
        self.connect_count: int = 0
        self.last_error: Optional[str] = None

    def ws_url(self) -> str:
        sym = self.symbol.lower()
        if self.market == "futures":
            return f"wss://fstream.binance.com/ws/{sym}@kline_{self.interval}"
        if self.market == "spot":
            return f"wss://stream.binance.com:9443/ws/{sym}@kline_{self.interval}"
        raise ValueError("market must be 'futures' or 'spot'")

    async def stream_closed(self, stop_event: Optional[asyncio.Event] = None) -> AsyncIterator[LiveBar]:
        """Async iterator of LiveBar objects (closed candles only)."""
        stop_event = stop_event or asyncio.Event()
        url = self.ws_url()
        backoff = 1.0

        while not stop_event.is_set():
            try:
                async with websockets.connect(
                    url,
                    ping_interval=self.ping_interval,
                    ping_timeout=self.ping_timeout,
                    close_timeout=10,
                    max_queue=1024,
                ) as ws:
                    self.last_connect_at = _utcnow()
                    self.connect_count += 1
                    self.last_error = None
                    backoff = 1.0
                    async for msg in ws:
                        if stop_event.is_set():
                            break

                        # Update heartbeat on *any* websocket message
                        self.last_message_at = _utcnow()

                        try:
                            data = json.loads(msg)
                        except Exception:
                            continue

                        k = data.get("k") if isinstance(data, dict) else None
                        if not isinstance(k, dict):
                            continue

                        # Only closed candles
                        if not bool(k.get("x", False)):
                            continue

                        # Times (ms)
                        try:
                            open_ts = pd.to_datetime(int(k["t"]), unit="ms", utc=True)
                            close_ts = pd.to_datetime(int(k["T"]), unit="ms", utc=True)
                        except Exception:
                            continue

                        # De-dupe / ordering
                        if self._last_emitted_open is not None and open_ts <= self._last_emitted_open:
                            continue
                        self._last_emitted_open = open_ts

                        try:
                            bar = LiveBar(
                                symbol=self.symbol,
                                interval=self.interval,
                                open_time=open_ts,
                                close_time=close_ts,
                                open=float(k.get("o")),
                                high=float(k.get("h")),
                                low=float(k.get("l")),
                                close=float(k.get("c")),
                                volume=float(k.get("v", 0.0)),
                            )
                        except Exception:
                            continue

                        yield bar

            except Exception:
                # Reconnect with jittered exponential backoff
                if stop_event.is_set():
                    break
                self.last_error = "websocket_disconnect"
                sleep_s = min(backoff, float(self.max_backoff_seconds))
                sleep_s = sleep_s + random.random()  # jitter
                await asyncio.sleep(sleep_s)
                backoff = min(backoff * 2.0, float(self.max_backoff_seconds))


def fetch_recent_klines_df(
    symbol: str,
    interval: str = "30m",
    limit: int = 1000,
    market: str = "futures",
    session: Optional[requests.Session] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fetch recent klines via REST (closed + possibly current open).

    Returns a DataFrame indexed by candle OPEN time (UTC) with columns:
      open, high, low, close, volume

    Metadata includes the REST endpoint and fetch timestamp.
    """
    symbol = symbol.upper().strip()
    interval = interval.strip().lower()
    market = market.strip().lower()
    limit = int(max(1, min(limit, 1500)))

    if market == "futures":
        base = "https://fapi.binance.com"
    elif market == "spot":
        base = "https://api.binance.com"
    else:
        raise ValueError("market must be 'futures' or 'spot'")

    url = f"{base}/api/v3/klines" if market == "spot" else f"{base}/fapi/v1/klines"

    params = {"symbol": symbol, "interval": interval, "limit": limit}
    sess = session or requests.Session()
    r = sess.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    rows = []
    for k in data:
        # kline schema: [openTime, open, high, low, close, volume, closeTime, ...]
        try:
            open_time = pd.to_datetime(int(k[0]), unit="ms", utc=True)
            close_time = pd.to_datetime(int(k[6]), unit="ms", utc=True)
            rows.append(
                {
                    "open_time": open_time,
                    "close_time": close_time,
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                }
            )
        except Exception:
            continue

    if not rows:
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df.index = pd.DatetimeIndex([], tz="UTC")
    else:
        df = pd.DataFrame(rows).set_index("open_time")
        df = df.sort_index()
        # Keep only closed candles (close_time <= now) for bootstrapping.
        now = pd.Timestamp(_utcnow())
        df = df[df["close_time"] <= now]
        df = df.drop(columns=["close_time"])

    meta = {
        "endpoint": url,
        "fetched_at": _utcnow().isoformat(),
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "market": market,
        "rows": int(len(df)),
    }

    return df, meta

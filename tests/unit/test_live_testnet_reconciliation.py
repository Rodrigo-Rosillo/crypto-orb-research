from __future__ import annotations

import asyncio
import json
from datetime import time
from pathlib import Path
from typing import Any

import pandas as pd

from forward.binance_live import LiveBar
from forward.live_testnet import run_live_testnet
from forward.state_store_sqlite import OpenPositionState, SQLiteStateStore
from forward.trader_service import TraderService


class FakeRuntimeBroker:
    def __init__(self, *, position_amts: list[float], position_risk_raise_on_call: int | None = None) -> None:
        self.position_amts = list(position_amts)
        self.position_risk_raise_on_call = position_risk_raise_on_call
        self.position_risk_call_count = 0
        self.cancel_all_called = False
        self.flatten_calls: list[dict[str, Any]] = []

    def set_leverage(self, symbol: str, leverage: int) -> dict[str, Any]:
        _ = symbol
        return {"leverage": int(leverage)}

    def position_risk(self, *, symbol: str) -> dict[str, Any]:
        if (
            self.position_risk_raise_on_call is not None
            and self.position_risk_call_count == int(self.position_risk_raise_on_call)
        ):
            raise RuntimeError("position_risk_failed")
        idx = min(self.position_risk_call_count, len(self.position_amts) - 1)
        amt = float(self.position_amts[idx])
        self.position_risk_call_count += 1
        return {
            "symbol": str(symbol),
            "positionAmt": f"{amt}",
            "entryPrice": "100.0" if abs(amt) > 1e-12 else "0.0",
            "unRealizedProfit": "0",
        }

    def account(self) -> dict[str, Any]:
        return {
            "availableBalance": "1000",
            "totalMarginBalance": "1000",
            "totalMaintMargin": "0",
        }

    def cancel_all_open_orders(self, *, symbol: str) -> dict[str, Any]:
        self.cancel_all_called = True
        return {"symbol": str(symbol), "status": "ok"}

    def place_market_order(self, *, symbol: str, side: str, quantity: float, reduce_only: bool = False) -> dict[str, Any]:
        self.flatten_calls.append(
            {
                "symbol": str(symbol),
                "side": str(side),
                "quantity": float(quantity),
                "reduce_only": bool(reduce_only),
            }
        )
        return {
            "orderId": 999,
            "status": "FILLED",
            "avgPrice": "100.0",
            "executedQty": f"{float(quantity)}",
            "origQty": f"{float(quantity)}",
        }


class FakeDataService:
    bars: list[LiveBar] = []

    def __init__(
        self,
        *,
        symbol: str,
        interval: str,
        market: str,
        stale_allowed_seconds: float,
        max_backoff_seconds: int,
        stale_check_interval_seconds: int,
        heartbeat_seconds: int,
        emit_event: Any,
        on_kill_switch: Any = None,
    ) -> None:
        _ = (
            symbol,
            interval,
            market,
            stale_allowed_seconds,
            max_backoff_seconds,
            stale_check_interval_seconds,
            heartbeat_seconds,
            emit_event,
            on_kill_switch,
        )
        self.connect_count = 1
        self.last_closed_bar_at = None

    async def heartbeat_task(self, stop_event: asyncio.Event) -> None:
        await stop_event.wait()

    async def stream_closed(self, stop_event: asyncio.Event):
        for bar in type(self).bars:
            if stop_event.is_set():
                return
            yield bar


def _runtime_timestamps() -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    anchor = pd.Timestamp.now(tz="UTC").floor("30min")
    bootstrap_open = anchor - pd.Timedelta(minutes=30)
    bootstrap_close = anchor
    live_open = anchor
    return bootstrap_open, bootstrap_close, live_open


def _bootstrap_df(bootstrap_open: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        [{"open": 99.0, "high": 101.0, "low": 98.0, "close": 100.0, "volume": 1.0}],
        index=pd.DatetimeIndex([bootstrap_open]),
    )


def _live_bar(live_open: pd.Timestamp) -> LiveBar:
    return LiveBar(
        symbol="SOLUSDT",
        interval="30m",
        open_time=live_open,
        close_time=live_open + pd.Timedelta(minutes=29, seconds=59),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        volume=2.0,
    )


def _build_signals_df(df_raw: pd.DataFrame, *, signal: int) -> pd.DataFrame:
    df_sig = df_raw.copy()
    df_sig["signal"] = int(signal)
    df_sig["signal_type"] = "unit_runtime_signal" if int(signal) != 0 else ""
    df_sig["orb_high"] = 200.0
    df_sig["close"] = df_sig["close"].astype(float)
    return df_sig


def _read_events(events_path: Path) -> list[dict[str, Any]]:
    if not events_path.exists():
        return []
    return [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run_cfg(tmp_path: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    run_dir = tmp_path / "run"
    cfg: dict[str, Any] = {}
    ft_cfg: dict[str, Any] = {
        "live": {
            "market": "futures",
            "bootstrap_limit": 10,
            "max_backoff_seconds": 1,
            "heartbeat_seconds": 60,
            "stale_check_interval_seconds": 60,
        },
        "testnet": {
            "base_url": "https://demo-fapi.binance.com",
            "recv_window_ms": 5000,
            "poll_interval_seconds": 0.0,
            "cancel_open_orders_on_exit": True,
            "flatten_on_mismatch": False,
            "max_retries": 1,
            "base_backoff_seconds": 0.01,
            "max_backoff_seconds": 0.01,
        },
    }
    return run_dir, cfg, ft_cfg


def _seed_open_position_state(run_dir: Path, *, side: str, qty: float, symbol: str = "SOLUSDT") -> None:
    db_path = run_dir / "state.db"
    with SQLiteStateStore(db_path=db_path) as store:
        state = store.load_state()
        state.open_position = OpenPositionState(
            symbol=str(symbol),
            side=str(side),
            qty=float(qty),
            entry_price=100.0,
            entry_time_utc="2024-01-01T00:30:00+00:00",
            entry_order_id=123,
        )
        store.save_state(state)


def test_runtime_reconciliation_blocks_new_entry_and_emits_runtime_reason(tmp_path: Path, monkeypatch) -> None:
    bootstrap_open, bootstrap_close, live_open = _runtime_timestamps()
    broker = FakeRuntimeBroker(position_amts=[0.0, 1.0])
    FakeDataService.bars = [_live_bar(live_open)]

    entry_calls: list[pd.Timestamp] = []

    async def _unexpected_entry(self: TraderService, bar_open_time: pd.Timestamp, row: pd.Series) -> None:
        _ = row
        entry_calls.append(bar_open_time)

    monkeypatch.setattr("forward.live_testnet.BinanceFuturesTestnetBroker", lambda cfg: broker)
    monkeypatch.setattr("forward.live_testnet.DataService", FakeDataService)
    monkeypatch.setattr(
        "forward.live_testnet.fetch_recent_klines_df",
        lambda **kwargs: (
            _bootstrap_df(bootstrap_open),
            {"last_close_time": bootstrap_close.isoformat()},
        ),
    )
    monkeypatch.setattr(
        "forward.live_testnet.fetch_server_time_ms",
        lambda market: (int(pd.Timestamp.now(tz="UTC").timestamp() * 1000), {"source": str(market)}),
    )
    monkeypatch.setattr(
        "forward.live_testnet.build_signals",
        lambda **kwargs: (_build_signals_df(kwargs["df_raw"], signal=1), pd.DataFrame()),
    )
    monkeypatch.setattr(TraderService, "maybe_place_trade_from_signal", _unexpected_entry)
    monkeypatch.setenv("HEARTBEAT_PATH", str(tmp_path / "heartbeat"))

    run_dir, cfg, ft_cfg = _run_cfg(tmp_path)
    rc = asyncio.run(
        run_live_testnet(
            run_dir=run_dir,
            cfg=cfg,
            ft_cfg=ft_cfg,
            risk_limits=None,
            symbol="SOLUSDT",
            timeframe="30m",
            orb_start=time(0, 0),
            orb_end=time(0, 30),
            orb_cutoff=time(23, 59),
            adx_period=14,
            adx_threshold=20.0,
            initial_capital=1000.0,
            position_size=0.1,
            taker_fee_rate=0.0005,
            leverage=1.0,
            delay_bars=1,
            slippage_bps=0.0,
            max_bars=1,
        )
    )

    events = _read_events(run_dir / "events.jsonl")
    recon_events = [e for e in events if str(e.get("type") or "") == "RECON_MISMATCH"]
    end_events = [e for e in events if str(e.get("type") or "") == "LIVE_RUN_END"]

    assert rc == 0
    assert entry_calls == []
    assert broker.position_risk_call_count == 2
    assert len(recon_events) == 1
    assert recon_events[0]["stage"] == "runtime"
    assert recon_events[0]["state"] == "FLAT"
    assert recon_events[0]["exchange"] == "LONG"
    assert end_events[-1]["reason"] == "RECON_MISMATCH"


def test_stop_event_from_margin_check_skips_entry_for_same_bar(tmp_path: Path, monkeypatch) -> None:
    bootstrap_open, bootstrap_close, live_open = _runtime_timestamps()
    broker = FakeRuntimeBroker(position_amts=[0.0, 0.0])
    FakeDataService.bars = [_live_bar(live_open)]

    entry_calls: list[pd.Timestamp] = []

    async def _unexpected_entry(self: TraderService, bar_open_time: pd.Timestamp, row: pd.Series) -> None:
        _ = row
        entry_calls.append(bar_open_time)

    def _trigger_stop(self: TraderService) -> None:
        self.stop_event.set()

    monkeypatch.setattr("forward.live_testnet.BinanceFuturesTestnetBroker", lambda cfg: broker)
    monkeypatch.setattr("forward.live_testnet.DataService", FakeDataService)
    monkeypatch.setattr(
        "forward.live_testnet.fetch_recent_klines_df",
        lambda **kwargs: (
            _bootstrap_df(bootstrap_open),
            {"last_close_time": bootstrap_close.isoformat()},
        ),
    )
    monkeypatch.setattr(
        "forward.live_testnet.fetch_server_time_ms",
        lambda market: (int(pd.Timestamp.now(tz="UTC").timestamp() * 1000), {"source": str(market)}),
    )
    monkeypatch.setattr(
        "forward.live_testnet.build_signals",
        lambda **kwargs: (_build_signals_df(kwargs["df_raw"], signal=1), pd.DataFrame()),
    )
    monkeypatch.setattr(TraderService, "maybe_place_trade_from_signal", _unexpected_entry)
    monkeypatch.setattr(TraderService, "maybe_kill_on_margin_ratio", _trigger_stop)
    monkeypatch.setenv("HEARTBEAT_PATH", str(tmp_path / "heartbeat"))

    run_dir, cfg, ft_cfg = _run_cfg(tmp_path)
    rc = asyncio.run(
        run_live_testnet(
            run_dir=run_dir,
            cfg=cfg,
            ft_cfg=ft_cfg,
            risk_limits=None,
            symbol="SOLUSDT",
            timeframe="30m",
            orb_start=time(0, 0),
            orb_end=time(0, 30),
            orb_cutoff=time(23, 59),
            adx_period=14,
            adx_threshold=20.0,
            initial_capital=1000.0,
            position_size=0.1,
            taker_fee_rate=0.0005,
            leverage=1.0,
            delay_bars=1,
            slippage_bps=0.0,
            max_bars=1,
        )
    )

    events = _read_events(run_dir / "events.jsonl")
    end_events = [e for e in events if str(e.get("type") or "") == "LIVE_RUN_END"]

    assert rc == 0
    assert entry_calls == []
    assert broker.position_risk_call_count == 2
    assert end_events[-1]["reason"] == "STOP"


def test_runtime_reconciliation_reuses_single_position_snapshot_per_bar(tmp_path: Path, monkeypatch) -> None:
    bootstrap_open, bootstrap_close, live_open = _runtime_timestamps()
    broker = FakeRuntimeBroker(position_amts=[0.0, 0.0])
    FakeDataService.bars = [_live_bar(live_open)]

    monkeypatch.setattr("forward.live_testnet.BinanceFuturesTestnetBroker", lambda cfg: broker)
    monkeypatch.setattr("forward.live_testnet.DataService", FakeDataService)
    monkeypatch.setattr(
        "forward.live_testnet.fetch_recent_klines_df",
        lambda **kwargs: (
            _bootstrap_df(bootstrap_open),
            {"last_close_time": bootstrap_close.isoformat()},
        ),
    )
    monkeypatch.setattr(
        "forward.live_testnet.fetch_server_time_ms",
        lambda market: (int(pd.Timestamp.now(tz="UTC").timestamp() * 1000), {"source": str(market)}),
    )
    monkeypatch.setattr(
        "forward.live_testnet.build_signals",
        lambda **kwargs: (_build_signals_df(kwargs["df_raw"], signal=0), pd.DataFrame()),
    )
    monkeypatch.setenv("HEARTBEAT_PATH", str(tmp_path / "heartbeat"))

    run_dir, cfg, ft_cfg = _run_cfg(tmp_path)
    rc = asyncio.run(
        run_live_testnet(
            run_dir=run_dir,
            cfg=cfg,
            ft_cfg=ft_cfg,
            risk_limits=None,
            symbol="SOLUSDT",
            timeframe="30m",
            orb_start=time(0, 0),
            orb_end=time(0, 30),
            orb_cutoff=time(23, 59),
            adx_period=14,
            adx_threshold=20.0,
            initial_capital=1000.0,
            position_size=0.1,
            taker_fee_rate=0.0005,
            leverage=1.0,
            delay_bars=1,
            slippage_bps=0.0,
            max_bars=1,
        )
    )

    assert rc == 0
    assert broker.position_risk_call_count == 2


def test_shutdown_guard_exception_recheck_flat_allows_final_cancel(tmp_path: Path, monkeypatch) -> None:
    bootstrap_open, bootstrap_close, _ = _runtime_timestamps()
    broker = FakeRuntimeBroker(position_amts=[1.0, 0.0])
    FakeDataService.bars = []

    def _raise_flatten(self: TraderService, *, reason: str) -> tuple[bool, str]:
        _ = reason
        raise RuntimeError("shutdown_flatten_failed")

    monkeypatch.setattr("forward.live_testnet.BinanceFuturesTestnetBroker", lambda cfg: broker)
    monkeypatch.setattr("forward.live_testnet.DataService", FakeDataService)
    monkeypatch.setattr(
        "forward.live_testnet.fetch_recent_klines_df",
        lambda **kwargs: (
            _bootstrap_df(bootstrap_open),
            {"last_close_time": bootstrap_close.isoformat()},
        ),
    )
    monkeypatch.setattr(
        "forward.live_testnet.fetch_server_time_ms",
        lambda market: (int(pd.Timestamp.now(tz="UTC").timestamp() * 1000), {"source": str(market)}),
    )
    monkeypatch.setattr(TraderService, "_emergency_flatten", _raise_flatten)
    monkeypatch.setenv("HEARTBEAT_PATH", str(tmp_path / "heartbeat"))

    run_dir, cfg, ft_cfg = _run_cfg(tmp_path)
    _seed_open_position_state(run_dir, side="LONG", qty=1.0)

    rc = asyncio.run(
        run_live_testnet(
            run_dir=run_dir,
            cfg=cfg,
            ft_cfg=ft_cfg,
            risk_limits=None,
            symbol="SOLUSDT",
            timeframe="30m",
            orb_start=time(0, 0),
            orb_end=time(0, 30),
            orb_cutoff=time(23, 59),
            adx_period=14,
            adx_threshold=20.0,
            initial_capital=1000.0,
            position_size=0.1,
            taker_fee_rate=0.0005,
            leverage=1.0,
            delay_bars=1,
            slippage_bps=0.0,
            max_bars=1,
        )
    )

    events = _read_events(run_dir / "events.jsonl")
    event_types = [str(event.get("type") or "") for event in events]
    with SQLiteStateStore(db_path=run_dir / "state.db") as store:
        loaded = store.load_state()

    assert rc == 0
    assert broker.cancel_all_called is True
    assert loaded.open_position is None
    assert "SHUTDOWN_GUARD_FLATTEN_ERROR" in event_types
    assert "CANCEL_ALL_OPEN_ORDERS" in event_types
    assert "CANCEL_ALL_OPEN_ORDERS_SKIPPED_RUNTIME_GUARD" not in event_types


def test_shutdown_guard_exception_recheck_open_keeps_cancel_skipped(tmp_path: Path, monkeypatch) -> None:
    bootstrap_open, bootstrap_close, _ = _runtime_timestamps()
    broker = FakeRuntimeBroker(position_amts=[1.0, 1.0])
    FakeDataService.bars = []

    def _raise_flatten(self: TraderService, *, reason: str) -> tuple[bool, str]:
        _ = reason
        raise RuntimeError("shutdown_flatten_failed")

    monkeypatch.setattr("forward.live_testnet.BinanceFuturesTestnetBroker", lambda cfg: broker)
    monkeypatch.setattr("forward.live_testnet.DataService", FakeDataService)
    monkeypatch.setattr(
        "forward.live_testnet.fetch_recent_klines_df",
        lambda **kwargs: (
            _bootstrap_df(bootstrap_open),
            {"last_close_time": bootstrap_close.isoformat()},
        ),
    )
    monkeypatch.setattr(
        "forward.live_testnet.fetch_server_time_ms",
        lambda market: (int(pd.Timestamp.now(tz="UTC").timestamp() * 1000), {"source": str(market)}),
    )
    monkeypatch.setattr(TraderService, "_emergency_flatten", _raise_flatten)
    monkeypatch.setenv("HEARTBEAT_PATH", str(tmp_path / "heartbeat"))

    run_dir, cfg, ft_cfg = _run_cfg(tmp_path)
    _seed_open_position_state(run_dir, side="LONG", qty=1.0)

    rc = asyncio.run(
        run_live_testnet(
            run_dir=run_dir,
            cfg=cfg,
            ft_cfg=ft_cfg,
            risk_limits=None,
            symbol="SOLUSDT",
            timeframe="30m",
            orb_start=time(0, 0),
            orb_end=time(0, 30),
            orb_cutoff=time(23, 59),
            adx_period=14,
            adx_threshold=20.0,
            initial_capital=1000.0,
            position_size=0.1,
            taker_fee_rate=0.0005,
            leverage=1.0,
            delay_bars=1,
            slippage_bps=0.0,
            max_bars=1,
        )
    )

    events = _read_events(run_dir / "events.jsonl")
    event_types = [str(event.get("type") or "") for event in events]
    with SQLiteStateStore(db_path=run_dir / "state.db") as store:
        loaded = store.load_state()

    assert rc == 0
    assert broker.cancel_all_called is False
    assert loaded.open_position is not None
    assert "SHUTDOWN_GUARD_FLATTEN_ERROR" in event_types
    assert "CANCEL_ALL_OPEN_ORDERS_SKIPPED_RUNTIME_GUARD" in event_types


def test_shutdown_guard_exception_recheck_failure_keeps_cancel_skipped(tmp_path: Path, monkeypatch) -> None:
    bootstrap_open, bootstrap_close, _ = _runtime_timestamps()
    broker = FakeRuntimeBroker(position_amts=[1.0], position_risk_raise_on_call=1)
    FakeDataService.bars = []

    def _raise_flatten(self: TraderService, *, reason: str) -> tuple[bool, str]:
        _ = reason
        raise RuntimeError("shutdown_flatten_failed")

    monkeypatch.setattr("forward.live_testnet.BinanceFuturesTestnetBroker", lambda cfg: broker)
    monkeypatch.setattr("forward.live_testnet.DataService", FakeDataService)
    monkeypatch.setattr(
        "forward.live_testnet.fetch_recent_klines_df",
        lambda **kwargs: (
            _bootstrap_df(bootstrap_open),
            {"last_close_time": bootstrap_close.isoformat()},
        ),
    )
    monkeypatch.setattr(
        "forward.live_testnet.fetch_server_time_ms",
        lambda market: (int(pd.Timestamp.now(tz="UTC").timestamp() * 1000), {"source": str(market)}),
    )
    monkeypatch.setattr(TraderService, "_emergency_flatten", _raise_flatten)
    monkeypatch.setenv("HEARTBEAT_PATH", str(tmp_path / "heartbeat"))

    run_dir, cfg, ft_cfg = _run_cfg(tmp_path)
    _seed_open_position_state(run_dir, side="LONG", qty=1.0)

    rc = asyncio.run(
        run_live_testnet(
            run_dir=run_dir,
            cfg=cfg,
            ft_cfg=ft_cfg,
            risk_limits=None,
            symbol="SOLUSDT",
            timeframe="30m",
            orb_start=time(0, 0),
            orb_end=time(0, 30),
            orb_cutoff=time(23, 59),
            adx_period=14,
            adx_threshold=20.0,
            initial_capital=1000.0,
            position_size=0.1,
            taker_fee_rate=0.0005,
            leverage=1.0,
            delay_bars=1,
            slippage_bps=0.0,
            max_bars=1,
        )
    )

    events = _read_events(run_dir / "events.jsonl")
    event_types = [str(event.get("type") or "") for event in events]
    with SQLiteStateStore(db_path=run_dir / "state.db") as store:
        loaded = store.load_state()

    assert rc == 0
    assert broker.cancel_all_called is False
    assert loaded.open_position is not None
    assert "SHUTDOWN_GUARD_FLATTEN_ERROR" in event_types
    assert "CANCEL_ALL_OPEN_ORDERS_SKIPPED_RUNTIME_GUARD" in event_types

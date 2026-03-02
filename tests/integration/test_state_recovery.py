from __future__ import annotations

import asyncio
import os
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

import pandas as pd
import pytest

from forward.binance_live import LiveBar
from forward.data_service import DataService
from forward.state_store_sqlite import SQLiteStateStore
from tests.integration.mocks import FakeBinanceClient, build_trader_service


def _entry_row() -> pd.Series:
    return pd.Series(
        {
            "signal": -1,
            "signal_type": "downtrend_breakdown",
            "close": 100.0,
            "orb_high": 200.0,
        }
    )



def _entry_row_missing_orb() -> pd.Series:
    return pd.Series(
        {
            "signal": -1,
            "signal_type": "downtrend_breakdown",
            "close": 100.0,
            "orb_high": None,
        }
    )

def _count_trade_log_events(db_path: Path, event_type: str) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM trade_log WHERE event_type = ?",
            (str(event_type),),
        ).fetchone()
        return int(row[0] if row is not None else 0)
    finally:
        conn.close()


def test_order_partial_fill_then_full_fill_records_once(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteStateStore(db_path=db_path) as store:
        state = store.load_state()
        broker = FakeBinanceClient(
            reject_entry=False,
            simulate_partial_fill_poll=True,
            fill_price=100.0,
        )
        trader = build_trader_service(
            broker=broker,
            store=store,
            state=state,
            work_dir=tmp_path,
            leverage=1.0,
            position_size=0.1,
            initial_capital=1000.0,
            max_order_rejects_per_day=10,
        )

        bar_t0 = pd.Timestamp("2024-01-01 00:30:00", tz="UTC")
        asyncio.run(trader.maybe_place_trade_from_signal(bar_t0, _entry_row()))

        assert broker.last_entry_order_id is not None
        first_poll = broker.poll_entry_order(int(broker.last_entry_order_id))
        second_poll = broker.poll_entry_order(int(broker.last_entry_order_id))
        assert first_poll["status"] == "PARTIALLY_FILLED"
        assert second_poll["status"] == "FILLED"

        # A second entry attempt while already open must not create duplicates.
        asyncio.run(
            trader.maybe_place_trade_from_signal(
                bar_t0 + pd.Timedelta(minutes=30),
                _entry_row(),
            )
        )

        loaded = store.load_state()
        assert loaded.open_position is not None
        assert loaded.open_position.qty == pytest.approx(1.0, abs=1e-12)
        assert loaded.open_position.entry_price == pytest.approx(100.0, abs=1e-12)

    assert _count_trade_log_events(db_path, "ENTRY") == 1


def test_order_rejection_increments_reject_counter_and_logs(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteStateStore(db_path=db_path) as store:
        state = store.load_state()
        broker = FakeBinanceClient(
            reject_entry=True,
            simulate_partial_fill_poll=False,
            fill_price=100.0,
        )
        trader = build_trader_service(
            broker=broker,
            store=store,
            state=state,
            work_dir=tmp_path,
            leverage=1.0,
            position_size=0.1,
            initial_capital=1000.0,
            max_order_rejects_per_day=10,
        )

        bar_t0 = pd.Timestamp("2024-01-01 00:30:00", tz="UTC")
        asyncio.run(trader.maybe_place_trade_from_signal(bar_t0, _entry_row()))

        loaded = store.load_state()
        assert loaded.order_rejects_today == 1
        assert loaded.open_position is None

    assert _count_trade_log_events(db_path, "REJECT") == 1



def test_tp_raise_but_land_recovers_protection_and_persists_open_position(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteStateStore(db_path=db_path) as store:
        state = store.load_state()
        broker = FakeBinanceClient(tp_raise_but_land=True, fill_price=100.0)
        trader = build_trader_service(
            broker=broker,
            store=store,
            state=state,
            work_dir=tmp_path,
            leverage=1.0,
            position_size=0.1,
            initial_capital=1000.0,
            max_order_rejects_per_day=10,
        )

        bar_t0 = pd.Timestamp("2024-01-01 00:30:00", tz="UTC")
        asyncio.run(trader.maybe_place_trade_from_signal(bar_t0, _entry_row()))

        loaded = store.load_state()
        assert loaded.open_position is not None
        assert loaded.open_position.tp_order_id is not None
        assert loaded.open_position.sl_order_id is not None
        assert trader.stop_event.is_set() is False


def test_missing_protection_flattens_and_continues(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteStateStore(db_path=db_path) as store:
        state = store.load_state()
        broker = FakeBinanceClient(reject_tp=True, fill_price=100.0)
        trader = build_trader_service(
            broker=broker,
            store=store,
            state=state,
            work_dir=tmp_path,
            leverage=1.0,
            position_size=0.1,
            initial_capital=1000.0,
            max_order_rejects_per_day=10,
        )

        bar_t0 = pd.Timestamp("2024-01-01 00:30:00", tz="UTC")
        asyncio.run(trader.maybe_place_trade_from_signal(bar_t0, _entry_row()))

        loaded = store.load_state()
        assert loaded.open_position is None
        assert trader.stop_event.is_set() is False
        # known_qty/known_side path should skip pre-fetch and only verify once post-close
        assert broker.position_risk_call_count == 1


def test_unknown_protection_halts_even_if_flatten_succeeds(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteStateStore(db_path=db_path) as store:
        state = store.load_state()
        broker = FakeBinanceClient(
            reject_tp=True,
            fail_get_algo_open_orders=True,
            fill_price=100.0,
        )
        trader = build_trader_service(
            broker=broker,
            store=store,
            state=state,
            work_dir=tmp_path,
            leverage=1.0,
            position_size=0.1,
            initial_capital=1000.0,
            max_order_rejects_per_day=10,
        )

        bar_t0 = pd.Timestamp("2024-01-01 00:30:00", tz="UTC")
        asyncio.run(trader.maybe_place_trade_from_signal(bar_t0, _entry_row()))

        loaded = store.load_state()
        assert loaded.open_position is None
        assert trader.stop_event.is_set() is True
        assert trader.skip_cancel_open_orders_on_exit_runtime is False


def test_bracket_skipped_flattens_first_and_kills_on_flatten_failure(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteStateStore(db_path=db_path) as store:
        state = store.load_state()
        broker = FakeBinanceClient(reject_flatten=True, fill_price=100.0)
        trader = build_trader_service(
            broker=broker,
            store=store,
            state=state,
            work_dir=tmp_path,
            leverage=1.0,
            position_size=0.1,
            initial_capital=1000.0,
            max_order_rejects_per_day=10,
        )

        bar_t0 = pd.Timestamp("2024-01-01 00:30:00", tz="UTC")
        asyncio.run(trader.maybe_place_trade_from_signal(bar_t0, _entry_row_missing_orb()))

        loaded = store.load_state()
        assert loaded.open_position is None
        assert trader.stop_event.is_set() is True
        assert trader.skip_cancel_open_orders_on_exit_runtime is True



def test_cancel_all_tracking_is_available_for_integration_cleanup(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteStateStore(db_path=db_path) as store:
        state = store.load_state()
        broker = FakeBinanceClient(fill_price=100.0)
        trader = build_trader_service(
            broker=broker,
            store=store,
            state=state,
            work_dir=tmp_path,
            leverage=1.0,
            position_size=0.1,
            initial_capital=1000.0,
            max_order_rejects_per_day=10,
        )

        bar_t0 = pd.Timestamp("2024-01-01 00:30:00", tz="UTC")
        asyncio.run(trader.maybe_place_trade_from_signal(bar_t0, _entry_row()))

        assert broker.cancel_all_called is False
        broker.cancel_all_open_orders(symbol="SOLUSDT")
        assert broker.cancel_all_called is True
        assert broker.cancel_all_call_count == 1
        assert broker.last_cancel_all_symbol == "SOLUSDT"



def _tp_candidate_row(
    algo_id: int,
    trigger_price: float,
    ts_ms: int | None,
    *,
    side: str = "BUY",
) -> dict[str, object]:
    row: dict[str, object] = {
        "algoId": int(algo_id),
        "symbol": "SOLUSDT",
        "type": "TAKE_PROFIT_MARKET",
        "side": str(side).upper(),
        "status": "NEW",
        "triggerPrice": f"{float(trigger_price)}",
    }
    if ts_ms is not None:
        row["time"] = int(ts_ms)
    return row


def test_fallback_tp_candidates_resolve_by_30s_retry_window(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteStateStore(db_path=db_path) as store:
        state = store.load_state()
        broker = FakeBinanceClient(tp_raise_but_land=True, fill_price=100.0)
        baseline_ms = int(broker._now_ms + 1000)
        broker.open_algo_orders_override = [
            _tp_candidate_row(algo_id=2201, trigger_price=98.05, ts_ms=baseline_ms - 120_000),
            _tp_candidate_row(algo_id=2202, trigger_price=98.08, ts_ms=baseline_ms - 10_000),
        ]

        trader = build_trader_service(
            broker=broker,
            store=store,
            state=state,
            work_dir=tmp_path,
            leverage=1.0,
            position_size=0.1,
            initial_capital=1000.0,
            max_order_rejects_per_day=10,
        )

        bar_t0 = pd.Timestamp("2024-01-01 00:30:00", tz="UTC")
        asyncio.run(trader.maybe_place_trade_from_signal(bar_t0, _entry_row()))

        loaded = store.load_state()
        assert loaded.open_position is not None
        assert loaded.open_position.tp_order_id == 2202
        assert trader.stop_event.is_set() is False


def test_fallback_tp_candidates_resolve_by_most_recent_timestamp(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteStateStore(db_path=db_path) as store:
        state = store.load_state()
        broker = FakeBinanceClient(tp_raise_but_land=True, fill_price=100.0)
        baseline_ms = int(broker._now_ms + 1000)
        broker.open_algo_orders_override = [
            _tp_candidate_row(algo_id=2301, trigger_price=98.01, ts_ms=baseline_ms - 25_000),
            _tp_candidate_row(algo_id=2302, trigger_price=98.07, ts_ms=baseline_ms - 5_000),
        ]

        trader = build_trader_service(
            broker=broker,
            store=store,
            state=state,
            work_dir=tmp_path,
            leverage=1.0,
            position_size=0.1,
            initial_capital=1000.0,
            max_order_rejects_per_day=10,
        )

        bar_t0 = pd.Timestamp("2024-01-01 00:30:00", tz="UTC")
        asyncio.run(trader.maybe_place_trade_from_signal(bar_t0, _entry_row()))

        loaded = store.load_state()
        assert loaded.open_position is not None
        assert loaded.open_position.tp_order_id == 2302
        assert trader.stop_event.is_set() is False


def test_fallback_tp_candidates_timestamp_tie_resolves_by_highest_algo_id(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteStateStore(db_path=db_path) as store:
        state = store.load_state()
        broker = FakeBinanceClient(tp_raise_but_land=True, fill_price=100.0)
        baseline_ms = int(broker._now_ms + 1000)
        tie_ts = baseline_ms - 8_000
        broker.open_algo_orders_override = [
            _tp_candidate_row(algo_id=2401, trigger_price=98.00, ts_ms=tie_ts),
            _tp_candidate_row(algo_id=2402, trigger_price=98.09, ts_ms=tie_ts),
        ]

        trader = build_trader_service(
            broker=broker,
            store=store,
            state=state,
            work_dir=tmp_path,
            leverage=1.0,
            position_size=0.1,
            initial_capital=1000.0,
            max_order_rejects_per_day=10,
        )

        bar_t0 = pd.Timestamp("2024-01-01 00:30:00", tz="UTC")
        asyncio.run(trader.maybe_place_trade_from_signal(bar_t0, _entry_row()))

        loaded = store.load_state()
        assert loaded.open_position is not None
        assert loaded.open_position.tp_order_id == 2402
        assert trader.stop_event.is_set() is False


def test_fallback_tp_non_comparable_timestamps_triggers_unknown_branch(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteStateStore(db_path=db_path) as store:
        state = store.load_state()
        broker = FakeBinanceClient(tp_raise_but_land=True, fill_price=100.0)
        broker.open_algo_orders_override = [
            _tp_candidate_row(algo_id=2501, trigger_price=98.05, ts_ms=None),
        ]

        trader = build_trader_service(
            broker=broker,
            store=store,
            state=state,
            work_dir=tmp_path,
            leverage=1.0,
            position_size=0.1,
            initial_capital=1000.0,
            max_order_rejects_per_day=10,
        )

        bar_t0 = pd.Timestamp("2024-01-01 00:30:00", tz="UTC")
        asyncio.run(trader.maybe_place_trade_from_signal(bar_t0, _entry_row()))

        loaded = store.load_state()
        assert loaded.open_position is None
        assert trader.stop_event.is_set() is True
        assert trader.skip_cancel_open_orders_on_exit_runtime is False


def test_missing_protection_with_flatten_failure_halts_and_sets_runtime_skip(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteStateStore(db_path=db_path) as store:
        state = store.load_state()
        broker = FakeBinanceClient(
            reject_tp=True,
            reject_flatten=True,
            open_algo_orders_override=[],
            fill_price=100.0,
        )

        trader = build_trader_service(
            broker=broker,
            store=store,
            state=state,
            work_dir=tmp_path,
            leverage=1.0,
            position_size=0.1,
            initial_capital=1000.0,
            max_order_rejects_per_day=10,
        )

        bar_t0 = pd.Timestamp("2024-01-01 00:30:00", tz="UTC")
        asyncio.run(trader.maybe_place_trade_from_signal(bar_t0, _entry_row()))

        loaded = store.load_state()
        assert loaded.open_position is None
        assert trader.stop_event.is_set() is True
        assert trader.skip_cancel_open_orders_on_exit_runtime is True


def test_protection_baseline_uses_exchange_then_local_fallback(tmp_path: Path) -> None:
    db_path_a = tmp_path / "state_a.db"
    with SQLiteStateStore(db_path=db_path_a) as store_a:
        state_a = store_a.load_state()
        broker_a = FakeBinanceClient(fill_price=100.0)
        trader_a = build_trader_service(
            broker=broker_a,
            store=store_a,
            state=state_a,
            work_dir=tmp_path / "a",
            leverage=1.0,
            position_size=0.1,
            initial_capital=1000.0,
            max_order_rejects_per_day=10,
        )
        epoch_a, source_a = trader_a._protection_baseline_epoch_seconds()
        assert source_a == "exchange"
        assert isinstance(epoch_a, float)
        assert epoch_a > 1_700_000_000.0

    db_path_b = tmp_path / "state_b.db"
    with SQLiteStateStore(db_path=db_path_b) as store_b:
        state_b = store_b.load_state()
        broker_b = FakeBinanceClient(fill_price=100.0, fail_server_time=True)
        trader_b = build_trader_service(
            broker=broker_b,
            store=store_b,
            state=state_b,
            work_dir=tmp_path / "b",
            leverage=1.0,
            position_size=0.1,
            initial_capital=1000.0,
            max_order_rejects_per_day=10,
        )
        t0 = time.time()
        epoch_b, source_b = trader_b._protection_baseline_epoch_seconds()
        t1 = time.time()
        assert source_b == "local_fallback"
        assert isinstance(epoch_b, float)
        assert max(0.0, t0 - 5.0) <= epoch_b <= (t1 + 5.0)

class _FakeReconnectSource:
    def __init__(self, bar_1: LiveBar, bar_2: LiveBar):
        self._bar_1 = bar_1
        self._bar_2 = bar_2
        self.connect_count = 0
        self.last_message_at = None
        self.last_connect_at = None

    async def stream_closed(self, stop_event: asyncio.Event) -> AsyncIterator[LiveBar]:
        backoff = 1.0
        while not stop_event.is_set():
            self.connect_count += 1
            self.last_connect_at = datetime.now(timezone.utc)
            if self.connect_count == 1:
                self.last_message_at = datetime.now(timezone.utc)
                yield self._bar_1
                try:
                    raise ConnectionError("simulated_disconnect")
                except ConnectionError:
                    time.sleep(backoff)
                    backoff = min(backoff * 2.0, 4.0)
                    continue

            self.last_message_at = datetime.now(timezone.utc)
            yield self._bar_1  # duplicate after reconnect
            self.last_message_at = datetime.now(timezone.utc)
            yield self._bar_2
            return


def test_stream_reconnect_does_not_duplicate_bar_processing(monkeypatch) -> None:
    bar_1 = LiveBar(
        symbol="SOLUSDT",
        interval="30m",
        open_time=pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
        close_time=pd.Timestamp("2024-01-01 00:29:59", tz="UTC"),
        open=100.0,
        high=100.0,
        low=100.0,
        close=100.0,
        volume=1.0,
    )
    bar_2 = LiveBar(
        symbol="SOLUSDT",
        interval="30m",
        open_time=pd.Timestamp("2024-01-01 00:30:00", tz="UTC"),
        close_time=pd.Timestamp("2024-01-01 00:59:59", tz="UTC"),
        open=101.0,
        high=101.0,
        low=101.0,
        close=101.0,
        volume=1.0,
    )

    recorded_sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda seconds: recorded_sleeps.append(float(seconds)))

    ds = DataService(
        symbol="SOLUSDT",
        interval="30m",
        market="futures",
        stale_allowed_seconds=3600.0,
        max_backoff_seconds=60,
        stale_check_interval_seconds=5,
        heartbeat_seconds=120,
        emit_event=lambda rows: None,
    )
    ds._src = _FakeReconnectSource(bar_1=bar_1, bar_2=bar_2)

    async def _run_loop() -> list[pd.Timestamp]:
        processed: list[pd.Timestamp] = []
        stop_event = asyncio.Event()
        last_bar_open: pd.Timestamp | None = None
        async for bar in ds.stream_closed(stop_event):
            if last_bar_open is not None and bar.open_time <= last_bar_open:
                continue
            processed.append(bar.open_time)
            last_bar_open = bar.open_time
            ds.last_closed_bar_at = datetime.now(timezone.utc)
            if len(processed) >= 2:
                stop_event.set()
        return processed

    processed = asyncio.run(_run_loop())
    assert processed == [bar_1.open_time, bar_2.open_time]
    assert recorded_sleeps
    assert all(v > 0 for v in recorded_sleeps)


@pytest.mark.skipif(os.name != "posix", reason="SIGKILL recovery test requires POSIX")
def test_sigkill_restart_does_not_reenter_open_position(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    ready_file = tmp_path / "ready.flag"

    run1 = [
        sys.executable,
        "-m",
        "tests.integration._recovery_harness",
        "--db-path",
        str(db_path),
        "--mode",
        "run1",
        "--ready-file",
        str(ready_file),
    ]
    proc = subprocess.Popen(run1, cwd=str(Path(__file__).resolve().parents[2]))
    try:
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if ready_file.exists():
                break
            if proc.poll() is not None:
                raise AssertionError("run1 exited before writing ready file")
            time.sleep(0.05)
        assert ready_file.exists(), "run1 did not reach committed ready state in time"

        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5.0)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)

    run2 = [
        sys.executable,
        "-m",
        "tests.integration._recovery_harness",
        "--db-path",
        str(db_path),
        "--mode",
        "run2",
        "--ready-file",
        str(ready_file),
    ]
    run2_result = subprocess.run(
        run2,
        cwd=str(Path(__file__).resolve().parents[2]),
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    assert run2_result.returncode == 0, run2_result.stdout + "\n" + run2_result.stderr

    conn = sqlite3.connect(str(db_path))
    try:
        open_row = conn.execute(
            """
            SELECT COUNT(*) FROM open_positions
            WHERE id = 1 AND symbol IS NOT NULL AND symbol <> ''
            """
        ).fetchone()
        entry_row = conn.execute(
            "SELECT COUNT(*) FROM trade_log WHERE event_type = 'ENTRY'"
        ).fetchone()
        assert int(open_row[0] if open_row is not None else 0) == 1
        assert int(entry_row[0] if entry_row is not None else 0) == 1
    finally:
        conn.close()

    with SQLiteStateStore(db_path=db_path) as store:
        loaded = store.load_state()
        assert loaded is not None
        assert loaded.open_position is not None
        assert loaded.bars_processed >= 0
        assert loaded.order_rejects_today >= 0


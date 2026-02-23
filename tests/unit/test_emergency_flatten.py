from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

from forward.state_store_sqlite import OpenPositionState, RunnerState, SQLiteStateStore
from scripts import emergency_flatten


class FakeBinanceClient:
    positions_sequence: List[List[Dict[str, Any]]] = []
    close_calls: int = 0

    def __init__(self, *, base_url: str, api_key: str, api_secret: str, recv_window_ms: int) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self._api_secret = api_secret
        self._recv_window_ms = recv_window_ms
        self._index = 0

    def get_positions(self) -> List[Dict[str, Any]]:
        seq = type(self).positions_sequence
        if not seq:
            return []
        i = min(self._index, len(seq) - 1)
        self._index += 1
        return seq[i]

    def close_position(self, row: Dict[str, Any]) -> None:
        _ = row
        type(self).close_calls += 1


def _seed_state_with_open_position(db_path: Path) -> None:
    state = RunnerState(
        open_position=OpenPositionState(
            symbol="BTCUSDT",
            side="LONG",
            qty=0.01,
            entry_price=50000.0,
            entry_time_utc="2024-01-01T00:00:00+00:00",
            entry_order_id=123,
        )
    )
    with SQLiteStateStore(db_path=db_path) as store:
        store.save_state(state)


def _load_open_position(db_path: Path) -> OpenPositionState | None:
    with SQLiteStateStore(db_path=db_path) as store:
        return store.load_state().open_position


def _patch_runtime(monkeypatch: pytest.MonkeyPatch, db_path: Path) -> None:
    monkeypatch.setattr(emergency_flatten, "BinanceClient", FakeBinanceClient)
    monkeypatch.setattr(emergency_flatten, "_get_credentials", lambda use_testnet: ("k", "s"))
    monkeypatch.setattr(emergency_flatten, "_select_base_url", lambda use_testnet: "https://example.test")
    monkeypatch.setattr(emergency_flatten, "_get_recv_window_ms", lambda: 5000)
    monkeypatch.setenv("STATE_DB_PATH", str(db_path))


def test_main_clears_sqlite_open_position_when_exchange_is_flat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "state.db"
    _seed_state_with_open_position(db_path)
    _patch_runtime(monkeypatch, db_path)

    FakeBinanceClient.positions_sequence = [
        [{"symbol": "BTCUSDT", "positionAmt": "0.01"}],
        [],
        [],
    ]
    FakeBinanceClient.close_calls = 0
    monkeypatch.setattr(sys, "argv", ["emergency_flatten.py", "--testnet"])

    rc = emergency_flatten.main()

    assert rc == 0
    assert FakeBinanceClient.close_calls == 1
    assert _load_open_position(db_path) is None


def test_main_exits_nonzero_and_does_not_clear_state_when_exchange_not_flat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "state.db"
    _seed_state_with_open_position(db_path)
    _patch_runtime(monkeypatch, db_path)

    FakeBinanceClient.positions_sequence = [
        [{"symbol": "BTCUSDT", "positionAmt": "0.01"}],
        [{"symbol": "BTCUSDT", "positionAmt": "0.01"}],
        [{"symbol": "BTCUSDT", "positionAmt": "0.01"}],
    ]
    FakeBinanceClient.close_calls = 0
    monkeypatch.setattr(sys, "argv", ["emergency_flatten.py", "--testnet"])

    with pytest.raises(SystemExit) as excinfo:
        raise SystemExit(emergency_flatten.main())

    assert excinfo.value.code == 1
    assert FakeBinanceClient.close_calls == 1
    assert _load_open_position(db_path) is not None

from __future__ import annotations

import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _write_state_json_snapshot_atomic(path: Path, state: "RunnerState") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / "state.json.tmp"
    tmp_path.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, path)


@dataclass
class OpenPositionState:
    symbol: str
    side: str
    qty: float
    entry_price: float
    entry_time_utc: str
    entry_order_id: Optional[int]
    tp_order_id: Optional[int] = None
    sl_order_id: Optional[int] = None
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None
    opened_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        # Preserve the historical JSON payload shape for forward-compatibility.
        return {
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "entry_price": self.entry_price,
            "entry_time_utc": self.entry_time_utc,
            "entry_order_id": self.entry_order_id,
            "tp_order_id": self.tp_order_id,
            "sl_order_id": self.sl_order_id,
            "tp_price": self.tp_price,
            "sl_price": self.sl_price,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "OpenPositionState":
        return OpenPositionState(
            symbol=str(data.get("symbol", "")),
            side=str(data.get("side", "")),
            qty=_coerce_float(data.get("qty"), 0.0),
            entry_price=_coerce_float(data.get("entry_price"), 0.0),
            entry_time_utc=str(data.get("entry_time_utc", "")),
            entry_order_id=_coerce_int(data.get("entry_order_id")),
            tp_order_id=_coerce_int(data.get("tp_order_id")),
            sl_order_id=_coerce_int(data.get("sl_order_id")),
            tp_price=_coerce_optional_float(data.get("tp_price")),
            sl_price=_coerce_optional_float(data.get("sl_price")),
            opened_at=data.get("opened_at"),
        )


@dataclass
class RunnerState:
    last_bar_open_time_utc: Optional[str] = None
    bars_processed: int = 0
    current_day_utc: Optional[str] = None
    order_rejects_today: int = 0
    daily_loss_halted: bool = False
    drawdown_halted: bool = False
    open_position: Optional[OpenPositionState] = None

    def to_dict(self) -> Dict[str, Any]:
        # Keep the legacy JSON schema (omit daily/drawdown halts in snapshots for now).
        return {
            "last_bar_open_time_utc": self.last_bar_open_time_utc,
            "bars_processed": int(self.bars_processed or 0),
            "current_day_utc": self.current_day_utc,
            "order_rejects_today": int(self.order_rejects_today or 0),
            "open_position": self.open_position.to_dict() if self.open_position is not None else None,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "RunnerState":
        op = data.get("open_position")
        open_position = OpenPositionState.from_dict(op) if isinstance(op, dict) else None
        return RunnerState(
            last_bar_open_time_utc=data.get("last_bar_open_time_utc"),
            bars_processed=int(data.get("bars_processed", 0) or 0),
            current_day_utc=data.get("current_day_utc"),
            order_rejects_today=int(data.get("order_rejects_today", 0) or 0),
            daily_loss_halted=bool(data.get("daily_loss_halted", False)),
            drawdown_halted=bool(data.get("drawdown_halted", False)),
            open_position=open_position,
        )


class SQLiteStateStore:
    def __init__(self, db_path: Path, events_path: Optional[Path] = None):
        self.db_path = Path(db_path)
        self.events_path = Path(events_path) if events_path is not None else None
        self.conn: Optional[sqlite3.Connection] = None

    def _require_conn(self) -> sqlite3.Connection:
        if self.conn is None:
            raise RuntimeError("SQLiteStateStore is not open")
        return self.conn

    def _emit_event(self, event_type: str, **fields: Any) -> None:
        if self.events_path is None:
            return
        event = {"ts": _utcnow_iso(), "type": event_type, **fields}
        try:
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass

    def open(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        try:
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA synchronous=NORMAL;")
            self.conn.execute("PRAGMA busy_timeout=5000;")
            self.init_schema()
            self.integrity_check_or_raise()
            self._emit_event("DB_INIT_OK", db_path=str(self.db_path))
        except Exception:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None
            raise

    def close(self) -> None:
        if self.conn is None:
            return
        conn = self.conn
        self.conn = None
        try:
            try:
                if getattr(conn, "in_transaction", False):
                    conn.rollback()
            except Exception:
                pass
        finally:
            conn.close()

    def __enter__(self) -> "SQLiteStateStore":
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def init_schema(self) -> None:
        conn = self._require_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runner_state (
              id INTEGER PRIMARY KEY,
              last_bar_open_time_utc TEXT,
              bars_processed INTEGER,
              current_day_utc TEXT,
              order_rejects_today INTEGER,
              daily_loss_halted INTEGER,
              drawdown_halted INTEGER,
              updated_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS open_positions (
              id INTEGER PRIMARY KEY,
              symbol TEXT,
              side TEXT,
              qty REAL,
              entry_price REAL,
              entry_time_utc TEXT,
              entry_order_id INTEGER,
              tp_order_id INTEGER,
              sl_order_id INTEGER,
              tp_price REAL,
              sl_price REAL,
              opened_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_log (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              event_type TEXT,
              symbol TEXT,
              side TEXT,
              qty REAL,
              price REAL,
              realized_pnl REAL,
              fee REAL,
              funding_applied REAL,
              reason TEXT,
              bar_time_utc TEXT,
              recorded_at TEXT
            )
            """
        )

    def integrity_check_or_raise(self) -> None:
        conn = self._require_conn()
        try:
            rows = conn.execute("PRAGMA integrity_check;").fetchall()
            passed = len(rows) == 1 and str(rows[0][0]).strip().lower() == "ok"
            if not passed:
                details = [str(r[0]) for r in rows]
                self._emit_event("DB_INTEGRITY_FAIL", db_path=str(self.db_path), result=details)
                raise RuntimeError(f"SQLite integrity check failed for {self.db_path}: {details}")
        except RuntimeError:
            raise
        except Exception as e:
            self._emit_event("DB_INTEGRITY_FAIL", db_path=str(self.db_path), error=str(e))
            raise RuntimeError(f"SQLite integrity check failed for {self.db_path}: {e}") from e

    def load_state(self) -> RunnerState:
        conn = self._require_conn()

        runner_row = conn.execute(
            """
            SELECT last_bar_open_time_utc, bars_processed, current_day_utc,
                   order_rejects_today, daily_loss_halted, drawdown_halted
            FROM runner_state
            WHERE id = 1
            """
        ).fetchone()

        if runner_row is None:
            state = RunnerState()
        else:
            state = RunnerState(
                last_bar_open_time_utc=runner_row["last_bar_open_time_utc"],
                bars_processed=int(runner_row["bars_processed"] or 0),
                current_day_utc=runner_row["current_day_utc"],
                order_rejects_today=int(runner_row["order_rejects_today"] or 0),
                daily_loss_halted=bool(int(runner_row["daily_loss_halted"] or 0)),
                drawdown_halted=bool(int(runner_row["drawdown_halted"] or 0)),
                open_position=None,
            )

        pos_row = conn.execute(
            """
            SELECT symbol, side, qty, entry_price, entry_time_utc, entry_order_id,
                   tp_order_id, sl_order_id, tp_price, sl_price, opened_at
            FROM open_positions
            WHERE id = 1
            """
        ).fetchone()

        if pos_row is not None:
            state.open_position = OpenPositionState(
                symbol=str(pos_row["symbol"] or ""),
                side=str(pos_row["side"] or ""),
                qty=_coerce_float(pos_row["qty"], 0.0),
                entry_price=_coerce_float(pos_row["entry_price"], 0.0),
                entry_time_utc=str(pos_row["entry_time_utc"] or ""),
                entry_order_id=_coerce_int(pos_row["entry_order_id"]),
                tp_order_id=_coerce_int(pos_row["tp_order_id"]),
                sl_order_id=_coerce_int(pos_row["sl_order_id"]),
                tp_price=_coerce_optional_float(pos_row["tp_price"]),
                sl_price=_coerce_optional_float(pos_row["sl_price"]),
                opened_at=pos_row["opened_at"],
            )
        else:
            state.open_position = None

        return state

    def save_state(self, state: RunnerState) -> None:
        conn = self._require_conn()
        now = _utcnow_iso()
        try:
            conn.execute("BEGIN")
            conn.execute(
                """
                INSERT INTO runner_state (
                  id, last_bar_open_time_utc, bars_processed, current_day_utc,
                  order_rejects_today, daily_loss_halted, drawdown_halted, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  last_bar_open_time_utc=excluded.last_bar_open_time_utc,
                  bars_processed=excluded.bars_processed,
                  current_day_utc=excluded.current_day_utc,
                  order_rejects_today=excluded.order_rejects_today,
                  daily_loss_halted=excluded.daily_loss_halted,
                  drawdown_halted=excluded.drawdown_halted,
                  updated_at=excluded.updated_at
                """,
                (
                    1,
                    state.last_bar_open_time_utc,
                    int(state.bars_processed or 0),
                    state.current_day_utc,
                    int(state.order_rejects_today or 0),
                    int(bool(state.daily_loss_halted)),
                    int(bool(state.drawdown_halted)),
                    now,
                ),
            )

            if state.open_position is None:
                conn.execute("DELETE FROM open_positions WHERE id = 1")
            else:
                op = state.open_position
                conn.execute(
                    """
                    INSERT INTO open_positions (
                      id, symbol, side, qty, entry_price, entry_time_utc, entry_order_id,
                      tp_order_id, sl_order_id, tp_price, sl_price, opened_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                      symbol=excluded.symbol,
                      side=excluded.side,
                      qty=excluded.qty,
                      entry_price=excluded.entry_price,
                      entry_time_utc=excluded.entry_time_utc,
                      entry_order_id=excluded.entry_order_id,
                      tp_order_id=excluded.tp_order_id,
                      sl_order_id=excluded.sl_order_id,
                      tp_price=excluded.tp_price,
                      sl_price=excluded.sl_price,
                      opened_at=excluded.opened_at
                    """,
                    (
                        1,
                        op.symbol,
                        op.side,
                        float(op.qty),
                        float(op.entry_price),
                        op.entry_time_utc,
                        _coerce_int(op.entry_order_id),
                        _coerce_int(op.tp_order_id),
                        _coerce_int(op.sl_order_id),
                        float(op.tp_price) if op.tp_price is not None else None,
                        float(op.sl_price) if op.sl_price is not None else None,
                        op.opened_at or now,
                    ),
                )

            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise

    def append_trade_log(
        self,
        *,
        event_type: str,
        symbol: Optional[str],
        side: Optional[str],
        qty: Optional[float],
        price: Optional[float],
        realized_pnl: Optional[float],
        fee: Optional[float],
        funding_applied: Optional[float],
        reason: Optional[str],
        bar_time_utc: Optional[str],
    ) -> None:
        conn = self.conn
        if conn is None:
            print("[state_store_sqlite] append_trade_log failed: store is not open", file=sys.stderr)
            return
        try:
            conn.execute("BEGIN")
            conn.execute(
                """
                INSERT INTO trade_log (
                  event_type, symbol, side, qty, price, realized_pnl,
                  fee, funding_applied, reason, bar_time_utc, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_type,
                    symbol,
                    side,
                    qty,
                    price,
                    realized_pnl,
                    fee,
                    funding_applied,
                    reason,
                    bar_time_utc,
                    _utcnow_iso(),
                ),
            )
            conn.commit()
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            print(f"[state_store_sqlite] append_trade_log failed: {e}", file=sys.stderr)

    def export_state_json_snapshot(self, path: Path, state: RunnerState) -> None:
        try:
            _write_state_json_snapshot_atomic(path, state)
        except Exception as e:
            print(f"[state_store_sqlite] export_state_json_snapshot failed: {e}", file=sys.stderr)


def _coerce_legacy_runner_state(legacy_state: Any) -> RunnerState:
    legacy_op = getattr(legacy_state, "open_position", None)
    open_position: Optional[OpenPositionState] = None
    if legacy_op is not None:
        open_position = OpenPositionState(
            symbol=str(getattr(legacy_op, "symbol", "")),
            side=str(getattr(legacy_op, "side", "")),
            qty=_coerce_float(getattr(legacy_op, "qty", 0.0), 0.0),
            entry_price=_coerce_float(getattr(legacy_op, "entry_price", 0.0), 0.0),
            entry_time_utc=str(getattr(legacy_op, "entry_time_utc", "")),
            entry_order_id=_coerce_int(getattr(legacy_op, "entry_order_id", None)),
            tp_order_id=_coerce_int(getattr(legacy_op, "tp_order_id", None)),
            sl_order_id=_coerce_int(getattr(legacy_op, "sl_order_id", None)),
            tp_price=_coerce_optional_float(getattr(legacy_op, "tp_price", None)),
            sl_price=_coerce_optional_float(getattr(legacy_op, "sl_price", None)),
            opened_at=None,
        )

    return RunnerState(
        last_bar_open_time_utc=getattr(legacy_state, "last_bar_open_time_utc", None),
        bars_processed=int(getattr(legacy_state, "bars_processed", 0) or 0),
        current_day_utc=getattr(legacy_state, "current_day_utc", None),
        order_rejects_today=int(getattr(legacy_state, "order_rejects_today", 0) or 0),
        daily_loss_halted=False,
        drawdown_halted=False,
        open_position=open_position,
    )


def migrate_state_json_to_sqlite(
    *,
    db_path: Path,
    json_path: Path,
    events_path: Optional[Path] = None,
    force: bool = False,
) -> bool:
    db_path = Path(db_path)
    json_path = Path(json_path)

    if db_path.exists() and not force:
        return False
    if not json_path.exists():
        raise FileNotFoundError(f"Legacy state JSON not found: {json_path}")

    from forward.state_store import load_state as load_legacy_state

    legacy_state = load_legacy_state(json_path)
    state = _coerce_legacy_runner_state(legacy_state)

    if force:
        for p in [db_path, Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm")]:
            if p.exists():
                p.unlink()

    with SQLiteStateStore(db_path=db_path, events_path=events_path) as store:
        store.save_state(state)

    backup_path = json_path.with_name(json_path.name + ".bak")
    os.replace(json_path, backup_path)
    _write_state_json_snapshot_atomic(json_path, state)
    return True

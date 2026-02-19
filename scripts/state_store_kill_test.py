from __future__ import annotations
import argparse, os, sqlite3, subprocess, sys, tempfile
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from forward.state_store_sqlite import OpenPositionState, RunnerState, SQLiteStateStore

def _sample_state(tag: str) -> RunnerState:
    return RunnerState(
        last_bar_open_time_utc=f"2026-02-19T00:00:00+00:00/{tag}",
        bars_processed=7 if tag == "baseline" else 11,
        current_day_utc="2026-02-19",
        order_rejects_today=1,
        open_position=OpenPositionState(
            symbol="SOLUSDT",
            side="SHORT",
            qty=0.42,
            entry_price=123.45,
            entry_time_utc="2026-02-19T00:00:00+00:00",
            entry_order_id=1001,
            tp_order_id=1002,
            sl_order_id=1003,
            tp_price=120.0,
            sl_price=130.0,
        ),
    )


def _state_key(state: RunnerState) -> tuple:
    op = state.open_position
    return (
        state.last_bar_open_time_utc,
        state.bars_processed,
        state.current_day_utc,
        state.order_rejects_today,
        None if op is None else (op.symbol, op.side, op.qty, op.entry_price, op.entry_order_id, op.tp_order_id, op.sl_order_id, op.tp_price, op.sl_price),
    )


def _child(mode: str, db_path: Path) -> None:
    if mode == "committed":
        with SQLiteStateStore(db_path=db_path) as store:
            store.save_state(_sample_state("committed"))
        os._exit(137)

    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.execute("BEGIN")
    conn.execute(
        "INSERT INTO runner_state (id,last_bar_open_time_utc,bars_processed,current_day_utc,order_rejects_today,daily_loss_halted,drawdown_halted,updated_at) "
        "VALUES (1,'BROKEN',999,'2099-01-01',9,0,0,'x') "
        "ON CONFLICT(id) DO UPDATE SET bars_processed=999,last_bar_open_time_utc='BROKEN'"
    )
    conn.execute(
        "INSERT INTO open_positions (id,symbol,side,qty,entry_price,entry_time_utc,entry_order_id,tp_order_id,sl_order_id,tp_price,sl_price,opened_at) "
        "VALUES (1,'BROKEN','LONG',9,9,'x',9,9,9,9,9,'x') "
        "ON CONFLICT(id) DO UPDATE SET symbol='BROKEN',qty=9"
    )
    os._exit(137)

def main() -> int:
    parser = argparse.ArgumentParser(description="Crash-safety test for SQLiteStateStore")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--mode", choices=["committed", "uncommitted"], default="committed")
    parser.add_argument("--db-path", default=None)
    args = parser.parse_args()

    if args.child:
        _child(args.mode, Path(args.db_path))
        return 0

    def _run_child(mode: str, db_path: Path) -> int:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--child", "--mode", mode, "--db-path", str(db_path)]
        return subprocess.run(cmd, check=False).returncode

    try:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "state.db"

            rc = _run_child("committed", db_path)
            if rc != 137:
                raise RuntimeError(f"committed child exit={rc}, expected 137")
            with SQLiteStateStore(db_path=db_path) as store:
                got = store.load_state()
            if _state_key(got) != _state_key(_sample_state("committed")):
                raise RuntimeError("committed crash recovery mismatch")

            with SQLiteStateStore(db_path=db_path) as store:
                store.save_state(_sample_state("baseline"))
            rc = _run_child("uncommitted", db_path)
            if rc != 137:
                raise RuntimeError(f"uncommitted child exit={rc}, expected 137")
            with SQLiteStateStore(db_path=db_path) as store:
                got2 = store.load_state()
            if _state_key(got2) != _state_key(_sample_state("baseline")):
                raise RuntimeError("uncommitted crash recovery mismatch")

        print("PASS")
        return 0
    except Exception as e:
        print(f"FAIL: {e}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forward.state_store_sqlite import migrate_state_json_to_sqlite


def _resolve_path(run_dir: Path, value: str | None, default_name: str) -> Path:
    if value is None:
        return run_dir / default_name
    p = Path(value)
    if not p.is_absolute():
        p = run_dir / p
    return p


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate run_dir/state.json to SQLite state.db")
    parser.add_argument("run_dir", help="Run directory containing state.json")
    parser.add_argument("--db-path", default=None, help="SQLite db path (default: run_dir/state.db)")
    parser.add_argument("--json-path", default=None, help="State JSON path (default: run_dir/state.json)")
    parser.add_argument("--force", action="store_true", help="Delete and recreate db file if it exists")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    db_path = _resolve_path(run_dir, args.db_path, "state.db")
    json_path = _resolve_path(run_dir, args.json_path, "state.json")

    migrated = migrate_state_json_to_sqlite(
        db_path=db_path,
        json_path=json_path,
        events_path=run_dir / "events.jsonl",
        force=bool(args.force),
    )
    if not migrated:
        print(f"[migrate] db already exists, skipping: {db_path}")
        return 0

    print(f"[migrate] migrated {json_path} -> {db_path}")
    print(f"[migrate] backup: {json_path.name}.bak")
    print(f"[migrate] snapshot refreshed: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

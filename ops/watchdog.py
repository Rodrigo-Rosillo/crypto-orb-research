#!/usr/bin/env python3
"""Host watchdog for trader heartbeat/container/trade_log alerts."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import shutil

DEFAULT_STATE = {
    "stale_since": None,
    "last_restart_issued_at": None,
    "last_container_restart_count": 0,
    "last_trade_log_id": 0,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_iso_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def ensure_state_shape(raw: dict[str, Any]) -> dict[str, Any]:
    state = dict(DEFAULT_STATE)
    state.update(raw)
    state["last_container_restart_count"] = to_int(state.get("last_container_restart_count"), 0)
    state["last_trade_log_id"] = to_int(state.get("last_trade_log_id"), 0)
    if state.get("stale_since") is not None and not isinstance(state.get("stale_since"), str):
        state["stale_since"] = None
    if state.get("last_restart_issued_at") is not None and not isinstance(
        state.get("last_restart_issued_at"), str
    ):
        state["last_restart_issued_at"] = None
    return state


def load_state(state_path: Path, dry_run: bool) -> tuple[dict[str, Any], bool]:
    if state_path.exists():
        try:
            with state_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except PermissionError:
            print(
                f"ERROR: Cannot read {state_path}. Is the watchdog running as root? See RUNBOOK.",
                file=sys.stderr,
            )
            raise
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARNING: Failed to read state file {state_path}: {exc}. Recreating state.")
            return dict(DEFAULT_STATE), True
        if not isinstance(raw, dict):
            print(f"WARNING: Invalid state structure in {state_path}. Recreating state.")
            return dict(DEFAULT_STATE), True
        return ensure_state_shape(raw), False

    state = dict(DEFAULT_STATE)
    if dry_run:
        print(f"DRY-RUN: Would create state file at {state_path}")
    else:
        save_state(state_path, state, dry_run=False)
    return state, True


def save_state(state_path: Path, state: dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        print(f"DRY-RUN: Would write state to {state_path}: {json.dumps(state, sort_keys=True)}")
        return
    if state_path.parent:
        state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp_path, state_path)


def run_command(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return subprocess.CompletedProcess(cmd, 127, "", str(exc))


def first_non_empty_line(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def get_container_id(
    compose_dir: Path, compose_file: str, service_name: str
) -> tuple[str | None, list[str] | None]:
    # Prefer docker compose (v2). Use -a so stopped containers are included.
    commands = [
        ["docker", "compose", "-f", compose_file, "ps", "-a", "-q", service_name],
        ["docker", "compose", "-f", compose_file, "ps", "-q", service_name],  # fallback
        ["docker-compose", "-f", compose_file, "ps", "-a", "-q", service_name],
        ["docker-compose", "-f", compose_file, "ps", "-q", service_name],     # fallback
    ]

    for cmd in commands:
        # If docker-compose isn't installed, skip it quietly to avoid rc=127 spam.
        if cmd[0] == "docker-compose" and shutil.which("docker-compose") is None:
            continue

        proc = run_command(cmd, cwd=compose_dir)

        if proc.returncode == 0:
            container_id = first_non_empty_line(proc.stdout)
            if container_id:
                # Return an identifier describing which compose flavor worked.
                return container_id, cmd[:2] if cmd[0] == "docker" else [cmd[0]]

            # rc=0 but empty output: not a failure, just "no container found" for this command
            continue

        stderr = (proc.stderr or "").strip()
        print(f"WARNING: Command failed: {' '.join(cmd)} (rc={proc.returncode}) {stderr}")

    print(
        f"WARNING: Could not resolve container ID for service '{service_name}' "
        f"using compose_file='{compose_file}' in '{compose_dir}'."
    )
    return None, None


def inspect_data_dir(container_id: str) -> str | None:
    cmd = [
        "docker",
        "inspect",
        container_id,
        "--format",
        '{{ range .Mounts }}{{ if eq .Destination "/data" }}{{ .Source }}{{ end }}{{ end }}',
    ]
    proc = run_command(cmd)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        print(f"WARNING: Failed to inspect /data mount for container {container_id}: {stderr}")
        return None
    data_dir = (proc.stdout or "").strip()
    if not data_dir:
        print(f"WARNING: /data mount source not found for container {container_id}.")
        return None
    return data_dir


def inspect_restart_count(container_id: str) -> int | None:
    cmd = ["docker", "inspect", container_id, "--format", "{{.RestartCount}}"]
    proc = run_command(cmd)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        print(f"WARNING: Failed to inspect restart count for {container_id}: {stderr}")
        return None
    value = (proc.stdout or "").strip()
    try:
        return int(value)
    except ValueError:
        print(f"WARNING: Non-integer restart count for {container_id}: {value!r}")
        return None


def send_telegram(bot_token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    body = json.dumps({"chat_id": chat_id, "text": text}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"ERROR: Telegram HTTP error ({exc.code}): {detail}")
    except urllib.error.URLError as exc:
        print(f"ERROR: Telegram request failed: {exc}")


def heartbeat_stale(heartbeat_path: Path, stale_seconds: int) -> tuple[bool, int]:
    try:
        mtime = heartbeat_path.stat().st_mtime
    except FileNotFoundError:
        print(f"WARNING: Heartbeat file is missing: {heartbeat_path}")
        return True, stale_seconds + 1
    except PermissionError:
        print(
            f"ERROR: Cannot read {heartbeat_path}. Is the watchdog running as root? See RUNBOOK.",
            file=sys.stderr,
        )
        raise
    except OSError as exc:
        print(f"WARNING: Failed to stat heartbeat file {heartbeat_path}: {exc}")
        return True, stale_seconds + 1
    age = max(0, int(time.time() - mtime))
    return age > stale_seconds, age


def maybe_seed_trade_log(
    state_freshly_created: bool,
    db_path: Path,
    state: dict[str, Any],
    state_path: Path,
    dry_run: bool,
) -> bool:
    if not state_freshly_created:
        return False
    seeded_max_id = 0
    try:
        db_path.stat()
    except FileNotFoundError:
        pass
    except PermissionError:
        print(
            f"ERROR: Cannot read {db_path}. Is the watchdog running as root? See RUNBOOK.",
            file=sys.stderr,
        )
        raise
    except OSError as exc:
        print(f"WARNING: Could not stat state.db at {db_path}: {exc}")
    else:
        try:
            with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
                row = conn.execute("SELECT MAX(id) FROM trade_log").fetchone()
                seeded_max_id = to_int(row[0] if row else 0, 0)
        except PermissionError:
            print(
                f"ERROR: Cannot read {db_path}. Is the watchdog running as root? See RUNBOOK.",
                file=sys.stderr,
            )
            raise
        except sqlite3.Error as exc:
            print(f"WARNING: Could not seed trade_log cursor from {db_path}: {exc}")
    state["last_trade_log_id"] = seeded_max_id
    save_state(state_path, state, dry_run=dry_run)
    print(
        f"{'DRY-RUN: ' if dry_run else ''}First run detected; seeded last_trade_log_id={seeded_max_id} and skipped trade_log alerts."
    )
    return True


def format_trade_alert(row: sqlite3.Row) -> str | None:
    event_type = row["event_type"]
    symbol = row["symbol"]
    side = row["side"]
    qty = row["qty"]
    price = row["price"]
    reason = row["reason"]
    bar_time_utc = row["bar_time_utc"]

    if event_type == "ENTRY":
        return f"ENTRY {symbol} {side} qty={qty} price={price} bar={bar_time_utc}"
    if event_type == "EXIT":
        return f"EXIT {symbol} {side} qty={qty} bar={bar_time_utc} reason={reason}"
    if event_type == "REJECT":
        return f"REJECT {symbol} reason={reason} bar={bar_time_utc}"
    if event_type == "KILL_SWITCH" and reason == "KILL_SWITCH_DATA_STALE":
        return f"KILL_SWITCH_DATA_STALE bar={bar_time_utc}"
    if event_type == "DRAWDOWN_HALT":
        return f"DRAWDOWN_HALT reason={reason}"
    return None


def process_trade_log(
    db_path: Path,
    state: dict[str, Any],
    state_path: Path,
    dry_run: bool,
    emit_alert: Any,
) -> None:
    try:
        db_path.stat()
    except FileNotFoundError:
        print(f"WARNING: state.db not found at {db_path}; skipping trade_log alerts.")
        return
    except PermissionError:
        print(
            f"ERROR: Cannot read {db_path}. Is the watchdog running as root? See RUNBOOK.",
            file=sys.stderr,
        )
        raise
    except OSError as exc:
        print(f"WARNING: Could not stat state.db at {db_path}: {exc}")
        return

    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, event_type, symbol, side, qty, price, realized_pnl, reason, bar_time_utc
                FROM trade_log
                WHERE id > ?
                ORDER BY id ASC
                """,
                (to_int(state.get("last_trade_log_id"), 0),),
            ).fetchall()
    except PermissionError:
        print(
            f"ERROR: Cannot read {db_path}. Is the watchdog running as root? See RUNBOOK.",
            file=sys.stderr,
        )
        raise
    except sqlite3.Error as exc:
        print(f"WARNING: Failed to query trade_log in {db_path}: {exc}")
        return

    if not rows:
        return

    max_id = to_int(state.get("last_trade_log_id"), 0)
    for row in rows:
        row_id = to_int(row["id"], max_id)
        max_id = max(max_id, row_id)
        message = format_trade_alert(row)
        if message:
            emit_alert(message)

    state["last_trade_log_id"] = max_id
    save_state(state_path, state, dry_run=dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(description="Host watchdog for trader container/process liveness.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without sending alerts or writing state.")
    args = parser.parse_args()

    dry_run = args.dry_run

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        print(
            "ERROR: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in environment.",
            file=sys.stderr,
        )
        return 1

    stale_seconds = to_int(os.getenv("WATCHDOG_HEARTBEAT_STALE_SECONDS"), 600)
    restart_on_stale = os.getenv("WATCHDOG_RESTART_ON_STALE", "0") == "1"
    restart_grace_seconds = to_int(os.getenv("WATCHDOG_RESTART_GRACE_SECONDS"), 300)
    state_path = Path(os.getenv("WATCHDOG_STATE_PATH", "/home/ubuntu/.watchdog_state.json"))
    compose_dir = Path(os.getenv("WATCHDOG_COMPOSE_DIR", os.getcwd()))
    compose_file = os.getenv("WATCHDOG_COMPOSE_FILE", "docker-compose.yml")
    service_name = os.getenv("WATCHDOG_SERVICE_NAME", "trader")
    data_dir_env = os.getenv("WATCHDOG_DATA_DIR")

    try:
        state, state_freshly_created = load_state(state_path, dry_run=dry_run)
    except PermissionError:
        return 1

    def emit_alert(text: str) -> None:
        if dry_run:
            print(f"DRY-RUN: Would send Telegram alert: {text}")
            return
        send_telegram(bot_token=bot_token, chat_id=chat_id, text=text)

    container_id: str | None = None
    compose_cmd_prefix: list[str] | None = None
    data_dir: str | None = None

    if data_dir_env:
        data_dir = data_dir_env
        print(f"Using WATCHDOG_DATA_DIR={data_dir}")
        container_id, compose_cmd_prefix = get_container_id(compose_dir, compose_file, service_name)
        if not container_id:
            print("WARNING: Container ID not found; container restart detection/actions will be skipped.")
    else:
        container_id, compose_cmd_prefix = get_container_id(compose_dir, compose_file, service_name)
        if not container_id:
            print("WARNING: Could not resolve container ID from docker compose.")
            print(
                "Skipping Docker-dependent checks: heartbeat and trade_log checks cannot run without data_dir."
            )
            return 0
        data_dir = inspect_data_dir(container_id)
        if not data_dir:
            print(
                "WARNING: Could not resolve /data mount source. "
                "Skipping Docker-dependent checks: heartbeat and trade_log checks cannot run without data_dir."
            )
            return 0

    data_dir_path = Path(data_dir)
    heartbeat_path = data_dir_path / "heartbeat"
    db_path = data_dir_path / "state.db"

    try:
        stale, age_seconds = heartbeat_stale(heartbeat_path, stale_seconds)
    except PermissionError:
        return 1

    state_changed = False
    now_iso = utc_now_iso()

    if stale:
        if not state.get("stale_since"):
            state["stale_since"] = now_iso
            state_changed = True
            save_state(state_path, state, dry_run=dry_run)
            emit_alert(f"Heartbeat stale for {service_name}: last seen {age_seconds}s ago")
    else:
        if state.get("stale_since"):
            state["stale_since"] = None
            state["last_restart_issued_at"] = None
            state_changed = True
            save_state(state_path, state, dry_run=dry_run)
            emit_alert(f"Heartbeat recovered for {service_name}")

    if container_id:
        restart_count = inspect_restart_count(container_id)
        if restart_count is not None:
            previous = to_int(state.get("last_container_restart_count"), 0)
            if restart_count > previous:
                emit_alert(f"Trader container restarted (restart_count={restart_count})")
            if restart_count != previous:
                state["last_container_restart_count"] = restart_count
                state_changed = True

    if stale and restart_on_stale:
        if not container_id or not compose_cmd_prefix:
            print("WARNING: Restart on stale requested, but container/compose command is unavailable.")
        else:
            stale_since_dt = parse_iso_utc(state.get("stale_since"))
            stale_for_seconds = 0
            if stale_since_dt:
                stale_for_seconds = max(0, int(time.time() - stale_since_dt.timestamp()))
            if stale_for_seconds >= restart_grace_seconds and not state.get("last_restart_issued_at"):
                restart_cmd = compose_cmd_prefix + ["-f", compose_file, "restart", service_name]
                if dry_run:
                    print(f"DRY-RUN: Would run: {' '.join(restart_cmd)} (cwd={compose_dir})")
                    print(
                        f"DRY-RUN: Would set last_restart_issued_at and alert restart for stale_for={stale_for_seconds}s"
                    )
                else:
                    proc = run_command(restart_cmd, cwd=compose_dir)
                    if proc.returncode == 0:
                        state["last_restart_issued_at"] = utc_now_iso()
                        state_changed = True
                        save_state(state_path, state, dry_run=dry_run)
                        emit_alert(f"Restart issued for {service_name} (stale for {stale_for_seconds}s)")
                    else:
                        stderr = (proc.stderr or "").strip()
                        print(f"WARNING: Failed to restart service {service_name}: {stderr}")

    try:
        seeded = maybe_seed_trade_log(
            state_freshly_created=state_freshly_created,
            db_path=db_path,
            state=state,
            state_path=state_path,
            dry_run=dry_run,
        )
    except PermissionError:
        return 1

    if not seeded:
        try:
            process_trade_log(
                db_path=db_path,
                state=state,
                state_path=state_path,
                dry_run=dry_run,
                emit_alert=emit_alert,
            )
        except PermissionError:
            return 1

    if state_changed:
        save_state(state_path, state, dry_run=dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

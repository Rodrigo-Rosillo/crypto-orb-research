#!/usr/bin/env python3
"""Daily performance report from trade_log, optionally sent to Telegram."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sqlite3
import statistics
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


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
    commands = [
        ["docker", "compose", "-f", compose_file, "ps", "-a", "-q", service_name],
        ["docker", "compose", "-f", compose_file, "ps", "-q", service_name],
        ["docker-compose", "-f", compose_file, "ps", "-a", "-q", service_name],
        ["docker-compose", "-f", compose_file, "ps", "-q", service_name],
    ]

    for cmd in commands:
        if cmd[0] == "docker-compose" and shutil.which("docker-compose") is None:
            continue

        proc = run_command(cmd, cwd=compose_dir)

        if proc.returncode == 0:
            container_id = first_non_empty_line(proc.stdout)
            if container_id:
                return container_id, cmd[:2] if cmd[0] == "docker" else [cmd[0]]
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


def send_telegram(bot_token: str, chat_id: str, text: str) -> bool:
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
        return True
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"ERROR: Telegram HTTP error ({exc.code}): {detail}")
    except urllib.error.URLError as exc:
        print(f"ERROR: Telegram request failed: {exc}")
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Telegram request failed: {exc}")
    return False


def _print_message(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(safe)


def _resolve_default_db_path() -> Path:
    compose_dir = Path(os.getenv("WATCHDOG_COMPOSE_DIR", os.getcwd()))
    compose_file = os.getenv("WATCHDOG_COMPOSE_FILE", "docker-compose.yml")
    service_name = os.getenv("WATCHDOG_SERVICE_NAME", "trader")
    data_dir_env = os.getenv("WATCHDOG_DATA_DIR")

    data_dir: str | None = None
    if data_dir_env:
        data_dir = data_dir_env
        print(f"Using WATCHDOG_DATA_DIR={data_dir}")
        container_id, _ = get_container_id(compose_dir, compose_file, service_name)
        if not container_id:
            print("WARNING: Container ID not found; proceeding with WATCHDOG_DATA_DIR override.")
    else:
        container_id, _ = get_container_id(compose_dir, compose_file, service_name)
        if not container_id:
            raise RuntimeError(
                "Could not resolve container ID from docker compose. "
                "Set WATCHDOG_DATA_DIR or pass --db-path."
            )
        data_dir = inspect_data_dir(container_id)
        if not data_dir:
            raise RuntimeError(
                "Could not resolve /data mount source. "
                "Set WATCHDOG_DATA_DIR or pass --db-path."
            )

    return Path(data_dir) / "state.db"


def _load_trades(db_path: Path, n: int = 30) -> list[dict]:
    sql = """
SELECT e.id              AS entry_id,
       e.symbol          AS symbol,
       e.side            AS side,
       e.qty             AS qty,
       e.price           AS entry_price,
       e.fee             AS entry_fee,
       e.bar_time_utc    AS entry_time,
       x.qty             AS exit_qty,
       x.price           AS exit_price,
       x.fee             AS exit_fee,
       x.realized_pnl    AS realized_pnl,
       x.funding_applied AS funding_applied,
       x.bar_time_utc    AS exit_time
FROM trade_log e
JOIN trade_log x ON x.id = (
    SELECT MIN(id) FROM trade_log
    WHERE id > e.id
      AND event_type = 'EXIT'
      AND symbol = e.symbol
)
WHERE e.event_type = 'ENTRY'
  AND x.realized_pnl IS NOT NULL
ORDER BY e.id DESC
LIMIT ?
"""

    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, (max(1, int(n)),)).fetchall()
    return [dict(r) for r in rows]


def _compute_metrics(trades: list[dict]) -> dict:
    if len(trades) < 2:
        return {"insufficient": True, "n": len(trades)}

    n = len(trades)
    pnls_net = [float(t["realized_pnl"]) - float(t.get("funding_applied") or 0.0) for t in trades]

    # Note: realized_pnl is already net-of-fees (Option A semantics).
    # funding_applied is a separate adjustment; include it when computing net performance.
    # Do NOT subtract fee columns again.

    wins = sum(1 for p in pnls_net if p > 0)
    losses = sum(1 for p in pnls_net if p < 0)
    win_rate = wins / n

    gross_wins = sum(p for p in pnls_net if p > 0)
    gross_losses = abs(sum(p for p in pnls_net if p < 0))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    net_return = sum(pnls_net)

    returns_frac: list[float] = []
    for t in trades:
        ep = float(t.get("entry_price") or 0)
        entry_qty = float(t.get("qty") or 0)
        exit_qty = float(t.get("exit_qty") or 0)
        qty_used = exit_qty if exit_qty > 0 else entry_qty
        pnl_net = float(t["realized_pnl"]) - float(t.get("funding_applied") or 0.0)
        if ep > 0 and qty_used > 0:
            returns_frac.append(pnl_net / (ep * qty_used))

    if len(returns_frac) >= 2:
        mean_r = statistics.mean(returns_frac)
        std_r = statistics.pstdev(returns_frac)
        sharpe = (mean_r / std_r) if std_r > 0 else None
        down = [r for r in returns_frac if r < 0]
        down_std = statistics.pstdev(down) if len(down) >= 2 else None
        sortino = (mean_r / down_std) if down_std and down_std > 0 else None
    else:
        sharpe = sortino = None

    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    longest_dd_trades = 0
    dd_start: int | None = None

    for i, pnl in enumerate(pnls_net):
        equity += pnl
        if equity > peak:
            peak = equity

        dd = equity - peak
        if dd < max_dd:
            max_dd = dd

        if equity < peak:
            if dd_start is None:
                dd_start = i
        else:
            if dd_start is not None:
                longest_dd_trades = max(longest_dd_trades, i - dd_start)
                dd_start = None

    if dd_start is not None:
        longest_dd_trades = max(longest_dd_trades, n - dd_start)

    max_consec_losses = 0
    cur = 0
    for p in pnls_net:
        if p < 0:
            cur += 1
            max_consec_losses = max(max_consec_losses, cur)
        else:
            cur = 0

    hold_hours: list[float] = []
    for t in trades:
        try:
            et = datetime.fromisoformat(str(t["entry_time"]))
            xt = datetime.fromisoformat(str(t["exit_time"]))
            if et.tzinfo is None:
                et = et.replace(tzinfo=timezone.utc)
            if xt.tzinfo is None:
                xt = xt.replace(tzinfo=timezone.utc)
            hold_hours.append((xt - et).total_seconds() / 3600)
        except Exception:
            pass
    avg_hold_h = (sum(hold_hours) / len(hold_hours)) if hold_hours else None

    try:
        first_t = datetime.fromisoformat(str(trades[0]["entry_time"]))
        last_t = datetime.fromisoformat(str(trades[-1]["entry_time"]))
        if first_t.tzinfo is None:
            first_t = first_t.replace(tzinfo=timezone.utc)
        if last_t.tzinfo is None:
            last_t = last_t.replace(tzinfo=timezone.utc)
        span_days = max(1, (last_t - first_t).days + 1)
        freq_per_day = n / span_days
        span_str = f"{first_t.strftime('%Y-%m-%d')} \u2192 {last_t.strftime('%Y-%m-%d')} ({span_days} days)"
    except Exception:
        freq_per_day = None
        span_str = "unknown"

    return {
        "n": n,
        "symbol": trades[0].get("symbol") if trades else None,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "net_return": net_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "drawdown_duration_trades": longest_dd_trades,
        "max_consec_losses": max_consec_losses,
        "avg_hold_hours": avg_hold_h,
        "freq_per_day": freq_per_day,
        "span_str": span_str,
        "insufficient": False,
    }


def _format_message(metrics: dict, symbol: str, report_dt: str) -> str:
    n = int(metrics.get("n") or 0)
    if metrics.get("insufficient"):
        return (
            f"📊 Daily Report — {symbol} — {report_dt}\n"
            f"Insufficient data ({n} completed trades in trade_log)."
        )

    freq = metrics.get("freq_per_day")
    freq_str = f"{float(freq):.2f} trades/day" if freq is not None else "N/A"

    pf = metrics.get("profit_factor")
    pf_str = "∞" if (isinstance(pf, float) and math.isinf(pf)) else f"{float(pf):.2f}"

    sharpe = metrics.get("sharpe")
    sharpe_str = f"{float(sharpe):.3f}" if sharpe is not None else "N/A"

    sortino = metrics.get("sortino")
    sortino_str = f"{float(sortino):.3f}" if sortino is not None else "N/A"

    avg_hold_hours = metrics.get("avg_hold_hours")
    hold_str = f"{float(avg_hold_hours):.1f} h" if avg_hold_hours is not None else "N/A"

    return "\n".join(
        [
            f"📊 Daily Report — {symbol} — {report_dt}",
            f"Rolling last {n} trades | Span: {metrics.get('span_str')}",
            f"Frequency: {freq_str}",
            "",
            f"Wins / Losses:   {int(metrics.get('wins') or 0)} / {int(metrics.get('losses') or 0)}",
            f"Win rate:        {float(metrics.get('win_rate') or 0.0):.1%}",
            f"Profit factor:   {pf_str}         (∞ if no losers)",
            f"Net return:      {float(metrics.get('net_return') or 0.0):+.2f} USD",
            "",
            f"Sharpe*:         {sharpe_str}     (* per-trade, not annualised)",
            f"Sortino*:        {sortino_str}",
            f"Max drawdown:    {float(metrics.get('max_drawdown') or 0.0):.2f} USD  (over {int(metrics.get('drawdown_duration_trades') or 0)} trades)",
            f"Consec. losses:  {int(metrics.get('max_consec_losses') or 0)} max",
            "",
            f"Avg hold time:   {hold_str}",
            "Avg slippage:    N/A (not recorded)",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate daily rolling trade performance report.")
    parser.add_argument("--dry-run", action="store_true", help="Print report and skip Telegram send.")
    parser.add_argument("--db-path", type=str, default=None, help="Override SQLite DB path.")
    parser.add_argument("--n", type=int, default=30, help="Rolling trade window size.")
    args = parser.parse_args()

    try:
        db_path = Path(args.db_path) if args.db_path else _resolve_default_db_path()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    try:
        trades_desc = _load_trades(db_path=db_path, n=int(args.n))
    except Exception as e:
        print(f"ERROR: Failed to load trades from {db_path}: {e}", file=sys.stderr)
        return 1

    trades = list(reversed(trades_desc))
    metrics = _compute_metrics(trades)

    symbol = str(metrics.get("symbol") or "UNKNOWN")
    report_dt = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    message = _format_message(metrics, symbol, report_dt)

    if args.dry_run:
        _print_message(message)
        return 0

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        print("ERROR: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set.", file=sys.stderr)
        return 1

    if not send_telegram(bot_token=bot_token, chat_id=chat_id, text=message):
        print("ERROR: Failed to send Telegram message.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

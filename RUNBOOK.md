# RUNBOOK — Crypto ORB Trader

## Quick Facts

- Exchange: Binance USD-M Futures (SOLUSDT perp)
- Timeframe: 30m, UTC
- Host: AWS Lightsail `<FILL_ME_PUBLIC_IP>`
- Service name: `trader` (`docker compose`)
- Data: `/data/state.db` (SQLite WAL), `/data/heartbeat` (inside container)
- Repo dir on host: `<FILL_ME_REPO_DIR>`

## System Overview

Crypto ORB Trader is a Dockerized ORB trading bot for Binance USD-M Futures (SOLUSDT perpetual) running on AWS Lightsail at `<FILL_ME_PUBLIC_IP>`. Production orchestration is from `<FILL_ME_REPO_DIR>` where `docker-compose.yml` lives. Connect to the host with:

```bash
ssh -i <FILL_ME_SSH_KEY_PATH> ubuntu@<FILL_ME_PUBLIC_IP>
```

## Normal Operations

Operate from the host, in the repo directory:

```bash
cd <FILL_ME_REPO_DIR>
```

Start:

```bash
docker compose up -d --build
```

Stop:

```bash
docker compose stop trader
```

Restart:

```bash
docker compose restart trader
```

Status:

```bash
docker compose ps
```

Logs (stream):

```bash
docker compose logs -f trader
```

Logs (last 200 lines):

```bash
docker compose logs trader --tail 200
```

Deploy code change:

```bash
git pull
docker compose up -d --build
docker compose ps
docker compose exec trader cat /data/heartbeat
```

`.env` handling:
- `.env` is host-only and must not be committed.
- `.env` contains secrets and `STATE_DB_PATH` / `HEARTBEAT_PATH`.
- Lock permissions:

```bash
chmod 600 .env
```

## Emergency Procedures

1. Stop trading immediately (bot responsive)

```bash
docker compose stop trader
```

2. Open position but bot unresponsive

### Emergency: Flatten all futures positions

Use this standalone script to close all open Binance USD-M Futures positions at market with `reduceOnly=true`.

Testnet/demo:

```bash
BINANCE_TESTNET_API_KEY=... BINANCE_TESTNET_API_SECRET=... python scripts/emergency_flatten.py --testnet
```

Mainnet:

```bash
BINANCE_API_KEY=... BINANCE_API_SECRET=... python scripts/emergency_flatten.py
```

Docker (works even if the main trader container is dead):

Testnet/demo:

```bash
BINANCE_TESTNET_API_KEY=... BINANCE_TESTNET_API_SECRET=... docker compose run --rm trader python scripts/emergency_flatten.py --testnet
```

Mainnet:

```bash
BINANCE_API_KEY=... BINANCE_API_SECRET=... docker compose run --rm trader python scripts/emergency_flatten.py
```

Verification:
- Confirm positions are flat in Binance UI, or rerun `emergency_flatten` and confirm output reports `No open positions.`

3. Container keeps restarting

```bash
docker compose ps
docker compose logs trader --tail 200
docker compose up -d --build
```

4. Suspected SQLite corruption

Stop bot:

```bash
docker compose stop trader
```

Resolve the host path for `/data` volume mount (inside-container `/data` is a Docker named volume, not host `/data`):

```bash
CID=$(docker compose ps -q trader | head -n 1)
DATA_DIR=$(docker inspect "$CID" \
  --format '{{ range .Mounts }}{{ if eq .Destination "/data" }}{{ .Source }}{{ end }}{{ end }}')
echo "$DATA_DIR"
```

Run integrity check:

```bash
sqlite3 "$DATA_DIR/state.db" "PRAGMA integrity_check;"
```

If result is not `ok`, move files aside:

```bash
sudo mv "$DATA_DIR/state.db" "$DATA_DIR/state.db.corrupt.$(date -u +%Y%m%dT%H%M%SZ)"
sudo mv "$DATA_DIR/state.db-wal" "$DATA_DIR/state.db-wal.corrupt.$(date -u +%Y%m%dT%H%M%SZ)" 2>/dev/null || true
sudo mv "$DATA_DIR/state.db-shm" "$DATA_DIR/state.db-shm.corrupt.$(date -u +%Y%m%dT%H%M%SZ)" 2>/dev/null || true
```

Restore latest backup (see Backup Procedure), then restart:

```bash
docker compose up -d
```

5. API key compromised

- Disable the compromised key in Binance console immediately.
- Close open positions manually via Binance UI (or run `emergency_flatten` if still safe).
- Generate a new API key with Read + Trade only, IP restricted to the Lightsail static IP, and withdrawals disabled.
- Update host `.env`, then redeploy:

```bash
docker compose up -d --build
```

## Backup Procedure (Daily)

Important: the SQLite DB is in a Docker named volume. Do not assume host `/data`; resolve the real mountpoint (under `/var/lib/docker/volumes/...`) each time, or access via `docker compose exec`.

Option 1: Local backups (no AWS required)

Resolve `DATA_DIR`:

```bash
CID=$(docker compose ps -q trader | head -n 1)
DATA_DIR=$(docker inspect "$CID" \
  --format '{{ range .Mounts }}{{ if eq .Destination "/data" }}{{ .Source }}{{ end }}{{ end }}')
echo "$DATA_DIR"
```

Backup copy:

```bash
sudo mkdir -p "$DATA_DIR/backups"
sudo cp "$DATA_DIR/state.db" "$DATA_DIR/backups/state.db.$(date -u +%Y%m%dT%H%M%SZ)"
```

Daily automation (root crontab):

```bash
sudo crontab -e
```

```cron
0 2 * * * CID=$(docker compose -f <FILL_ME_REPO_DIR>/docker-compose.yml ps -q trader | head -n 1) && \
DATA_DIR=$(docker inspect "$CID" --format '{{ range .Mounts }}{{ if eq .Destination "/data" }}{{ .Source }}{{ end }}{{ end }}') && \
mkdir -p "$DATA_DIR/backups" && \
cp "$DATA_DIR/state.db" "$DATA_DIR/backups/state.db.$(date -u +%Y%m%dT%H%M%SZ)"
```

Retention (delete local backups older than 30 days):

```bash
find "$DATA_DIR/backups" -name "state.db.*" -mtime +30 -delete
```

Option 2: S3 backups (optional)

```bash
S3_BUCKET=s3://<FILL_ME_BUCKET>
CID=$(docker compose ps -q trader | head -n 1)
DATA_DIR=$(docker inspect "$CID" \
  --format '{{ range .Mounts }}{{ if eq .Destination "/data" }}{{ .Source }}{{ end }}{{ end }}')
aws s3 cp "$DATA_DIR/state.db" \
  "$S3_BUCKET/state.db.$(date -u +%Y%m%dT%H%M%SZ)"
```

Restore steps:

```bash
docker compose stop trader
CID=$(docker compose ps -q trader | head -n 1)
DATA_DIR=$(docker inspect "$CID" \
  --format '{{ range .Mounts }}{{ if eq .Destination "/data" }}{{ .Source }}{{ end }}{{ end }}')
sudo cp /path/to/backup/state.db "$DATA_DIR/state.db"
sudo rm -f "$DATA_DIR/state.db-wal" "$DATA_DIR/state.db-shm"
docker compose up -d
```

## Contacts and Escalation

- Primary operator: `<FILL_ME_NAME>` `<FILL_ME_CONTACT>`
- Secondary: `<FILL_ME_NAME>` `<FILL_ME_CONTACT>`
- Where to check status: Binance System Status page
- Where to check status: AWS Lightsail status page
- If urgent and a position is open: run `emergency_flatten` first, then investigate root cause.

## Monitoring / Watchdog

### Watchdog (host cron)

This watchdog runs on the host (outside Docker) using system Python:

```bash
/usr/bin/python3 /home/ubuntu/watchdog.py
```

Installation:

```bash
cp ops/watchdog.py /home/ubuntu/watchdog.py
```

Required env vars:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

Optional env vars:

- `WATCHDOG_COMPOSE_DIR` (default: cwd; set to repo root on Lightsail)
- `WATCHDOG_COMPOSE_FILE` (default: `docker-compose.yml`)
- `WATCHDOG_SERVICE_NAME` (default: `trader`)
- `WATCHDOG_DATA_DIR` (overrides auto-detect; useful for local dev/testing)
- `WATCHDOG_HEARTBEAT_STALE_SECONDS` (default: `600`)
- `WATCHDOG_RESTART_ON_STALE` (default: `0`; set to `1` to enable auto-restart)
- `WATCHDOG_RESTART_GRACE_SECONDS` (default: `300`)
- `WATCHDOG_STATE_PATH` (default: `/home/ubuntu/.watchdog_state.json`)

CRITICAL FILE PERMISSIONS:

Docker named volume data is owned by `root:root` with mode `701`. The `ubuntu` user cannot read it even if they are in the `docker` group.

RECOMMENDED: run cron as root (`sudo crontab -e`):

```cron
TELEGRAM_BOT_TOKEN=xxx
TELEGRAM_CHAT_ID=xxx
WATCHDOG_COMPOSE_DIR=/home/ubuntu/crypto-orb-research
*/5 * * * * /usr/bin/python3 /home/ubuntu/watchdog.py >> /var/log/watchdog.log 2>&1
```

ALTERNATIVE: relax volume permissions (less secure; must repeat after `docker compose down/up`):

```bash
sudo chmod o+x /var/lib/docker/volumes/
sudo chmod -R o+r /var/lib/docker/volumes/crypto-orb-research_trader_data/
```

Testing:

```bash
# Dry-run (safe, no Telegram):
sudo WATCHDOG_COMPOSE_DIR=/home/ubuntu/crypto-orb-research \
  TELEGRAM_BOT_TOKEN=test TELEGRAM_CHAT_ID=test \
  /usr/bin/python3 /home/ubuntu/watchdog.py --dry-run

# Simulate stale heartbeat:
VOLUME=$(docker volume inspect crypto-orb-research_trader_data \
  --format '{{ .Mountpoint }}')
sudo touch -t 202001010000 $VOLUME/heartbeat
sudo /usr/bin/python3 /home/ubuntu/watchdog.py
# Expect: stale alert in Telegram

# Verify cron installed:
sudo crontab -l

# Simulate trade log alerts:
# Open + close a small testnet position, then run watchdog manually.
# Expect: ENTRY and EXIT alerts in Telegram.
```

Log rotation install:

```bash
sudo cp ops/logrotate-watchdog.conf /etc/logrotate.d/watchdog
```

## Threat Model

| Scenario | Existing Mitigation | Response |
| --- | --- | --- |
| 1. API Key Leak | IP-restricted API key, host-only `.env`, least-privilege key settings | Disable key, flatten exposure, rotate key, redeploy |
| 2. Runaway Orders | Reject-count kill switch and single-open-position enforcement | Stop bot, flatten, inspect run artifacts before restart |
| 3. Lightsail Instance Reboot | `restart: unless-stopped`, watchdog cron, SQLite WAL consistency | Verify container/heartbeat and auto-enable instance behavior |
| 4. SQLite File Corrupted | Startup integrity checks and transactional state writes | Stop bot, inspect DB integrity, restore backup, restart |
| 5. WebSocket Drops / Reconnect Fails Repeatedly | Exponential backoff, heartbeat updates, data-stale kill switch, watchdog alerting | Confirm stale event/status, restart after connectivity is healthy |
| 6. Unexpected Open Position on Startup | Startup reconciliation vs exchange position, mismatch halt, optional auto-flatten | Reconcile manually, verify state alignment, restart |
| 7. Binance System Outage | Data-stale kill switch; halts trading without fresh data | Wait for recovery, verify TP/SL state, then restart |
| 8. Large Unexpected Loss | Partial guards (reject/margin kill switches), alerting; drawdown breaker deferred | Stop bot, assess position risk, perform postmortem before resume |

### 1. API Key Leak
(committed to git, exposed in logs, environment, etc.)

Mitigation:
- API key is IP-restricted to Lightsail static IP in Binance console.
- Key is stored in host-only `.env` (`chmod 600`, not committed).
- Key has Read + Trade only; withdrawals disabled.
- Secret scanning in CI: DEFERRED to Phase 7.

Response:
1. Disable compromised key in Binance console immediately.
2. Close open positions manually via Binance UI or `emergency_flatten.py`.
3. Audit what was exposed and how.
4. Generate new key (Read + Trade only, IP-restricted, withdrawals disabled).
5. Update host `.env`, redeploy: `docker compose up -d --build`.

Status: PARTIAL (secret scanning deferred to Phase 7 CI setup)

### 2. Runaway Orders (bug causes repeated order placement)

Mitigation:
- `max_order_rejects_per_day` kill switch halts trading after N consecutive
  order rejects (default 3). Fires `KILL_SWITCH_ORDER_REJECTS` event and stops
  the main loop.
- One open position at a time enforced: bot will not place a new entry if
  `state.open_position` is not `None`.

Response:
1. `docker compose stop trader`
2. Run `emergency_flatten.py` to close any open position.
3. Inspect `events.jsonl` and `orders.csv` in the run directory to determine
   root cause before restarting.

Note: A max daily order count limit (separate from reject count) is not yet
implemented. The reject kill switch is the primary guard.

Status: PARTIAL

### 3. Lightsail Instance Reboot

Mitigation:
- `restart: unless-stopped` in `docker-compose.yml` causes Docker to restart
  the trader container automatically after host reboot.
- Watchdog cron (root crontab) resumes automatically on host reboot.
- SQLite WAL mode ensures state is consistent after unclean shutdown.

Response:
1. Confirm container restarted: `docker compose ps`
2. Confirm heartbeat is fresh: `docker compose exec trader cat /data/heartbeat`
3. Watchdog will send a Telegram alert when heartbeat recovers.

Operator action: Ensure "Auto-enable instance" is active in the Lightsail
console so the instance itself restarts after an AWS host failure.

Status: ACTIVE

### 4. SQLite File Corrupted

Mitigation:
- SQLite WAL mode with `synchronous=NORMAL` and `PRAGMA integrity_check` on
  every startup. Bot halts if integrity check fails rather than operating on
  a corrupt state.
- WAL atomic writes: `save_state()` uses explicit `BEGIN/COMMIT` covering both
  `runner_state` and `open_positions` in a single transaction.

Response:
See "Suspected SQLite corruption" under Emergency Procedures. Summary:
1. `docker compose stop trader`
2. Resolve `DATA_DIR` via `docker inspect`.
3. Run `PRAGMA integrity_check;` via `sqlite3`.
4. If not `ok`: move aside `state.db` + WAL sidecars, restore from backup.
5. `docker compose up -d`

Status: ACTIVE

### 5. WebSocket Drops / Reconnect Fails Repeatedly

Mitigation:
- `BinanceLiveKlineSource` implements exponential backoff reconnect
  (`max_backoff_seconds` configurable).
- Heartbeat background task writes `/data/heartbeat` every 60 seconds.
- Data staleness kill switch in `DataService.heartbeat_task`: if no bar
  arrives within `max_data_gap_bars * bar_seconds` seconds, fires
  `KILL_SWITCH_DATA_STALE` and sets `stop_event`.
- Watchdog detects stale heartbeat (>10 min) and sends Telegram alert.

Response:
1. Watchdog Telegram alert will fire within 10–15 minutes of stale heartbeat.
2. Check `events.jsonl` for `KILL_SWITCH_DATA_STALE` entries.
3. Check Binance System Status page for exchange-side WebSocket issues.
4. Restart bot when connectivity is confirmed: `docker compose restart trader`

Status: ACTIVE

### 6. Unexpected Open Position on Startup
(local state says flat, exchange has an open position — or vice versa)

Mitigation:
- Startup reconciliation block in `run_live_testnet()` fetches
  `/fapi/v2/positionRisk` and compares to `state.open_position`.
- On any mismatch, bot emits `RECON_MISMATCH` event and halts (returns 0)
  before entering the main trading loop.
- `flatten_on_mismatch` config option (default: false) controls whether the
  bot auto-flattens the exchange position on mismatch.

Response:
1. Check `events.jsonl` for `RECON_MISMATCH` event and its payload.
2. Manually reconcile: close the unexpected position via Binance UI or
   `emergency_flatten.py` if needed.
3. Confirm `state.open_position` in `state.json` matches reality.
4. Restart bot: `docker compose up -d`

`flatten_on_mismatch` setting: review `config_forward_test.yaml` and make a
conscious choice. Default (`false`) is recommended — auto-flatten can hide bugs.

Status: ACTIVE

### 7. Binance System Outage

Mitigation:
- Data staleness kill switch fires when no bars arrive within the gap
  threshold (same as scenario 5).
- Bot halts gracefully; does not attempt to place orders without data.
- Open position may remain unhedged during the outage (TP/SL orders
  were placed on Binance at entry; those remain on exchange if exchange
  is partially functional).

Response:
1. Check Binance System Status page.
2. Do NOT restart the bot during the outage — it will halt again immediately.
3. When exchange recovers: if position is open, verify TP/SL orders are
   still active in Binance UI before restarting the bot.
4. If TP/SL orders were cancelled during the outage: use `emergency_flatten.py`
   or manually manage the position before restarting.
5. Restart: `docker compose up -d`

Status: ACTIVE

### 8. Large Unexpected Loss

Mitigation:
- `daily_loss_halted` and `drawdown_halted` fields exist in the SQLite
  schema (`runner_state` table) but the drawdown circuit breaker logic is
  NOT YET WIRED to trading decisions. These are stubs reserved for a
  future phase.
- Current guards: order reject kill switch, margin ratio kill switch
  (`KILL_SWITCH_MARGIN_RATIO` fires if `maint_margin/balance >= threshold`).
- Watchdog sends ENTRY/EXIT/REJECT Telegram alerts on every trade.

Response:
1. `docker compose stop trader`
2. If position is open: assess whether to hold or flatten via `emergency_flatten.py`.
3. Do not attempt to recover losses by resuming trading immediately.
4. Conduct a postmortem: review `trade_log`, `events.jsonl`, `signals.csv`.
5. Identify root cause before resuming. Resume only after understanding
   why the circuit breaker did not prevent the loss.

Status: PARTIAL — drawdown circuit breaker deferred to future phase.


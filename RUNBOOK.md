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

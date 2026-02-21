# Emergency: Flatten all futures positions

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

# Watchdog (host cron)

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

# crypto-orb-research

Research and deployment repo for an Opening Range Breakout (ORB) strategy on **SOLUSDT 30m** Binance USD-M Futures (UTC). Covers the full pipeline from reproducible backtests through production deployment: data quality, spot + isolated-margin futures execution models, Phase-3 robustness (walk-forward, bootstraps, benchmarks, regime diagnostics), live shadow and testnet trading, Dockerized production deployment on AWS Lightsail, and a CI-gated test suite (unit, property-based, integration, and replay regression).

> **Educational/research only.** Nothing here is financial advice. If you ever run this live, start with paper/shadow mode and strict risk limits.

---

## What this repo is (and is not)

**It is:**
- A “research artifact” where rerunning scripts produces auditable outputs (JSON/CSV/HTML) under `reports/`.
- A pipeline from **raw CSV** → **data quality checks** → **processed parquet + valid-day list** → **baseline backtest** → **stress/robustness/walk-forward**.
- A **Dockerized production deployment** (Phase 6), with SQLite WAL state, a host-side watchdog, Telegram alerting, and a RUNBOOK + threat model.
- A **CI-gated test suite** (Phase 7): unit tests, property-based tests (Hypothesis), replay regression, and state-recovery integration tests running on every push.

**It is not:**
- A framework for multiple assets/timeframes (currently configured for SOLUSDT 30m only).
- Financially advised. Run shadow/testnet mode before any real money.

---

## Current “registered” baseline

**Asset:** SOLUSDT perpetual (data is Binance OHLCV 30m)  
**Timezone:** UTC for everything  
**Signals:** only `downtrend_breakdown` rule is active in `strategy.py`  
**ORB window:** `13:30 → 14:00` (cutoff `14:00`)  
**ADX:** period `14`, threshold `43`  
**Execution assumption:** **next-candle open** (delay_bars=1)  
**Trading days:** only **valid days** (`48` bars present) are eligible for entries  
**Futures baseline defaults:** 1x leverage, MMR=0.5%, funding per 8h = 0.01% (configurable for research, but 1x is the baseline)

---

## Repo structure (high level)

- `config.yaml` — single source of truth for strategy params (symbol/timeframe/ORB/ADX/fees/risk).
- `strategy.py` — pure signal logic and indicator computation.
- `backtester/futures_engine.py` — isolated-margin futures simulator (fees/slippage/delay/funding/liquidation approx).
- `forward/` — live pipeline: data service, trader service, risk engine, stream engine, SQLite state store.
- `scripts/*.py` — runnable research scripts that write artifacts.
- `ops/watchdog.py` — host-side cron watchdog (heartbeat + Telegram alerts + optional restart).
- `scripts/emergency_flatten.py` — standalone emergency position flattener (works without main container).
- `tests/unit/` — unit tests (indicators, signals, fees/funding, SQLite state, risk controls, property-based).
- `tests/integration/` — replay regression + state-recovery integration tests.
- `.github/workflows/ci.yml` — CI pipeline (lint, type-check, unit tests, replay regression, Docker build, secret scan).
- `data/manifest.json` — describes **raw CSV dataset** location + per-file hashes.
- `data/processed/` — parquet dataset + valid/invalid day lists + processed manifest.
- `reports/` — outputs (baseline, scenarios, walk-forward, robustness, etc.).
- `RUNBOOK.md` — operational runbook (start/stop, emergency procedures, backup, threat model).

---

## Setup

### 1) Create & activate venv
```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

> `pyarrow` is required for parquet. It’s already listed in `requirements.txt`.

### 3) Install dev / test dependencies (optional, for running tests locally)
```bash
pip install -r requirements-dev.txt
```

Includes: `pytest`, `pytest-cov`, `hypothesis`, `ruff`, `mypy`.

---

## Data: how it’s versioned

### Raw data (CSV)
Raw files live outside git (usually). This repo tracks them via `data/manifest.json`:
- `data_root` (your local folder path)
- per-file sha256
- combined `dataset_sha256`

**If you move the raw folder**, update `data/manifest.json` (or regenerate it).

#### Build / refresh the raw manifest (recommended when raw data changes)
Use when: you added/removed CSVs or want a fresh dataset hash.

```bash
python scripts/hash_data.py --data-dir data/raw --patterns *.csv --out data/manifest.json
```

Outcome:
- Updates `data/manifest.json` with file hashes + `dataset_sha256`.

### Processed data (parquet + valid days)
The Phase-3 scripts use processed parquet + valid-day lists.

```bash
python scripts/build_parquet.py --raw-manifest data/manifest.json --out-dir data/processed
```

Outcome:
- `data/processed/SOLUSDT_30m.parquet`
- `data/processed/valid_days.csv` (valid day = exactly **48** bars present)
- `data/processed/invalid_days.csv`
- `data/processed/manifest.json` (hashes of processed outputs)

---

## Quickstart (first run)

If you want to reproduce the baseline + view a human-readable report:

```bash
python scripts/data_quality.py
python scripts/build_parquet.py
python scripts/run_baseline.py --engine futures --leverage 1 --mmr 0.005 --funding-per-8h 0.0001
python scripts/render_report.py
```

Open:
- `reports/baseline/report.html`

---

## Command cookbook (what to run, when, and what you get)

### Help for any script
Use when: you forget flags / defaults.

```bash
python scripts/<script_name>.py --help
```

---

## Phase 0 / 1 essentials

### 1) Data quality report
Use when: you want to confirm no weird timestamp gaps/dupes/misalignment.

```bash
python scripts/data_quality.py --manifest data/manifest.json --data-dir "" --out-dir reports/data_quality
```

Outcome:
- `reports/data_quality/quality.json`
- `reports/data_quality/quality.html`
- Includes missing bars, gaps sample, and **missing_by_day** list.

---

### 2) Build parquet + valid days
Use when: you want to run Phase-3 walk-forward / regime analysis.

```bash
python scripts/build_parquet.py --raw-manifest data/manifest.json --out-dir data/processed
```

Outcome:
- `data/processed/SOLUSDT_30m.parquet`
- `data/processed/valid_days.csv`
- `data/processed/invalid_days.csv`
- `data/processed/manifest.json`

---

### 3) Baseline backtest (spot engine)
Use when: you want a “spot-like” execution model (fees/slippage/delay, no funding/liquidation).

```bash
python scripts/run_baseline.py --engine spot
```

Common variations:
```bash
python scripts/run_baseline.py --engine spot --slippage-bps 1 --delay-bars 2
python scripts/run_baseline.py --engine spot --fee-mult 1.25
```

Outcome (in `reports/baseline/`):
- `results.json` (top-level metrics + params + dataset hash)
- `trades.csv`
- `equity_curve.csv` (+ `equity_curve.png`)
- `orb_ranges.csv`
- `run_metadata.json`
- `hashes.json`

---

### 4) Baseline backtest (futures engine)
Use when: you want futures execution effects (funding, liquidation approximation, margin).

```bash
python scripts/run_baseline.py --engine futures --leverage 1 --mmr 0.005 --funding-per-8h 0.0001
```

Common variations (research only):
```bash
python scripts/run_baseline.py --engine futures --leverage 1 --funding-per-8h 0
python scripts/run_baseline.py --engine futures --slippage-bps 3 --delay-bars 2
```

Outcome: same baseline folder artifacts, but includes futures fields in `results.json` (funding/liquidations/etc).

---

### 5) Render HTML report from latest baseline artifacts
Use when: you want a quick visual summary.

```bash
python scripts/render_report.py --in-dir reports/baseline --out reports/baseline/report.html
```

Outcome:
- `reports/baseline/report.html`

---

## Phase 2: cost / execution stress testing

### 6) Cost grid for both engines
Use when: you want to see sensitivity to fees/slippage/delay and (for futures) funding.

```bash
python scripts/run_cost_grid.py
```

Customize grids:
```bash
python scripts/run_cost_grid.py --fee-mults 1.0,1.25,1.5 --slippages-bps 0,1,3,5 --delays 1,2
python scripts/run_cost_grid.py --funding-list 0,0.0001,-0.0001 --leverage-list 1
```

Outcome:
- Per-scenario folders:
  - `reports/scenarios/spot/<scenario>/results.json`
  - `reports/scenarios/futures/<scenario>/results.json`
- Summary CSVs:
  - `reports/scenarios/grid_summary_spot.csv`
  - `reports/scenarios/grid_summary_futures.csv`
  - plus one CSV per funding value (e.g. `grid_summary_futures_funding_0p0001.csv`)

---

## Phase 3: robustness (walk-forward, benchmarks, bootstrap, regime)

### 7) Robustness table (parameter perturbations)
Use when: you want to see if small parameter changes destroy performance.

```bash
python scripts/robustness_table.py --out-dir reports/robustness
```

Outcome:
- `reports/robustness/robustness_table.csv`
- `reports/robustness/robustness_summary.json`
- `reports/robustness/run_metadata.json`

---

### 8) Bootstrap confidence intervals (equity curve resampling)
Use when: you want uncertainty estimates on performance metrics.

```bash
python scripts/bootstrap_ci.py --in-dir reports/baseline --out-dir reports/bootstrap
```

Outcome:
- `reports/bootstrap/bootstrap_report.json`
- `reports/bootstrap/bootstrap_samples.csv`

---

### 9) Benchmarks (sanity comparisons)
Use when: you want basic comparators (e.g., buy/hold-style, simple timing rules) under consistent cost assumptions.

```bash
python scripts/run_benchmarks.py --out-dir reports/benchmarks
```

Outcome:
- `reports/benchmarks/benchmarks_summary.csv`
- `reports/benchmarks/run_metadata.json`
- `reports/benchmarks/hashes.json`

---

### 10) Walk-forward evaluation (fixed params, optional light tuning)
Use when: you want rolling train/test windows (default: train=24m, test=6m, step=6m).

```bash
python scripts/walk_forward.py --engine futures --out-dir reports/walk_forward
```

Optional: tune only ADX threshold on TRAIN (reports TEST):
```bash
python scripts/walk_forward.py --engine futures --tune-adx-threshold 38,43,48 --out-dir reports/walk_forward
```

Outcome:
- `reports/walk_forward/walk_forward_folds.csv`
- `reports/walk_forward/walk_forward_report.html`
- `reports/walk_forward/walk_forward_metadata.json`

---

### 11) Walk-forward funding sweep wrapper
Use when: you want walk-forward stats across multiple funding values.

```bash
python scripts/walk_forward_sweep.py --funding-list 0,0.0001,-0.0001 --out-dir reports/walk_forward_sweep
```

Outcome:
- `reports/walk_forward_sweep/walk_forward_funding_sweep.csv`
- `reports/walk_forward_sweep/walk_forward_funding_summary.csv`

---

### 12) Nested walk-forward tuner (TRAIN-only selection, TEST evaluation)
Use when: you want to test whether “tuning” (e.g., ADX threshold / ORB start) survives out-of-sample.

```bash
python scripts/walk_forward_tune.py --engine futures --out-dir reports/walk_forward_tune
```

Common variants:
- Lock ORB start grid to **13:30 only** (recommended for avoiding ORB-start overfit):
```bash
python scripts/walk_forward_tune.py --engine futures --orb-start-grid 13:30 --out-dir reports/walk_forward_tune
```

Outcome:
- `reports/walk_forward_tune/walk_forward_tune_summary.csv`
- `reports/walk_forward_tune/selection_counts.csv`
- `reports/walk_forward_tune/aggregate_test_stats.json`
- `reports/walk_forward_tune/folds/` (per-fold artifacts)
- `reports/walk_forward_tune/run_metadata.json`

> Note: `scripts/walk_forward_tune.py` remains the legacy single-rule tuner. It does **not** generate the staged multi-rule workflow below.

---

### 12b) Multi-rule tuning workflow (4-rule manifest pipeline)
Use when: you want a reproducible staged search across the four registered rules, with concrete YAML snapshots, per-stage manifests, resumable execution, and deterministic leaderboard promotion.

Initialize a run root from the current four-rule baseline config:
```bash
python scripts/tune_manifest.py init --run-root reports/tuning_20260310 --config config.yaml
```

Generate later-stage manifests after the previous stage leaderboard exists:
```bash
python scripts/tune_manifest.py stage1 --run-root reports/tuning_20260310
python scripts/tune_manifest.py stage2 --run-root reports/tuning_20260310
python scripts/tune_manifest.py stage3 --run-root reports/tuning_20260310
python scripts/tune_manifest.py stage4 --run-root reports/tuning_20260310
python scripts/tune_manifest.py stage5 --run-root reports/tuning_20260310
python scripts/tune_manifest.py holdout --run-root reports/tuning_20260310
```

Run any generated stage manifest serially and resumably:
```bash
python scripts/tune_run.py --run-root reports/tuning_20260310 --stage stage1_marginal
python scripts/tune_run.py --run-root reports/tuning_20260310 --stage stage4_joint_fragility
```

Aggregate completed runs into leaderboard artifacts:
```bash
python scripts/tune_leaderboard.py --run-root reports/tuning_20260310 --stage stage1_marginal
python scripts/tune_leaderboard.py --run-root reports/tuning_20260310 --stage stage4_joint_fragility
```

Workflow stages:
- `baseline` â€” fixed development walk-forward on the baseline 4-rule config.
- `fragility` â€” local baseline robustness neighborhood using `robustness_table.py`.
- `stage1_marginal` â€” coarse one-rule-at-a-time sweep scored on the full 4-rule strategy.
- `stage1_isolated` â€” coarse single-rule diagnostic sweep; diagnostic only, no promotion.
- `stage2_marginal` â€” fine one-rule-at-a-time sweep around the best Stage 1 passing candidate per rule.
- `stage3_joint` â€” Cartesian recombination of the top 3 Stage 2 candidates per rule.
- `stage4_joint_fragility` â€” local robustness check for the top 5 Stage 3 joint configs.
- `stage5_order` â€” all 24 rule-order permutations for the top fragility-passing joint configs.
- `holdout` â€” fixed untouched holdout evaluation for the top 3 ordered finalists.

Run-root layout:
- `run_settings.json`
- `scenario_manifest.csv`
- `<stage>/manifest.csv`
- `<stage>/configs/<run_id>.yaml`
- `<stage>/runs/<run_id>/...`
- `<stage>/leaderboard.csv`
- `<stage>/summary.json`
- `<stage>/selected.csv`

Selection rules:
- Walk-forward stages rank by `hard_pass`, `median_test_daily_sharpe`, `positive_folds`, `median_test_total_return_pct`, `worst_fold_max_drawdown_pct`, `std_test_total_return_pct`, `total_test_trades`, `distance_from_baseline`, then `scenario_id`.
- Fragility stages require `neighbor_median_sharpe_ratio >= 0.85`, `neighbor_positive_share >= 0.60`, and `neighbor_dd_buffer >= -5`.
- The workflow is intentionally scoped to the registered `SOLUSDT 30m` four-rule baseline in canonical order metadata: `uptrend_reversion`, `downtrend_reversion`, `downtrend_breakdown`, `uptrend_continuation`.

---

### 13) Regime analysis on baseline trades
Use when: you want to understand which day “regimes” drive results (volatility buckets, ADX buckets, day-of-week, etc).

```bash
python scripts/regime_analysis.py --funding-per-8h 0
```

Outcome:
- `reports/regime/regime_summary.json`
- Multiple CSV breakdowns:
  - `by_volatility.csv`, `by_adx_bucket.csv`, `by_day_of_week.csv`, `by_month.csv`, etc.
- `reports/regime/trades_enriched.csv` (baseline trades with derived features)

---

### 14) Walk-forward regime filter experiment (TRAIN-derived thresholds, applied to TEST)
Use when: you want to test “skip bad days” rules without leaking future info.

```bash
python scripts/walk_forward_regime_filter.py --out-dir reports/walk_forward_regime_filter
```

Outcome:
- `reports/walk_forward_regime_filter/walk_forward_regime_filter_summary.csv`
- `reports/walk_forward_regime_filter/blocked_days.csv`
- `reports/walk_forward_regime_filter/run_metadata.json`

> Note: In our Phase-3 results, regime filters did **not** beat the fixed baseline consistently, so they should remain **off by default** unless revalidated.

---

## Where results live (by default)

- Baseline: `reports/baseline/`
- Data quality: `reports/data_quality/`
- Cost scenarios: `reports/scenarios/`
- Robustness grid: `reports/robustness/`
- Bootstrap: `reports/bootstrap/`
- Benchmarks: `reports/benchmarks/`
- Walk-forward: `reports/walk_forward/`
- Walk-forward tuning: `reports/walk_forward_tune/`
- Multi-rule tuning: user-chosen run root, e.g. `reports/tuning_20260310/`
- Walk-forward sweep: `reports/walk_forward_sweep/`
- Regime analysis: `reports/regime/`
- Walk-forward regime filter: `reports/walk_forward_regime_filter/`

---

## Determinism & reproducibility notes

- Key scripts set `PYTHONHASHSEED=0` and seed RNGs (Python + NumPy).
- Most outputs include:
  - dataset hashes (raw manifest `dataset_sha256`)
  - run metadata (OS/Python/Git commit)
  - parameters used

If you change **anything** (data, config, code), expect hashes and metrics to change.

---

## Troubleshooting

- **“Raw data directory not found”**  
  Update `data/manifest.json` → `data_root`, or run:
  ```bash
  python scripts/run_baseline.py --data-dir /absolute/path/to/raw/csvs
  ```

- **Parquet build fails**  
  Ensure `pyarrow` installed:
  ```bash
  pip install pyarrow
  ```

- **Too few trades in tuning**  
  The tuner enforces `--min-trades` (default 20). Lower it only for diagnostics.

---

---

## Phase 5 — Forward test (paper/shadow)

Phase 5 validates the end-to-end pipeline in live-like conditions.

Run a deterministic replay forward-test (Step 2: replay + shadow execution):
```bash
python scripts/forward_test.py --config config_forward_test.yaml --mode shadow --source replay
```

This creates a timestamped run folder under `reports/forward_test/` with:

- `signals.csv` — strategy signals (timestamped)
- `orders.csv` — shadow “orders” (scheduled at signal time; filled if executed)
- `fills.csv` — hypothetical fills (entry/exit)
- `positions.csv` — basic position snapshots (entry/exit)
- `events.jsonl` — structured event log (signals/fills/risk events)
- `shadow_stats.json` — engine stats (fees, liquidations, risk snapshot)

Generate the **Phase 5 Step 5 divergence report** (HTML + JSON) for a run:

```bash
python scripts/forward_test_report.py --run-id <RUN_ID>
```

Outputs (inside the run folder):
- `forward_test_report.json`
- `forward_test_report.html`

Optional: run only a window of the dataset:
```bash
python scripts/forward_test.py --config config_forward_test.yaml --mode shadow --source replay --start 2025-01-01 --end 2025-06-30
```

Acceptance check (Phase 5 / Step 2): compare forward-test vs baseline

```bash
# Compare the latest forward-test run folder automatically
python scripts/compare_forward_vs_baseline.py

# Or specify a run id
python scripts/compare_forward_vs_baseline.py --run-id 20260214T222953Z

# Strict mode also requires qty + fees to match (use only when leverage/sizing configs are identical)
python scripts/compare_forward_vs_baseline.py --run-id 20260214T222953Z --strict
```

### Step 3: Live ingestion (shadow trading on live market)

This step connects to Binance **live** kline data and writes the same artifacts as replay mode.

1) Edit `config_forward_test.yaml` and set:
- `forward_test.source: live`
- (optional) adjust `forward_test.live.*` settings

2) Run:
```bash
python scripts/forward_test.py --config config_forward_test.yaml --mode shadow --source live
```

Useful stop controls:
```bash
# Stop after 10 closed candles
python scripts/forward_test.py --config config_forward_test.yaml --mode shadow --source live --max-bars 10

# Stop after 180 minutes
python scripts/forward_test.py --config config_forward_test.yaml --mode shadow --source live --duration-minutes 180
```

Acceptance check (Phase 5 / Step 3):
- candles arrive on schedule (every 30m)
- no duplicate candle processing (check `events.jsonl` for `BAR_DUPLICATE_IGNORED`)
- simulate a disconnect (disable internet or block the domain), then restore it:
  - the runner should reconnect and resume without duplicating candles
  - `events.jsonl` will show heartbeat warnings if data pauses

### Step 4: Testnet broker (optional second mode)

This step adds a **Binance USD-M futures TESTNET broker**. Market data is still taken from Binance live klines.

**Secrets (required)**

Set these environment variables (never put keys in config files):

- `BINANCE_TESTNET_API_KEY`
- `BINANCE_TESTNET_API_SECRET`

**Smoke test (places a tiny testnet order and flattens it)**

1) Set config:
- `forward_test.mode: testnet`
- `forward_test.source: live`

2) Run:
```bash
python scripts/forward_test.py --config config_forward_test.yaml --mode testnet --source live --smoke-test
```

Check `reports/forward_test/<RUN_ID>/events.jsonl` for:
- `TESTNET_SMOKE_START`
- `TESTNET_SMOKE_ENTRY`
- `TESTNET_SMOKE_FLATTEN`
- `LIVE_RUN_END`

**Live testnet run (strategy-driven)**

```bash
python scripts/forward_test.py --config config_forward_test.yaml --mode testnet --source live --duration-minutes 180
```

Acceptance check (Phase 5 / Step 4):
- you can place a tiny testnet order (smoke test)
- reconciliation confirms exchange position matches internal state (see `state.json` + `positions.csv`)
- forced restart resumes safely:
  - stop the program, then re-run with the same `--run-id` and it should load `state.json` and reconcile

## Docker run (Step 6.4)

```bash
docker build -t crypto-orb-trader .
# Create .env from RUNBOOK template, then fill in secrets (never commit this file)
docker compose up -d
docker compose logs -f trader
docker compose down
# Verify state persisted:
docker compose run --rm trader ls -la /data
```

### Phase 6 — Host-side watchdog (Telegram alerts + optional restart)

This repo includes a **host-side watchdog** that monitors the running `trader` container and the `/data` volume heartbeat, and sends **Telegram alerts** on incident transitions (stale → recovered). It is designed to be **non-spammy**.

#### What it monitors

* Resolves the container via `docker compose` for service `WATCHDOG_SERVICE_NAME` (default: `trader`)
* Resolves the host `/data` directory by inspecting the container’s `/data` mount
* Reads:

  * `/data/heartbeat` (freshness)
  * `/data/state.db` (optional, for trade_log-based checks if enabled in your watchdog)

#### Why it’s often silent

The watchdog is intentionally **stateful** and alerts only on **transitions**:

* **healthy → stale**: sends **one** “Heartbeat stale…” alert and sets `stale_since`
* **stale → healthy**: sends **one** “Heartbeat recovered…” alert and clears `stale_since`
* If it stays stale or stays healthy: **no repeated alerts** (silent is expected)

This is why you may see **no stdout output** and still get exit code `0`.

#### Files and paths (host)

* Env file (root-only): `/etc/watchdog.env`
* Wrapper: `/usr/local/bin/run_watchdog.sh`
* Script: `/home/ubuntu/watchdog.py`
* Default state file: `/home/ubuntu/.watchdog_state.json`

  * You can override with `WATCHDOG_STATE_PATH=/tmp/...` for testing.

#### Example `/etc/watchdog.env`

Keep secrets here, plus compose directory:

* `TELEGRAM_BOT_TOKEN="..."`
* `TELEGRAM_CHAT_ID="..."`
* `WATCHDOG_COMPOSE_DIR="/home/ubuntu/crypto-orb-research"`

Optional knobs (can live here, but note overrides below):

* `WATCHDOG_HEARTBEAT_STALE_SECONDS="600"`
* `WATCHDOG_RESTART_ON_STALE="1"`
* `WATCHDOG_RESTART_GRACE_SECONDS="300"`
* `WATCHDOG_SERVICE_NAME="trader"`
* `WATCHDOG_COMPOSE_FILE="docker-compose.yml"`

#### Wrapper behavior (important for overrides)

`run_watchdog.sh` sources `/etc/watchdog.env`. If you set `WATCHDOG_HEARTBEAT_STALE_SECONDS` in that file, then a command like:

`WATCHDOG_HEARTBEAT_STALE_SECONDS=5 /usr/local/bin/run_watchdog.sh`

may be overwritten by the env file (depending on how/when it’s sourced). For one-off testing, prefer the “source then export overrides” pattern below.

---

### Testing alerts (recommended)

#### 1) Confirm it runs (may be quiet)

```bash
sudo /usr/local/bin/run_watchdog.sh --dry-run; echo "exit_code=$?"
```

Exit code `0` with no output is normal when healthy/no transition.

#### 2) Force a *stale* alert safely (fresh state file test)

This forces a clean transition by using a new state file. Works even if your normal state is already “stale” or “healthy”.

```bash
cd /home/ubuntu/crypto-orb-research
sudo docker compose stop trader
sleep 20

STATE="/tmp/watchdog_stoptest_$(date -u +%s).json"
sudo bash -lc "
set -a; source /etc/watchdog.env; set +a
export WATCHDOG_STATE_PATH='$STATE'
export WATCHDOG_HEARTBEAT_STALE_SECONDS=5
export WATCHDOG_RESTART_ON_STALE=0
cd /home/ubuntu/crypto-orb-research
/usr/bin/python3 /home/ubuntu/watchdog.py
"
```

Expected: one Telegram “Heartbeat stale…” alert.

#### 3) Force a *recovery* alert

```bash
cd /home/ubuntu/crypto-orb-research
sudo docker compose start trader
sleep 30

sudo bash -lc "
set -a; source /etc/watchdog.env; set +a
export WATCHDOG_STATE_PATH='$STATE'
export WATCHDOG_HEARTBEAT_STALE_SECONDS=5
export WATCHDOG_RESTART_ON_STALE=0
cd /home/ubuntu/crypto-orb-research
/usr/bin/python3 /home/ubuntu/watchdog.py
"
```

Expected: one Telegram “Heartbeat recovered…” alert (may take a few tries if the app takes time to write the first heartbeat after startup).

#### Avoid flapping in production

Do **not** use tiny stale thresholds like `5s` outside testing. Set `WATCHDOG_HEARTBEAT_STALE_SECONDS` comfortably above the normal heartbeat interval (common values: `120–600` seconds).

---

### Cron installation and setup (Ubuntu)

#### Install cron

```bash
sudo apt-get update
sudo apt-get install -y cron
sudo systemctl enable --now cron
```

#### Cron entry (recommended: `/etc/cron.d/crypto_orb_watchdog`)

```bash
sudo tee /etc/cron.d/crypto_orb_watchdog >/dev/null <<'EOF'
*/5 * * * * root /usr/local/bin/run_watchdog.sh >> /var/log/watchdog.log 2>&1
EOF
sudo chmod 644 /etc/cron.d/crypto_orb_watchdog
sudo systemctl restart cron
```

#### Verify cron is running

```bash
sudo journalctl -u cron --no-pager -n 50
sudo tail -n 200 /var/log/watchdog.log
```

**Note:** `watchdog.log` can be empty when everything is healthy.

#### Prevent duplicate cron jobs

Make sure you **do not** also have a root crontab entry running watchdog directly (this can cause confusing logs / wrong working directory):

```bash
sudo crontab -l
```

If you see a watchdog line there, remove it and rely on `/etc/cron.d/crypto_orb_watchdog`.

---

## Phase 7 — Testing & CI

Phase 7 adds a CI-gated test suite covering the most failure-prone logic in the pipeline. All tests run on every push to `main` and every pull request via GitHub Actions.

### Running tests locally

```bash
# Install dev dependencies first
pip install -r requirements-dev.txt

# Unit tests only (fast, no network, no data files required)
pytest tests/unit -v

# Replay regression (requires data/processed/SOLUSDT_30m.parquet)
pytest tests/integration/test_replay.py -v

# State recovery integration tests (POSIX only for SIGKILL test)
pytest tests/integration/test_state_recovery.py -v

# All tests with coverage report
pytest tests/ -v --cov=core --cov=forward --cov=backtester --cov=strategy --cov-report=term-missing
```

If `data/processed/SOLUSDT_30m.parquet` is not present, the replay regression test skips automatically with a clear message. Run `python scripts/build_parquet.py` to generate it.

### What is tested

**Unit tests** (`tests/unit/`):
- `test_indicators.py` — ADX flat-market stability, trend-direction sanity, ORB boundary detection
- `test_signals.py` — post-cutoff rule, one-signal-per-day lock, partial-condition non-firing
- `test_fees_funding.py` — deterministic round-trip fee + funding math against manual formula
- `test_state_store_sqlite.py` — WAL crash-safety: write → crash-close → reopen → exact equality
- `test_risk_controls.py` — daily loss halt, drawdown halt, consecutive-loss reset on UTC day boundary
- `test_forward_risk_engine.py` — kill-switch boundary semantics (`>` vs `>=` for each check)
- `test_properties_futures.py` — Hypothesis properties: position size cap, accounting identity, no-lookahead entries
- `test_properties_risk.py` — Hypothesis properties: consecutive-loss day semantics, kill-switch latch

**Integration tests** (`tests/integration/`):
- `test_replay.py` — replay regression: batch backtest path == streaming shadow-engine path (trade count, entry/exit timestamps, per-trade PnL, final equity)
- `test_state_recovery.py` — partial fill handling, rejection counter, reconnect dedupe, SIGKILL restart without re-entry (POSIX only)

### CI pipeline (`.github/workflows/ci.yml`)

Runs on every push to `main` and every pull request. All steps except coverage threshold are **blocking**.

| Step | Command | Blocks? |
|---|---|---|
| Lint | `ruff check .` | Yes |
| Type check | `mypy core forward --strict ...` | Yes |
| Unit tests | `pytest tests/unit` | Yes |
| Coverage threshold | `coverage report --fail-under=80` | Warning only |
| Replay regression | `pytest tests/integration/test_replay.py` | Yes |
| State recovery | `pytest tests/integration/test_state_recovery.py` | Yes |
| Docker build | `docker build --no-cache .` | Yes |
| Secret scan | `gitleaks` | Yes |

> The replay regression test skips in CI if the parquet data file is not present (it is not committed to the repo). This is expected — the test is meaningful locally and in environments where the data pipeline has been run.

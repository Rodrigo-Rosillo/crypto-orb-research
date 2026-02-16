# crypto-orb-research

Research repo for an Opening Range Breakout (ORB) strategy on **SOLUSDT 30m** Binance data (UTC), with **reproducible** backtests, **spot + isolated-margin futures** execution models, and Phase-3 robustness (walk-forward, bootstraps, benchmarks, regime diagnostics).

> **Educational/research only.** Nothing here is financial advice. If you ever run this live, start with paper/shadow mode and strict risk limits.

---

## What this repo is (and is not)

**It is:**
- A “research artifact” where rerunning scripts produces auditable outputs (JSON/CSV/HTML) under `reports/`.
- A pipeline from **raw CSV** → **data quality checks** → **processed parquet + valid-day list** → **baseline backtest** → **stress/robustness/walk-forward**.

**It is not:**
- A production trading bot (Phase 4 covers that separately).
- A framework for multiple assets/timeframes (it’s currently configured for SOLUSDT 30m).

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
- `scripts/spot_engine.py` — spot execution simulator (fees/slippage/delay).
- `scripts/*.py` — runnable research scripts that write artifacts.
- `data/manifest.json` — describes **raw CSV dataset** location + per-file hashes.
- `data/processed/` — parquet dataset + valid/invalid day lists + processed manifest.
- `reports/` — outputs (baseline, scenarios, walk-forward, robustness, etc.).

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

---

### 13) Regime analysis on baseline trades
Use when: you want to understand which day “regimes” drive results (volatility buckets, ADX buckets, day-of-week, etc).

```bash
python scripts/regime_analysis.py --in-dir reports/baseline --out-dir reports/regime
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


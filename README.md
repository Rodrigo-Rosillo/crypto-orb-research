# crypto-orb-research

Research repo for testing an Opening Range Breakout (ORB) strategy on Binance OHLCV data, with deterministic outputs and two execution models (spot + isolated-margin futures).

## Quick mental model

The workflow is:

1. **Raw CSV dataset** is described by `data/manifest.json`.
2. `scripts/build_parquet.py` builds a normalized parquet dataset and valid/invalid trading-day lists under `data/processed/`.
3. `scripts/run_baseline.py` loads config + data, computes indicators/signals, runs either spot or futures backtest, and writes reports into `reports/baseline/`.
4. Optional scripts generate data-quality reports, cost-stress scenario grids, and an HTML summary report.

## Project structure

- `config.yaml` — central strategy + risk configuration (symbol/timeframe, ORB window, ADX settings, fees/risk).  
- `strategy.py` — pure signal logic:
  - ADX/+DI/-DI calculation
  - ORB high/low extraction
  - signal generation rules (currently only `downtrend_breakdown` is active)
- `backtester/futures_engine.py` — futures execution simulator with:
  - isolated margin
  - leverage
  - fees/slippage/delay
  - optional funding
  - liquidation approximation
- `scripts/spot_engine.py` — spot-style execution simulator used by baseline when `--engine spot`.
- `scripts/run_baseline.py` — main orchestrator and primary entrypoint for a research run.
- `scripts/build_parquet.py` — preprocessing + valid day classification.
- `scripts/data_quality.py` — checks dataset integrity and emits JSON/HTML report.
- `scripts/run_cost_grid.py` — parameter sweep for fee/slippage/delay (+ leverage/funding for futures).
- `scripts/render_report.py` — turns baseline outputs into `report.html`.
- `data/` — manifests and processed dataset artifacts.
- `reports/` — output artifacts from baseline/quality/scenario runs.

## Important concepts newcomers should know

- **Determinism is intentional:** baseline script locks hash seed and RNG seeds so repeated runs are reproducible.
- **Day validity matters:** trades are only allowed on “valid days” (days with expected bar count).
- **Engine switch:** same signals can be executed via `spot` or `futures` engine (`--engine`).
- **Signal scope right now:** in `generate_orb_signals`, the other two rules are disabled and only downtrend breakdown is active.
- **Reports are artifact-first:** scripts write JSON/CSV/HTML artifacts under `reports/` so results can be inspected without rerunning analysis.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Useful commands

### 1) Build processed dataset and valid-day lists

```bash
python scripts/build_parquet.py
```

Useful flags:

```bash
python scripts/build_parquet.py --help
python scripts/build_parquet.py --raw-manifest data/manifest.json --out-dir data/processed
```

### 2) Run baseline backtest (spot)

```bash
PYTHONHASHSEED=0 python scripts/run_baseline.py --engine spot
```

### 3) Run baseline backtest (futures)

```bash
PYTHONHASHSEED=0 python scripts/run_baseline.py --engine futures --leverage 1 --mmr 0.005 --funding-per-8h 0.0001
```

### 4) Generate a data quality report

```bash
python scripts/data_quality.py
```

### 5) Sweep cost assumptions (scenario grid)

```bash
python scripts/run_cost_grid.py
```

This writes per-scenario outputs under `reports/scenarios/` plus summary CSVs.

### 6) Render an HTML report from baseline artifacts

```bash
python scripts/render_report.py
```

## What to inspect after a run

- `reports/baseline/results.json` — top-level metrics and parameters.
- `reports/baseline/trades.csv` — trade-by-trade outcomes.
- `reports/baseline/equity_curve.csv` — timestamped equity evolution.
- `reports/baseline/report.html` — human-readable summary page.

## Recommended “first day” workflow

1. Open `config.yaml` and confirm symbol, timeframe, ORB times, ADX threshold.
2. Run `python scripts/build_parquet.py`.
3. Run `PYTHONHASHSEED=0 python scripts/run_baseline.py --engine spot`.
4. Run `python scripts/render_report.py` and inspect `reports/baseline/report.html`.
5. Compare with `--engine futures` to understand execution-model sensitivity.

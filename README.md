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
# pyarrow is required for parquet. It’s already listed in requirements.txt.
```

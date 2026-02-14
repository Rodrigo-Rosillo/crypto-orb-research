# Phase 4 — Risk controls policy (survivable futures)

This repo is a **research artifact**. These controls are meant to make futures-style testing survivable and to serve as the blueprint for live risk governance.

All logic is **UTC**.

## What is enforced (hard controls)

### 1) Max position size + max leverage
- **Max position size** is enforced as a cap on **initial margin used per trade**: `max_position_margin_frac` (fraction of current equity).
- **Max leverage** is a hard cap: if the engine is configured with higher leverage, it is clamped down.

### 2) Max daily loss (stop trading for the day)
- If equity falls below `day_start_equity * (1 - max_daily_loss_pct)` then the engine:
  - halts **new entries** for that UTC day
  - continues to manage existing positions (and may flatten if other limits trigger)

### 3) Max drawdown circuit breaker (stop until manual review)
- If equity falls below `peak_equity * (1 - max_drawdown_pct)` then the engine:
  - halts globally (no new entries)
  - attempts to **flatten immediately** at market (next available bar open in backtest)

### 4) Max consecutive losses / max exposure duration
- After `max_consecutive_losses` losing trades in a row, stop trading for that UTC day.
- If a position is held for `max_exposure_bars` bars or more, it is force-closed at market.

### 5) Kill switch
Backtest approximations for live-safety events:
- **Data feed stale**: if the gap between bars exceeds `max_data_gap_bars * expected_bar_seconds`, halt globally and flatten.
- **Order rejects**: counts blocked/failed entries per day; if they exceed `max_order_rejects_per_day`, halt globally.
- **Margin ratio spike**: if `maintenance_margin / margin_balance` exceeds `max_margin_ratio`, halt globally and flatten.

## Configuration

- Default research config: `config.yaml` (risk controls **disabled**)
- Phase 4 preset: `config_phase4.yaml` (risk controls **enabled**)

## Where it lives in code

- Risk logic: `backtester/risk.py`
- Futures engine enforcement: `backtester/futures_engine.py`

## How to run with Phase 4 controls

Baseline (spot engine):
```bash
python scripts/run_baseline.py --engine spot --config config_phase4.yaml
```

Futures (example, still deterministic):
```bash
python scripts/run_baseline.py --engine futures --config config_phase4.yaml --leverage 2 --funding-per-8h 0.0001
```

The futures `results.json` includes an `engine_stats.risk` section with the full event log and halt reasons.

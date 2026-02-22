from __future__ import annotations

import argparse
import sys
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.utils import parse_hhmm, sha256_file, stable_json  # noqa: E402
from strategy import add_trend_indicators  # noqa: E402


# ---------------------------
# Utilities
# ---------------------------
def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Dataset must have a DatetimeIndex (UTC candle open timestamps).")
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    return df.sort_index()


def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def day_bucket_from_date(d: pd.Timestamp) -> str:
    # Mon..Sun
    return d.day_name()


def time_of_day_bucket(ts: pd.Timestamp) -> str:
    h = int(ts.hour)
    if 0 <= h <= 5:
        return "00-05"
    if 6 <= h <= 11:
        return "06-11"
    if 12 <= h <= 17:
        return "12-17"
    return "18-23"


# ---------------------------
# Regime features (day-level)
# ---------------------------
def compute_daily_realized_vol(df: pd.DataFrame) -> pd.Series:
    """
    Realized volatility proxy from intraday log returns:
      rv_day = sqrt(sum_{intraday} (logret^2))
    Indexed by python date.
    """
    close = df["close"].astype(float)
    logret = np.log(close).diff()

    tmp = pd.DataFrame({"logret": logret})
    tmp["date"] = tmp.index.date

    # Sum of squared intraday returns per day
    rv2 = tmp.groupby("date")["logret"].apply(lambda x: float(np.nansum(np.square(x.values))))
    rv = np.sqrt(rv2)
    return rv


def compute_day_features(
    df_ind: pd.DataFrame,
    valid_days: set,
    orb_end: time,
    adx_threshold: float,
    vol_lookback_days: int,
) -> pd.DataFrame:
    """
    Returns day_features indexed by python date with:
      - vol_score (trailing realized vol)
      - vol_bucket (low/med/high)
      - adx_at_orb_end
      - adx_bucket (low/med/high)
      - trend_label_at_orb_end (uptrend/downtrend/sideways)
      - trend_binary (trend/chop based on adx_threshold)
    """
    df_ind = ensure_utc_index(df_ind)

    # Only consider valid days for bucketing (matches your backtest policy)
    day_index = pd.Index(sorted(valid_days), name="date")

    # Volatility score: trailing mean realized vol (shifted 1 day to avoid leakage)
    rv = compute_daily_realized_vol(df_ind)
    rv = rv.reindex(day_index)  # align to valid days
    vol_score = rv.rolling(vol_lookback_days, min_periods=vol_lookback_days).mean().shift(1)

    # ADX snapshot at orb_end (use exact timestamp if present; otherwise last prior candle)
    # Build a timestamp per day at orb_end.
    day_ts = pd.to_datetime(
        [datetime(d.year, d.month, d.day, orb_end.hour, orb_end.minute, tzinfo=timezone.utc) for d in day_index]
    )
    snap = df_ind.reindex(day_ts, method="ffill")[["adx", "trend"]].copy()
    snap.index = day_index
    snap = snap.rename(columns={"adx": "adx_at_orb_end", "trend": "trend_label_at_orb_end"})

    out = pd.DataFrame(index=day_index)
    out["vol_score"] = vol_score
    out["adx_at_orb_end"] = snap["adx_at_orb_end"]
    out["trend_label_at_orb_end"] = snap["trend_label_at_orb_end"]

    # Binary trend/chop using the SAME threshold as strategy (default 43)
    out["trend_binary"] = np.where(out["adx_at_orb_end"] >= float(adx_threshold), "trend", "chop")

    # Buckets (low/med/high) using qcut over available values
    def qcut3(s: pd.Series, label_base: str) -> pd.Series:
        s2 = s.dropna()
        if len(s2) < 10:
            # Too few values to bucket meaningfully
            return pd.Series(index=s.index, dtype="object")
        b = pd.qcut(s2, q=[0.0, 1 / 3, 2 / 3, 1.0], labels=["low", "med", "high"])
        return b.reindex(s.index)

    out["vol_bucket"] = qcut3(out["vol_score"], "vol")
    out["adx_bucket"] = qcut3(out["adx_at_orb_end"], "adx")

    return out


# ---------------------------
# Funding attribution (per trade)
# ---------------------------
def is_funding_bar(ts: pd.Timestamp) -> bool:
    # Binance funding times: 00:00, 08:00, 16:00 UTC
    return ts.minute == 0 and ts.second == 0 and ts.microsecond == 0 and ts.hour in (0, 8, 16)


def estimate_trade_funding(
    df_prices: pd.DataFrame,
    entry_ts: pd.Timestamp,
    exit_ts: pd.Timestamp,
    qty: float,
    side: str,
    funding_per_8h: float,
) -> float:
    """
    Approximate funding using the same sign convention as futures_engine:
      pay = notional * rate * (1 if long else -1)
      total_funding += pay  (positive => paid, negative => received)

    Important timing detail to match engine:
    - funding at timestamp occurs BEFORE entry on that bar
      => exclude ts == entry_ts
    - funding at timestamp occurs BEFORE exit management on that bar
      => include ts == exit_ts if funding bar
    So we sum funding bars with: entry_ts < ts <= exit_ts
    """
    if funding_per_8h == 0.0 or qty <= 0:
        return 0.0

    side = side.upper()
    if side not in ("LONG", "SHORT"):
        return 0.0

    # Candidate funding timestamps within range
    idx = df_prices.index
    # slice with a slightly wider range then filter precisely
    window = df_prices[(idx > entry_ts) & (idx <= exit_ts)]
    if window.empty:
        return 0.0

    fbars = window[window.index.map(is_funding_bar)]
    if fbars.empty:
        return 0.0

    # Engine uses bar open as mark proxy
    prices = fbars["open"].astype(float).values
    notionals = float(qty) * prices
    rate = float(funding_per_8h)

    # + means pay for longs, - means receive; opposite for shorts
    sign = 1.0 if side == "LONG" else -1.0
    pay = notionals * rate * sign
    return float(np.nansum(pay))


# ---------------------------
# Group summaries
# ---------------------------
def profit_factor(pnl: pd.Series) -> float:
    wins = pnl[pnl > 0].sum()
    losses = pnl[pnl < 0].sum()
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / abs(losses))


def summarize_group(df: pd.DataFrame, initial_capital: float) -> Dict[str, Any]:
    pnl = df["pnl_net"].astype(float)

    trades = int(len(df))
    wins = int((pnl > 0).sum())
    losses = int((pnl <= 0).sum())
    win_rate = float((wins / trades) * 100.0) if trades else 0.0

    total_pnl = float(pnl.sum())
    avg_pnl = float(pnl.mean()) if trades else 0.0
    med_pnl = float(pnl.median()) if trades else 0.0

    total_fees = float(df["fees_total"].astype(float).sum()) if "fees_total" in df.columns else 0.0
    total_funding = float(df["funding_est"].astype(float).sum()) if "funding_est" in df.columns else 0.0

    hold_min = float(df["hold_minutes"].astype(float).mean()) if "hold_minutes" in df.columns and trades else 0.0

    liqs = int((df["exit_reason"] == "liquidation").sum()) if "exit_reason" in df.columns else 0

    return {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": win_rate,
        "total_pnl_net": total_pnl,
        "total_return_pct_on_initial": float((total_pnl / initial_capital) * 100.0) if initial_capital else 0.0,
        "avg_pnl_net": avg_pnl,
        "median_pnl_net": med_pnl,
        "profit_factor": profit_factor(pnl),
        "avg_hold_minutes": hold_min,
        "total_fees": total_fees,
        "total_funding_est": total_funding,
        "liquidations": liqs,
    }


def grouped_table(df: pd.DataFrame, group_col: str, initial_capital: float) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_col, dropna=False):
        row = {"group": str(key)}
        row.update(summarize_group(g, initial_capital))
        rows.append(row)
    out = pd.DataFrame(rows)
    # Sort: most trades first, then best pnl
    if not out.empty:
        out = out.sort_values(by=["trades", "total_pnl_net"], ascending=[False, False]).reset_index(drop=True)
    return out


# ---------------------------
# Main
# ---------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Regime / fragility analysis for baseline trades (Phase 3 Step 5).")

    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data", default="", help="Parquet path. Default: data/processed/<symbol>_<timeframe>.parquet")
    ap.add_argument("--valid-days", default="data/processed/valid_days.csv")

    ap.add_argument("--trades", default="reports/baseline/trades.csv")
    ap.add_argument("--out-dir", default="reports/regime")

    ap.add_argument("--vol-lookback-days", type=int, default=30)
    ap.add_argument("--funding-per-8h", type=float, default=0.0001)

    args = ap.parse_args()

    config_path = (REPO_ROOT / args.config).resolve()
    cfg_text = config_path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(cfg_text) or {}

    symbol = str(cfg.get("symbol", "SOLUSDT"))
    timeframe = str(cfg.get("timeframe", "30m"))
    orb_end = parse_hhmm(str(cfg["orb"]["end"]))
    adx_period = int(cfg["adx"]["period"])
    adx_threshold = float(cfg["adx"]["threshold"])
    initial_capital = float(cfg["risk"]["initial_capital"])

    data_path = Path(args.data) if args.data else Path(f"data/processed/{symbol}_{timeframe}.parquet")
    data_path = (REPO_ROOT / data_path).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Parquet not found: {data_path}. Run: python scripts/build_parquet.py")

    valid_days_path = (REPO_ROOT / args.valid_days).resolve()
    if not valid_days_path.exists():
        raise FileNotFoundError(f"Valid days not found: {valid_days_path}. Run: python scripts/build_parquet.py")

    trades_path = (REPO_ROOT / args.trades).resolve()
    if not trades_path.exists():
        raise FileNotFoundError(f"Trades CSV not found: {trades_path}. Run: python scripts/run_baseline.py")

    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df = pd.read_parquet(data_path)
    df = ensure_utc_index(df)

    needed = {"open", "high", "low", "close", "volume"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Parquet missing required columns: {sorted(missing)}")

    df = df[["open", "high", "low", "close", "volume"]].copy()

    # Load valid days
    vdf = pd.read_csv(valid_days_path)
    if "date_utc" not in vdf.columns:
        raise ValueError(f"{valid_days_path} must contain column 'date_utc'")
    valid_days = set(pd.to_datetime(vdf["date_utc"], utc=True).dt.date)

    # Add indicators
    df_ind = add_trend_indicators(df, period=adx_period)

    # Day-level regimes
    day_features = compute_day_features(
        df_ind=df_ind,
        valid_days=valid_days,
        orb_end=orb_end,
        adx_threshold=adx_threshold,
        vol_lookback_days=int(args.vol_lookback_days),
    )
    day_features.to_csv(out_dir / "day_features.csv", index=True)

    # Load trades
    trades = pd.read_csv(trades_path)

    entry_col = pick_col(trades, ["entry_time", "entry_ts", "entry_timestamp", "entry_datetime"])
    exit_col = pick_col(trades, ["exit_time", "exit_ts", "exit_timestamp", "exit_datetime"])
    if entry_col is None or exit_col is None:
        raise ValueError(f"Trades file must include entry/exit time columns. Found columns: {list(trades.columns)}")

    pnl_col = pick_col(trades, ["pnl_net", "pnl", "net_pnl"])
    if pnl_col is None:
        raise ValueError("Trades file must include pnl column (pnl_net or pnl).")

    fees_col = pick_col(trades, ["fees_total", "fee_total", "fees", "total_fees"])
    if fees_col is None:
        # allow missing; we’ll set 0
        trades["fees_total"] = 0.0
        fees_col = "fees_total"

    qty_col = pick_col(trades, ["qty", "quantity"])
    side_col = pick_col(trades, ["type", "side", "position"])
    reason_col = pick_col(trades, ["exit_reason", "reason"])

    trades["entry_time_utc"] = pd.to_datetime(trades[entry_col], utc=True, errors="coerce")
    trades["exit_time_utc"] = pd.to_datetime(trades[exit_col], utc=True, errors="coerce")
    trades = trades.dropna(subset=["entry_time_utc", "exit_time_utc"]).copy()

    trades["pnl_net"] = pd.to_numeric(trades[pnl_col], errors="coerce").fillna(0.0)
    trades["fees_total"] = pd.to_numeric(trades[fees_col], errors="coerce").fillna(0.0)

    if reason_col is not None:
        trades["exit_reason"] = trades[reason_col].astype(str)
    else:
        trades["exit_reason"] = ""

    if side_col is not None:
        trades["side"] = trades[side_col].astype(str)
    else:
        trades["side"] = ""

    if qty_col is not None:
        trades["qty"] = pd.to_numeric(trades[qty_col], errors="coerce").fillna(0.0)
    else:
        trades["qty"] = 0.0

    # Enrich: time buckets
    trades["entry_date"] = trades["entry_time_utc"].dt.date
    trades["entry_dow"] = trades["entry_time_utc"].dt.day_name()
    trades["entry_hour"] = trades["entry_time_utc"].dt.hour.astype(int)
    trades["entry_time_bucket"] = trades["entry_time_utc"].apply(time_of_day_bucket)
    trades["entry_month"] = trades["entry_time_utc"].dt.to_period("M").astype(str)
    trades["hold_minutes"] = (trades["exit_time_utc"] - trades["entry_time_utc"]).dt.total_seconds() / 60.0

    # Join regimes by entry day
    reg = day_features.copy()
    reg = reg.reset_index().rename(columns={"date": "entry_date"})
    trades = trades.merge(reg, on="entry_date", how="left")

    # Estimate per-trade funding if possible (futures engine columns present)
    funding_rate = float(args.funding_per_8h)
    funding_est = []
    if funding_rate != 0.0 and "qty" in trades.columns and "side" in trades.columns:
        for _, r in trades.iterrows():
            f = estimate_trade_funding(
                df_prices=df_ind[["open"]],
                entry_ts=pd.Timestamp(r["entry_time_utc"]),
                exit_ts=pd.Timestamp(r["exit_time_utc"]),
                qty=float(r["qty"]),
                side=str(r["side"]),
                funding_per_8h=funding_rate,
            )
            funding_est.append(f)
    else:
        funding_est = [0.0] * len(trades)

    trades["funding_est"] = funding_est

    # Save enriched trade table (super useful for debugging)
    trades.to_csv(out_dir / "trades_enriched.csv", index=False)

    # ---------------------------
    # Group summaries (deliverable)
    # ---------------------------
    by_vol = grouped_table(trades.dropna(subset=["vol_bucket"]), "vol_bucket", initial_capital)
    by_vol.to_csv(out_dir / "by_volatility.csv", index=False)

    by_adx = grouped_table(trades.dropna(subset=["adx_bucket"]), "adx_bucket", initial_capital)
    by_adx.to_csv(out_dir / "by_adx_bucket.csv", index=False)

    by_trend_binary = grouped_table(trades.dropna(subset=["trend_binary"]), "trend_binary", initial_capital)
    by_trend_binary.to_csv(out_dir / "by_trend_vs_chop.csv", index=False)

    by_trend_label = grouped_table(trades.dropna(subset=["trend_label_at_orb_end"]), "trend_label_at_orb_end", initial_capital)
    by_trend_label.to_csv(out_dir / "by_trend_direction.csv", index=False)

    by_dow = grouped_table(trades, "entry_dow", initial_capital)
    by_dow.to_csv(out_dir / "by_day_of_week.csv", index=False)

    by_tod = grouped_table(trades, "entry_time_bucket", initial_capital)
    by_tod.to_csv(out_dir / "by_time_of_day.csv", index=False)

    by_month = grouped_table(trades, "entry_month", initial_capital)
    by_month.to_csv(out_dir / "by_month.csv", index=False)

    by_side = grouped_table(trades, "side", initial_capital)
    by_side.to_csv(out_dir / "by_side.csv", index=False)

    by_reason = grouped_table(trades, "exit_reason", initial_capital)
    by_reason.to_csv(out_dir / "by_exit_reason.csv", index=False)

    # Cross-tab: vol bucket x trend bucket (trade count + pnl)
    cross = (
        trades.dropna(subset=["vol_bucket", "adx_bucket"])
        .groupby(["vol_bucket", "adx_bucket"], dropna=False)
        .agg(trades=("pnl_net", "size"), total_pnl_net=("pnl_net", "sum"))
        .reset_index()
        .sort_values(["vol_bucket", "adx_bucket"])
    )
    cross.to_csv(out_dir / "crosstab_vol_x_adx.csv", index=False)

    # High-level summary JSON
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "timeframe": timeframe,
        "initial_capital": initial_capital,
        "assumptions": {
            "vol_lookback_days": int(args.vol_lookback_days),
            "orb_end": cfg["orb"]["end"],
            "adx_period": adx_period,
            "adx_threshold": adx_threshold,
            "funding_per_8h_used_for_estimate": funding_rate,
            "note": "Funding is estimated from candles + qty + funding schedule; matches engine timing: entry_ts < funding_ts <= exit_ts.",
        },
        "counts": {
            "trades_total": int(len(trades)),
            "days_with_vol_bucket": int(trades["vol_bucket"].notna().sum()),
            "days_with_adx_bucket": int(trades["adx_bucket"].notna().sum()),
        },
        "files": {
            "day_features.csv": "day_features.csv",
            "trades_enriched.csv": "trades_enriched.csv",
            "by_volatility.csv": "by_volatility.csv",
            "by_adx_bucket.csv": "by_adx_bucket.csv",
            "by_trend_vs_chop.csv": "by_trend_vs_chop.csv",
            "by_trend_direction.csv": "by_trend_direction.csv",
            "by_day_of_week.csv": "by_day_of_week.csv",
            "by_time_of_day.csv": "by_time_of_day.csv",
            "by_month.csv": "by_month.csv",
            "by_side.csv": "by_side.csv",
            "by_exit_reason.csv": "by_exit_reason.csv",
            "crosstab_vol_x_adx.csv": "crosstab_vol_x_adx.csv",
        },
    }

    (out_dir / "regime_summary.json").write_text(stable_json(summary), encoding="utf-8")

    # Metadata + hashes (Phase 0 style)
    outputs = [
        "day_features.csv",
        "trades_enriched.csv",
        "by_volatility.csv",
        "by_adx_bucket.csv",
        "by_trend_vs_chop.csv",
        "by_trend_direction.csv",
        "by_day_of_week.csv",
        "by_time_of_day.csv",
        "by_month.csv",
        "by_side.csv",
        "by_exit_reason.csv",
        "crosstab_vol_x_adx.csv",
        "regime_summary.json",
    ]
    out_hashes = {f: sha256_file(out_dir / f) for f in outputs if (out_dir / f).exists()}

    run_meta = {
        "generated_utc": summary["generated_utc"],
        "inputs": {
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "parquet_path": str(data_path),
            "parquet_sha256": sha256_file(data_path),
            "valid_days_path": str(valid_days_path),
            "valid_days_sha256": sha256_file(valid_days_path),
            "trades_path": str(trades_path),
            "trades_sha256": sha256_file(trades_path),
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "outputs": {f: str(out_dir / f) for f in outputs},
        "output_sha256": out_hashes,
    }
    (out_dir / "run_metadata.json").write_text(stable_json(run_meta), encoding="utf-8")

    print(f"[OK] Regime reports written to: {out_dir}")
    print("Key outputs:")
    for f in ["by_volatility.csv", "by_trend_vs_chop.csv", "by_day_of_week.csv", "by_month.csv", "crosstab_vol_x_adx.csv"]:
        print(f" - {out_dir / f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

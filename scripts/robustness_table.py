import os

# Determinism locks (must be set before Python does much work)
os.environ["PYTHONHASHSEED"] = "0"

import argparse
import copy
import hashlib
import platform
import random
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import yaml


# Determinism: RNG seeds
random.seed(0)
np.random.seed(0)


# Ensure repo root is importable when running: python scripts/robustness_table.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtester.futures_engine import FuturesEngineConfig, backtest_futures_orb  # noqa: E402
from backtester.risk import risk_limits_from_config  # noqa: E402
from core.utils import load_valid_days_csv, parse_hhmm, sha256_file, stable_json  # noqa: E402
from strategy import SignalRule, build_signals_from_config, load_signal_rules_from_config, serialize_signal_rules  # noqa: E402


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def get_git_info() -> Dict[str, Any]:
    def run(cmd: List[str]) -> str:
        return subprocess.check_output(cmd, cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL).decode().strip()

    info: Dict[str, Any] = {}
    try:
        info["commit"] = run(["git", "rev-parse", "HEAD"])
        info["branch"] = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        porcelain = run(["git", "status", "--porcelain"])
        info["dirty"] = bool(porcelain)
    except Exception:
        info["commit"] = None
        info["branch"] = None
        info["dirty"] = None
    return info


def fmt_hhmm(t) -> str:
    return f"{t.hour:02d}:{t.minute:02d}"


def add_minutes_to_time(t, minutes: int):
    base = datetime(2000, 1, 1, t.hour, t.minute)
    out = base + timedelta(minutes=int(minutes))
    return out.time()


@dataclass(frozen=True)
class RobustnessScenario:
    scenario_id: str
    cfg: Dict[str, Any]
    perturbed_rule_signal_type: str | None
    perturbed_adx_threshold: float | None
    perturbed_orb_start: str | None
    strategy_rules: List[Dict[str, Any]]


def _minutes_between(start: time, end: time) -> int:
    start_dt = datetime(2000, 1, 1, start.hour, start.minute)
    end_dt = datetime(2000, 1, 1, end.hour, end.minute)
    return int((end_dt - start_dt).total_seconds() // 60)


def _scenario_cfg_with_rules(cfg: Dict[str, Any], strategy_rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    scenario_cfg = copy.deepcopy(cfg)
    signals_cfg = dict(scenario_cfg.get("signals") or {})
    signals_cfg["rules"] = copy.deepcopy(strategy_rules)
    scenario_cfg["signals"] = signals_cfg
    return scenario_cfg


def _scenario_id_value(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def multi_rule_orb_start_offsets_for_timeframe(timeframe: str) -> List[int]:
    if str(timeframe) == "30m":
        return [-30, 0, 30]
    return [-15, 0, 15]


def _build_rule_variant(
    rule: SignalRule,
    *,
    adx_threshold: float,
    orb_start: time,
    orb_window_min: int,
    cutoff_offset_min: int,
) -> SignalRule:
    orb_end = add_minutes_to_time(orb_start, orb_window_min)
    orb_cutoff = add_minutes_to_time(orb_end, cutoff_offset_min)
    return replace(
        rule,
        adx_threshold=float(adx_threshold),
        orb_start=orb_start,
        orb_end=orb_end,
        orb_cutoff=orb_cutoff,
    )


def _baseline_scenario(cfg: Dict[str, Any], rules: List[SignalRule]) -> RobustnessScenario:
    strategy_rules = serialize_signal_rules(rules)
    return RobustnessScenario(
        scenario_id="baseline",
        cfg=_scenario_cfg_with_rules(cfg, strategy_rules),
        perturbed_rule_signal_type=None,
        perturbed_adx_threshold=None,
        perturbed_orb_start=None,
        strategy_rules=strategy_rules,
    )


def _build_single_rule_grid_scenarios(
    cfg: Dict[str, Any],
    rules: List[SignalRule],
    adx_threshold_grid: List[float],
    orb_start_grid: List[str],
    orb_window_min: int,
    cutoff_offset_min: int,
) -> List[RobustnessScenario]:
    base_rule = rules[0]
    scenarios: List[RobustnessScenario] = [_baseline_scenario(cfg, rules)]

    for orb_start_str in orb_start_grid:
        orb_start = parse_hhmm(orb_start_str)
        for thr in adx_threshold_grid:
            perturbed_rule = _build_rule_variant(
                base_rule,
                adx_threshold=float(thr),
                orb_start=orb_start,
                orb_window_min=int(orb_window_min),
                cutoff_offset_min=int(cutoff_offset_min),
            )
            if perturbed_rule == base_rule:
                continue

            strategy_rules = serialize_signal_rules([perturbed_rule])
            scenario_id = (
                f"{base_rule.signal_type}_adx{_scenario_id_value(float(thr))}"
                f"_orb{orb_start.hour:02d}{orb_start.minute:02d}"
            )
            scenarios.append(
                RobustnessScenario(
                    scenario_id=scenario_id,
                    cfg=_scenario_cfg_with_rules(cfg, strategy_rules),
                    perturbed_rule_signal_type=base_rule.signal_type,
                    perturbed_adx_threshold=float(thr),
                    perturbed_orb_start=fmt_hhmm(orb_start),
                    strategy_rules=strategy_rules,
                )
            )

    return scenarios


def _build_multi_rule_neighborhood_scenarios(
    cfg: Dict[str, Any],
    rules: List[SignalRule],
    orb_start_offsets_min: List[int],
) -> List[RobustnessScenario]:
    scenarios: List[RobustnessScenario] = [_baseline_scenario(cfg, rules)]

    for idx, rule in enumerate(rules):
        orb_window_min = _minutes_between(rule.orb_start, rule.orb_end)
        cutoff_offset_min = _minutes_between(rule.orb_end, rule.orb_cutoff)
        adx_values = [float(rule.adx_threshold) + delta for delta in (-1, 0, 1)]
        # Multi-rule ORB perturbations should stay on the bar grid. On 30m data, +/-15 minutes
        # would create off-grid starts, so callers can provide +/-30-minute offsets instead.
        orb_starts = [add_minutes_to_time(rule.orb_start, delta) for delta in orb_start_offsets_min]

        for thr in adx_values:
            for orb_start in orb_starts:
                if float(thr) == float(rule.adx_threshold) and orb_start == rule.orb_start:
                    continue

                perturbed_rule = _build_rule_variant(
                    rule,
                    adx_threshold=float(thr),
                    orb_start=orb_start,
                    orb_window_min=orb_window_min,
                    cutoff_offset_min=cutoff_offset_min,
                )
                scenario_rules = list(rules)
                scenario_rules[idx] = perturbed_rule
                strategy_rules = serialize_signal_rules(scenario_rules)
                scenario_id = (
                    f"{rule.signal_type}_adx{_scenario_id_value(float(thr))}"
                    f"_orb{orb_start.hour:02d}{orb_start.minute:02d}"
                )
                scenarios.append(
                    RobustnessScenario(
                        scenario_id=scenario_id,
                        cfg=_scenario_cfg_with_rules(cfg, strategy_rules),
                        perturbed_rule_signal_type=rule.signal_type,
                        perturbed_adx_threshold=float(thr),
                        perturbed_orb_start=fmt_hhmm(orb_start),
                        strategy_rules=strategy_rules,
                    )
                )

    return scenarios


def build_robustness_scenarios(
    cfg: Dict[str, Any],
    rules: List[SignalRule],
    *,
    adx_threshold_grid: List[float],
    orb_start_grid: List[str],
    orb_window_min: int,
    cutoff_offset_min: int,
    multi_rule_orb_start_offsets_min: List[int],
) -> List[RobustnessScenario]:
    if len(rules) == 1:
        return _build_single_rule_grid_scenarios(
            cfg,
            rules,
            adx_threshold_grid=adx_threshold_grid,
            orb_start_grid=orb_start_grid,
            orb_window_min=orb_window_min,
            cutoff_offset_min=cutoff_offset_min,
        )
    return _build_multi_rule_neighborhood_scenarios(
        cfg,
        rules,
        orb_start_offsets_min=multi_rule_orb_start_offsets_min,
    )


def compute_max_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    roll_max = equity.cummax()
    dd = (equity / roll_max) - 1.0
    return float(dd.min() * 100.0)


def compute_daily_sharpe(equity_df: pd.DataFrame) -> float:
    """Annualized Sharpe using daily equity returns (sqrt(365))."""
    if equity_df.empty:
        return 0.0
    s = equity_df.set_index("timestamp")["equity"]
    daily = s.resample("1D").last().dropna()
    rets = daily.pct_change().dropna()
    if len(rets) < 2:
        return 0.0
    std = float(rets.std(ddof=1))
    if std == 0.0 or np.isnan(std):
        return 0.0
    return float((rets.mean() / std) * np.sqrt(365.0))


def compute_cagr(equity_df: pd.DataFrame, initial_capital: float) -> float:
    if equity_df.empty or initial_capital <= 0:
        return 0.0
    start_ts = pd.to_datetime(equity_df["timestamp"].iloc[0], utc=True)
    end_ts = pd.to_datetime(equity_df["timestamp"].iloc[-1], utc=True)
    days = (end_ts - start_ts).total_seconds() / 86400.0
    if days <= 0:
        return 0.0
    years = days / 365.0
    final_equity = float(equity_df["equity"].iloc[-1])
    return float((final_equity / initial_capital) ** (1.0 / years) - 1.0)


def maybe_write_equity_plot(equity_df: pd.DataFrame, out_path: Path) -> None:
    """Best-effort plot writer.

    Matplotlib is optional. Some environments have binary wheels mismatches that can emit noisy
    tracebacks on import; we suppress stderr to keep runs clean.
    """
    try:
        import io
        import contextlib

        with contextlib.redirect_stderr(io.StringIO()):
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt  # noqa: F401

        # Plot (suppress any backend noise)
        with contextlib.redirect_stderr(io.StringIO()):
            plt.figure()
            plt.plot(equity_df["timestamp"], equity_df["equity"])
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
    except Exception:
        return


def summarize_run(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    initial_capital: float,
    total_fees: float,
    total_funding: float,
    liquidations: int,
) -> Dict[str, Any]:
    total_trades = int(len(trades_df))
    pnl_col = "pnl_net" if "pnl_net" in trades_df.columns else None

    if total_trades and pnl_col:
        pnl = trades_df[pnl_col]
        wins = int((pnl > 0).sum())
        losses = int((pnl <= 0).sum())
        win_rate = (wins / total_trades) * 100.0
        avg_win = float(pnl[pnl > 0].mean()) if wins else 0.0
        avg_loss = float(pnl[pnl <= 0].mean()) if losses else 0.0
        total_pnl_net = float(pnl.sum())
        expectancy = float(total_pnl_net / total_trades)
    else:
        wins = losses = 0
        win_rate = avg_win = avg_loss = total_pnl_net = expectancy = 0.0

    final_equity = float(equity_df["equity"].iloc[-1]) if not equity_df.empty else float(initial_capital)
    total_return_pct = (final_equity / initial_capital - 1.0) * 100.0 if initial_capital else 0.0
    max_dd = compute_max_drawdown_pct(equity_df["equity"]) if not equity_df.empty else 0.0
    sharpe = compute_daily_sharpe(equity_df)
    cagr = compute_cagr(equity_df, initial_capital)

    return {
        "initial_capital": float(initial_capital),
        "final_equity": float(final_equity),
        "total_return_pct": float(total_return_pct),
        "cagr": float(cagr),
        "max_drawdown_pct": float(max_dd),
        "daily_sharpe": float(sharpe),
        "total_trades": int(total_trades),
        "winning_trades": int(wins),
        "losing_trades": int(losses),
        "win_rate_pct": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "total_pnl_net": float(total_pnl_net),
        "expectancy_per_trade": float(expectancy),
        "total_fees": float(total_fees),
        "total_funding": float(total_funding),
        "liquidations": int(liquidations),
    }


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_time_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


@dataclass(frozen=True)
class RobustnessRunConfig:
    config: str | Path = "config.yaml"
    data: str | Path = ""
    valid_days: str | Path = "data/processed/valid_days.csv"
    out_dir: str | Path = "reports/robustness"
    adx_threshold_grid: Sequence[float] = (35.0, 38.0, 43.0, 48.0, 55.0)
    orb_start_grid: Sequence[str] = ("13:00", "13:30", "14:00")
    orb_window_min: int = 30
    cutoff_offset_min: int = 0
    engine: str = "futures"
    fee_mult: float = 1.0
    slippage_bps: float = 0.0
    delay_bars: int = 1
    leverage: float = 1.0
    mmr: float = 0.005
    funding_per_8h: float = 0.0001
    start: str = ""
    end: str = ""
    objective: str = "daily_sharpe"
    max_scenarios: int = 0


@dataclass(frozen=True)
class RobustnessRunResult:
    out_dir: Path
    table_csv: Path
    summary_json: Path
    metadata_json: Path
    scenario_count: int
    objective: str


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Phase 3: robustness table over a small parameter neighborhood (guards against overfitting)."
    )

    ap.add_argument("--config", default="config.yaml")
    ap.add_argument(
        "--data",
        default="",
        help="Processed parquet path. Default: data/processed/<symbol>_<timeframe>.parquet",
    )
    ap.add_argument("--valid-days", default="data/processed/valid_days.csv")
    ap.add_argument("--out-dir", default="reports/robustness")

    # Parameter neighborhood
    ap.add_argument("--adx-threshold-grid", default="35,38,43,48,55")
    ap.add_argument("--orb-start-grid", default="13:00,13:30,14:00")
    ap.add_argument("--orb-window-min", type=int, default=30)
    ap.add_argument("--cutoff-offset-min", type=int, default=0, help="cutoff = orb_end + offset (minutes)")

    # Engine + cost model knobs (defaults match your Phase 2/3 work)
    ap.add_argument("--engine", choices=["spot", "futures"], default="futures")
    ap.add_argument("--fee-mult", type=float, default=1.0)
    ap.add_argument("--slippage-bps", type=float, default=0.0)
    ap.add_argument("--delay-bars", type=int, default=1)

    # Futures knobs
    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--mmr", type=float, default=0.005)
    ap.add_argument("--funding-per-8h", type=float, default=0.0001)

    # Optional range filter
    ap.add_argument("--start", default="", help="Optional ISO start UTC")
    ap.add_argument("--end", default="", help="Optional ISO end UTC (exclusive)")

    # Reporting
    ap.add_argument("--objective", choices=["daily_sharpe", "total_return_pct", "cagr"], default="daily_sharpe")
    ap.add_argument("--max-scenarios", type=int, default=0, help="0 = run all")
    return ap


def parse_run_config(argv: Sequence[str] | None = None) -> RobustnessRunConfig:
    args = build_arg_parser().parse_args(argv)
    return RobustnessRunConfig(
        config=args.config,
        data=args.data,
        valid_days=args.valid_days,
        out_dir=args.out_dir,
        adx_threshold_grid=tuple(parse_float_list(args.adx_threshold_grid)),
        orb_start_grid=tuple(parse_time_list(args.orb_start_grid)),
        orb_window_min=int(args.orb_window_min),
        cutoff_offset_min=int(args.cutoff_offset_min),
        engine=args.engine,
        fee_mult=float(args.fee_mult),
        slippage_bps=float(args.slippage_bps),
        delay_bars=int(args.delay_bars),
        leverage=float(args.leverage),
        mmr=float(args.mmr),
        funding_per_8h=float(args.funding_per_8h),
        start=str(args.start),
        end=str(args.end),
        objective=str(args.objective),
        max_scenarios=int(args.max_scenarios),
    )


def run_robustness_table(run_cfg: RobustnessRunConfig) -> RobustnessRunResult:
    config_path = Path(run_cfg.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()

    valid_days_path = Path(run_cfg.valid_days)
    if not valid_days_path.is_absolute():
        valid_days_path = (REPO_ROOT / valid_days_path).resolve()

    out_dir = Path(run_cfg.out_dir)
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios_dir = out_dir / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    cfg_text = config_path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(cfg_text) or {}
    rules = load_signal_rules_from_config(cfg)

    symbol = str(cfg.get("symbol", "SOLUSDT"))
    timeframe = str(cfg.get("timeframe", "30m"))
    adx_period = int(cfg["adx"]["period"])
    multi_rule_orb_start_offsets_min = multi_rule_orb_start_offsets_for_timeframe(timeframe)

    initial_capital = float(cfg["risk"]["initial_capital"])
    position_size = float(cfg["risk"]["position_size"])
    taker_fee_rate = float(cfg["fees"]["taker_fee_rate"])

    # Phase 4 risk controls
    risk_limits = risk_limits_from_config(cfg)

    # Load parquet
    data_path = Path(run_cfg.data) if str(run_cfg.data).strip() else Path(f"data/processed/{symbol}_{timeframe}.parquet")
    if not data_path.is_absolute():
        data_path = (REPO_ROOT / data_path).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Parquet not found: {data_path}. Run: python scripts/build_parquet.py")

    df = pd.read_parquet(data_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Parquet dataset must have a DatetimeIndex")
    if df.index.tz is None:
        df = df.tz_localize("UTC")

    needed = ["open", "high", "low", "close", "volume"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing required column in parquet: {c}")
    df = df[needed].copy().sort_index()

    if run_cfg.start:
        start_ts = pd.to_datetime(run_cfg.start, utc=True)
        df = df[df.index >= start_ts]
    if run_cfg.end:
        end_ts = pd.to_datetime(run_cfg.end, utc=True)
        df = df[df.index < end_ts]

    if df.empty:
        raise RuntimeError("No candles after applying start/end filters.")

    # Valid day list (exactly 48 bars/day)
    valid_days = load_valid_days_csv(valid_days_path)
    print(f"[OK] Valid days loaded: {len(valid_days)} ({valid_days_path})")

    adx_threshold_grid = [float(x) for x in run_cfg.adx_threshold_grid]
    orb_start_grid = [str(x) for x in run_cfg.orb_start_grid]
    scenarios = build_robustness_scenarios(
        cfg,
        rules,
        adx_threshold_grid=adx_threshold_grid,
        orb_start_grid=orb_start_grid,
        orb_window_min=int(run_cfg.orb_window_min),
        cutoff_offset_min=int(run_cfg.cutoff_offset_min),
        multi_rule_orb_start_offsets_min=multi_rule_orb_start_offsets_min,
    )
    if run_cfg.max_scenarios:
        scenarios = scenarios[: int(run_cfg.max_scenarios)]

    # Engine config
    futures_cfg = FuturesEngineConfig(
        initial_capital=initial_capital,
        position_size=position_size,
        leverage=float(run_cfg.leverage),
        taker_fee_rate=taker_fee_rate,
        fee_mult=float(run_cfg.fee_mult),
        slippage_bps=float(run_cfg.slippage_bps),
        delay_bars=int(run_cfg.delay_bars),
        maintenance_margin_rate=float(run_cfg.mmr),
        funding_rate_per_8h=float(run_cfg.funding_per_8h),
    )

    # Run scenarios
    rows: List[Dict[str, Any]] = []
    outputs: Dict[str, str] = {}

    for scenario in scenarios:
        sid = scenario.scenario_id
        sdir = scenarios_dir / sid
        sdir.mkdir(parents=True, exist_ok=True)

        df_sig, _, scenario_rules = build_signals_from_config(df, scenario.cfg, valid_days)

        signal_counts = df_sig.loc[df_sig["signal"] != 0, "signal_type"].value_counts(dropna=False).to_dict()
        signals_total = int((df_sig["signal"] != 0).sum())

        total_fees = 0.0
        total_funding = 0.0
        liquidations = 0
        engine_stats: Dict[str, Any] = {}

        if run_cfg.engine == "spot":
            from backtester.spot_engine import backtest_orb_strategy  # type: ignore

            trades, equity_curve, final_capital, total_fees = backtest_orb_strategy(
                df=df_sig,
                orb_ranges=None,
                initial_capital=initial_capital,
                position_size=position_size,
                taker_fee_rate=taker_fee_rate,
                valid_days=valid_days,
                fee_mult=float(run_cfg.fee_mult),
                slippage_bps=float(run_cfg.slippage_bps),
                delay_bars=int(run_cfg.delay_bars),
            )
            engine_stats = {
                "final_capital": float(final_capital),
                "total_fees": float(total_fees),
                "total_funding": 0.0,
                "liquidations": 0,
            }
        else:
            trades, equity_curve, stats = backtest_futures_orb(
                df=df_sig,
                orb_ranges=None,
                valid_days=valid_days,
                cfg=futures_cfg,
                risk_limits=risk_limits,
            )
            total_fees = float(stats.get("total_fees", 0.0))
            total_funding = float(stats.get("total_funding", 0.0))
            liquidations = int(stats.get("liquidations", 0))
            engine_stats = stats

        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame({"timestamp": df_sig.index, "equity": equity_curve})
        strategy_rules = serialize_signal_rules(scenario_rules)

        metrics = summarize_run(
            trades_df=trades_df,
            equity_df=equity_df,
            initial_capital=initial_capital,
            total_fees=total_fees,
            total_funding=total_funding,
            liquidations=liquidations,
        )

        trades_path = sdir / "trades.csv"
        equity_path = sdir / "equity_curve.csv"
        results_path = sdir / "results.json"
        plot_path = sdir / "equity_curve.png"

        trades_df.to_csv(trades_path, index=False)
        equity_df.to_csv(equity_path, index=False)
        maybe_write_equity_plot(equity_df, plot_path)

        result_obj = {
            "scenario_id": sid,
            "symbol": symbol,
            "timeframe": timeframe,
            "range": {
                "start": equity_df["timestamp"].min().isoformat(),
                "end": equity_df["timestamp"].max().isoformat(),
                "candles": int(len(equity_df)),
            },
            "params": {
                "adx_period": int(adx_period),
                "engine": run_cfg.engine,
                "initial_capital": float(initial_capital),
                "position_size": float(position_size),
                "taker_fee_rate": float(taker_fee_rate),
                "fee_mult": float(run_cfg.fee_mult),
                "slippage_bps": float(run_cfg.slippage_bps),
                "delay_bars": int(run_cfg.delay_bars),
                "leverage": float(run_cfg.leverage),
                "mmr": float(run_cfg.mmr),
                "funding_per_8h": float(run_cfg.funding_per_8h),
                "perturbed_rule_signal_type": scenario.perturbed_rule_signal_type,
                "perturbed_adx_threshold": scenario.perturbed_adx_threshold,
                "perturbed_orb_start": scenario.perturbed_orb_start,
                "strategy_rules": strategy_rules,
            },
            "signals": {
                "signals_total": int(signals_total),
                "signal_type_counts": signal_counts,
            },
            "engine_stats": engine_stats,
            "metrics": metrics,
        }
        results_path.write_text(stable_json(result_obj), encoding="utf-8")

        rows.append(
            {
                "scenario_id": sid,
                "perturbed_rule_signal_type": scenario.perturbed_rule_signal_type,
                "perturbed_adx_threshold": scenario.perturbed_adx_threshold,
                "perturbed_orb_start": scenario.perturbed_orb_start,
                "signals_total": int(signals_total),
                "strategy_rules": stable_json(strategy_rules),
                **metrics,
            }
        )

        outputs[f"scenarios/{sid}/results.json"] = str(results_path)
        outputs[f"scenarios/{sid}/trades.csv"] = str(trades_path)
        outputs[f"scenarios/{sid}/equity_curve.csv"] = str(equity_path)
        if plot_path.exists():
            outputs[f"scenarios/{sid}/equity_curve.png"] = str(plot_path)

        label = scenario.perturbed_rule_signal_type or "baseline"
        print(
            f"[OK] {sid} [{label}] | sharpe={metrics['daily_sharpe']:.2f} | ret={metrics['total_return_pct']:.1f}% | dd={metrics['max_drawdown_pct']:.1f}%"
        )

    # Write robustness table
    table_df = pd.DataFrame(rows)
    if table_df.empty:
        raise RuntimeError("No scenarios ran. Check your grids / filters.")

    table_df = table_df.reset_index(drop=True)
    table_csv = out_dir / "robustness_table.csv"
    table_df.to_csv(table_csv, index=False)
    outputs["robustness_table.csv"] = str(table_csv)

    # Summary (best/median/worst for the chosen objective)
    obj = str(run_cfg.objective)
    if obj not in table_df.columns:
        raise ValueError(f"Objective '{obj}' not in table columns")

    best_row = table_df.sort_values(obj, ascending=False).iloc[0].to_dict()
    worst_row = table_df.sort_values(obj, ascending=True).iloc[0].to_dict()
    median_row = table_df.sort_values(obj).iloc[len(table_df) // 2].to_dict()

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "objective": obj,
        "scenario_count": int(len(table_df)),
        "best": best_row,
        "median": median_row,
        "worst": worst_row,
        "notes": {
            "interpretation": "If 'best' is much better than 'median' and 'worst', you may be overfitting/cherry-picking. We want a strong median and not-too-bad worst.",
        },
    }

    summary_path = out_dir / "robustness_summary.json"
    summary_path.write_text(stable_json(summary), encoding="utf-8")
    outputs["robustness_summary.json"] = str(summary_path)

    # Repro metadata + hashes
    hashes = {k: sha256_file(Path(v)) for k, v in outputs.items() if Path(v).exists()}
    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git": get_git_info(),
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": {"platform": platform.platform()},
        "inputs": {
            "config_path": str(config_path),
            "config_sha256": sha256_bytes(cfg_text.encode("utf-8")),
            "data_path": str(data_path),
            "data_sha256": sha256_file(data_path),
            "valid_days_path": str(valid_days_path),
            "valid_days_sha256": sha256_file(valid_days_path),
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "grids": {
            "mode": "legacy_single_rule_grid" if len(rules) == 1 else "multi_rule_one_rule_at_a_time",
            "adx_threshold_grid": adx_threshold_grid,
            "orb_start_grid": orb_start_grid,
            "orb_window_min": int(run_cfg.orb_window_min),
            "cutoff_offset_min": int(run_cfg.cutoff_offset_min),
            "multi_rule_adx_offsets": [-1, 0, 1],
            "multi_rule_orb_start_offsets_min": multi_rule_orb_start_offsets_min,
        },
        "assumptions": {
            "engine": run_cfg.engine,
            "fee_mult": float(run_cfg.fee_mult),
            "slippage_bps": float(run_cfg.slippage_bps),
            "delay_bars": int(run_cfg.delay_bars),
            "leverage": float(run_cfg.leverage),
            "mmr": float(run_cfg.mmr),
            "funding_per_8h": float(run_cfg.funding_per_8h),
        },
        "outputs": outputs,
        "output_sha256": hashes,
    }

    meta_path = out_dir / "run_metadata.json"
    meta_path.write_text(stable_json(meta), encoding="utf-8")

    print(f"\n[OK] Wrote: {table_csv}")
    print(f"[OK] Wrote: {summary_path}")
    return RobustnessRunResult(
        out_dir=out_dir,
        table_csv=table_csv,
        summary_json=summary_path,
        metadata_json=meta_path,
        scenario_count=int(len(table_df)),
        objective=obj,
    )


def main(argv: Sequence[str] | None = None) -> int:
    run_robustness_table(parse_run_config(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

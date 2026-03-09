from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest
import yaml

from scripts import walk_forward, walk_forward_regime_filter, walk_forward_tune

TEST_ROOT = Path(__file__).resolve().parents[2] / "reports"


def single_rule_cfg(*, adx_threshold: float) -> dict:
    return {
        "symbol": "SOLUSDT",
        "timeframe": "30m",
        "adx": {"period": 14, "threshold": 99},
        "orb": {"start": "23:00", "end": "23:30", "cutoff": "23:30"},
        "signals": {
            "rules": [
                {
                    "signal_type": "uptrend_reversion",
                    "signal": 1,
                    "trend": "uptrend",
                    "trigger": "close_below_orb_low",
                    "adx_threshold": float(adx_threshold),
                    "orb": {"start": "12:30", "end": "13:00", "cutoff": "13:00"},
                }
            ]
        },
        "risk": {"initial_capital": 10_000, "position_size": 0.95},
        "fees": {"taker_fee_rate": 0.0005},
        "risk_controls": {"enabled": False},
    }


def multi_rule_cfg() -> dict:
    cfg = single_rule_cfg(adx_threshold=35.0)
    cfg["signals"]["rules"].append(
        {
            "signal_type": "downtrend_breakdown",
            "signal": -1,
            "trend": "downtrend",
            "trigger": "close_below_orb_low",
            "adx_threshold": 43.0,
            "orb": {"start": "13:30", "end": "14:00", "cutoff": "14:00"},
        }
    )
    return cfg


def market_df() -> pd.DataFrame:
    rows = [
        {"ts": "2024-01-10 12:30", "open": 105.0, "high": 110.0, "low": 100.0, "close": 105.0, "volume": 1.0},
        {"ts": "2024-01-10 13:00", "open": 106.0, "high": 108.0, "low": 101.0, "close": 106.0, "volume": 1.0},
        {"ts": "2024-01-10 13:30", "open": 99.0, "high": 100.0, "low": 98.0, "close": 99.0, "volume": 1.0},
        {"ts": "2024-02-10 12:30", "open": 105.0, "high": 111.0, "low": 100.0, "close": 105.0, "volume": 1.0},
        {"ts": "2024-02-10 13:00", "open": 106.0, "high": 109.0, "low": 101.0, "close": 106.0, "volume": 1.0},
        {"ts": "2024-02-10 13:30", "open": 99.0, "high": 100.0, "low": 98.0, "close": 99.0, "volume": 1.0},
    ]
    idx = pd.DatetimeIndex([pd.Timestamp(row["ts"], tz="UTC") for row in rows])
    payload = [{k: v for k, v in row.items() if k != "ts"} for row in rows]
    return pd.DataFrame(payload, index=idx)


def fake_add_trend_indicators(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    del period
    out = df.copy()
    out["adx"] = [40.0 if ts.strftime("%H:%M") == "13:30" else 50.0 for ts in out.index]
    out["trend"] = "uptrend"
    out["plus_di"] = 1.0
    out["minus_di"] = 0.0
    return out


def fake_backtest_futures_orb(df, orb_ranges=None, valid_days=None, cfg=None, risk_limits=None):
    del orb_ranges, valid_days, risk_limits
    signal_types = df.loc[df["signal"] != 0, "signal_type"].tolist()
    trades = [{"signal_type": signal_type, "pnl_net": 1.0, "pnl": 1.0} for signal_type in signal_types]
    initial_capital = float(cfg.initial_capital)
    equity_curve = [initial_capital for _ in range(len(df))]
    if equity_curve:
        equity_curve[-1] = initial_capital + 1.0 * len(signal_types)
    return trades, equity_curve, {"total_fees": 0.0, "total_funding": 0.0, "liquidations": 0}


def case_prefix(case_name: str) -> str:
    return f"codex_{case_name}_{uuid4().hex[:8]}"


def cleanup_paths(paths: dict[str, Path]) -> None:
    for key, path in paths.items():
        if key == "out_dir":
            shutil.rmtree(path, ignore_errors=True)
            continue
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def write_runtime_files(prefix: str, cfg: dict, *, include_folds: bool) -> dict[str, Path]:
    cfg_path = TEST_ROOT / f"{prefix}_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    data_path = TEST_ROOT / f"{prefix}_data.parquet"
    data_path.write_bytes(b"placeholder")

    valid_days = pd.DataFrame({"date_utc": ["2024-01-10", "2024-02-10"]})
    valid_days_path = TEST_ROOT / f"{prefix}_valid_days.csv"
    valid_days.to_csv(valid_days_path, index=False)

    paths = {
        "config": cfg_path,
        "data": data_path,
        "valid_days": valid_days_path,
        "out_dir": TEST_ROOT / f"{prefix}_out",
    }

    if include_folds:
        folds = pd.DataFrame(
            [
                {
                    "fold_id": "fold_01",
                    "train_start": "2024-01-01",
                    "train_end": "2024-01-31",
                    "test_start": "2024-02-01",
                    "test_end": "2024-02-29",
                }
            ]
        )
        folds_path = TEST_ROOT / f"{prefix}_folds.csv"
        folds.to_csv(folds_path, index=False)
        paths["folds"] = folds_path

    return paths


def patch_runtime(monkeypatch: pytest.MonkeyPatch, df: pd.DataFrame, *modules: object) -> None:
    monkeypatch.setattr(pd, "read_parquet", lambda path: df.copy())
    for module in modules:
        monkeypatch.setattr(module, "add_trend_indicators", fake_add_trend_indicators)
        monkeypatch.setattr(module, "backtest_futures_orb", fake_backtest_futures_orb)


def run_main(monkeypatch: pytest.MonkeyPatch, module, args: list[str]) -> int:
    monkeypatch.setattr(sys, "argv", [str(Path(module.__file__).name), *args])
    return module.main()


def test_walk_forward_tuning_updates_effective_single_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    paths = write_runtime_files(case_prefix("walk_forward_single_rule"), single_rule_cfg(adx_threshold=45.0), include_folds=False)
    try:
        patch_runtime(monkeypatch, market_df(), walk_forward)

        rc = run_main(
            monkeypatch,
            walk_forward,
            [
                "--config",
                str(paths["config"]),
                "--data",
                str(paths["data"]),
                "--valid-days",
                str(paths["valid_days"]),
                "--out-dir",
                str(paths["out_dir"]),
                "--engine",
                "futures",
                "--train-months",
                "1",
                "--test-months",
                "1",
                "--step-months",
                "1",
                "--start",
                "2024-01-01",
                "--end",
                "2024-03-01",
                "--tune-adx-threshold",
                "35,45",
            ],
        )

        assert rc == 0

        results = json.loads((paths["out_dir"] / "folds" / "fold_01" / "results.json").read_text(encoding="utf-8"))
        assert results["chosen_adx_threshold"] == pytest.approx(35.0)
        assert results["params"]["adx_threshold"] == pytest.approx(35.0)
        assert results["params"]["strategy_rules"][0]["adx_threshold"] == pytest.approx(35.0)
        assert results["params"]["strategy_rules"][0]["signal_type"] == "uptrend_reversion"
        assert results["metrics"]["total_trades"] == 1
    finally:
        cleanup_paths(paths)


def test_walk_forward_tune_accepts_explicit_single_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    paths = write_runtime_files(case_prefix("walk_forward_tune_single_rule"), single_rule_cfg(adx_threshold=45.0), include_folds=True)
    try:
        patch_runtime(monkeypatch, market_df(), walk_forward_tune)

        rc = run_main(
            monkeypatch,
            walk_forward_tune,
            [
                "--config",
                str(paths["config"]),
                "--folds-csv",
                str(paths["folds"]),
                "--data",
                str(paths["data"]),
                "--valid-days",
                str(paths["valid_days"]),
                "--out-dir",
                str(paths["out_dir"]),
                "--engine",
                "futures",
                "--adx-threshold-grid",
                "35,45",
                "--orb-start-grid",
                "12:30,13:00",
                "--objective",
                "total_return_pct",
                "--min-trades",
                "0",
            ],
        )

        assert rc == 0

        summary = pd.read_csv(paths["out_dir"] / "walk_forward_tune_summary.csv")
        row = summary.iloc[0]
        assert row["chosen_scenario_id"] == "adx35_orb1230"
        assert row["chosen_adx_threshold"] == pytest.approx(35.0)
        assert row["chosen_orb_start"] == "12:30"
        assert int(row["test_total_trades"]) == 1

        test_results = json.loads((paths["out_dir"] / "folds" / "fold_01" / "test_results.json").read_text(encoding="utf-8"))
        assert test_results["chosen_params"]["adx_threshold"] == pytest.approx(35.0)
        assert test_results["signals"]["signal_type_counts"] == {"uptrend_reversion": 1}
    finally:
        cleanup_paths(paths)


def test_walk_forward_regime_filter_accepts_explicit_single_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    paths = write_runtime_files(
        case_prefix("walk_forward_regime_filter_single_rule"),
        single_rule_cfg(adx_threshold=35.0),
        include_folds=True,
    )
    try:
        patch_runtime(monkeypatch, market_df(), walk_forward_regime_filter)

        rc = run_main(
            monkeypatch,
            walk_forward_regime_filter,
            [
                "--config",
                str(paths["config"]),
                "--folds-csv",
                str(paths["folds"]),
                "--data",
                str(paths["data"]),
                "--valid-days",
                str(paths["valid_days"]),
                "--out-dir",
                str(paths["out_dir"]),
            ],
        )

        assert rc == 0

        summary = pd.read_csv(paths["out_dir"] / "walk_forward_regime_filter_summary.csv")
        row = summary.iloc[0]
        assert int(row["signals_total"]) == 1
        assert json.loads(row["signal_type_counts_json"]) == {"uptrend_reversion": 1}
        assert int(row["test_total_trades"]) == 1

        metadata = json.loads((paths["out_dir"] / "run_metadata.json").read_text(encoding="utf-8"))
        assert metadata["settings"]["fixed_params"]["adx_threshold"] == pytest.approx(35.0)
        assert metadata["settings"]["fixed_params"]["orb_start"] == "12:30"
    finally:
        cleanup_paths(paths)


def test_walk_forward_tuning_rejects_multi_rule_config(monkeypatch: pytest.MonkeyPatch) -> None:
    paths = write_runtime_files(case_prefix("walk_forward_multi_rule_reject"), multi_rule_cfg(), include_folds=False)
    try:
        monkeypatch.setattr(pd, "read_parquet", lambda path: market_df().copy())

        with pytest.raises(ValueError, match="exactly one signal rule"):
            run_main(
                monkeypatch,
                walk_forward,
                [
                    "--config",
                    str(paths["config"]),
                    "--data",
                    str(paths["data"]),
                    "--valid-days",
                    str(paths["valid_days"]),
                    "--out-dir",
                    str(paths["out_dir"]),
                    "--start",
                    "2024-01-01",
                    "--end",
                    "2024-03-01",
                    "--tune-adx-threshold",
                    "35,45",
                ],
            )
    finally:
        cleanup_paths(paths)


@pytest.mark.parametrize("module", [walk_forward_tune, walk_forward_regime_filter])
def test_single_rule_only_tools_reject_multi_rule_configs(
    monkeypatch: pytest.MonkeyPatch,
    module,
) -> None:
    paths = write_runtime_files(case_prefix(f"{module.__name__.split('.')[-1]}_multi_rule_reject"), multi_rule_cfg(), include_folds=True)
    try:

        with pytest.raises(ValueError, match="exactly one signal rule"):
            run_main(
                monkeypatch,
                module,
                [
                    "--config",
                    str(paths["config"]),
                    "--folds-csv",
                    str(paths["folds"]),
                    "--out-dir",
                    str(paths["out_dir"]),
                ],
            )
    finally:
        cleanup_paths(paths)

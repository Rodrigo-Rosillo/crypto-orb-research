from __future__ import annotations

from pathlib import Path
from uuid import uuid4
import shutil

import pandas as pd
import pytest
import yaml

from core.tuning import (
    STAGE_BASELINE,
    STAGE_STAGE1_ISOLATED,
    STAGE_STAGE1_MARGINAL,
    STAGE_STAGE2_MARGINAL,
    STAGE_STAGE3_JOINT,
    STAGE_STAGE4_JOINT_FRAGILITY,
    EVAL_ROBUSTNESS,
    EVAL_WALK_FORWARD,
    RuleParameter,
    ScenarioManifestRow,
    aggregate_fragility_stage,
    aggregate_walk_forward_stage,
    baseline_distance,
    build_holdout_manifest,
    build_stage1_manifests,
    build_stage3_manifest,
    build_stage4_manifest,
    build_stage5_manifest,
    create_run_settings,
    load_base_strategy_definition,
    load_manifest,
    manifest_row_from_params,
    order_id,
    param_id_from_params,
    scenario_id,
    stage_manifest_path,
    write_manifest,
    write_run_settings,
    write_stage_artifacts,
)
from scripts import tune_leaderboard, tune_manifest, tune_run

TEST_ROOT = Path(__file__).resolve().parents[2] / "reports"


def multi_rule_cfg() -> dict:
    return {
        "symbol": "SOLUSDT",
        "timeframe": "30m",
        "adx": {"period": 14},
        "signals": {
            "rules": [
                {
                    "signal_type": "uptrend_reversion",
                    "signal": 1,
                    "trend": "uptrend",
                    "trigger": "close_below_orb_low",
                    "adx_threshold": 35,
                    "orb": {"start": "12:30", "end": "13:00", "cutoff": "13:00"},
                },
                {
                    "signal_type": "downtrend_reversion",
                    "signal": -2,
                    "trend": "downtrend",
                    "trigger": "close_above_orb_high",
                    "adx_threshold": 44,
                    "orb": {"start": "13:00", "end": "13:30", "cutoff": "13:30"},
                },
                {
                    "signal_type": "downtrend_breakdown",
                    "signal": -1,
                    "trend": "downtrend",
                    "trigger": "close_below_orb_low",
                    "adx_threshold": 43,
                    "orb": {"start": "13:30", "end": "14:00", "cutoff": "14:00"},
                },
                {
                    "signal_type": "uptrend_continuation",
                    "signal": 2,
                    "trend": "uptrend",
                    "trigger": "close_above_orb_high",
                    "adx_threshold": 29,
                    "orb": {"start": "14:00", "end": "14:30", "cutoff": "14:30"},
                },
            ]
        },
        "risk": {"initial_capital": 10_000, "position_size": 0.95},
        "fees": {"taker_fee_rate": 0.0005},
        "risk_controls": {"enabled": False},
    }


def write_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(multi_rule_cfg(), sort_keys=False), encoding="utf-8")
    return path


def case_root(case_name: str) -> Path:
    root = TEST_ROOT / f"codex_{case_name}_{uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def cleanup_case(root: Path) -> None:
    shutil.rmtree(root, ignore_errors=True)


def manifest_params(adx_shift: float = 0.0, start_shift_min: int = 0, *, code: str = "ur") -> dict[str, RuleParameter]:
    params = {
        "ur": RuleParameter("ur", "uptrend_reversion", 35.0, "12:30"),
        "dr": RuleParameter("dr", "downtrend_reversion", 44.0, "13:00"),
        "db": RuleParameter("db", "downtrend_breakdown", 43.0, "13:30"),
        "uc": RuleParameter("uc", "uptrend_continuation", 29.0, "14:00"),
    }
    current = params[code]
    hour, minute = map(int, current.orb_start.split(":"))
    shifted = pd.Timestamp(2000, 1, 1, hour, minute) + pd.Timedelta(minutes=start_shift_min)
    params[code] = RuleParameter(code, current.signal_type, current.adx_threshold + adx_shift, shifted.strftime("%H:%M"))
    return params


def write_walk_forward_folds(out_dir: Path, score: float, *, trades: int = 12, drawdown: float = -10.0, liquidations: int = 0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "fold_id": f"fold_{idx:02d}",
                "train_start": "2021-01-01",
                "train_end": "2022-12-31",
                "test_start": "2023-01-01",
                "test_end": "2023-06-30",
                "chosen_adx_threshold": 0.0,
                "initial_capital": 10_000.0,
                "final_equity": 10_000.0 + score * 100,
                "total_return_pct": score + idx,
                "cagr": score / 100.0,
                "max_drawdown_pct": drawdown + idx * 0.1,
                "daily_sharpe": score / 10.0 + idx * 0.01,
                "total_trades": trades,
                "winning_trades": trades - 2,
                "losing_trades": 2,
                "win_rate_pct": ((trades - 2) / trades) * 100.0,
                "avg_win": 100.0,
                "avg_loss": -50.0,
                "total_pnl_net": score * 10.0,
                "expectancy_per_trade": score / max(trades, 1),
                "total_fees": 1.0,
                "total_funding": -0.5,
                "liquidations": liquidations,
                "fold_dir": str(out_dir / f"fold_{idx:02d}"),
            }
            for idx in range(1, 5)
        ]
    )
    df.to_csv(out_dir / "walk_forward_folds.csv", index=False)


def write_robustness_table(out_dir: Path, *, baseline_sharpe: float, neighbor_sharpes: list[float], baseline_dd: float = -10.0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "scenario_id": "baseline",
            "daily_sharpe": baseline_sharpe,
            "total_return_pct": 30.0,
            "max_drawdown_pct": baseline_dd,
        }
    ]
    for idx, sharpe in enumerate(neighbor_sharpes, start=1):
        rows.append(
            {
                "scenario_id": f"n{idx}",
                "daily_sharpe": sharpe,
                "total_return_pct": 5.0 + idx,
                "max_drawdown_pct": baseline_dd - 1.0,
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "robustness_table.csv", index=False)


def test_ids_and_distance_tokens() -> None:
    params = manifest_params(adx_shift=1.0, start_shift_min=30, code="ur")
    assert param_id_from_params(params) == "ur_a36_t1300__dr_a44_t1300__db_a43_t1330__uc_a29_t1400"
    assert order_id(("ur", "dr", "db", "uc")) == "ord_ur_dr_db_uc"
    assert scenario_id("stage1_marginal", params, ("ur", "dr", "db", "uc")).startswith("stage1_marginal__ur_a36_t1300")
    assert baseline_distance(params) == pytest.approx(1.5)


def test_stage1_generation_and_isolated_snapshots() -> None:
    root = case_root("tuning_stage1")
    try:
        base_definition = load_base_strategy_definition(write_config(root))
        run_root = root / "run"

        manifests = build_stage1_manifests(run_root, base_definition)
        write_stage_artifacts(run_root, base_definition, STAGE_STAGE1_MARGINAL, manifests[STAGE_STAGE1_MARGINAL])
        write_stage_artifacts(run_root, base_definition, STAGE_STAGE1_ISOLATED, manifests[STAGE_STAGE1_ISOLATED])

        assert len(manifests[STAGE_STAGE1_MARGINAL]) == 308
        assert len(manifests[STAGE_STAGE1_ISOLATED]) == 308

        first_marginal = yaml.safe_load(Path(manifests[STAGE_STAGE1_MARGINAL][0].config_path).read_text(encoding="utf-8"))
        first_isolated = yaml.safe_load(Path(manifests[STAGE_STAGE1_ISOLATED][0].config_path).read_text(encoding="utf-8"))

        assert len(first_marginal["signals"]["rules"]) == 4
        assert len(first_isolated["signals"]["rules"]) == 1
        assert manifests[STAGE_STAGE1_MARGINAL][0].run_id == "s1m_ur_001"
        assert manifests[STAGE_STAGE1_ISOLATED][0].run_id == "s1i_ur_001"
    finally:
        cleanup_case(root)


def test_stage_promotions_generate_expected_counts() -> None:
    root = case_root("tuning_promotions")
    try:
        base_definition = load_base_strategy_definition(write_config(root))
        run_root = root / "run"

        stage1_rows = []
        for code in ["ur", "dr", "db", "uc"]:
            for idx in range(3):
                params = manifest_params(adx_shift=float(idx), start_shift_min=30 * idx, code=code)
                stage1_rows.append(
                    manifest_row_from_params(
                        run_root=run_root,
                        stage=STAGE_STAGE2_MARGINAL,
                        run_id=f"{code}_{idx}",
                        params_by_code=params,
                        changed_rule=code,
                        evaluation_type=EVAL_WALK_FORWARD,
                        order_codes=("ur", "dr", "db", "uc"),
                    ).to_record()
                    | {"hard_pass": True, "rank": idx + 1}
                )
        stage1_df = pd.DataFrame(stage1_rows)
        stage2_manifest = build_stage3_manifest(run_root, base_definition, stage1_df)
        stage4_manifest = build_stage4_manifest(run_root, pd.DataFrame(stage1_rows[:5]))

        fragility_rows = []
        for idx in range(3):
            row = stage4_manifest[idx].to_record() | {"fragility_pass": True, "rank": idx + 1}
            fragility_rows.append(row)
        stage5_manifest = build_stage5_manifest(run_root, pd.DataFrame(fragility_rows))

        order_rows = []
        for parent in ["s4f_001", "s4f_002", "s4f_003"]:
            for idx in range(2):
                row = stage5_manifest[idx].to_record() | {"hard_pass": True, "rank": idx + 1, "parent_run_id": parent}
                order_rows.append(row)
        holdout_manifest = build_holdout_manifest(run_root, pd.DataFrame(order_rows))

        assert len(stage2_manifest) == 81
        assert len(stage4_manifest) == 5
        assert len(stage5_manifest) == 72
        assert len(holdout_manifest) == 3
    finally:
        cleanup_case(root)


def test_aggregate_walk_forward_stage_ranks_and_gates() -> None:
    root = case_root("tuning_agg_wf")
    try:
        row_good = ScenarioManifestRow(
            run_id="good",
            scenario_id="s",
            stage=STAGE_STAGE3_JOINT,
            parent_run_id="",
            changed_rule="joint",
            evaluation_type=EVAL_WALK_FORWARD,
            param_id="p",
            order_id="ord_ur_dr_db_uc",
            ur_adx=35.0,
            ur_start="12:30",
            dr_adx=44.0,
            dr_start="13:00",
            db_adx=43.0,
            db_start="13:30",
            uc_adx=29.0,
            uc_start="14:00",
            config_path=str(root / "good.yaml"),
            out_dir=str(root / "good"),
            status="done",
        )
        row_bad = ScenarioManifestRow(**{**row_good.to_record(), "run_id": "bad", "out_dir": str(root / "bad")})

        write_walk_forward_folds(Path(row_good.out_dir), score=30.0, trades=12, drawdown=-10.0, liquidations=0)
        write_walk_forward_folds(Path(row_bad.out_dir), score=-5.0, trades=5, drawdown=-30.0, liquidations=1)

        leaderboard = aggregate_walk_forward_stage([row_good, row_bad], STAGE_STAGE3_JOINT)

        assert leaderboard.iloc[0]["run_id"] == "good"
        assert bool(leaderboard.iloc[0]["hard_pass"]) is True
        assert bool(leaderboard.iloc[1]["hard_pass"]) is False
    finally:
        cleanup_case(root)


def test_aggregate_fragility_stage_computes_pass() -> None:
    root = case_root("tuning_agg_frag")
    try:
        row = ScenarioManifestRow(
            run_id="frag",
            scenario_id="frag_s",
            stage=STAGE_STAGE4_JOINT_FRAGILITY,
            parent_run_id="parent",
            changed_rule="joint",
            evaluation_type=EVAL_ROBUSTNESS,
            param_id="p",
            order_id="ord_ur_dr_db_uc",
            ur_adx=35.0,
            ur_start="12:30",
            dr_adx=44.0,
            dr_start="13:00",
            db_adx=43.0,
            db_start="13:30",
            uc_adx=29.0,
            uc_start="14:00",
            config_path=str(root / "frag.yaml"),
            out_dir=str(root / "frag"),
            status="done",
        )
        write_robustness_table(Path(row.out_dir), baseline_sharpe=2.0, neighbor_sharpes=[1.8, 1.9, 1.7])
        leaderboard = aggregate_fragility_stage([row])
        assert bool(leaderboard.iloc[0]["fragility_pass"]) is True
    finally:
        cleanup_case(root)


def test_tune_run_updates_status_and_handles_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    root = case_root("tune_run_status")
    try:
        config_path = write_config(root)
        run_root = root / "run"
        base_definition = load_base_strategy_definition(config_path)
        settings = create_run_settings(run_root, config_path)
        write_run_settings(settings)

        row = manifest_row_from_params(
            run_root=run_root,
            stage=STAGE_BASELINE,
            run_id="base_001",
            params_by_code=base_definition.baseline_params,
            changed_rule="joint",
            evaluation_type=EVAL_WALK_FORWARD,
            order_codes=base_definition.base_order_codes,
        )
        write_manifest(stage_manifest_path(run_root, STAGE_BASELINE), [row])
        write_manifest(run_root / "scenario_manifest.csv", [row])

        def fake_success(run_cfg) -> None:
            write_walk_forward_folds(Path(run_cfg.out_dir), score=10.0)

        monkeypatch.setattr(tune_run, "run_walk_forward", fake_success)
        assert tune_run.main(["--run-root", str(run_root), "--stage", STAGE_BASELINE]) == 0
        updated = load_manifest(stage_manifest_path(run_root, STAGE_BASELINE))[0]
        assert updated.status == "done"

        monkeypatch.setattr(tune_run, "run_walk_forward", lambda run_cfg: (_ for _ in ()).throw(RuntimeError("boom")))
        assert tune_run.main(["--run-root", str(run_root), "--stage", STAGE_BASELINE, "--overwrite"]) == 1
        failed = load_manifest(stage_manifest_path(run_root, STAGE_BASELINE))[0]
        assert failed.status == "failed"
        assert (Path(failed.out_dir) / "run_error.txt").exists()
    finally:
        cleanup_case(root)


def test_stage1_to_stage2_integration(monkeypatch: pytest.MonkeyPatch) -> None:
    root = case_root("tuning_integration")
    try:
        config_path = write_config(root)
        run_root = root / "run"

        assert tune_manifest.main(["init", "--run-root", str(run_root), "--config", str(config_path)]) == 0
        assert tune_manifest.main(["stage1", "--run-root", str(run_root)]) == 0

        targets = {
            "uptrend_reversion": (36.0, "13:00"),
            "downtrend_reversion": (48.0, "14:00"),
            "downtrend_breakdown": (41.0, "14:30"),
            "uptrend_continuation": (30.0, "15:00"),
        }
        baseline_rules = {rule["signal_type"]: rule for rule in multi_rule_cfg()["signals"]["rules"]}

        def fake_walk_forward(run_cfg) -> None:
            cfg = yaml.safe_load(Path(run_cfg.config).read_text(encoding="utf-8"))
            rules = cfg["signals"]["rules"]
            changed = rules[0]
            for rule in rules:
                baseline = baseline_rules[rule["signal_type"]]
                if float(rule["adx_threshold"]) != float(baseline["adx_threshold"]) or rule["orb"]["start"] != baseline["orb"]["start"]:
                    changed = rule
                    break
            target_adx, target_start = targets[changed["signal_type"]]
            start_delta = abs((pd.Timestamp(f"2000-01-01 {changed['orb']['start']}") - pd.Timestamp(f"2000-01-01 {target_start}")).total_seconds()) / 1800.0
            score = 50.0 - (abs(float(changed["adx_threshold"]) - target_adx) * 2.0 + start_delta)
            write_walk_forward_folds(Path(run_cfg.out_dir), score=score)

        monkeypatch.setattr(tune_run, "run_walk_forward", fake_walk_forward)
        assert tune_run.main(["--run-root", str(run_root), "--stage", STAGE_STAGE1_MARGINAL]) == 0
        assert tune_leaderboard.main(["--run-root", str(run_root), "--stage", STAGE_STAGE1_MARGINAL]) == 0
        assert tune_manifest.main(["stage2", "--run-root", str(run_root)]) == 0

        stage1_leaderboard = pd.read_csv(run_root / STAGE_STAGE1_MARGINAL / "leaderboard.csv")
        stage2_manifest = pd.read_csv(run_root / STAGE_STAGE2_MARGINAL / "manifest.csv")

        assert len(stage1_leaderboard) == 308
        assert len(pd.read_csv(run_root / STAGE_STAGE1_MARGINAL / "selected.csv")) == 20
        assert len(stage2_manifest) == 180
    finally:
        cleanup_case(root)

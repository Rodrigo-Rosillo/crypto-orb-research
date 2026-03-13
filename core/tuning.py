from __future__ import annotations

import copy
import itertools
from dataclasses import asdict, dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from core.utils import parse_hhmm, stable_json
from strategy import load_signal_rules_from_config

STAGE_BASELINE = "baseline"
STAGE_FRAGILITY = "fragility"
STAGE_STAGE1_MARGINAL = "stage1_marginal"
STAGE_STAGE1_ISOLATED = "stage1_isolated"
STAGE_STAGE2_MARGINAL = "stage2_marginal"
STAGE_STAGE3_JOINT = "stage3_joint"
STAGE_STAGE4_JOINT_FRAGILITY = "stage4_joint_fragility"
STAGE_STAGE5_ORDER = "stage5_order"
STAGE_HOLDOUT = "holdout"

EVAL_WALK_FORWARD = "walk_forward"
EVAL_ROBUSTNESS = "robustness"

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_REJECTED = "rejected"
STATUS_SELECTED = "selected"

MANIFEST_COLUMNS = [
    "run_id",
    "scenario_id",
    "stage",
    "parent_run_id",
    "changed_rule",
    "evaluation_type",
    "param_id",
    "order_id",
    "ur_adx",
    "ur_start",
    "dr_adx",
    "dr_start",
    "db_adx",
    "db_start",
    "uc_adx",
    "uc_start",
    "config_path",
    "out_dir",
    "status",
]


@dataclass(frozen=True)
class CanonicalRuleDefinition:
    code: str
    signal_type: str
    baseline_adx: float
    baseline_orb_start: str


CANONICAL_RULES: tuple[CanonicalRuleDefinition, ...] = (
    CanonicalRuleDefinition("ur", "uptrend_reversion", 35.0, "12:30"),
    CanonicalRuleDefinition("dr", "downtrend_reversion", 44.0, "13:00"),
    CanonicalRuleDefinition("db", "downtrend_breakdown", 43.0, "13:30"),
    CanonicalRuleDefinition("uc", "uptrend_continuation", 29.0, "14:00"),
)

RULE_BY_CODE = {rule.code: rule for rule in CANONICAL_RULES}
RULE_BY_SIGNAL_TYPE = {rule.signal_type: rule for rule in CANONICAL_RULES}
CANONICAL_RULE_CODES = tuple(rule.code for rule in CANONICAL_RULES)
CANONICAL_SIGNAL_TYPES = tuple(rule.signal_type for rule in CANONICAL_RULES)

STAGE1_ORB_GRIDS: dict[str, tuple[str, ...]] = {
    "ur": ("11:00", "11:30", "12:00", "12:30", "13:00", "13:30", "14:00"),
    "dr": ("11:30", "12:00", "12:30", "13:00", "13:30", "14:00", "14:30"),
    "db": ("12:00", "12:30", "13:00", "13:30", "14:00", "14:30", "15:00"),
    "uc": ("12:30", "13:00", "13:30", "14:00", "14:30", "15:00", "15:30"),
}
STAGE1_ADX_GRID: tuple[float, ...] = (20.0, 24.0, 28.0, 32.0, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0)
STAGE2_ADX_OFFSETS: tuple[int, ...] = (-4, -3, -2, -1, 0, 1, 2, 3, 4)
STAGE2_START_OFFSETS_MIN: tuple[int, ...] = (-60, -30, 0, 30, 60)


@dataclass(frozen=True)
class RuleParameter:
    code: str
    signal_type: str
    adx_threshold: float
    orb_start: str


@dataclass(frozen=True)
class BaseRuleTemplate:
    code: str
    signal_type: str
    raw_rule: dict[str, Any]
    orb_window_min: int
    cutoff_offset_min: int


@dataclass(frozen=True)
class BaseStrategyDefinition:
    config: dict[str, Any]
    base_config_path: Path
    templates_by_code: dict[str, BaseRuleTemplate]
    baseline_params: dict[str, RuleParameter]
    base_order_codes: tuple[str, ...]


@dataclass(frozen=True)
class ScenarioManifestRow:
    run_id: str
    scenario_id: str
    stage: str
    parent_run_id: str
    changed_rule: str
    evaluation_type: str
    param_id: str
    order_id: str
    ur_adx: float
    ur_start: str
    dr_adx: float
    dr_start: str
    db_adx: float
    db_start: str
    uc_adx: float
    uc_start: str
    config_path: str
    out_dir: str
    status: str = STATUS_PENDING

    def to_record(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ScenarioManifestRow":
        return cls(
            run_id=str(record["run_id"]),
            scenario_id=str(record["scenario_id"]),
            stage=str(record["stage"]),
            parent_run_id=str(record.get("parent_run_id", "")),
            changed_rule=str(record.get("changed_rule", "")),
            evaluation_type=str(record["evaluation_type"]),
            param_id=str(record["param_id"]),
            order_id=str(record["order_id"]),
            ur_adx=float(record["ur_adx"]),
            ur_start=str(record["ur_start"]),
            dr_adx=float(record["dr_adx"]),
            dr_start=str(record["dr_start"]),
            db_adx=float(record["db_adx"]),
            db_start=str(record["db_start"]),
            uc_adx=float(record["uc_adx"]),
            uc_start=str(record["uc_start"]),
            config_path=str(record["config_path"]),
            out_dir=str(record["out_dir"]),
            status=str(record.get("status", STATUS_PENDING)),
        )


@dataclass(frozen=True)
class StageExecutionSettings:
    stage: str
    evaluation_type: str
    engine: str = "futures"
    start: str = "2021-01-01"
    end: str = "2025-01-01"
    train_months: int | None = 24
    test_months: int | None = 6
    step_months: int | None = 6
    fee_mult: float = 1.0
    slippage_bps: float = 0.0
    delay_bars: int = 1
    leverage: float = 1.0
    mmr: float = 0.005
    funding_per_8h: float = 0.0001
    adx_threshold_grid: tuple[float, ...] = (35.0, 38.0, 43.0, 48.0, 55.0)
    orb_start_grid: tuple[str, ...] = ("13:00", "13:30", "14:00")
    orb_window_min: int = 30
    cutoff_offset_min: int = 0
    objective: str = "daily_sharpe"
    max_scenarios: int = 0


@dataclass(frozen=True)
class TuningRunSettings:
    run_root: str
    base_config_path: str
    stages: dict[str, StageExecutionSettings]


@dataclass(frozen=True)
class LeaderboardRow:
    run_id: str
    scenario_id: str
    stage: str
    rank: int
    status: str


@dataclass(frozen=True)
class PromotionOutput:
    stage: str
    selected_run_ids: tuple[str, ...] = field(default_factory=tuple)


def _optional_int(value: Any) -> int | None:
    if value in ("", None):
        return None
    return int(value)


def fmt_hhmm(value: time) -> str:
    return f"{value.hour:02d}:{value.minute:02d}"


def minutes_between(start: time, end: time) -> int:
    start_dt = datetime(2000, 1, 1, start.hour, start.minute)
    end_dt = datetime(2000, 1, 1, end.hour, end.minute)
    return int((end_dt - start_dt).total_seconds() // 60)


def shift_hhmm(hhmm: str, minutes: int) -> str:
    parsed = parse_hhmm(hhmm)
    base = datetime(2000, 1, 1, parsed.hour, parsed.minute)
    out = base + timedelta(minutes=int(minutes))
    return f"{out.hour:02d}:{out.minute:02d}"


def numeric_token(value: float) -> str:
    value = float(value)
    sign = ""
    if value < 0:
        sign = "m"
        value = abs(value)
    text = f"{value:.10f}".rstrip("0").rstrip(".")
    return f"{sign}{text.replace('.', 'p')}"


def time_token(value: str) -> str:
    parsed = parse_hhmm(value)
    return f"{parsed.hour:02d}{parsed.minute:02d}"


def _minutes_of(hhmm: str) -> int:
    parsed = parse_hhmm(hhmm)
    return parsed.hour * 60 + parsed.minute


def order_id(order_codes: Sequence[str]) -> str:
    return "ord_" + "_".join(order_codes)


def _order_codes_from_id(order_id_value: str) -> list[str]:
    parts = [part for part in str(order_id_value).split("_") if part]
    if not parts or parts[0] != "ord":
        raise ValueError(f"Invalid order_id: {order_id_value}")
    return parts[1:]


def param_id_from_params(params_by_code: Mapping[str, RuleParameter]) -> str:
    parts = []
    for code in CANONICAL_RULE_CODES:
        rule = params_by_code[code]
        parts.append(f"{code}_a{numeric_token(rule.adx_threshold)}_t{time_token(rule.orb_start)}")
    return "__".join(parts)


def scenario_id(stage: str, params_by_code: Mapping[str, RuleParameter], order_codes: Sequence[str]) -> str:
    return f"{stage}__{param_id_from_params(params_by_code)}__{order_id(order_codes)}"


def default_stage_settings() -> dict[str, StageExecutionSettings]:
    return {
        STAGE_BASELINE: StageExecutionSettings(stage=STAGE_BASELINE, evaluation_type=EVAL_WALK_FORWARD),
        STAGE_FRAGILITY: StageExecutionSettings(
            stage=STAGE_FRAGILITY,
            evaluation_type=EVAL_ROBUSTNESS,
            train_months=None,
            test_months=None,
            step_months=None,
            adx_threshold_grid=(35.0, 38.0, 43.0, 48.0, 55.0),
            orb_start_grid=("13:00", "13:30", "14:00"),
            orb_window_min=30,
            cutoff_offset_min=0,
            objective="daily_sharpe",
            max_scenarios=0,
        ),
        STAGE_STAGE1_MARGINAL: StageExecutionSettings(stage=STAGE_STAGE1_MARGINAL, evaluation_type=EVAL_WALK_FORWARD),
        STAGE_STAGE1_ISOLATED: StageExecutionSettings(stage=STAGE_STAGE1_ISOLATED, evaluation_type=EVAL_WALK_FORWARD),
        STAGE_STAGE2_MARGINAL: StageExecutionSettings(stage=STAGE_STAGE2_MARGINAL, evaluation_type=EVAL_WALK_FORWARD),
        STAGE_STAGE3_JOINT: StageExecutionSettings(stage=STAGE_STAGE3_JOINT, evaluation_type=EVAL_WALK_FORWARD),
        STAGE_STAGE4_JOINT_FRAGILITY: StageExecutionSettings(
            stage=STAGE_STAGE4_JOINT_FRAGILITY,
            evaluation_type=EVAL_ROBUSTNESS,
            train_months=None,
            test_months=None,
            step_months=None,
            adx_threshold_grid=(35.0, 38.0, 43.0, 48.0, 55.0),
            orb_start_grid=("13:00", "13:30", "14:00"),
            orb_window_min=30,
            cutoff_offset_min=0,
            objective="daily_sharpe",
            max_scenarios=0,
        ),
        STAGE_STAGE5_ORDER: StageExecutionSettings(stage=STAGE_STAGE5_ORDER, evaluation_type=EVAL_WALK_FORWARD),
        STAGE_HOLDOUT: StageExecutionSettings(
            stage=STAGE_HOLDOUT,
            evaluation_type=EVAL_WALK_FORWARD,
            start="2023-01-01",
            end="2026-02-01",
            train_months=24,
            test_months=13,
            step_months=13,
        ),
    }


def _stage_settings_to_dict(settings: StageExecutionSettings) -> dict[str, Any]:
    return {
        "stage": settings.stage,
        "evaluation_type": settings.evaluation_type,
        "engine": settings.engine,
        "start": settings.start,
        "end": settings.end,
        "train_months": settings.train_months,
        "test_months": settings.test_months,
        "step_months": settings.step_months,
        "fee_mult": settings.fee_mult,
        "slippage_bps": settings.slippage_bps,
        "delay_bars": settings.delay_bars,
        "leverage": settings.leverage,
        "mmr": settings.mmr,
        "funding_per_8h": settings.funding_per_8h,
        "adx_threshold_grid": list(settings.adx_threshold_grid),
        "orb_start_grid": list(settings.orb_start_grid),
        "orb_window_min": settings.orb_window_min,
        "cutoff_offset_min": settings.cutoff_offset_min,
        "objective": settings.objective,
        "max_scenarios": settings.max_scenarios,
    }


def _stage_settings_from_dict(payload: Mapping[str, Any]) -> StageExecutionSettings:
    return StageExecutionSettings(
        stage=str(payload["stage"]),
        evaluation_type=str(payload["evaluation_type"]),
        engine=str(payload.get("engine", "futures")),
        start=str(payload.get("start", "2021-01-01")),
        end=str(payload.get("end", "2025-01-01")),
        train_months=_optional_int(payload.get("train_months")),
        test_months=_optional_int(payload.get("test_months")),
        step_months=_optional_int(payload.get("step_months")),
        fee_mult=float(payload.get("fee_mult", 1.0)),
        slippage_bps=float(payload.get("slippage_bps", 0.0)),
        delay_bars=int(payload.get("delay_bars", 1)),
        leverage=float(payload.get("leverage", 1.0)),
        mmr=float(payload.get("mmr", 0.005)),
        funding_per_8h=float(payload.get("funding_per_8h", 0.0001)),
        adx_threshold_grid=tuple(float(x) for x in payload.get("adx_threshold_grid", [35.0, 38.0, 43.0, 48.0, 55.0])),
        orb_start_grid=tuple(str(x) for x in payload.get("orb_start_grid", ["13:00", "13:30", "14:00"])),
        orb_window_min=int(payload.get("orb_window_min", 30)),
        cutoff_offset_min=int(payload.get("cutoff_offset_min", 0)),
        objective=str(payload.get("objective", "daily_sharpe")),
        max_scenarios=int(payload.get("max_scenarios", 0)),
    )


def create_run_settings(run_root: Path, base_config_path: Path) -> TuningRunSettings:
    return TuningRunSettings(
        run_root=str(run_root.resolve()),
        base_config_path=str(base_config_path.resolve()),
        stages=default_stage_settings(),
    )


def run_settings_path(run_root: Path) -> Path:
    return run_root / "run_settings.json"


def master_manifest_path(run_root: Path) -> Path:
    return run_root / "scenario_manifest.csv"


def stage_dir(run_root: Path, stage: str) -> Path:
    return run_root / stage


def stage_manifest_path(run_root: Path, stage: str) -> Path:
    return stage_dir(run_root, stage) / "manifest.csv"


def stage_configs_dir(run_root: Path, stage: str) -> Path:
    return stage_dir(run_root, stage) / "configs"


def stage_runs_dir(run_root: Path, stage: str) -> Path:
    return stage_dir(run_root, stage) / "runs"


def stage_leaderboard_path(run_root: Path, stage: str) -> Path:
    return stage_dir(run_root, stage) / "leaderboard.csv"


def stage_selected_path(run_root: Path, stage: str) -> Path:
    return stage_dir(run_root, stage) / "selected.csv"


def stage_summary_path(run_root: Path, stage: str) -> Path:
    return stage_dir(run_root, stage) / "summary.json"


def ensure_stage_layout(run_root: Path, stage: str) -> None:
    stage_dir(run_root, stage).mkdir(parents=True, exist_ok=True)
    stage_configs_dir(run_root, stage).mkdir(parents=True, exist_ok=True)
    stage_runs_dir(run_root, stage).mkdir(parents=True, exist_ok=True)


def write_run_settings(settings: TuningRunSettings) -> Path:
    out_path = run_settings_path(Path(settings.run_root))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_root": settings.run_root,
        "base_config_path": settings.base_config_path,
        "canonical_rule_codes": list(CANONICAL_RULE_CODES),
        "canonical_signal_types": list(CANONICAL_SIGNAL_TYPES),
        "stages": {stage: _stage_settings_to_dict(stage_settings) for stage, stage_settings in settings.stages.items()},
    }
    out_path.write_text(stable_json(payload, indent=2), encoding="utf-8")
    return out_path


def load_run_settings(run_root: Path) -> TuningRunSettings:
    payload = yaml.safe_load(run_settings_path(run_root).read_text(encoding="utf-8")) or {}
    stages = {
        str(stage_name): _stage_settings_from_dict(stage_payload)
        for stage_name, stage_payload in (payload.get("stages") or {}).items()
    }
    return TuningRunSettings(
        run_root=str(payload["run_root"]),
        base_config_path=str(payload["base_config_path"]),
        stages=stages,
    )


def load_base_strategy_definition(base_config_path: Path) -> BaseStrategyDefinition:
    cfg = yaml.safe_load(base_config_path.read_text(encoding="utf-8")) or {}
    rules = load_signal_rules_from_config(cfg)
    raw_rules = list(((cfg.get("signals") or {}).get("rules") or []))
    if len(rules) != len(CANONICAL_RULES) or len(raw_rules) != len(CANONICAL_RULES):
        raise ValueError("Multi-rule tuning requires a config with exactly four explicit signal rules.")

    templates_by_code: dict[str, BaseRuleTemplate] = {}
    baseline_params: dict[str, RuleParameter] = {}
    order_codes: list[str] = []

    for idx, (parsed_rule, raw_rule, expected_rule) in enumerate(zip(rules, raw_rules, CANONICAL_RULES)):
        if parsed_rule.signal_type != expected_rule.signal_type:
            raise ValueError(
                f"signals.rules[{idx}] must be {expected_rule.signal_type!r} for the multi-rule tuning workflow."
            )
        parsed_start = fmt_hhmm(parsed_rule.orb_start)
        if float(parsed_rule.adx_threshold) != expected_rule.baseline_adx or parsed_start != expected_rule.baseline_orb_start:
            raise ValueError(
                f"signals.rules[{idx}] must match the registered baseline {expected_rule.signal_type!r} values "
                f"({expected_rule.baseline_adx:g} @ {expected_rule.baseline_orb_start})."
            )

        code = expected_rule.code
        order_codes.append(code)
        templates_by_code[code] = BaseRuleTemplate(
            code=code,
            signal_type=parsed_rule.signal_type,
            raw_rule=copy.deepcopy(raw_rule),
            orb_window_min=minutes_between(parsed_rule.orb_start, parsed_rule.orb_end),
            cutoff_offset_min=minutes_between(parsed_rule.orb_end, parsed_rule.orb_cutoff),
        )
        baseline_params[code] = RuleParameter(
            code=code,
            signal_type=parsed_rule.signal_type,
            adx_threshold=float(parsed_rule.adx_threshold),
            orb_start=parsed_start,
        )

    return BaseStrategyDefinition(
        config=cfg,
        base_config_path=base_config_path.resolve(),
        templates_by_code=templates_by_code,
        baseline_params=baseline_params,
        base_order_codes=tuple(order_codes),
    )


def baseline_distance(params_by_code: Mapping[str, RuleParameter]) -> float:
    score = 0.0
    for canonical in CANONICAL_RULES:
        current = params_by_code[canonical.code]
        score += abs(float(current.adx_threshold) - canonical.baseline_adx) / 2.0
        score += abs(_minutes_of(current.orb_start) - _minutes_of(canonical.baseline_orb_start)) / 30.0
    return float(score)


def params_from_manifest_row(row: ScenarioManifestRow) -> dict[str, RuleParameter]:
    return {
        "ur": RuleParameter("ur", RULE_BY_CODE["ur"].signal_type, float(row.ur_adx), row.ur_start),
        "dr": RuleParameter("dr", RULE_BY_CODE["dr"].signal_type, float(row.dr_adx), row.dr_start),
        "db": RuleParameter("db", RULE_BY_CODE["db"].signal_type, float(row.db_adx), row.db_start),
        "uc": RuleParameter("uc", RULE_BY_CODE["uc"].signal_type, float(row.uc_adx), row.uc_start),
    }


def manifest_row_from_params(
    *,
    run_root: Path,
    stage: str,
    run_id: str,
    params_by_code: Mapping[str, RuleParameter],
    changed_rule: str,
    evaluation_type: str,
    order_codes: Sequence[str],
    parent_run_id: str = "",
    status: str = STATUS_PENDING,
) -> ScenarioManifestRow:
    scenario = scenario_id(stage, params_by_code, order_codes)
    config_path = stage_configs_dir(run_root, stage) / f"{run_id}.yaml"
    out_dir = stage_runs_dir(run_root, stage) / run_id
    return ScenarioManifestRow(
        run_id=run_id,
        scenario_id=scenario,
        stage=stage,
        parent_run_id=parent_run_id,
        changed_rule=changed_rule,
        evaluation_type=evaluation_type,
        param_id=param_id_from_params(params_by_code),
        order_id=order_id(order_codes),
        ur_adx=float(params_by_code["ur"].adx_threshold),
        ur_start=params_by_code["ur"].orb_start,
        dr_adx=float(params_by_code["dr"].adx_threshold),
        dr_start=params_by_code["dr"].orb_start,
        db_adx=float(params_by_code["db"].adx_threshold),
        db_start=params_by_code["db"].orb_start,
        uc_adx=float(params_by_code["uc"].adx_threshold),
        uc_start=params_by_code["uc"].orb_start,
        config_path=str(config_path.resolve()),
        out_dir=str(out_dir.resolve()),
        status=status,
    )


def _materialize_rule(template: BaseRuleTemplate, params: RuleParameter) -> dict[str, Any]:
    rule_cfg = copy.deepcopy(template.raw_rule)
    rule_cfg["adx_threshold"] = float(params.adx_threshold)
    start = parse_hhmm(params.orb_start)
    end = (datetime(2000, 1, 1, start.hour, start.minute) + timedelta(minutes=template.orb_window_min)).time()
    cutoff = (datetime(2000, 1, 1, end.hour, end.minute) + timedelta(minutes=template.cutoff_offset_min)).time()
    orb_cfg = dict(rule_cfg.get("orb") or {})
    orb_cfg["start"] = params.orb_start
    orb_cfg["end"] = fmt_hhmm(end)
    orb_cfg["cutoff"] = fmt_hhmm(cutoff)
    rule_cfg["orb"] = orb_cfg
    return rule_cfg


def write_config_snapshot(
    base_definition: BaseStrategyDefinition,
    row: ScenarioManifestRow,
    *,
    isolated_code: str | None = None,
) -> Path:
    cfg = copy.deepcopy(base_definition.config)
    order_codes = tuple(_order_codes_from_id(row.order_id))
    params_by_code = params_from_manifest_row(row)
    selected_codes = (isolated_code,) if isolated_code is not None else order_codes
    signals_cfg = dict(cfg.get("signals") or {})
    signals_cfg["rules"] = [_materialize_rule(base_definition.templates_by_code[code], params_by_code[code]) for code in selected_codes]
    cfg["signals"] = signals_cfg
    out_path = Path(row.config_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out_path


def rows_to_dataframe(rows: Sequence[ScenarioManifestRow]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    return pd.DataFrame([row.to_record() for row in rows], columns=MANIFEST_COLUMNS)


def write_manifest(path: Path, rows: Sequence[ScenarioManifestRow]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_to_dataframe(rows).to_csv(path, index=False)
    return path


def load_manifest(path: Path) -> list[ScenarioManifestRow]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []
    return [ScenarioManifestRow.from_record(record) for record in df.to_dict(orient="records")]


def sync_master_manifest(run_root: Path, stage_rows: Sequence[ScenarioManifestRow], stage: str) -> Path:
    existing = load_manifest(master_manifest_path(run_root))
    retained = [row for row in existing if row.stage != stage]
    combined = retained + list(stage_rows)
    return write_manifest(master_manifest_path(run_root), combined)


def update_manifest_statuses(run_root: Path, stage: str, status_by_run_id: Mapping[str, str]) -> None:
    if not status_by_run_id:
        return
    for path in [stage_manifest_path(run_root, stage), master_manifest_path(run_root)]:
        rows = load_manifest(path)
        updated: list[ScenarioManifestRow] = []
        changed = False
        for row in rows:
            if row.stage == stage and row.run_id in status_by_run_id:
                updated.append(ScenarioManifestRow(**{**row.to_record(), "status": status_by_run_id[row.run_id]}))
                changed = True
            else:
                updated.append(row)
        if changed:
            write_manifest(path, updated)


def load_stage_leaderboard(run_root: Path, stage: str) -> pd.DataFrame:
    path = stage_leaderboard_path(run_root, stage)
    if not path.exists():
        raise FileNotFoundError(f"Leaderboard not found: {path}")
    return pd.read_csv(path)


def initialize_run_root(run_root: Path, base_definition: BaseStrategyDefinition) -> dict[str, list[ScenarioManifestRow]]:
    run_root.mkdir(parents=True, exist_ok=True)
    settings = create_run_settings(run_root, base_definition.base_config_path)
    write_run_settings(settings)

    baseline_rows = build_baseline_stage(run_root, base_definition)
    fragility_rows = build_initial_fragility_stage(run_root, base_definition)
    write_stage_artifacts(run_root, base_definition, STAGE_BASELINE, baseline_rows)
    write_stage_artifacts(run_root, base_definition, STAGE_FRAGILITY, fragility_rows)
    return {STAGE_BASELINE: baseline_rows, STAGE_FRAGILITY: fragility_rows}


def write_stage_artifacts(
    run_root: Path,
    base_definition: BaseStrategyDefinition,
    stage: str,
    rows: Sequence[ScenarioManifestRow],
) -> Path:
    ensure_stage_layout(run_root, stage)
    for row in rows:
        isolated_code = row.changed_rule if stage == STAGE_STAGE1_ISOLATED else None
        write_config_snapshot(base_definition, row, isolated_code=isolated_code)
    manifest = write_manifest(stage_manifest_path(run_root, stage), rows)
    sync_master_manifest(run_root, rows, stage)
    return manifest


def build_baseline_stage(run_root: Path, base_definition: BaseStrategyDefinition) -> list[ScenarioManifestRow]:
    params = copy.deepcopy(base_definition.baseline_params)
    return [
        manifest_row_from_params(
            run_root=run_root,
            stage=STAGE_BASELINE,
            run_id="base_001",
            params_by_code=params,
            changed_rule="joint",
            evaluation_type=EVAL_WALK_FORWARD,
            order_codes=base_definition.base_order_codes,
        )
    ]


def build_initial_fragility_stage(run_root: Path, base_definition: BaseStrategyDefinition) -> list[ScenarioManifestRow]:
    params = copy.deepcopy(base_definition.baseline_params)
    return [
        manifest_row_from_params(
            run_root=run_root,
            stage=STAGE_FRAGILITY,
            run_id="frag_001",
            params_by_code=params,
            changed_rule="joint",
            evaluation_type=EVAL_ROBUSTNESS,
            order_codes=base_definition.base_order_codes,
            parent_run_id="base_001",
        )
    ]


def build_stage1_manifests(run_root: Path, base_definition: BaseStrategyDefinition) -> dict[str, list[ScenarioManifestRow]]:
    marginal_rows: list[ScenarioManifestRow] = []
    isolated_rows: list[ScenarioManifestRow] = []
    for code in CANONICAL_RULE_CODES:
        for idx, (adx_value, start_value) in enumerate(itertools.product(STAGE1_ADX_GRID, STAGE1_ORB_GRIDS[code]), start=1):
            params = copy.deepcopy(base_definition.baseline_params)
            params[code] = RuleParameter(code, RULE_BY_CODE[code].signal_type, float(adx_value), str(start_value))
            marginal_rows.append(
                manifest_row_from_params(
                    run_root=run_root,
                    stage=STAGE_STAGE1_MARGINAL,
                    run_id=f"s1m_{code}_{idx:03d}",
                    params_by_code=params,
                    changed_rule=code,
                    evaluation_type=EVAL_WALK_FORWARD,
                    order_codes=base_definition.base_order_codes,
                )
            )
            isolated_rows.append(
                manifest_row_from_params(
                    run_root=run_root,
                    stage=STAGE_STAGE1_ISOLATED,
                    run_id=f"s1i_{code}_{idx:03d}",
                    params_by_code=params,
                    changed_rule=code,
                    evaluation_type=EVAL_WALK_FORWARD,
                    order_codes=base_definition.base_order_codes,
                )
            )
    return {STAGE_STAGE1_MARGINAL: marginal_rows, STAGE_STAGE1_ISOLATED: isolated_rows}


def build_stage2_manifest(run_root: Path, base_definition: BaseStrategyDefinition, leaderboard: pd.DataFrame) -> list[ScenarioManifestRow]:
    centers = best_candidate_per_rule(leaderboard)
    rows: list[ScenarioManifestRow] = []
    for code in CANONICAL_RULE_CODES:
        center = centers[code]
        center_adx = float(center[f"{code}_adx"])
        center_start = str(center[f"{code}_start"])
        for idx, (adx_offset, start_offset) in enumerate(itertools.product(STAGE2_ADX_OFFSETS, STAGE2_START_OFFSETS_MIN), start=1):
            params = copy.deepcopy(base_definition.baseline_params)
            params[code] = RuleParameter(
                code,
                RULE_BY_CODE[code].signal_type,
                center_adx + float(adx_offset),
                shift_hhmm(center_start, int(start_offset)),
            )
            rows.append(
                manifest_row_from_params(
                    run_root=run_root,
                    stage=STAGE_STAGE2_MARGINAL,
                    run_id=f"s2m_{code}_{idx:03d}",
                    params_by_code=params,
                    changed_rule=code,
                    evaluation_type=EVAL_WALK_FORWARD,
                    order_codes=base_definition.base_order_codes,
                    parent_run_id=str(center["run_id"]),
                )
            )
    return rows


def build_stage3_manifest(run_root: Path, base_definition: BaseStrategyDefinition, leaderboard: pd.DataFrame) -> list[ScenarioManifestRow]:
    selected = top_n_per_rule(leaderboard, per_rule=3)
    combinations = itertools.product(
        selected["ur"].to_dict(orient="records"),
        selected["dr"].to_dict(orient="records"),
        selected["db"].to_dict(orient="records"),
        selected["uc"].to_dict(orient="records"),
    )
    rows: list[ScenarioManifestRow] = []
    for idx, combo in enumerate(combinations, start=1):
        params = copy.deepcopy(base_definition.baseline_params)
        parent_ids: list[str] = []
        for code, record in zip(CANONICAL_RULE_CODES, combo):
            params[code] = RuleParameter(code, RULE_BY_CODE[code].signal_type, float(record[f"{code}_adx"]), str(record[f"{code}_start"]))
            parent_ids.append(str(record["run_id"]))
        rows.append(
            manifest_row_from_params(
                run_root=run_root,
                stage=STAGE_STAGE3_JOINT,
                run_id=f"s3j_{idx:03d}",
                params_by_code=params,
                changed_rule="joint",
                evaluation_type=EVAL_WALK_FORWARD,
                order_codes=base_definition.base_order_codes,
                parent_run_id="|".join(parent_ids),
            )
        )
    return rows


def build_stage4_manifest(run_root: Path, leaderboard: pd.DataFrame) -> list[ScenarioManifestRow]:
    selected = top_n_overall(leaderboard, count=5, pass_column="hard_pass")
    rows: list[ScenarioManifestRow] = []
    for idx, record in enumerate(selected.to_dict(orient="records"), start=1):
        source = ScenarioManifestRow.from_record(record)
        rows.append(
            ScenarioManifestRow(
                **{
                    **source.to_record(),
                    "stage": STAGE_STAGE4_JOINT_FRAGILITY,
                    "run_id": f"s4f_{idx:03d}",
                    "evaluation_type": EVAL_ROBUSTNESS,
                    "config_path": str((stage_configs_dir(run_root, STAGE_STAGE4_JOINT_FRAGILITY) / f"s4f_{idx:03d}.yaml").resolve()),
                    "out_dir": str((stage_runs_dir(run_root, STAGE_STAGE4_JOINT_FRAGILITY) / f"s4f_{idx:03d}").resolve()),
                    "parent_run_id": source.run_id,
                    "status": STATUS_PENDING,
                }
            )
        )
    return rows


def build_stage5_manifest(run_root: Path, leaderboard: pd.DataFrame) -> list[ScenarioManifestRow]:
    selected = top_n_overall(leaderboard, count=3, pass_column="fragility_pass")
    rows: list[ScenarioManifestRow] = []
    for parent_idx, record in enumerate(selected.to_dict(orient="records"), start=1):
        source = ScenarioManifestRow.from_record(record)
        params = params_from_manifest_row(source)
        for perm_idx, order_codes in enumerate(itertools.permutations(CANONICAL_RULE_CODES), start=1):
            rows.append(
                manifest_row_from_params(
                    run_root=run_root,
                    stage=STAGE_STAGE5_ORDER,
                    run_id=f"s5o_{parent_idx:02d}_{perm_idx:02d}",
                    params_by_code=params,
                    changed_rule="order",
                    evaluation_type=EVAL_WALK_FORWARD,
                    order_codes=order_codes,
                    parent_run_id=source.run_id,
                )
            )
    return rows


def build_holdout_manifest(run_root: Path, leaderboard: pd.DataFrame) -> list[ScenarioManifestRow]:
    stage5_winners = best_order_per_parent(leaderboard)
    finalists = top_n_overall(stage5_winners, count=3)
    rows: list[ScenarioManifestRow] = []
    for idx, record in enumerate(finalists.to_dict(orient="records"), start=1):
        source = ScenarioManifestRow.from_record(record)
        rows.append(
            ScenarioManifestRow(
                **{
                    **source.to_record(),
                    "stage": STAGE_HOLDOUT,
                    "run_id": f"hold_{idx:03d}",
                    "config_path": str((stage_configs_dir(run_root, STAGE_HOLDOUT) / f"hold_{idx:03d}.yaml").resolve()),
                    "out_dir": str((stage_runs_dir(run_root, STAGE_HOLDOUT) / f"hold_{idx:03d}").resolve()),
                    "parent_run_id": source.run_id,
                    "status": STATUS_PENDING,
                }
            )
        )
    return rows


def best_candidate_per_rule(leaderboard: pd.DataFrame) -> dict[str, Mapping[str, Any]]:
    df = _passed_rows(leaderboard, "hard_pass")
    selected: dict[str, Mapping[str, Any]] = {}
    for code in CANONICAL_RULE_CODES:
        group = df[df["changed_rule"] == code]
        if group.empty:
            raise ValueError(f"No passing candidate found for rule {code}.")
        selected[code] = group.iloc[0].to_dict()
    return selected


def top_n_per_rule(leaderboard: pd.DataFrame, per_rule: int) -> dict[str, pd.DataFrame]:
    df = _passed_rows(leaderboard, "hard_pass")
    selected: dict[str, pd.DataFrame] = {}
    for code in CANONICAL_RULE_CODES:
        group = df[df["changed_rule"] == code].head(int(per_rule)).reset_index(drop=True)
        if len(group) < int(per_rule):
            raise ValueError(f"Need at least {per_rule} passing rows for rule {code}.")
        selected[code] = group
    return selected


def top_n_overall(leaderboard: pd.DataFrame, count: int, pass_column: str | None = None) -> pd.DataFrame:
    df = leaderboard.copy()
    if pass_column:
        df = _passed_rows(df, pass_column)
    if df.empty:
        raise ValueError("No qualifying rows available for promotion.")
    return df.head(int(count)).reset_index(drop=True)


def best_order_per_parent(leaderboard: pd.DataFrame) -> pd.DataFrame:
    df = _passed_rows(leaderboard, "hard_pass")
    rows = []
    for _, group in df.groupby("parent_run_id", dropna=False):
        rows.append(group.iloc[0])
    if not rows:
        raise ValueError("No passing order rows available for holdout promotion.")
    return pd.DataFrame(rows).reset_index(drop=True)


def _passed_rows(df: pd.DataFrame, flag_column: str) -> pd.DataFrame:
    if flag_column not in df.columns:
        raise ValueError(f"Missing required column {flag_column!r} in leaderboard.")
    out = df[df[flag_column].astype(bool)].copy()
    if out.empty:
        raise ValueError(f"No rows passed {flag_column}.")
    if "rank" in out.columns:
        out = out.sort_values("rank", ascending=True)
    return out.reset_index(drop=True)


def aggregate_walk_forward_stage(manifest_rows: Sequence[ScenarioManifestRow], stage: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for manifest_row in manifest_rows:
        folds_path = Path(manifest_row.out_dir) / "walk_forward_folds.csv"
        if manifest_row.status not in {STATUS_DONE, STATUS_SELECTED, STATUS_REJECTED} or not folds_path.exists():
            continue
        folds = pd.read_csv(folds_path)
        if folds.empty:
            continue
        total_trades = float(folds["total_trades"].sum()) if "total_trades" in folds.columns else 0.0
        wins = float(folds["winning_trades"].sum()) if "winning_trades" in folds.columns else 0.0
        row = {
            **manifest_row.to_record(),
            "fold_count": int(len(folds)),
            "positive_folds": int((folds["total_return_pct"] > 0).sum()),
            "positive_fold_pct": float((folds["total_return_pct"] > 0).mean() * 100.0),
            "median_test_daily_sharpe": float(folds["daily_sharpe"].median()),
            "mean_test_daily_sharpe": float(folds["daily_sharpe"].mean()),
            "median_test_total_return_pct": float(folds["total_return_pct"].median()),
            "mean_test_total_return_pct": float(folds["total_return_pct"].mean()),
            "std_test_total_return_pct": float(folds["total_return_pct"].std(ddof=1)) if len(folds) >= 2 else 0.0,
            "worst_fold_total_return_pct": float(folds["total_return_pct"].min()),
            "worst_fold_max_drawdown_pct": float(folds["max_drawdown_pct"].min()),
            "median_test_max_drawdown_pct": float(folds["max_drawdown_pct"].median()),
            "total_test_trades": int(total_trades),
            "median_test_trades": float(folds["total_trades"].median()),
            "liquidations_sum": int(folds["liquidations"].sum()) if "liquidations" in folds.columns else 0,
            "fees_sum": float(folds["total_fees"].sum()) if "total_fees" in folds.columns else 0.0,
            "funding_sum": float(folds["total_funding"].sum()) if "total_funding" in folds.columns else 0.0,
            "weighted_win_rate_pct": float((wins / total_trades) * 100.0) if total_trades else 0.0,
            "weighted_expectancy": float(folds["total_pnl_net"].sum() / total_trades) if total_trades and "total_pnl_net" in folds.columns else 0.0,
            "distance_from_baseline": baseline_distance(params_from_manifest_row(manifest_row)),
        }
        row["hard_pass"] = bool(
            row["liquidations_sum"] == 0
            and row["worst_fold_max_drawdown_pct"] >= -25.0
            and row["positive_folds"] >= 3
            and row["total_test_trades"] >= 40
        )
        if stage == STAGE_HOLDOUT:
            first = folds.iloc[0].to_dict()
            row["daily_sharpe"] = float(first.get("daily_sharpe", 0.0))
            row["total_return_pct"] = float(first.get("total_return_pct", 0.0))
            row["max_drawdown_pct"] = float(first.get("max_drawdown_pct", 0.0))
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=MANIFEST_COLUMNS + ["rank"])

    df = pd.DataFrame(rows)
    if stage == STAGE_HOLDOUT:
        df = df.sort_values(
            by=["daily_sharpe", "total_return_pct", "max_drawdown_pct", "scenario_id"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
    else:
        df = df.sort_values(
            by=[
                "hard_pass",
                "median_test_daily_sharpe",
                "positive_folds",
                "median_test_total_return_pct",
                "worst_fold_max_drawdown_pct",
                "std_test_total_return_pct",
                "total_test_trades",
                "distance_from_baseline",
                "scenario_id",
            ],
            ascending=[False, False, False, False, False, True, False, True, True],
        ).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


def aggregate_fragility_stage(manifest_rows: Sequence[ScenarioManifestRow]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for manifest_row in manifest_rows:
        table_path = Path(manifest_row.out_dir) / "robustness_table.csv"
        if manifest_row.status not in {STATUS_DONE, STATUS_SELECTED, STATUS_REJECTED} or not table_path.exists():
            continue
        table = pd.read_csv(table_path)
        if table.empty:
            continue
        baseline = table.loc[table["scenario_id"] == "baseline"]
        if baseline.empty:
            raise ValueError(f"robustness_table.csv missing baseline row for {manifest_row.run_id}")
        baseline_row = baseline.iloc[0]
        neighbors = table.loc[table["scenario_id"] != "baseline"].copy()
        if neighbors.empty:
            raise ValueError(f"robustness_table.csv has no neighbor rows for {manifest_row.run_id}")

        baseline_sharpe = max(float(baseline_row["daily_sharpe"]), 0.01)
        fragility_row = {
            **manifest_row.to_record(),
            "baseline_daily_sharpe": float(baseline_row["daily_sharpe"]),
            "baseline_total_return_pct": float(baseline_row["total_return_pct"]),
            "baseline_max_drawdown_pct": float(baseline_row["max_drawdown_pct"]),
            "neighbor_median_sharpe_ratio": float(neighbors["daily_sharpe"].median() / baseline_sharpe),
            "neighbor_positive_share": float((neighbors["total_return_pct"] > 0).mean()),
            "neighbor_dd_buffer": float(neighbors["max_drawdown_pct"].median() - float(baseline_row["max_drawdown_pct"])),
            "distance_from_baseline": baseline_distance(params_from_manifest_row(manifest_row)),
        }
        fragility_row["fragility_pass"] = bool(
            fragility_row["neighbor_median_sharpe_ratio"] >= 0.85
            and fragility_row["neighbor_positive_share"] >= 0.60
            and fragility_row["neighbor_dd_buffer"] >= -5.0
        )
        rows.append(fragility_row)

    if not rows:
        return pd.DataFrame(columns=MANIFEST_COLUMNS + ["rank"])

    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=[
            "fragility_pass",
            "baseline_daily_sharpe",
            "neighbor_median_sharpe_ratio",
            "neighbor_positive_share",
            "neighbor_dd_buffer",
            "distance_from_baseline",
            "scenario_id",
        ],
        ascending=[False, False, False, False, False, True, True],
    ).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


def selected_rows_for_stage(leaderboard: pd.DataFrame, stage: str) -> pd.DataFrame:
    if leaderboard.empty:
        return leaderboard.copy()

    if stage == STAGE_STAGE1_MARGINAL:
        frames = [_passed_rows(leaderboard[leaderboard["changed_rule"] == code], "hard_pass").head(5) for code in CANONICAL_RULE_CODES]
        return pd.concat(frames, ignore_index=True)
    if stage == STAGE_STAGE2_MARGINAL:
        frames = [_passed_rows(leaderboard[leaderboard["changed_rule"] == code], "hard_pass").head(3) for code in CANONICAL_RULE_CODES]
        return pd.concat(frames, ignore_index=True)
    if stage == STAGE_STAGE3_JOINT:
        return top_n_overall(leaderboard, count=5, pass_column="hard_pass")
    if stage == STAGE_STAGE4_JOINT_FRAGILITY:
        return top_n_overall(leaderboard, count=3, pass_column="fragility_pass")
    if stage == STAGE_STAGE5_ORDER:
        return top_n_overall(best_order_per_parent(leaderboard), count=3)
    if stage == STAGE_HOLDOUT:
        return top_n_overall(leaderboard, count=1)
    if stage == STAGE_STAGE1_ISOLATED:
        return leaderboard.iloc[0:0].copy()
    return top_n_overall(leaderboard, count=1)


def write_stage_leaderboard_outputs(run_root: Path, stage: str, leaderboard: pd.DataFrame) -> PromotionOutput:
    stage_dir(run_root, stage).mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(stage_leaderboard_path(run_root, stage), index=False)

    selected = selected_rows_for_stage(leaderboard, stage)
    selected.to_csv(stage_selected_path(run_root, stage), index=False)

    selected_run_id_list: list[str] = [str(x) for x in selected["run_id"].tolist()] if not selected.empty else []
    summary = {
        "stage": stage,
        "rows": int(len(leaderboard)),
        "selected_rows": int(len(selected)),
        "selected_run_ids": selected_run_id_list,
    }
    stage_summary_path(run_root, stage).write_text(stable_json(summary, indent=2), encoding="utf-8")

    selected_run_ids = set(selected_run_id_list)
    status_by_run_id: dict[str, str] = {}
    if stage == STAGE_STAGE1_ISOLATED:
        for run_id in leaderboard["run_id"].tolist():
            status_by_run_id[str(run_id)] = STATUS_DONE
    elif stage == STAGE_STAGE4_JOINT_FRAGILITY:
        for _, row in leaderboard.iterrows():
            status_by_run_id[str(row["run_id"])] = STATUS_SELECTED if str(row["run_id"]) in selected_run_ids else (STATUS_REJECTED if not bool(row["fragility_pass"]) else STATUS_DONE)
    elif stage == STAGE_HOLDOUT:
        for _, row in leaderboard.iterrows():
            status_by_run_id[str(row["run_id"])] = STATUS_SELECTED if str(row["run_id"]) in selected_run_ids else STATUS_DONE
    else:
        for _, row in leaderboard.iterrows():
            if str(row["run_id"]) in selected_run_ids:
                status_by_run_id[str(row["run_id"])] = STATUS_SELECTED
            elif "hard_pass" in leaderboard.columns and not bool(row["hard_pass"]):
                status_by_run_id[str(row["run_id"])] = STATUS_REJECTED
            else:
                status_by_run_id[str(row["run_id"])] = STATUS_DONE
    update_manifest_statuses(run_root, stage, status_by_run_id)
    return PromotionOutput(stage=stage, selected_run_ids=tuple(selected_run_id_list))

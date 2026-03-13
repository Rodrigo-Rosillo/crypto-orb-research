from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.tuning import (  # noqa: E402
    EVAL_ROBUSTNESS,
    EVAL_WALK_FORWARD,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_REJECTED,
    STATUS_RUNNING,
    STATUS_SELECTED,
    load_manifest,
    load_run_settings,
    stage_manifest_path,
    update_manifest_statuses,
)
from scripts.robustness_table import RobustnessRunConfig, run_robustness_table  # noqa: E402
from scripts.walk_forward import WalkForwardRunConfig, run_walk_forward  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run a generated tuning stage manifest.")
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--overwrite", action="store_true")
    return ap


def _result_exists(row) -> bool:
    out_dir = Path(row.out_dir)
    if row.evaluation_type == EVAL_WALK_FORWARD:
        return (out_dir / "walk_forward_folds.csv").exists()
    if row.evaluation_type == EVAL_ROBUSTNESS:
        return (out_dir / "robustness_table.csv").exists()
    raise ValueError(f"Unsupported evaluation type: {row.evaluation_type}")


def _run_walk_forward(row, settings) -> None:
    run_walk_forward(
        WalkForwardRunConfig(
            config=row.config_path,
            data="",
            valid_days="data/processed/valid_days.csv",
            out_dir=row.out_dir,
            engine=settings.engine,
            train_months=int(settings.train_months or 24),
            test_months=int(settings.test_months or 6),
            step_months=int(settings.step_months or 6),
            start=settings.start,
            end=settings.end,
            fee_mult=float(settings.fee_mult),
            slippage_bps=float(settings.slippage_bps),
            delay_bars=int(settings.delay_bars),
            leverage=float(settings.leverage),
            mmr=float(settings.mmr),
            funding_per_8h=float(settings.funding_per_8h),
            tune_adx_threshold=(),
        )
    )


def _run_robustness(row, settings) -> None:
    run_robustness_table(
        RobustnessRunConfig(
            config=row.config_path,
            data="",
            valid_days="data/processed/valid_days.csv",
            out_dir=row.out_dir,
            adx_threshold_grid=tuple(settings.adx_threshold_grid),
            orb_start_grid=tuple(settings.orb_start_grid),
            orb_window_min=int(settings.orb_window_min),
            cutoff_offset_min=int(settings.cutoff_offset_min),
            engine=settings.engine,
            fee_mult=float(settings.fee_mult),
            slippage_bps=float(settings.slippage_bps),
            delay_bars=int(settings.delay_bars),
            leverage=float(settings.leverage),
            mmr=float(settings.mmr),
            funding_per_8h=float(settings.funding_per_8h),
            start=settings.start,
            end=settings.end,
            objective=str(settings.objective),
            max_scenarios=int(settings.max_scenarios),
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_root = Path(args.run_root)
    settings = load_run_settings(run_root)
    stage_settings = settings.stages[str(args.stage)]
    manifest_rows = load_manifest(stage_manifest_path(run_root, str(args.stage)))

    total = 0
    skipped = 0
    failed = 0

    for row in manifest_rows:
        if not args.overwrite and row.status in {STATUS_DONE, STATUS_SELECTED, STATUS_REJECTED} and _result_exists(row):
            skipped += 1
            continue

        update_manifest_statuses(run_root, row.stage, {row.run_id: STATUS_RUNNING})
        total += 1
        try:
            if row.evaluation_type == EVAL_WALK_FORWARD:
                _run_walk_forward(row, stage_settings)
            elif row.evaluation_type == EVAL_ROBUSTNESS:
                _run_robustness(row, stage_settings)
            else:
                raise ValueError(f"Unsupported evaluation type: {row.evaluation_type}")
            update_manifest_statuses(run_root, row.stage, {row.run_id: STATUS_DONE})
        except Exception:
            failed += 1
            out_dir = Path(row.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "run_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            update_manifest_statuses(run_root, row.stage, {row.run_id: STATUS_FAILED})

    print(f"[OK] Stage {args.stage}: ran={total} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

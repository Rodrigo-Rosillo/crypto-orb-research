from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.tuning import (  # noqa: E402
    EVAL_ROBUSTNESS,
    EVAL_WALK_FORWARD,
    aggregate_fragility_stage,
    aggregate_walk_forward_stage,
    load_manifest,
    load_run_settings,
    stage_manifest_path,
    write_stage_leaderboard_outputs,
)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Aggregate tuning stage outputs into leaderboard artifacts.")
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--stage", required=True)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_root = Path(args.run_root)
    settings = load_run_settings(run_root)
    stage_settings = settings.stages[str(args.stage)]
    manifest_rows = load_manifest(stage_manifest_path(run_root, str(args.stage)))

    if stage_settings.evaluation_type == EVAL_WALK_FORWARD:
        leaderboard = aggregate_walk_forward_stage(manifest_rows, str(args.stage))
    elif stage_settings.evaluation_type == EVAL_ROBUSTNESS:
        leaderboard = aggregate_fragility_stage(manifest_rows)
    else:
        raise ValueError(f"Unsupported evaluation type: {stage_settings.evaluation_type}")

    promotion = write_stage_leaderboard_outputs(run_root, str(args.stage), leaderboard)
    print(f"[OK] Wrote leaderboard for {args.stage} with {len(leaderboard)} rows")
    print(f"[OK] Selected {len(promotion.selected_run_ids)} run(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

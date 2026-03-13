from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.tuning import (  # noqa: E402
    STAGE_HOLDOUT,
    STAGE_STAGE1_ISOLATED,
    STAGE_STAGE1_MARGINAL,
    STAGE_STAGE2_MARGINAL,
    STAGE_STAGE3_JOINT,
    STAGE_STAGE4_JOINT_FRAGILITY,
    STAGE_STAGE5_ORDER,
    build_holdout_manifest,
    build_stage1_manifests,
    build_stage2_manifest,
    build_stage3_manifest,
    build_stage4_manifest,
    build_stage5_manifest,
    initialize_run_root,
    load_base_strategy_definition,
    load_run_settings,
    load_stage_leaderboard,
    stage_manifest_path,
    write_stage_artifacts,
)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate multi-rule tuning manifests and YAML snapshots.")
    sub = ap.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Initialize run root with baseline and fragility stages.")
    init.add_argument("--run-root", required=True)
    init.add_argument("--config", default="config.yaml")

    for command in ["stage1", "stage2", "stage3", "stage4", "stage5", "holdout"]:
        cmd = sub.add_parser(command, help=f"Generate {command} manifests.")
        cmd.add_argument("--run-root", required=True)

    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_root = Path(args.run_root)

    if args.command == "init":
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = (REPO_ROOT / config_path).resolve()
        base_definition = load_base_strategy_definition(config_path)
        stage_rows = initialize_run_root(run_root, base_definition)
        print(f"[OK] Initialized tuning root: {run_root.resolve()}")
        for stage_name, rows in stage_rows.items():
            print(f"  - {stage_manifest_path(run_root, stage_name)} ({len(rows)} scenario(s))")
        return 0

    settings = load_run_settings(run_root)
    base_definition = load_base_strategy_definition(Path(settings.base_config_path))

    if args.command == "stage1":
        manifests = build_stage1_manifests(run_root, base_definition)
        for stage_name in [STAGE_STAGE1_MARGINAL, STAGE_STAGE1_ISOLATED]:
            rows = manifests[stage_name]
            write_stage_artifacts(run_root, base_definition, stage_name, rows)
            print(f"[OK] Wrote {stage_manifest_path(run_root, stage_name)} ({len(rows)} scenario(s))")
        return 0

    if args.command == "stage2":
        leaderboard = load_stage_leaderboard(run_root, STAGE_STAGE1_MARGINAL)
        rows = build_stage2_manifest(run_root, base_definition, leaderboard)
        write_stage_artifacts(run_root, base_definition, STAGE_STAGE2_MARGINAL, rows)
        print(f"[OK] Wrote {stage_manifest_path(run_root, STAGE_STAGE2_MARGINAL)} ({len(rows)} scenario(s))")
        return 0

    if args.command == "stage3":
        leaderboard = load_stage_leaderboard(run_root, STAGE_STAGE2_MARGINAL)
        rows = build_stage3_manifest(run_root, base_definition, leaderboard)
        write_stage_artifacts(run_root, base_definition, STAGE_STAGE3_JOINT, rows)
        print(f"[OK] Wrote {stage_manifest_path(run_root, STAGE_STAGE3_JOINT)} ({len(rows)} scenario(s))")
        return 0

    if args.command == "stage4":
        leaderboard = load_stage_leaderboard(run_root, STAGE_STAGE3_JOINT)
        rows = build_stage4_manifest(run_root, leaderboard)
        write_stage_artifacts(run_root, base_definition, STAGE_STAGE4_JOINT_FRAGILITY, rows)
        print(f"[OK] Wrote {stage_manifest_path(run_root, STAGE_STAGE4_JOINT_FRAGILITY)} ({len(rows)} scenario(s))")
        return 0

    if args.command == "stage5":
        leaderboard = load_stage_leaderboard(run_root, STAGE_STAGE4_JOINT_FRAGILITY)
        rows = build_stage5_manifest(run_root, leaderboard)
        write_stage_artifacts(run_root, base_definition, STAGE_STAGE5_ORDER, rows)
        print(f"[OK] Wrote {stage_manifest_path(run_root, STAGE_STAGE5_ORDER)} ({len(rows)} scenario(s))")
        return 0

    if args.command == "holdout":
        leaderboard = load_stage_leaderboard(run_root, STAGE_STAGE5_ORDER)
        rows = build_holdout_manifest(run_root, leaderboard)
        write_stage_artifacts(run_root, base_definition, STAGE_HOLDOUT, rows)
        print(f"[OK] Wrote {stage_manifest_path(run_root, STAGE_HOLDOUT)} ({len(rows)} scenario(s))")
        return 0

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

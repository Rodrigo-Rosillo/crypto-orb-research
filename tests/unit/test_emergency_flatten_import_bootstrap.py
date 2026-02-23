from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import pytest


def _norm_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def test_run_path_bootstraps_repo_root_for_forward_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "emergency_flatten.py"
    repo_root_norm = _norm_path(str(repo_root))

    # Simulate invoking the script by file path from a different working directory.
    monkeypatch.chdir(tmp_path)

    filtered_path = []
    for entry in sys.path:
        if not entry:
            continue
        if _norm_path(entry) == repo_root_norm:
            continue
        filtered_path.append(entry)
    monkeypatch.setattr(sys, "path", [str(script_path.parent), *filtered_path])

    removed_forward_modules: dict[str, object] = {}
    for name in list(sys.modules):
        if name == "forward" or name.startswith("forward."):
            removed_forward_modules[name] = sys.modules.pop(name)

    try:
        runpy.run_path(str(script_path), run_name="emergency_flatten_import_test")
    finally:
        sys.modules.update(removed_forward_modules)

    assert _norm_path(sys.path[0]) == repo_root_norm

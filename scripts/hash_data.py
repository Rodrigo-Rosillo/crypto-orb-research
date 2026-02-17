from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.utils import sha256_file  # noqa: E402


def compute_dataset_hash(file_entries: List[Dict[str, Any]]) -> str:
    """
    Combined hash = sha256 of the concatenation of each file's sha256,
    in a stable order (by relative path).
    """
    h = hashlib.sha256()
    for e in sorted(file_entries, key=lambda x: x["path"]):
        h.update(e["sha256"].encode("utf-8"))
    return h.hexdigest()


def build_manifest(data_dir: Path, patterns: List[str]) -> Dict[str, Any]:
    files: List[Path] = []
    for pat in patterns:
        files.extend(data_dir.rglob(pat))

    # Keep only real files, stable order
    files = sorted([p for p in files if p.is_file()], key=lambda p: str(p).lower())

    entries: List[Dict[str, Any]] = []
    for p in files:
        stat = p.stat()
        entries.append(
            {
                "path": p.relative_to(data_dir).as_posix(),
                "size_bytes": stat.st_size,
                "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "sha256": sha256_file(p),
            }
        )

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "generated_utc": datetime.now(tz=timezone.utc).isoformat(),
        "data_root": data_dir.as_posix(),
        "include_globs": patterns,
        "file_count": len(entries),
        "files": entries,
        "dataset_sha256": compute_dataset_hash(entries),
    }
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Hash raw dataset files into data/manifest.json")
    parser.add_argument("--data-dir", default="data/raw", help="Directory containing raw data files")
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["*.csv"],
        help="Glob patterns to include (searched recursively)",
    )
    parser.add_argument(
        "--out",
        default="data/manifest.json",
        help="Output manifest path (committed)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_path = Path(args.out).resolve()

    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    manifest = build_manifest(data_dir, args.patterns)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote manifest: {out_path}")
    print(f"Files: {manifest['file_count']}")
    print(f"Dataset hash: {manifest['dataset_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

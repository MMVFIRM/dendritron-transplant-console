#!/usr/bin/env python3
"""Build deterministic SHA-256 manifests for demo assets and release files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "release",
    "build",
    ".cache",
}


def sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def records(
    root: Path,
    *,
    relative_to: Path,
    excluded_names: set[str] | None = None,
) -> list[dict[str, object]]:
    excluded_names = excluded_names or set()
    output = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(relative_to)
        if relative.name in excluded_names:
            continue
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if relative.suffix in {".pyc", ".pyo"}:
            continue
        output.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return output


def write(path: Path, value: object) -> None:
    # LF on every platform, so the manifests match the repository bytes.
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    assets = root / "assets"
    asset_records = records(
        assets, relative_to=root, excluded_names={"ASSET_MANIFEST.json", "RELEASE_MANIFEST.json"}
    )
    write(
        root / "ASSET_MANIFEST.json",
        {
            "schema_version": 1,
            "release": "dendritron-transplant-console-v1.1.0",
            "kind": "frozen_demo_assets",
            "files": asset_records,
            "total_bytes": sum(int(item["bytes"]) for item in asset_records),
        },
    )
    release_records = records(
        root, relative_to=root, excluded_names={"RELEASE_MANIFEST.json"}
    )
    write(
        root / "RELEASE_MANIFEST.json",
        {
            "schema_version": 1,
            "release": "dendritron-transplant-console-v1.1.0",
            "files": release_records,
            "total_bytes": sum(int(item["bytes"]) for item in release_records),
        },
    )
    print(f"Wrote {len(asset_records)} asset hashes and {len(release_records)} release hashes.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verify every file represented by RELEASE_MANIFEST.json."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    errors = []
    for record in manifest["files"]:
        path = root / record["path"]
        if not path.is_file():
            errors.append(f"missing {record['path']}")
        elif path.stat().st_size != record["bytes"]:
            errors.append(f"size {record['path']}")
        elif digest(path) != record["sha256"]:
            errors.append(f"hash {record['path']}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(1)
    print(f"Verified {len(manifest['files'])} release files.")


if __name__ == "__main__":
    main()

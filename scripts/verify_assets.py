#!/usr/bin/env python3
"""Verify every frozen demo asset against the packaged SHA-256 manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


def sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "ASSET_MANIFEST.json"
    if not manifest_path.is_file():
        raise SystemExit("ASSET_MANIFEST.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    total_bytes = 0
    for record in manifest.get("files", []):
        relative = str(record["path"])
        path = root / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        size = path.stat().st_size
        total_bytes += size
        if size != int(record["bytes"]):
            failures.append(f"size mismatch: {relative}")
            continue
        observed = sha256(path)
        if observed != str(record["sha256"]):
            failures.append(f"SHA-256 mismatch: {relative}")
    if total_bytes != int(manifest.get("total_bytes", total_bytes)):
        failures.append("aggregate asset byte count mismatch")
    if failures:
        print("Asset verification failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        raise SystemExit(1)
    print(
        f"Verified {len(manifest.get('files', []))} frozen assets "
        f"({total_bytes:,} bytes)."
    )


if __name__ == "__main__":
    main()

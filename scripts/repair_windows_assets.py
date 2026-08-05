#!/usr/bin/env python3
"""Repair CRLF conversion of frozen text assets in older Windows clones."""
from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    root = parser.parse_args().root.resolve()
    manifest = json.loads((root / "ASSET_MANIFEST.json").read_text(encoding="utf-8"))
    repaired: list[str] = []

    for record in manifest["files"]:
        path = root / record["path"]
        if path.suffix not in {".json", ".jsonl"} or not path.is_file():
            continue
        current = path.read_bytes()
        if len(current) == record["bytes"] and sha256(current) == record["sha256"]:
            continue
        normalized = current.replace(b"\r\n", b"\n")
        if len(normalized) == record["bytes"] and sha256(normalized) == record["sha256"]:
            path.write_bytes(normalized)
            repaired.append(record["path"])

    if repaired:
        print(f"Repaired Windows line endings in {len(repaired)} frozen assets.")


if __name__ == "__main__":
    main()

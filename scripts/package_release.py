#!/usr/bin/env python3
"""Create a clean, manifest-bound release ZIP with one top-level directory."""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path
import zipfile

EXCLUDED = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "release",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("dist"))
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    subprocess.run(["python", str(root / "scripts/build_manifests.py")], check=True)
    subprocess.run(["python", str(root / "scripts/verify_assets.py")], check=True)
    subprocess.run(["python", str(root / "scripts/verify_release.py")], check=True)
    if not args.skip_tests:
        subprocess.run(["python", "-m", "pytest", "-q"], cwd=root, check=True)

    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f"{root.name}.zip"
    archive.unlink(missing_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(root)
            if any(part in EXCLUDED for part in relative.parts):
                continue
            if relative.suffix in {".pyc", ".pyo"}:
                continue
            handle.write(path, Path(root.name) / relative)

    record = destination / f"{root.name}-SHA256.txt"
    record.write_text(
        f"{sha256(archive)}  {archive.name}\n",
        encoding="utf-8",
    )

    with tempfile.TemporaryDirectory(prefix="dendritron-console-verify-") as temporary:
        extracted = Path(temporary)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(extracted)
        release = extracted / root.name
        subprocess.run(["python", "scripts/verify_assets.py"], cwd=release, check=True)
        subprocess.run(["python", "scripts/verify_release.py"], cwd=release, check=True)
        if not args.skip_tests:
            subprocess.run(["python", "-m", "pytest", "-q"], cwd=release, check=True)

    print(f"Release: {archive}")
    print(f"SHA-256: {sha256(archive)}")
    print(f"Record: {record}")


if __name__ == "__main__":
    main()

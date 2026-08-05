from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def test_asset_manifest_is_complete(root: Path) -> None:
    manifest = json.loads((root / "ASSET_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["files"]
    for record in manifest["files"]:
        path = root / record["path"]
        assert path.is_file(), record["path"]
        assert path.stat().st_size == record["bytes"]
        assert sha256(path) == record["sha256"]


def test_frontend_and_third_party_notice_files_are_present(root: Path) -> None:
    for relative in (
        "web/index.html",
        "web/styles.css",
        "web/app.js",
        "THIRD_PARTY_NOTICES.md",
        "licenses/APACHE-2.0-QWEN.txt",
        "licenses/PSF-LICENSE.txt",
    ):
        assert (root / relative).is_file(), relative

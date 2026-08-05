from __future__ import annotations

from pathlib import Path

import pytest

from demo_server.engine import DemoEngine


@pytest.fixture(scope="session")
def root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def engine(root: Path) -> DemoEngine:
    return DemoEngine(root, threads=1)

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
  .venv/bin/python -m pip install -r requirements.txt
fi
.venv/bin/python scripts/verify_assets.py
exec .venv/bin/python launch.py "$@"

#!/usr/bin/env bash
# One-command setup for macOS/Linux: create the venv, install, verify, configure.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python3}
"$PYTHON" -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
pip install -e ".[all]"
echo
recon3d setup "$@" || true
recon3d doctor || true
echo
echo "Done. Activate the environment with:  . .venv/bin/activate"
echo "Then try:  python scripts/make_demo_dataset.py --out ./demo/refs --views 9 --masks"

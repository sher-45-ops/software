#!/usr/bin/env bash
# End-to-end walkthrough on a synthetic reference set (no photos required).
set -euo pipefail

python scripts/make_demo_dataset.py --out ./demo/refs --views 9 --masks > /dev/null

recon3d doctor | head -12
recon3d create walkthrough --subject robot
recon3d add-images walkthrough ./demo/refs
recon3d reconstruct walkthrough --preset draft --texture-resolution 1024 --json \
  | python -c 'import json,sys; d=json.load(sys.stdin); print(d["quality"]["overall"], d["outputs"]["version_dir"])'
recon3d status walkthrough
recon3d export walkthrough --formats fbx,usdz
recon3d versions walkthrough

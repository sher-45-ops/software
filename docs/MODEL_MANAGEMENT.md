# Model management

Recon3D's reconstruction is classical and self-contained: **no model download is required
for any core feature.** Everything in the pipeline (segmentation, camera solve, carving,
depth plane sweep, UVs, texturing, materials, rigging, LODs, export) is implemented with
numpy/OpenCV/scikit-image/trimesh and runs offline.

On top of that, optional open models can improve specific stages. They are managed by
`recon3d models`, recorded in `recon3d/modelzoo/registry.json`, and **never downloaded
implicitly**.

## Commands

```bash
recon3d models list                    # ids, purpose, licence, size, install state
recon3d models download                # the recommended default (depth_anything_v2_small)
recon3d models download u2netp_salient # a specific model
recon3d models verify                  # re-check every installed checksum
recon3d models delete <id> --yes       # remove files (requires confirmation)
recon3d setup --with-models            # optional: fetch the recommended set at setup time
```

`--json` on any of these returns the machine-readable form (used by the API and agents).

## Registry contents

| id | Model | Stage | Runtime | Licence | Size |
| --- | --- | --- | --- | --- | --- |
| `depth_anything_v2_small` | Depth Anything V2 (Small, ONNX) | `depth_estimation` | onnxruntime | **Apache-2.0** | ~99 MB |
| `u2netp_salient` | U²-Net-p salient object detection (ONNX) | `segmentation` | onnxruntime | **Apache-2.0** | ~4 MB |

Deliberate omissions: Depth Anything V2 **Base/Large/Giant are CC-BY-NC-4.0** and UniDepth
is CC-BY-NC-4.0, so they are not offered — the engine will not hand you a licence trap.
Rigging models (e.g. UniRig, MIT) and metric-depth models (Metric3D, BSD-2-Clause) are
candidates for future optional backends; nothing in the product depends on them.

## Storage, checksums and integrity

Models live in `<data root>/models/<id>/`. The first successful download records the file's
SHA-256 into `<data root>/models/state.json`; every later use verifies it, and
`recon3d models verify` re-checks on demand. Upstream files are pinned by URL in the
registry; if you prefer your own export, drop the file in place and run
`recon3d models verify` — a mismatch is reported rather than ignored.

## Offline and air-gapped use

Three layers of protection, all default-safe:

1. `allow_network_downloads = false` (config) blocks the downloader outright.
2. `RECON3D_OFFLINE=1` / `recon3d setup --offline` sets `offline = true`, which blocks
   downloads *and* any runtime attempt to fetch anything.
3. `recon3d models download` requires an explicit model id (or the documented default) and
   prints exactly what it will fetch, from where, and under which licence.

When a model is missing, the engine falls back to the classical implementation for that
stage and records the fallback in the stage report — never an error, never a silent
quality claim.

## Using a downloaded model

```bash
recon3d models download depth_anything_v2_small
recon3d reconstruct hero --preset high --param use_neural_depth=true
```

The stage report then includes a `backends` block naming the model, its licence and whether
it was used. `recon3d doctor` lists which optional runtimes (onnxruntime, torch) and models
are present, and `recon3d info --json` documents the fallbacks.

## Security

Download URLs must be `http(s)`. Model ids are validated as identifiers, so
`../../etc/passwd` is rejected before any filesystem work. Downloads are written to a
temporary file inside the model directory and only moved into place after the checksum
passes. Nothing is ever executed from a downloaded file: ONNX models are loaded by
onnxruntime as data.

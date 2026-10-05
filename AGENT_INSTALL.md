# AGENT_INSTALL - bootstrap from a bare machine

This file is written for an AI agent (or a human following an agent's steps) that has been
given nothing but this repository URL. Following it end-to-end produces a working
reconstruction engine and a verified asset. No prior context, no hidden config, no external
AI service.

> **Repository:** <https://github.com/sher-45-ops/software> (branch
> `arena/01a10ae3-software`, which is where this project lives)

## 0. Prerequisites

- Python **3.9+** (3.10-3.12 recommended) with `venv` and `pip`.
- ~2 GB of free disk (plus your images; plus ~100 MB if you later download an optional model).
- No GPU, no CUDA, no Docker, no network *after* installation if you prefer.

## 1. Clone and install (one command each)

```bash
git clone -b arena/01a10ae3-software https://github.com/sher-45-ops/software.git recon3d-engine
cd recon3d-engine

python -m venv .venv
# Windows (PowerShell): .\.venv\Scripts\Activate.ps1
# Windows (cmd):        .venv\Scripts\activate.bat
# macOS / Linux:        . .venv/bin/activate

pip install -e ".[all]"
```

`.[all]` adds the API, MCP, UV/GLB quality extras and onnxruntime. The engine also runs with
plain `pip install -e .`; the missing pieces simply fall back (and `doctor` says so).

Verify:

```bash
recon3d --version      # recon3d 1.0.0
recon3d doctor --json  # hardware, dependencies, backends, data root
recon3d info --json    # stages, presets, parameters, output tree, formats
```

Windows-only notes: use `python` from an official installer; if you are on a machine where
`pip install` needs a proxy, set `HTTPS_PROXY` for the install and then run
`recon3d setup --offline` so the engine itself never touches the network.

## 2. Configure

```bash
recon3d setup                       # writes ~/.recon3d/config.json and creates the tree
recon3d setup --data-root D:\recon3d-data   # or an explicit location
```

The data root holds everything the engine writes: `projects/`, `models/`, `cache/`, `logs/`,
`config.json`. Override per run with the `RECON3D_HOME` environment variable.

## 3. Prove it works (no images needed)

```bash
python scripts/make_demo_dataset.py --out ./demo/refs --views 9 --masks
recon3d create demo --subject robot
recon3d add-images demo ./demo/refs
recon3d reconstruct demo --preset draft --json
```

The last command prints a JSON object with `quality`, `statistics` and `outputs`. Expect:
`state: completed`, a mesh with real geometry (thousands of triangles), textures written,
and a `version_dir` containing `final/`, `textures/`, `reports/`. Save the paths.

## 4. Reconstruct from real images

```bash
recon3d create hero --subject character
recon3d add-images hero /path/to/front.png /path/to/right.png /path/to/back.png \
                     /path/to/left.png /path/to/front_right.png /path/to/back_left.png
recon3d reconstruct hero --preset standard --texture-resolution 2048 \
    --formats glb,obj --json
```

Rules that keep the result honest:

- **Turntable sets work best**: 4-12 views around the subject at roughly equal angles,
  plus a top view if available. Images must show the *same* subject and *same* style.
- **Read `quality` before using the asset.** `mean_silhouette_iou` below ~0.6, a `grade` of
  `poor`/`bad`, or a non-empty `stages_failed` means the asset is not ready; report that to
  the user instead of shipping it.
- `missing_regions` lists what no reference image observed. It is not invented.

## 5. Retrieve the result

```bash
recon3d status hero --json | python -c "import json,sys; print(json.load(sys.stdin)['outputs'])"
```

The output tree is fixed and documented:

```
<data root>/projects/<project>/versions/<version>/
├── final/       model.glb, model.obj (+ .mtl), fbx/stl/ply/usd/usdz as requested
├── textures/    basecolor, normal, roughness, metallic, ao, orm (PNG)
├── lod/         lod0.glb … lod3.glb
├── rig/         rig.json, skin_weights.npz, skin.json
├── previews/    turntable.gif, stills, wireframe/normal/uv/coverage renders
├── renders/     reference-angle renders used by the comparison loop
├── reports/     quality.json, statistics.json, stages.json, reference_comparison.json,
│                image_quality.json, validation_report.json
└── intermediate/ point cloud, per-stage dumps, checkpoints
```

## 6. Interfaces for automation

```bash
recon3d serve --port 8760      # REST + WebSocket  (docs/API.md)
recon3d studio                 # the same server + browser UI
python -m recon3d.mcpserver    # MCP over stdio   (docs/MCP.md)
python -m recon3d.mcpserver --json   # lists the 5 tools without starting a client
```

MCP client config (Claude Code / Desktop style):

```json
{"mcpServers": {"recon3d": {
  "command": "/absolute/path/.venv/bin/python",
  "args": ["-m", "recon3d.mcpserver"],
  "env": {"RECON3D_HOME": "/absolute/path/data-root"}}}}
```

Tools: `recon3d_doctor`, `recon3d_create_project`, `recon3d_reconstruct`,
`recon3d_job_status`, `recon3d_list_outputs`. No shell execution is required, and none of
them can escape the data root.

## 7. Verify the installation the way CI does

```bash
pytest -q                                   # unit + integration tests
python scripts/e2e_check.py --preset draft  # end-to-end gate; prints PASS/FAIL per check
```

`scripts/e2e_check.py` renders a synthetic reference set, runs the full pipeline, and
compares the result against the references and the ground truth. It exits non-zero when the
reconstruction is not genuinely good, so it is safe to gate on.

## 8. Troubleshooting fast paths

| Symptom | Fix |
| --- | --- |
| `ModuleNotFoundError: recon3d` | activate the venv, or run `python -m recon3d.cli.main` |
| `the API server needs FastAPI + uvicorn` | `pip install "recon3d[api]"` |
| Out-of-memory during a run | `--preset draft`, lower `--texture-resolution`, or a machine with more RAM |
| Warnings about feature matches | references are textureless/repetitive; the silhouette solver still works for turntable sets |
| `stage_failed` | the report names the stage; fix the cause and re-run - checkpoints resume |
| Model download refused | offline mode is on; it is a deliberate default |

Full list: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

## 9. What this engine will never do

- Call OpenAI/Anthropic/Gemini/Replicate/Meshy/Tripo/Stability or any cloud service.
- Return a primitive, billboard, depth-map trick or placeholder and call it a reconstruction.
- Require a GUI: every operation above is headless.
- Hide a failure: skipped stages, missing regions and unverified exports are all reported.
- Execute shell commands on behalf of an agent request.

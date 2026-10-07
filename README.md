# Recon3D Engine

**Turn a handful of reference images into a real 3D asset — locally, on your CPU, with no
generative AI service involved.**

`recon3d` is a local-first multi-view reconstruction engine: you give it front/back/left/
right/three-quarter/top photographs (or renders) of a character, creature, vehicle, prop or
object, and it produces an actual textured mesh — real geometry, real UVs, real PBR maps,
optional rig and LODs — plus machine-readable reports about how good the result actually is.

- **Genuinely 3D.** The output is carved and fused from your images: silhouette carving,
  photometric shell carving, multi-view depth fusion, marching cubes, clean-up, decimation.
  No billboards, no depth-map fakes, no primitives standing in for your subject.
- **Local-first.** Everything runs on the machine you are on. No OpenAI/Anthropic/Gemini/
  Replicate/Meshy/Tripo/Stability calls, no cloud GPU, no account, no API key. Optional
  neural helpers (depth, saliency) are open models you download yourself with checksums.
- **Headless by design.** Every operation is available through the CLI, the REST/WebSocket
  API and an MCP server — a GUI is a convenience, never a requirement.
- **Honest.** Quality is measured from the produced files and reported, including what is
  missing. A bad reconstruction is reported as bad, not polished into a fake success.

---

## Quick start

```bash
git clone -b arena/01a10ae3-software https://github.com/sher-45-ops/software.git recon3d-engine
cd recon3d-engine
bash scripts/install.sh                           # Windows: powershell scripts/install.ps1
# ...or step by step:
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[all]"

recon3d setup                                     # create data root + config
recon3d doctor                                    # check hardware/deps/backends

recon3d create robot --subject robot              # a project
recon3d add-images robot ./photos                 # front.png, back.png, ...
recon3d reconstruct robot --preset standard --formats glb,obj
```

The asset lands in your data root (`~/.recon3d` by default, override with
`RECON3D_DATA_ROOT` or `recon3d setup --data-root`):

```
~/.recon3d/projects/robot/versions/v001/
├── final/            model.glb, model.obj + .mtl, (fbx/stl/ply/usd as requested)
├── textures/         basecolor.png normal.png roughness.png metallic.png ao.png orm.png
├── lod/              lod0.glb … lod3.glb
├── rig/              rig.json, skin_weights.npz, skin.json      (when rigging is on)
├── previews/         turntable.gif, stills, wireframe/normal/uv/coverage renders
├── renders/          full-resolution reference-angle renders
├── reports/          quality.json statistics.json stages.json reference_comparison.json …
└── intermediate/     point cloud, carved-volume report, per-stage dumps
```

No images at hand? Generate a synthetic reference set and try the whole thing offline:

```bash
python scripts/make_demo_dataset.py --out ./demo/refs --views 9 --masks
recon3d create demo --subject robot
recon3d add-images demo ./demo/refs
recon3d reconstruct demo --preset draft
```

---

## Interfaces

| Interface | How | Docs |
| --- | --- | --- |
| **CLI** | `recon3d doctor \| create \| add-images \| reconstruct \| status \| jobs \| cancel \| retry \| versions \| export \| serve \| studio \| models \| info \| projects \| project \| setup`, every command supports `--json` | [docs/CLI.md](docs/CLI.md) |
| **REST + WebSocket** | `recon3d serve --port 8760` → `POST /v1/projects/{id}/reconstruct`, `GET /v1/jobs/{id}`, `WS /v1/ws/jobs/{id}` | [docs/API.md](docs/API.md) |
| **Studio (web UI)** | `recon3d studio` → browser at `http://127.0.0.1:8760` | [docs/API.md](docs/API.md) |
| **MCP server** | `python -m recon3d.mcpserver` → tools `recon3d_doctor`, `recon3d_create_project`, `recon3d_reconstruct`, `recon3d_job_status`, `recon3d_list_outputs` | [docs/MCP.md](docs/MCP.md) |
| **Python** | `from recon3d.core.pipeline import run_pipeline` | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |

Agents that have never seen this repository should start at **[AGENT_INSTALL.md](AGENT_INSTALL.md)**
for a self-contained "clone → install → reconstruct → retrieve output" recipe, and read
**[docs/AGENT_USAGE.md](docs/AGENT_USAGE.md)** for the full operating contract.

---

## What the pipeline does

18 stages, each checkpointed, resumable, cancellable and reported individually:

| # | Stage | What happens |
| --- | --- | --- |
| 1 | `ingestion` | load images, EXIF/orientation normalisation, hashing, dedupe |
| 2 | `validation` | blur, exposure, resolution, duplicate and coverage checks per image |
| 3 | `segmentation` | subject masks (classical threshold/saliency + grabcut; optional local salient model) |
| 4 | `analysis` | subject-type classification, symmetry detection, material hints |
| 5 | `features` | feature extraction and cross-view matching evidence |
| 6 | `camera_estimation` | ring geometry, **lens (FOV) estimation**, scale gauge |
| 7 | `mesh_reconstruction` | silhouette carve + photometric shell carve + depth fusion + **symmetry completion** → mesh |
| 8 | `silhouette_refine` | re-render, compare, nudge cameras — accepted only if agreement improves |
| 9 | `mesh_cleanup` | floaters, holes, non-manifold, normals, weld, degenerate faces |
| 10 | `optimization` | topology + polycount budget |
| 11 | `uv` | xatlas unwrap (built-in fallback) + quality report |
| 12 | `texture` | multi-view projection, **diffusion inpainting of unobserved texels** → basecolor/normal/roughness/metallic/AO + ORM |
| 13 | `materials` | evidence-based PBR material estimate + agent overrides |
| 14 | `rigging` | humanoid/creature skeleton, auto skin weights, deformation stress test |
| 15 | `lod` | LOD0–LOD3 chain with per-level budgets and surface-error measurement |
| 16 | `preview` | turntable, stills, wireframe/normal/UV/coverage renders |
| 17 | `comparison` | model re-rendered from reference angles, compared with the references |
| 18 | `export` | GLB/GLTF/OBJ(+MTL)/FBX/STL/PLY/USD(A)/USDZ, each verified by re-reading it |

Run a subset with `--stages uv,texture`; resume an interrupted run with the same command
(checkpoints in `intermediate/<stage>/_checkpoint.json` are reused automatically).

---

## Parameters and presets

`--preset fast|draft|standard|high|ultra|game_ready|cinematic` sets geometry effort, texture
resolution, carve levels, refinement passes and LOD policy in one word. `fast` is the
iteration preset (smaller carve, 1K textures): use it while tuning views, then re-run with
`standard` for the final asset. Override any
individual parameter on the CLI (`--texture-resolution 4096 --target-polycount 30000`), in
the API body, or via MCP arguments. Full list: `recon3d info --json`.

```bash
recon3d reconstruct hero --preset high --texture-resolution 4096 \
    --target-polycount 60000 --units meters --subject-height 1.8 \
    --formats glb,usdz,fbx --rig --param photo_consistency=true
```

The presets never silently ignore a request: `policy` in each preset JSON says whether a
parameter is honoured, and `reports/quality.json` records what was actually produced.

---

## Quality and honesty

Every run writes `reports/quality.json` with a 0–100 score broken into geometry quality,
reference similarity, mesh health and texture quality, computed from the produced files
(not from intent). It includes:

- mean silhouette IoU against the reference masks, per-view,
- mean colour RMSE for the textured model re-rendered from reference angles,
- coverage delta (which parts of the subject were never observed),
- `missing_regions` — the areas no reference image saw, so nothing is invented there,
- `inferred` — **every region the engine filled in rather than observed**, each with its
  kind (`symmetry_completion`, `texture_inpainting`), size and a `confidence` measured from
  the distance to real evidence; the same entries are echoed into `missing_regions` as
  `inferred: …` so a caller that only reads that list still sees them,
- `budgets` — for presets that promise named budgets (`game_ready` ships a mobile and a
  desktop one), the target, the measured actual value and a per-metric pass/fail,
- warnings for every degraded or skipped stage, and
- `reports/reference_comparison.json` for the render-vs-reference loop.

Two things are deliberately *not* silent: a texture region no reference observed is filled
by classical diffusion (no neural model, no generative API) and reported with a confidence,
and a mirror-completed volume is kept only when it renders **better** against your
references than the carve it replaces — otherwise it is discarded and the report says so.
Inferred regions never raise the quality score.

Reproduce the numbers yourself with the end-to-end gate:

```bash
python scripts/e2e_check.py --preset standard --views 9 --min-iou 0.55
pytest -q                       # unit + integration tests
```

### Verified end-to-end (this machine)

The gate below was run in this repository on a **2-core / 4 GB, CPU-only** Linux box against
a synthetic 9-view robot reference set with ground-truth masks
(`scripts/make_demo_dataset.py`). It builds the references, runs the full `standard` pipeline
through the public API and checks the produced files — scores are read from
`reports/quality.json` / `reports/statistics.json` of that run, never supplied by hand:

| Measurement | Value |
| --- | --- |
| Overall quality | **96.3 / 100 (excellent)** — geometry 99.7, reference similarity 94.0, mesh health 100.0, texture 82.0 |
| Mean silhouette IoU (8 ring views vs ground truth) | **0.8955** (min 0.8713) |
| Mean colour RMSE / coverage delta | 0.1677 / −0.0142 |
| Exported mesh | 62 018 vertices / **124 032 triangles**, **1 component, watertight, genus 0** |
| Textures | 1024² basecolor, normal, roughness, metallic, AO, packed ORM |
| UV atlas | 807 islands, mean distortion 0.236 |
| LODs | 3 (124 032 / 62 016 / 31 008 triangles) |
| End-to-end wall clock | 1 188 s (~20 min) for the standard preset on the CI runner; **2 097 s (35 min) and 688 MB peak RSS** re-measured on a 2-core / 4 GB CPU box for 1.1.0, where the carve budget is clamped to 224³ |
| `fast` preset (same 9 views, 1024² textures) | **1 117 s (18.6 min)**, 481 MB peak RSS, quality 93.3 / mean IoU 0.882 — use it while tuning the reference set |
| Gate result | `RESULT: PASS` (job completed, no failed stages, mesh + textures + GLB verified) |

The full report of that run ships with the release (`recon3d-e2e-report-v1.0.0.json`) and in
`.e2e/e2e-standard.json` if you run the gate yourself — every number above is read back out of
`reports/quality.json` / `reports/statistics.json`, never supplied by hand. Health metrics are
measured on a welded copy of the surface, so the UV seam split (807 islands) is not mistaken
for loose components.

Three rows are worth reading as *behaviour*, not just numbers — each is a decision the engine
made against its own candidate geometry, and each is recorded verbatim in
`reports/stages.json`:

- The silhouette refinement **rejected its own candidate**: the rotation-only refinement
  looked better against the mesh it was tuned on, but on a freshly re-carved mesh it scored
  0.8999 against the original solve's 0.9037, so the original cameras and mesh were kept.
- The depth sweep's volume fusion was **not accepted** either: the fused surface scored
  0.797 against the silhouette hull's 0.9037, so the hull survived and the depth point cloud
  stayed a diagnostic artefact (`reports/stages.json` → `mesh_reconstruction.depth`).
- The reference-comparison alignment pass ran, measured **no** improvement (0.8813 → 0.8813)
  and therefore returned the model untouched rather than "improving" it.

That is the contract: the engine ships the geometry that measures best against your
references, and reports what it discarded and why.

## Hardware

Runs on a CPU. 4 GB RAM is enough for `draft`/`standard` with modest image counts; more
cores, more RAM and higher `--performance-mode` raise the carve resolution and texture
size. NVIDIA GPUs are used *if present* by optional backends (torch/onnxruntime), never
required. `recon3d doctor` prints the detected profile and the budget the engine will use.

## Optional accelerators

The engine works with none of these installed, and uses them when it finds them: COLMAP
(SfM), Open3D (decimation/Poisson), xatlas (UVs — the built-in unwrapper is the fallback),
Blender (binary FBX/renders), ONNX Runtime + Depth-Anything-V2-Small (Apache-2.0) and a
salient-object model for better masks. Licences and download policy:
[docs/LICENSES.md](docs/LICENSES.md) and [docs/MODEL_MANAGEMENT.md](docs/MODEL_MANAGEMENT.md).

## Documentation

| Document | Contents |
| --- | --- |
| [INSTALL.md](docs/INSTALL.md) | Windows/macOS/Linux install, venv, offline installs, data root |
| [CLI.md](docs/CLI.md) | every command, flag and exit code |
| [API.md](docs/API.md) | REST endpoints, WebSocket protocol, studio UI |
| [MCP.md](docs/MCP.md) | MCP server setup for Claude Code / Codex / any client |
| [AGENT_USAGE.md](docs/AGENT_USAGE.md) | how an agent should drive the engine, safety rules, recipes |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | module map, data flow, algorithms, extension points |
| [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | symptoms → causes → fixes |
| [MODEL_MANAGEMENT.md](docs/MODEL_MANAGEMENT.md) | local model registry, checksums, offline mode |
| [LICENSES.md](docs/LICENSES.md) | dependency/licence table and obligations |
| [AGENT_INSTALL.md](AGENT_INSTALL.md) | single-file bootstrap for a fresh agent |

## License

Apache-2.0. See [LICENSE](LICENSE) and [docs/LICENSES.md](docs/LICENSES.md) for the
third-party component table.

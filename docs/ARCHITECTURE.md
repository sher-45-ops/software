# Architecture

Recon3D is a local-first reconstruction **platform**, not a script. Every interface
(CLI, REST, MCP, Python) drives the same core, and every stage writes artefacts + JSON
reports to a predictable tree on disk.

```
recon3d/
├── __init__.py            VERSION and package metadata
├── config.py              Config, data root, env vars, hardware limits
├── errors.py              typed errors (code + HTTP status + details)
├── agent/manifest.py      capability_report(): stages, presets, params, formats
├── core/
│   ├── project.py         Project, Version, ReferenceImage, ProjectManager (on-disk truth)
│   ├── pipeline.py        StageSpec table, PipelineContext, run_pipeline, 18 stages
│   ├── jobs.py            Job/JobState/JobManager: progress, cancel, wait, checkpoints
│   ├── progress.py        ProgressEvent, ProgressReporter, logging setup
│   ├── resources.py       hardware profile, performance modes, voxel/RAM budgets
│   ├── security.py        PathSandbox, validation, destructive-op consent
│   └── store.py           atomic JSON/JSONL/npz helpers, hashing, timestamps
├── engine/
│   ├── ingestion/         loading, EXIF, masks
│   ├── validation/        image quality + artefact validation (validate_version)
│   ├── segmentation/      classical masks (+ optional ONNX salient model)
│   ├── analysis/          subject classification, view assignment, symmetry
│   ├── cameras/           SIFT ring solve, lens estimation, scale gauge, refinement
│   ├── reconstruction/    visual hull carve, photo shell carve, depth plane-sweep, mesh
│   ├── geometry/          cleanup (floaters/holes/normals/weld), quadric decimation
│   ├── textures/          xatlas/builtin UV, multi-view projection, PBR derivation
│   ├── materials/         19-entry PBR library, evidence-based assignment, overrides
│   ├── rigging/           humanoid/creature skeletons, auto weights, stress test
│   ├── optimization/      LOD chain, polycount budgets, silhouette retention
│   ├── preview/           turntable/stills/wireframe/normal/UV/coverage renders
│   ├── compare/           rasteriser, silhouette IoU, colour RMSE, quality scoring
│   ├── export/            GLB/GLTF/OBJ/FBX/STL/PLY/USD/USDZ writers + verifiers
│   └── backends/          optional-accelerator detection (COLMAP, Blender, Open3D…)
├── presets/*.json         draft, standard, high, ultra, game_ready, cinematic
├── modelzoo/              optional model registry + download manager
├── api/server.py          FastAPI REST + WebSocket + static studio UI
├── cli/main.py            14 subcommands, --json everywhere
├── mcpserver/server.py    5 whitelisted MCP tools
└── studio/static/         dependency-free single-page UI
```

## Data flow

```
images ──► ingestion ──► validation ──► segmentation ──► analysis
                                                             │
                                              subject type, views, symmetry
                                                             ▼
        features ──► camera_estimation ◄─────────── masks, images, EXIF
                          │  ring geometry, lens (FOV), scale gauge
                          ▼
   mesh_reconstruction ──► silhouette_refine ──► mesh_cleanup ──► optimization
        carve + photo shell + depth fusion      floaters/holes/normals/weld
                          ▼
        uv ──► texture ──► materials ──► rigging ──► lod ──► preview
                          ▼
        comparison (render vs references) ──► export (+ verify) ──► reports
```

State travels in `PipelineContext`, which holds the project, resolved parameters, caches
for expensive intermediates (images, masks, rig, mesh, UVs, textures) and the stage
reports. Anything large is written to disk and referenced by path, so a crashed run can be
resumed (`intermediate/<stage>/_checkpoint.json` records an inputs hash per stage).

## Camera solve (the part that decides everything)

1. **Views** — filenames, mask geometry and (when they agree) SIFT matches assign each
   image to a ring position (azimuth/elevation).
2. **Ring geometry** — azimuth from pair-wise essential matrices when features exist,
   else from view assignment priors; distances from the ring consistency.
3. **Lens** — `calibrate_lens_and_rig` sweeps FOV candidates, solving the rig and closing
   the scale gauge for each, carving a coarse hull and scoring silhouette agreement. The
   best candidate is refined. It is a *coarse* estimate: it is accurate to roughly ±4° and
   is reported honestly, not silently trusted.
4. **Scale gauge** — the reconstruction is only defined up to a similarity, so
   `calibrate_rig_height` measures the carved height and rescales until the subject is 1.0
   engine unit tall (or `--subject-height` metres on export).
5. **Refinement** — `refine_rig_silhouette` nudges **azimuth and elevation per view** against
   the reference mask, bounded and budgeted. The distance axis is deliberately disabled for
   this step: silhouette IoU of a *fixed* mesh rises when every camera simply walks closer,
   which is a scale-gauge change rather than a camera correction, and it cannot be validated
   against the very mesh it was tuned on (that made the exported model ~6 % too small while
   the stage reported "improved"). The refined rig is therefore re-carved and re-extracted at
   the full budget and the new (rig, mesh) pair is kept **only if the freshly carved geometry
   matches the references better**; otherwise the original pair is restored. Both numbers and
   the decision are written to `reports/stages.json` under `silhouette_refine`.

## Geometry

- **Silhouette carve**: hierarchical visual hull (coarse pass, then a refinement grid inside
  the occupied bbox) with an occupancy grid in `(nz, ny, nx)` order. The final level spends
  the whole `voxel_max` budget on the *tight* subject box (a low-RAM machine caps at 224³,
  see `mode_budget`): marching cubes places vertices inside cells, so a coarse grid loses
  roughly a voxel of thickness everywhere, which the comparison report sees as "thinner than
  the reference".
- **Photometric shell carve**: voxels near the surface are kept only when the sampled
  colours agree across views; the removal is capped so one bad mask cannot destroy the
  model.
- **Depth fusion** (optional, *validated*): a classical multi-view plane sweep per view;
  candidates are scored by cross-view colour agreement, fused and filtered, and then used to
  carve voxels no measurement supports. A sparse point cloud cannot support a surface, so the
  support radius scales with the **measured** sample spacing and the carve is skipped (with
  the reason recorded in `statistics["depth_fusion"]`) when samples are more than ~2.5 voxels
  apart. Even then the fused volume must *earn* its place: the fused surface is re-extracted
  and compared with the silhouette hull, and whichever scores higher against the references
  is the one that continues down the pipeline. Measured on the benchmark: a 22k-sample sweep
  shredded 82 % of a clean 224³ hull into 384 fragments (IoU 0.90 → 0.60); the guard now
  rejects it and the hull is kept.
- **Volume cleanup**: a morphological closing pass plus a small-blob filter removes
  pinhole/tunnel noise; the field is then blurred (`volume_sigma`, per-mode: 0.45–0.6) so
  marching cubes produces a clean shell instead of a genus-400 sponge, and extracted at
  `iso_level` (0.40–0.45 on the blurred field, which recovers the half-voxel the blur costs).
  Measured on the 9-view benchmark: 128³/unblurred → silhouette IoU 0.746;
  224³ + `volume_sigma 0.5` + `iso_level 0.42` → IoU ≈ 0.90 with a watertight, single-shell,
  genus-0 mesh.
- **Mesh cleanup**: floaters, isolated vertices, welds, hole filling, normal repair,
  degenerate/duplicate face removal and Laplacian smoothing (sharp-edge aware).

## Texturing and materials

`uv` runs xatlas (MIT) when installed, else the built-in chart packer, and reports island
count, packing efficiency, distortion, overlap and texel density. `texture` rasterises the
atlas, renders visibility buffers per reference view, projects the reference colours, fills
seams, and derives normal/roughness/metallic/AO from the projected result. `materials`
assigns a PBR profile from measured evidence (colour, saturation, gloss) and honours agent
overrides such as `material_request="brushed aluminium"`.

## Rigging, LODs and previews

`rigging` builds a humanoid or quadruped skeleton fitted to the subject bounds, computes
smooth skin weights with distance falloff, checks the bind pose and runs a deformation
stress test; output is `rig.json`, `skin_weights.npz`, `skin.json`. `lod` builds a chain
(1.0, 0.5, 0.25, 0.125 of LOD0) with per-level budgets, measures surface error against
LOD0 and (optionally) silhouette retention from the solved cameras. `preview` renders
turntables, stills and diagnostic maps; `comparison` re-renders the finished model from the
reference angles and compares silhouettes/colours with the references.

## Interfaces

`cli/main.py` is a thin formatting layer over the core; `--json` prints the same object the
API returns. `api/server.py` exposes those operations over HTTP and pushes job progress
over a WebSocket; artefact downloads are path-checked against the version directory.
`mcpserver/server.py` exposes five whitelisted tools, with no shell execution anywhere.
`core/security.py` confines all writes to a `PathSandbox`, validates images before copying
them into a project, and gates destructive operations behind explicit confirmation.

## Failure model

Stages declare whether they are optional. An optional stage that fails logs a warning,
records itself in the stage report and the pipeline continues with a degraded result; a
required stage failure aborts with `stage_failed` and marks the job failed, and a retry
resumes from the last valid checkpoint. `run_pipeline` never returns a success payload for a
run that produced no mesh.

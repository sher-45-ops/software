# Changelog

All notable changes to Recon3D Engine. This project follows
[Keep a Changelog](https://keepachangelog.com/) and [Semantic Versioning](https://semver.org/).

## [1.1.0] - 2026-10-07

Adds the three "finish the hard parts" features - all of them extensions of stages that
already existed, all of them measured, all of them disclosed.

### Added

- **Symmetry completion** (`symmetry_completion`, `symmetry_min_score`). Stage 4 already
  measures bilateral symmetry per reference and picks the mirror plane; the reconstruction
  stage now *uses* that evidence: it locates the plane offset that best explains the carved
  volume (Dice overlap of the volume with its own mirror), fills the voxels whose mirror
  image is occupied but which the carve left empty, and keeps the completed surface **only
  when it re-renders better against the references** than the carve it replaces. Refused or
  degenerate mirrors are reported with the reason, never applied silently. `--symmetry x|y|z`
  forces an axis, `--no-symmetry-completion` disables the fill, `--min-symmetry 0.65` tunes
  the threshold.
- **Texture inpainting** (`texture_inpainting`). The atlas texels no reference observed used
  to ship as magenta. They are now filled by a classical multi-scale (pull-push) diffusion
  fill - no neural network, no model weights, no external API - and each filled patch is
  measured: bounding box in UV space, texel count and a `confidence` that decays with the
  distance to the nearest texel a reference actually saw (a one-texel seam gap is trusted at
  ~0.75, a whole occluded flank much less). `--no-inpainting` restores the magenta + warning
  behaviour.
- **Honest inference reporting.** `reports/quality.json` gains `inferred` (one entry per
  filled region: kind, stage, region, size, confidence, evidence, verification) and
  `inferred_summary`; the same regions are echoed into `missing_regions` as `inferred: …`, so
  an agent that reads only that list cannot mistake a filled patch for an observed surface.
  Inferred regions add a warning and never raise a score.
- **A `fast` preset** (and a `fast` quality alias) for iteration: the existing `performance`
  mode (160³ carve, 1K textures, one refinement pass, 6 comparison renders) instead of
  `standard`'s 224-256³ carve. Measured on a 2-core / 4 GB CPU box with the same 9-view
  fixture: **1 117 s (18.6 min) and 481 MB peak RSS at quality 93.3 / IoU 0.882**, versus
  **2 097 s (35.0 min) and 688 MB at quality 93.7 / IoU 0.888** for `standard`.
- **Measured asset budgets on `game_ready`.** The preset now declares a `mobile` budget
  (12 000 triangles, 1 024 px texture, 4 LODs) alongside `desktop` (40 000 / 2 048 / 4); the
  LOD stage exports each as `lod/<name>.glb` and reports target-vs-actual with a per-metric
  pass/fail in `quality.json:budgets`. A budget that cannot be met is reported as failed
  instead of being quietly dropped. `recon3d info --json` exposes the budgets
  (`available_presets()[*].asset_budgets`).

### Changed

- `[all]` no longer installs `onnxruntime`; neural backends are opt-in
  (`pip install recon3d[neural]`) and stay disabled unless a model is downloaded on purpose.
- New CLI flags on `reconstruct`: `--symmetry`, `--min-symmetry`, `--no-symmetry-completion`,
  `--no-inpainting`.
- Version bumped to **1.1.0**. `v1.0.0` remains the original release: the first upload of it
  was scored 85.2/100, the re-cut of the same tag scores 96.3/100 (see release notes); this
  revision is published as `v1.1.0` so the two are never confused again.

### Tests

- 7 new tests: mirror completion on synthetic volumes (fills the missing lobe, is a no-op on
  a symmetric volume, refuses to invent a second subject), the inpainting fill and its
  per-region confidences, the disclosure path into `quality.json`/`missing_regions`, and the
  budget export (pass and fail).

## [1.0.0] - 2026-10-05

First release: a complete, local-first multi-view image-to-3D reconstruction engine.

> This release was re-cut on 2026-10-06 after a hardening pass (same version, because the
> first upload was never announced). The tag, the wheel/sdist and the attached gate report
> all point at the hardened commit.

### Fixed (hardening pass)

- **Checkpoints are now genuinely resumable.** The input hash no longer mixes in anything
  the pipeline itself derives during a run (analysed subject type, solved scale hint,
  measured scores) - it hashes the *requested* parameters, the image set and the input
  hashes of the stage's dependencies. A crashed or cancelled run now really continues
  instead of redoing every stage.
- **A stage whose artefacts vanished is rebuilt**, not silently re-used:
  `checkpoint_valid(..., require_outputs=...)` verifies the files each checkpoint promises,
  and unusable cache entries are invalidated and re-executed.
- **Every geometry stage persists its surface** (`intermediate/<stage>/mesh.ply`), so
  `mesh_reconstruction` (the expensive carve) can be resumed directly.
- `Project.add_images` no longer crashes with `SameFileError` when a reference already
  lives inside `input/original` - exactly what the REST upload path does.
- `GET /health` returned 500 because `JobManager.active_count` was serialised as a bound
  method; it is a property again.
- **`recon3d setup` crashed on a fresh machine**: `save_config` called
  `Config.to_dict(redact=False)` while `to_dict` took no arguments, so the first command of
  every install guide raised `TypeError` on an empty data root. `to_dict(redact=...)` now
  exists (redaction masks credential-like `extra` keys for diagnostics) and the documented
  `setup -> create -> add-images -> doctor` path is covered by a test.
- WebSocket progress (`/v1/ws/jobs/{id}`) failed with HTTP 403: FastAPI could not resolve
  annotations that were only imported inside functions. Every name in an annotation is now
  importable at module scope.

### Added (hardening pass)

- **Cancellation across processes**: `recon3d cancel JOB` writes a request file that a
  running job watches, so a second terminal or an agent can stop a reconstruction running
  under the API server - finished stages stay checkpointed.
- **Bounded retries**: a stage that fails for a transient reason is retried
  (`--stage-retries`, default 1 extra attempt); input errors, sandbox rejections and
  cancellations are never retried. Attempt counts land in the job record and `stages.json`.
- **Job recovery surface**: `recon3d jobs` lists every recorded run (including those from
  earlier processes) and flags resumable ones; `recon3d retry JOB` re-runs a failed or
  cancelled job from its persisted record, resuming from surviving checkpoints.
- Job records are written both to the global log directory (a discoverable index) and to
  the project's `intermediate/jobs/` (self-describing project folders).
- **A re-used stage brings its outputs along**: textures, rig, LODs and comparison renders
  are copied into the new version, and the version's asset pointers are rebuilt from disk,
  so every version is a complete snapshot. A fully cached re-run now takes seconds, produces
  the same files and the same measured quality as the original run, and the export stage is
  regenerated because its files are named after the version.
- **Resumed exports keep their UVs**: a mesh reloaded from disk gets its texture coordinates
  re-attached, and the `uv` stage persists its seam-split mesh, so a resumed run no longer
  emits a GLB without `TEXCOORD_0` while still shipping texture maps.
- Mesh health (components, watertightness, Euler number) is measured on a welded copy of the
  surface, so UV seam splitting is no longer reported as dozens of loose components.
- New tests: checkpoint/resume (`tests/test_resume.py`), cross-process cancellation and
  retry (`tests/test_robustness.py`), REST-only workflow incl. multipart upload, artefact
  download and the model registry (`tests/test_rest_workflow.py`), WebSocket progress
  (`tests/test_api_ws.py`), and documentation drift guards (`tests/test_manifest.py`:
  the CLI command list, the REST endpoints, the recovery promises and every pipeline
  parameter are checked against the code, so `agent-manifest.json` cannot go stale).

### Added

- **Pipeline (18 stages, checkpointed, resumable, cancellable)**: ingestion, validation,
  segmentation, analysis, features, camera_estimation, mesh_reconstruction,
  silhouette_refine, mesh_cleanup, optimization, uv, texture, materials, rigging, lod,
  preview, comparison, export.
- **Reconstruction**: hierarchical silhouette carving, photometric shell carving,
  multi-view depth fusion (plane sweep), morphological volume cleanup and field smoothing,
  marching cubes, hole filling, normal repair, quadric decimation.
- **Camera solving**: SIFT ring geometry, analysis-by-synthesis lens (FOV) estimation,
  scale-gauge closure against the reference silhouettes, bounded per-view silhouette
  refinement that is accepted only when agreement improves.
- **Texturing**: xatlas UV unwrapping with a dependency-free fallback, multi-view projection
  with seam filling, derived normal/roughness/metallic/AO maps and packed ORM.
- **Materials**: 19-entry PBR library, evidence-based assignment, agent overrides.
- **Rigging**: humanoid and creature skeletons, automatic skin weights, deformation stress
  test.
- **LODs**: LOD0–LOD3 with per-level texture budgets, measured surface error and optional
  silhouette retention.
- **Export**: GLB, GLTF, OBJ (+MTL), FBX (ASCII, binary with Blender), STL, PLY, USD, USDA,
  USDZ — each file re-read and verified after writing.
- **Quality reporting**: `reports/quality.json` with geometry / reference-similarity /
  mesh-health / texture scores, per-view silhouette IoU, colour RMSE, coverage delta and
  `missing_regions`; `reports/reference_comparison.json` for the render-vs-reference loop.
- **Interfaces**: CLI (14 subcommands, `--json` everywhere), REST API + WebSocket progress,
  dependency-free studio web UI, MCP server (5 whitelisted tools), Python API.
- **Presets**: draft, standard, high, ultra, game_ready, cinematic, with hardware-aware
  budgets and explicit parameter policy.
- **Robustness**: per-stage checkpoints, inputs hashing, caching, cancellation, retry,
  crash recovery, RAM/voxel budgets, performance modes.
- **Security**: path sandboxing, image validation, no shell execution, explicit confirmation
  for destructive operations, offline mode, checksum-verified optional model downloads.
- **Documentation**: README, INSTALL, CLI, API, MCP, AGENT_USAGE, ARCHITECTURE,
  TROUBLESHOOTING, MODEL_MANAGEMENT, LICENSES, AGENT_INSTALL, capability manifest.
- **Tests**: unit, security, CLI and end-to-end accuracy gate (`scripts/e2e_check.py`).

### Fixed during the 1.0.0 cycle

- `voxelize_mesh` used a ray parameter as a world coordinate and closed open meshes with
  infinity, corrupting every voxel measurement.
- Depth fusion crashed on broadcasting and shadowed its own output buffer, so depth
  evidence never reached the reconstruction.
- LOD generation emptied the pipeline mesh (LOD0 *is* the pipeline mesh) and re-exported
  full-resolution textures on every level, exhausting memory.
- Component counting on a textured mesh duplicated every texture per component, which could
  consume gigabytes; the engine no longer attaches texture images to the in-memory mesh.
- Carved volumes riddled with pinholes produced swiss-cheese surfaces (genus ~400); a
  morphological closing pass, small-blob filter and field blur (`volume_sigma`) plus an
  `iso_level` tuned for the blurred field now yield a clean, watertight, genus-0 shell.
- Silhouette refinement moved *every* camera closer (`delta_log_distance ≈ −0.06`) and
  accepted the change because it was scored against the stale mesh it had been tuned on,
  leaving the exported model ~6 % too small. Refinement now optimises rotations only,
  re-carves at full resolution, and keeps the new (rig, mesh) pair only when the freshly
  carved geometry matches the references better — otherwise the original solve is kept.
- The final carve level doubled blindly (64³ → 128³) instead of spending the hardware
  budget: it now targets `voxel_max` on the tight subject box, which measurably recovers
  the "thinner than the reference" surface deficit.
- `fuse_depth_points` mutated the volume it was given and returned the same object, so the
  caller could not tell "nothing was carved" from "carved a lot" and the validation step
  silently accepted an unvalidated carve. It now returns a new `HullResult` and leaves its
  input untouched.
- Depth fusion kept only voxels within ~1.5 voxels of a depth sample, which turned a clean
  224³ hull into a string of beads (22k samples → 82 % of the volume removed, IoU 0.90 →
  0.60, 384 components). Support radius now scales with the measured sample spacing, sparse
  sweeps are skipped outright, and the fused surface is re-extracted and compared against
  the silhouette hull so only the better-scoring volume survives.
- `recon3d <cmd> --json` mixed progress events into stdout, so the output was not one
  parseable JSON document; events now go to stderr and the reconstruct result is a flat
  envelope (`status`, `quality`, `statistics`, `outputs`, …).

[1.1.0]: https://github.com/sher-45-ops/software/releases/tag/v1.1.0
[1.0.0]: https://github.com/sher-45-ops/software/releases/tag/v1.0.0

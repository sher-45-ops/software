# Changelog

All notable changes to Recon3D Engine. This project follows
[Keep a Changelog](https://keepachangelog.com/) and [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-10-05

First release: a complete, local-first multi-view image-to-3D reconstruction engine.

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

[1.0.0]: https://github.com/sher-45-ops/software/releases/tag/v1.0.0

# Agent usage guide

This document is written for an autonomous agent (or the human configuring one) that is
driving Recon3D. It states what the engine guarantees, what it refuses to do, and the exact
calls that work.

## The contract

1. **Local only.** No call leaves the machine unless you explicitly download an optional
   model. Never send reference images to a third-party service "to help" - the engine's
   whole point is that it does not need one.
2. **No shell execution.** You never need to run a shell command as part of a
   reconstruction. Use the CLI, the REST API, or the MCP tools. If your harness only has
   MCP, that is enough.
3. **Never fabricate results.** Quality numbers come from `reports/quality.json` and the
   job result. If a stage failed or was skipped, say so; do not describe an asset as
   "high quality" without citing the measured numbers.
4. **Respect consent gates.** Deletion requires `--yes` / `confirm=true`. Paths are
   sandboxed. Attempting to escape the sandbox is an error, not a challenge.
5. **Report what is missing.** `missing_regions` lists the parts of the subject no reference
   image observed. Pass that on: it is information, not a failure to hide.

## Discovery

```bash
recon3d doctor --json      # hardware, deps, backends, data root
recon3d info --json        # stages, presets, parameters, formats, output tree, policy
```

MCP equivalents: `recon3d_doctor()`. Everything else follows from those two payloads — they
are designed to be the only documentation an agent needs at runtime.

## The standard workflow

```bash
# 1. project
recon3d create hero --subject character --json

# 2. images (paths are validated; non-images are refused)
recon3d add-images hero ./refs/front.png ./refs/right.png ./refs/back.png ./refs/left.png \
    ./refs/front_right.png ./refs/back_left.png --json

# 3. fast sanity run - its quality numbers tell you whether the references are usable
recon3d reconstruct hero --preset draft --json

# 4. the asset you keep
recon3d reconstruct hero --preset standard --texture-resolution 2048 \
    --formats glb,obj --json

# 5. retrieve
recon3d status hero --json                       # job state, quality, output paths
recon3d export hero --formats fbx,usdz --json    # extra formats without re-running
```

MCP: `recon3d_create_project(name, image_paths)` → `recon3d_reconstruct(project, preset)` →
`recon3d_list_outputs(job_id)`.

## Choosing parameters

| Situation | Recommendation |
| --- | --- |
| First look at a new reference set | `draft` (fast; quality numbers already meaningful) |
| Default production asset | `standard` |
| Detail matters (faces, mechanical edges) | `high`, texture 4096 |
| Real-time engine target | `game_ready` with `--target-polycount` |
| Offline render / film | `cinematic` (no polycount cap) |
| Fast iteration while tuning the references | `fast` (measured: 18.6 min / 481 MB / quality 93.3 vs `standard` 35 min / 688 MB / 93.7 on a 2-core 4 GB box) |
| Small RAM (<8 GB) | `draft` or `standard`, texture ≤ 2048; the engine also clamps to the detected hardware |
| Metric output | `--units meters --subject-height 1.8` (or centimetres/millimetres) |
| Rigged character | add `--rig`; check `reports/rig.json` before use |
| Stylised input | keep `style` as supplied; the engine does not photorealise |

`--param key=value` reaches any pipeline parameter (`photo_consistency=false`,
`symmetry=x`, `material_request="brushed aluminium"`, `preserve_sharp_edges=false`, …).
The full list with types and defaults is in `recon3d info --json → parameters`.

## Reading the result

```json
{
  "quality": {
    "overall": 78.1, "grade": "good",
    "geometry_quality": 79.6, "reference_similarity": 83.5,
    "mesh_health": 68.0, "texture_quality": 70.2,
    "metrics": {"mean_silhouette_iou": 0.795, "mean_color_rmse": 0.226,
                "mean_coverage_delta": -0.029, "views_compared": 8},
    "missing_regions": ["top",
      "inferred: texture atlas region uv[0.100,0.100]-[0.200,0.200] (12 texels) (confidence 0.42, texture_inpainting)"],
    "inferred": [{"kind": "texture_inpainting", "stage": "texture", "region": "…",
                  "texels": 12, "confidence": 0.42, "evidence": "…", "verified": "…"}],
    "inferred_summary": {"regions": 1, "kinds": ["texture_inpainting"], "mean_confidence": 0.42},
    "budgets": [{"name": "mobile", "pass": true, "checks": {"target_polycount":
                 {"target": 12000, "actual": 11999, "pass": true}}}],
    "warnings": ["…"]
  },
  "statistics": {"triangles": 64968, "vertices": 52017, "watertight": true,
                 "lod_levels": 3, "texture_resolution": 1024},
  "outputs": {"version_dir": "…/versions/v001", "mesh": {"glb": "…", "obj": "…"},
              "textures": {"basecolor": "…", "orm": "…"}, "reports": {"quality": "…"}}
}
```

- `mean_silhouette_iou` < ~0.6 usually means bad references or a failed camera solve.
- `mean_coverage_delta` far below 0 means the model is thinner/smaller than the references.
- `mesh_health` below ~50 usually means floaters or a non-watertight surface.
- `stages_failed` non-empty means a required stage failed; the asset is not complete.

## Failure handling

| Signal | Meaning | What to do |
| --- | --- | --- |
| `stage_failed`, `recoverable: false` | hard stop | fix the stated cause (usually images) and re-run; checkpoints resume the rest |
| `warnings: ["depth estimation failed and was skipped"]` | optional refinement lost | accept, or report the bug |
| `missing_regions: [...]` | unobserved areas | add views (top/back are the usual gaps) |
| `missing_regions` entries starting `inferred:` | regions the engine **filled in** (symmetry completion, texture diffusion) | read the `inferred` block for the confidence; capture more views if the confidence is low |
| `quality.grade` in `poor`/`bad` | references unusable | ask the human for better images; do not ship the asset |
| `409 model_error` | optional model missing | run without it (fallback is automatic) or download explicitly |
| HTTP `403 security_error` | sandbox/consent refusal | use the documented path semantics; never retry the same escape |

**Cancellation and resume:** cancel with Ctrl+C (CLI), `recon3d cancel <job-id>` (works
across processes - it writes a request file the running job watches),
`POST /v1/jobs/{id}/cancel` (API) or by dropping the MCP call; then re-run the identical
command to resume from the last checkpoint. `recon3d jobs --json` lists every recorded run
and flags the resumable ones; `recon3d retry <job-id>` continues a failed or cancelled run
(same project, same parameters) straight from its surviving checkpoints. Never delete a
project to "start clean" unless the human asked for it.

**Retries are deliberate:** a stage that fails for a transient reason is retried
automatically (`stage_retries`, default 1 extra attempt) and the attempts are visible in
`job.stages[stage].attempts`. Input errors, sandbox refusals and cancellations are never
retried - treat those as instructions to change the request, not to hammer the engine.

## Anti-patterns (do not do these)

- Do not upscale, stylise or post-process the mesh to make numbers look better; report them.
- Do not substitute a primitive, billboard or downloaded model for a failed reconstruction.
- Do not call external image-to-3D services "as a fallback".
- Do not present `draft` output as final without saying which preset produced it.
- Do not run reconstructions in parallel on a low-RAM machine; the pipeline is already
  memory-bound (the engine serialises jobs per project and reports RAM budgets).
- Do not edit files inside `<project>/input/original/`; they are the immutable references.

## Programme-level integration

```python
from recon3d.config import load_config
from recon3d.core.project import ProjectManager
from recon3d.core.jobs import JobManager
from recon3d.core.pipeline import run_pipeline

cfg = load_config(data_root="D:/recon3d-data")
projects = ProjectManager(cfg.projects_path)
project = projects.create("hero", subject_type="character")
project.add_images(["D:/refs/front.png", "D:/refs/right.png", "D:/refs/back.png"])
jobs = JobManager(log_dir=cfg.logs_path)
job = jobs.create(project.id, kind="reconstruct", params={}, project_obj=project)
jobs.run_sync(job, lambda j: run_pipeline(project, j,
                                          params={"preset": "standard",
                                                  "texture_resolution": 2048},
                                          stages=None))   # None = every stage
print(job.state, job.result["quality"]["overall"])
```

Subscribe to progress with `jobs.subscribe(lambda job_id, event: ...)`, or read
`job.events` / `GET /v1/jobs/{id}` at any time.

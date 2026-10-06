# CLI reference

```
recon3d {doctor,setup,create,add-images,projects,project,reconstruct,status,versions,export,serve,studio,models,info} ...
```

Every command accepts `--json` and prints a **single** JSON object on stdout instead of prose
(progress events are emitted as one JSON object per line on *stderr*, so
`recon3d reconstruct --json | jq` always parses), so agents
and CI never have to parse human text. Exit codes:

| Code | Meaning |
| --- | --- |
| `0` | success |
| `1` | the operation failed (validation error, failed stage, missing asset) |
| `2` | usage error (unknown flags, bad arguments — argparse) |
| `130` | interrupted by the user (Ctrl+C); the job is marked cancelled and checkpoints are kept |

Global: `recon3d --version` prints the engine version. Configuration comes from flags →
`RECON3D_*` environment variables → `config.json` → defaults (see [INSTALL.md](INSTALL.md)).

---

## Diagnostics and setup

### `recon3d doctor [--json]`

Hardware, dependency, backend and data-root diagnosis. This is the first command to run on
a new machine, and the first tool an agent should call before anything else.

```bash
$ recon3d doctor
recon3d 1.0.0 on Python 3.11.2 (linux)
data root: /home/user/.recon3d
cpu: 2 cores, RAM 3.8 GB, device: cpu

dependencies:
  [ok  ] numpy: 2.4.6
  [ok  ] cv2: 5.0.0
  ...
optional local backends:
  [no ] colmap               structure-from-motion camera solve
  [yes] xatlas               production UV unwrapping
advice:
  - only 3.8 GB RAM detected - use --preset draft or --performance-mode balanced ...
```

### `recon3d setup [--data-root PATH] [--performance-mode MODE] [--offline] [--with-models] [--json]`

Creates the data root, writes `config.json`, detects hardware and picks a performance mode,
and optionally downloads the recommended optional models (never without `--with-models`).

## Projects

### `recon3d create NAME [--subject TYPE] [--style STYLE] [--notes TEXT] [--json]`

`--subject` is one of `auto character creature vehicle prop object environment robot`.
The subject type only sets priors (symmetry expectations, classification hints); it is not
trusted over the measured analysis.

### `recon3d add-images PROJECT IMAGES... [--views front back ...] [--json]`

Copies images into the project (`input/original`, read-only) and registers them. View
labels are optional: they are inferred from filenames (`front.png`, `left_90.jpg`,
`back-right.png`, …) and from the images themselves; `--views` overrides the inference for
sets with opaque names (`IMG_0421.jpg`).

### `recon3d projects [--delete ID] [--yes] [--json]`
### `recon3d project PROJECT [--json]`
### `recon3d versions PROJECT [--json]`

List projects with image/version counts, inspect one project's manifest, or list versions
with their parameters and asset paths. `--delete` requires `--yes`; without it the command
refuses (destructive operations are always explicit).

## Reconstruction

### `recon3d reconstruct PROJECT [options]`

| Flag | Default | Meaning |
| --- | --- | --- |
| `--preset` | `standard` | `draft standard high ultra game_ready cinematic` |
| `--quality` | = preset | alias kept for scripts that speak "quality" |
| `--target-polycount N` | `auto` | triangle budget for the exported mesh |
| `--texture-resolution N` | preset | 512–8192 (clamped by hardware limits, reported) |
| `--formats LIST` | `glb,obj` | comma-separated `glb gltf obj fbx stl ply usd usda usdz` |
| `--units` | `normalized` | `normalized meters centimeters millimeters` |
| `--subject-height M` | — | real height in metres, for metric export |
| `--stages LIST` | all | run a subset, e.g. `uv,texture,lod` |
| `--param KEY=VALUE` | — | any pipeline parameter; JSON values allowed (`--param preserve_sharp_edges=false`) |
| `--rig` / `--no-rig` | preset | generate a skeleton + skin weights |
| `--lods` / `--no-lods` | preset | build the LOD chain |
| `--no-uvs` | — | skip UV/texture stages (geometry only) |
| `--preview` / `--no-preview` | on | previews and reference-comparison renders |
| `--stage-retries N` | `1` | extra attempts for a stage that fails for a transient reason (bad input is never retried) |
| `--watch` | on | stream stage/progress lines to stdout |
| `--json` | — | one JSON object on stdout: `status`, `quality`, `statistics`, `outputs`, `stages_failed`, `validation`, `job` (events on stderr as JSON lines) |

Runs are checkpointed per stage - and a checkpoint is re-used whenever the requested
parameters, the images and the upstream stages are unchanged. Re-running the identical
command on a finished project is therefore a *snapshot*: every stage reports `re-used cached
results`, the version directory is filled with the same files, the measured quality is
identical, and it takes seconds instead of minutes (the final export is rewritten because
its files are named after the version). Re-running a stopped command resumes where the last
attempt stopped; changing an input or parameter invalidates only the affected stages.
`Ctrl+C` cancels cleanly and keeps the checkpoints. A stage that fails for a transient
reason (busy disk, momentary allocation failure) is retried up to `--stage-retries` times;
input errors, sandbox rejections and cancellations are never retried.

```bash
# Geometry only, fast
recon3d reconstruct hero --preset draft --no-uvs --no-lods

# Final asset for a game engine
recon3d reconstruct hero --preset game_ready --target-polycount 25000 \
    --formats glb,fbx --texture-resolution 2048 --json
```

### `recon3d status PROJECT [--job JOB] [--watch] [--json]`

Job state, progress, per-stage timings and the measured quality of the last finished
version. `--watch` follows a running job.

### `recon3d jobs [--project PROJECT] [--json]`

Lists every job this machine has recorded - including runs started by an earlier process,
by the REST API or by an agent - newest first. Each record shows the state, progress,
stage, errors and whether the job is **resumable** (finish it by re-running the same
command, or with `recon3d retry`).

```bash
recon3d jobs --json | jq '.jobs[] | select(.resumable) | .id'
```

### `recon3d cancel JOB [--reason TEXT] [--json]`

Asks a running job to stop at the next checkpoint boundary. The request is a file written
next to the job's state, so it works across processes: an agent or a second terminal can
stop a reconstruction running inside the API server.

```bash
recon3d cancel job-1a2b3c4d5e6f --reason "switching to the high preset"
```

Everything finished before the stop is kept, so `recon3d retry` (or re-running the same
`reconstruct` command) continues from there instead of starting over. Cancelling a job
that already finished is refused.

### `recon3d retry JOB [--stages LIST] [--force-stage LIST] [--json]`

Re-runs a failed or cancelled job from its persisted record - same project, same
parameters - resuming from the checkpoints that survived. Use `--force-stage` to rebuild
a stage you don't trust (for example after changing something outside the project).

```bash
recon3d jobs                       # find the id of the run that died
recon3d retry job-1a2b3c4d5e6f     # continue; finished stages are re-used
```

## Outputs, servers, models

### `recon3d export PROJECT [--version V] [--formats LIST] [--out DIR] [--json]`

Re-exports an existing version (no reconstruction re-run). Each file is re-read after
writing and verified; the report lists any format that failed and why.

### `recon3d serve [--host H] [--port P] [--reload] [--open-browser] [--json]`

Starts the REST + WebSocket API ([API.md](API.md)). Binds `127.0.0.1:8760` by default;
`--host 0.0.0.0` exposes it to your network (do this only on a trusted machine — the API
has no authentication, by design: it is a local tool).

### `recon3d studio [--host H] [--port P] [--json]`

`serve` plus a browser window: upload images, pick a preset, watch stage progress live over
a WebSocket, and download the results.

### `recon3d models {list,download,verify,delete} [IDS...] [--force] [--yes] [--json]`

The local model manager: lists optional open models with their licence, size and install
state; `download` fetches them and records SHA-256 checksums (`state.json`); `verify`
re-checks them; `delete --yes` removes them. Offline mode and
`allow_network_downloads=false` block every download. See [MODEL_MANAGEMENT.md](MODEL_MANAGEMENT.md).

### `recon3d info [--json]`

Machine-readable capability report: version, stages, presets, parameters and defaults,
output tree, formats, materials, LOD ratios, interfaces, backends and the honesty policy.
This is the single call an agent needs to discover what the engine can do:

```bash
recon3d info --json | jq '.parameters, .stages[].name'
```

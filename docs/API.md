# REST API and studio UI

Start the server:

```bash
recon3d serve --port 8760              # API only
recon3d studio                         # API + browser UI
recon3d serve --host 0.0.0.0 --port 8760   # expose to your LAN (trusted networks only)
```

The API has **no authentication on purpose**: it is a local tool that manages files in your
data root. Bind it to `127.0.0.1` unless you understand the consequences. Destructive
endpoints require `confirm=true`.

Run the server on `0.0.0.0` when a reverse proxy or container must reach it; the studio UI
is served from the same origin at `/`, and the browser only ever talks to that origin
(the UI never calls `localhost` for backend data — everything is relative and proxied by
the server).

## Conventions

- All bodies and responses are JSON (`Content-Type: application/json`).
- All paths are relative to the server's data root; the API never accepts arbitrary output
  paths from a browser or agent.
- Errors follow the shape
  `{"ok": false, "code": "...", "message": "...", "details": {...}}` with an HTTP status
  from the error table below.
- Long operations return a **job object** immediately; poll it or subscribe over WebSocket.
- Artefact URLs are stable for the lifetime of the version directory.

| HTTP | `code` | Raised when |
| --- | --- | --- |
| 400 | `validation_error` | bad parameters, unusable images, unsupported format |
| 403 | `security_error` | path traversal, destructive operation without confirmation |
| 404 | `not_found` | unknown project, version, job or artefact |
| 409 | `conflict_error` / `model_error` | duplicate project, model not installed |
| 422 | `stage_failed` | a pipeline stage failed (details carry the stage and whether it is recoverable) |

## Endpoints

### Meta

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | liveness, version, data root, offline flag, active jobs |
| `GET` | `/v1/capabilities` | same payload as `recon3d info --json` |
| `GET` | `/v1/doctor` | hardware, dependencies, optional backends |
| `GET` | `/v1/presets` | presets with descriptions, params and resolved budgets |

### Projects

| Method | Path | Body / notes |
| --- | --- | --- |
| `POST` | `/v1/projects` | `{"name": "hero", "subject_type": "character", "style": "realistic", "description": "…"}` |
| `GET` | `/v1/projects` | list with image/version counts and disk usage |
| `GET` | `/v1/projects/{id}` | full manifest, including every registered image |
| `DELETE` | `/v1/projects/{id}?confirm=true` | delete the project directory (requires confirmation) |
| `POST` | `/v1/projects/{id}/images` | `multipart/form-data` with `files`, or `{"paths": ["/abs/a.png", …]}` |
| `GET` | `/v1/projects/{id}/images` | registered images with metadata, masks and quality flags |
| `GET` | `/v1/projects/{id}/versions` | versions, parameters and asset paths |

### Reconstruction and jobs

| Method | Path | Body / notes |
| --- | --- | --- |
| `POST` | `/v1/projects/{id}/reconstruct` | pipeline parameters (`preset`, `texture_resolution`, `target_polycount`, `export_formats`, `generate_rig`, `stages`, …); returns `{"job": {...}}` |
| `GET` | `/v1/jobs?project=hero` | all jobs, optionally filtered |
| `GET` | `/v1/jobs/{job}?events=50` | state, progress, stage table, recent events, `resumable`, `result` when finished |
| `GET` | `/v1/jobs/{job}` (failed/cancelled) | `resumable: true` means re-running the same request - or `recon3d retry {job}` - continues from the surviving checkpoints instead of restarting |
| `POST` | `/v1/jobs/{job}/cancel` | cooperative cancel; the job stops at its next stage boundary and keeps every finished checkpoint. The same request can be made from another process with `recon3d cancel {job}` |
| `GET` | `/v1/jobs/{job}/artifacts` | every produced file with sizes and download URLs |
| `GET` | `/v1/jobs/{job}/artifacts/{path}` | download one artefact (path is confined to the version directory) |
| `POST` | `/v1/export` | `{"project": "hero", "version": "v001", "formats": ["fbx","usdz"], "output_dir": "…"}` — re-export without re-running |

### Models

| Method | Path | Notes |
| --- | --- | --- |
| `GET` | `/v1/models` | registry entries, licence, size, checksum, install state |
| `POST` | `/v1/models/{id}/download` | download (blocked in offline mode / when downloads are disabled) |

## Example session

```bash
# 1. create a project
curl -s -X POST http://127.0.0.1:8760/v1/projects \
  -H 'Content-Type: application/json' \
  -d '{"name":"hero","subject_type":"character"}' | jq .id

# 2. upload reference images
curl -s -X POST http://127.0.0.1:8760/v1/projects/hero/images \
  -F files=@front.png -F files=@right.png -F files=@back.png -F files=@left.png

# 3. reconstruct (returns a job id immediately)
JOB=$(curl -s -X POST http://127.0.0.1:8760/v1/projects/hero/reconstruct \
  -H 'Content-Type: application/json' \
  -d '{"preset":"standard","texture_resolution":2048,"export_formats":["glb","obj"]}' \
  | jq -r .job.id)

# 4. poll
curl -s http://127.0.0.1:8760/v1/jobs/$JOB | jq '{state, progress, message}'

# 5. list and download artefacts
curl -s http://127.0.0.1:8760/v1/jobs/$JOB/artifacts | jq '.files[] | {path, bytes}'
curl -sO http://127.0.0.1:8760/v1/jobs/$JOB/artifacts/final/model.glb
```

## WebSocket progress

```
ws://127.0.0.1:8760/v1/ws/jobs/{job_id}
```

The server pushes one JSON object per progress event, then a terminal message and closes:

```json
{"kind":"progress","stage":"texture","progress":67.8,"message":"projected front.png (995744 texels)","ts":"2026-10-05T11:44:39Z"}
{"kind":"progress","stage":"lod","progress":12.0,"message":"LOD0: 20000 faces","ts":"…"}
{"kind":"done","state":"completed","result":{"version":"v001","quality":{…},"outputs":{…}}}
```

`state` is `completed`, `failed` or `cancelled`. If the job is unknown the socket receives
`{"kind":"error","message":"…"}` before closing. Clients that reconnect after a drop simply
reconnect to the same URL — events are replayed from the job registry, and `GET /v1/jobs/{id}`
always carries the authoritative state.

## Studio UI

`recon3d studio` opens the same server plus a single-page UI at `/` that:

- lists and creates projects, uploads reference images,
- exposes preset, texture resolution, polycount, units, rig/LOD/preview switches,
- streams stage progress live over the WebSocket above,
- renders the quality report (overall score, geometry/reference/mesh/texture breakdown,
  silhouette IoU, triangle count, warnings and missing regions),
- links every produced artefact and preview for download.

The UI is static HTML/CSS/JS served from `recon3d/studio/static`; it needs no build step and
no external CDN, so it works offline.

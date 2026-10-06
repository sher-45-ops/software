# Troubleshooting

Every failure below is one the engine reports itself. Exit codes and machine-readable
errors are in [CLI.md](CLI.md) / [API.md](API.md); the engine never fails silently, and it
prefers a degraded-but-honest result over a fake one.

Start with `recon3d doctor` — it answers half of these questions directly.

---

## Installation and startup

**`ModuleNotFoundError: recon3d`**
The package is not installed in the Python you are running. Activate the venv
(`.venv\Scripts\activate` on Windows, `. .venv/bin/activate` elsewhere) and
`pip install -e ".[all]"`. Check with `python -c "import recon3d; print(recon3d.VERSION)"`.

**`the API server needs FastAPI + uvicorn`**
`pip install "recon3d[api]"`. The CLI works without it; only `serve`/`studio` need it.

**`the MCP server needs the mcp package`**
`pip install "recon3d[mcp]"`. Both MCP SDK generations are supported (`FastMCP` and the 2.x
`MCPServer`); if your platform pins an unusual version, `pip install "mcp>=1.2"`.

**`Address already in use` when running `serve`/`studio`**
Another process owns port 8760. `recon3d serve --port 8761`, or stop the other server.

**`[no] xatlas` in `doctor`**
UV unwrapping falls back to the built-in chart packer. It works, but islands are less
tidy. `pip install xatlas` (MIT, no system dependencies) for production-quality UVs.

## Running the pipeline

**`stage_failed` / `recoverable: false` in a report**
The stage could not continue. The report names the stage and the exception; the run keeps
every earlier checkpoint, so re-running the same command resumes instead of restarting.
Known-recoverable stages (`preview`, `comparison`, `lod`, `rigging`, `texture` fallbacks)
degrade with a warning instead of failing the job.

**Out of memory (process killed, or `MemoryError`)**
Reconstruction is memory-hungry at high carve resolutions. In order of impact:
1. `--preset draft` (or `recon3d setup --performance-mode draft`).
2. Lower texture size: `--texture-resolution 1024`.
3. Fewer/smaller images: the engine caps working resolution itself (`RECON3D_LIMIT_MAX_IMAGE_DIM`).
4. `--stages` to run the heavy parts separately.
`recon3d doctor --json` shows the RAM budget the engine assumes; a 4 GB machine should
stay at `draft`/`balanced` and ≤2048 textures.

**"very few geometrically consistent feature matches between views"**
The reference set is textureless, repetitive, or the images do not show the same subject.
The camera solve then relies on the silhouette/ring geometry, which is fine for turntable
sets and poor for arbitrary camera arrangements. Add images with texture and overlap.

**"most image pairs share almost no features; cross-view appearance consistency is low"**
Same cause, weaker signal. The reconstruction still runs; expect lower
`reference_similarity` in `reports/quality.json`.

**"lens estimate … silhouette IoU 0.82"** (or any FOV estimate)
The lens estimate is analysis-by-synthesis and coarse: it is accurate to roughly ±4° for
clean turntable sets and less for noisy ones. It affects depth, not silhouette fit. The
report `reports/quality.json → camera.lens` lists every candidate that was tried.

**"scale gauge corrected by 1.06x"**
Expected: the gauge is closed by measuring the carved subject and rescaling, so the result
really is 1.0 engine unit tall (`units: normalized`). With `--units meters
--subject-height 1.8` the export is scaled to real-world metres.

**"removed N disconnected component(s)"**
Debris from carving (thin shells, isolated fragments). Removal is normal. A very large N
(hundreds) means the reference set is inconsistent — check for images of a different
subject or background clutter leaking into the masks.

**"high UV distortion; texture stretching is likely"**
Install `xatlas` for better islands, or increase the carve resolution / texture size. The
report includes `distortion`, `islands`, `packing_efficiency` and `texel_density` per run.

**"depth estimation failed and was skipped"**
Depth fusion is an *optional* refinement of the carved volume; the reconstruction still
completes. If the warning names an error, that is a bug worth reporting with the log.

**Job seems stuck**
Long stages (`texture` on 8192² maps, `ultra` carving) can take many minutes. Watch with
`recon3d status PROJECT --watch`, the WebSocket, or `recon3d status PROJECT` for per-stage
timings. Cancel with Ctrl+C, `recon3d cancel <job-id>` from any terminal, or
`POST /v1/jobs/{id}/cancel`; checkpoints survive and the run can be continued with
`recon3d retry <job-id>`.

**Resume after a crash**
Run the same command again. Checkpoints live in
`<project>/intermediate/<stage>/_checkpoint.json` and are reused when the inputs and
parameters are unchanged. `--force` (CLI) or deleting a stage directory forces a re-run.

To find out what a crashed process left behind:

```bash
recon3d jobs --json          # every recorded run, newest first; resumable ones are flagged
recon3d retry <job-id>       # continue the failed/cancelled run from its checkpoints
recon3d retry <job-id> --force-stage mesh_reconstruction   # rebuild a stage you distrust
```

A checkpoint is only trusted when the artefacts it promises still exist; if a mesh was
deleted or a stage's inputs changed, that stage (and only that stage) is rebuilt.

**Stopping a long run from another terminal or agent**
`recon3d cancel <job-id>` (or `POST /v1/jobs/{id}/cancel`) writes a cancel request that the
running job watches - it stops at the next stage boundary and keeps every finished stage.
A stage that fails for a *transient* reason (busy disk, momentary allocation failure) is
retried automatically (`--stage-retries`, default one extra attempt); bad input, sandbox
refusals and cancellations are never retried.

## Image and coverage problems

**"coverage 45%" / `missing_regions` non-empty**
The references did not observe part of the subject (usually the back or top). Those regions
are reported rather than invented — add missing views (a top view helps vehicles and
creatures a lot) and re-run.

**Subject is small in frame**
Crop or move closer. Masks from a subject occupying <5% of the frame lose detail and the
depth stage has too few pixels to work with.

**Background clutter gets captured**
Check `<project>/input/masks/*.png` (written by the segmentation stage) — if the subject
mask includes background, supply your own masks in that folder and re-run; they are
honoured as-is. A plain, contrasting background cuts most of this.

**"the reconstruction is systematically smaller than the references" /
`mean_coverage_delta` below ≈ −0.02**
The silhouette carve is the only source of shape, and it can only be as large as the masks
say. Check, in order: (1) the reported masks actually cover the subject
(`<project>/input/masks/*.png` — supply your own if they are tight or loose);
(2) the comparison report's per-view IoU — one weak view drags the mean down;
(3) the quality report's `metrics.mean_coverage_delta` — a few percent is normal
(marching cubes rounds convex surfaces inwards), tens of percent means missing views.
Camera refinement never trades scale for IoU: it refines rotations only and re-carves to
prove the change, so a systematic size error is an evidence problem, not a tuning one.
If you need the subject to be exactly a known height, pass `--subject-height` (metres) or
`--units meters` on export and the asset is rescaled to that measurement.

**Textures look washed out / flat**
The texture stage normalises per-view colour; strong specular highlights and shadows across
views are averaged. Shoot with diffuse light, and keep `style` at the supplied style
(`--param style=stylized` avoids any photorealistic correction).

## Export problems

**"export verification failed"**
The file was written but re-reading it did not reproduce the expected geometry. The report
lists what differed. Retry once (`recon3d export PROJECT --formats glb`) — a partial disk
or an interrupted write is the usual cause; if it repeats, report the format and the log.

**FBX is ASCII, not binary**
Binary FBX needs Blender (or the FBX SDK). Without Blender the engine writes ASCII FBX —
importable everywhere, but larger. `recon3d export PROJECT --formats glb` is the
recommended interchange format.

**`usd` vs `usda` vs `usdz`**
`usda` = text USD, `usd` = the same text file with the `.usd` extension, `usdz` = the zipped
package. All three are produced without extra dependencies.

## Security and sandboxing

**"path escapes the sandbox"**
Every write is confined to the project/version directory. Use `--out` for exports, and note
that the API refuses absolute paths outside the data root by design.

**"operation 'delete_project' is destructive and requires explicit confirmation"**
Pass `--yes` (CLI) or `confirm=true` (API). This gate exists so an agent cannot delete your
work by accident.

**Model download refused**
Offline mode (`RECON3D_OFFLINE=1` or `recon3d setup --offline`) or
`allow_network_downloads=false` blocks downloads by design. Re-enable explicitly with
`recon3d models download <id> --force` after unsetting offline mode.

## Still stuck?

Collect `reports/quality.json`, `reports/stages.json` and the job log
(`<data root>/logs/`), then open an issue with
`recon3d doctor --json`, the exact command, and the image count/resolution. Never attach
images you do not own.

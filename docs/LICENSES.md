# Licences

Recon3D Engine itself is **Apache-2.0** (see [../LICENSE](../LICENSE)). This page lists every
dependency, its licence and what that means for the assets you produce with it.

## Core dependencies (installed by `pip install recon3d`)

| Package | Licence | Used for |
| --- | --- | --- |
| numpy | BSD-3-Clause | all numerics |
| scipy | BSD-3-Clause | morphology, filters (volume cleanup), signal |
| opencv-python-headless | Apache-2.0 (OpenCV 4.x: Apache-2.0; 5.x: Apache-2.0) | image I/O, resizing, SIFT, GrabCut, morphology |
| Pillow | MIT-CMU | PNG/JPEG I/O for textures and previews |
| scikit-image | BSD-3-Clause | marching cubes, measure |
| trimesh | MIT | mesh container, I/O, repair helpers |
| psutil | BSD-3-Clause | hardware detection |
| PyYAML | MIT | config/report files |
| fast-simplification | MIT | quadric decimation (with a built-in fallback) |

## Optional extras

| Package | Extra | Licence | Notes |
| --- | --- | --- | --- |
| fastapi | `api` | MIT | REST server |
| uvicorn | `api` | BSD-3-Clause | ASGI server |
| python-multipart | `api` | Apache-2.0 | image uploads |
| mcp | `mcp` | MIT | MCP server (both 1.x and 2.x SDKs supported) |
| xatlas | `quality` | MIT | UV unwrapping (built-in fallback when absent) |
| pygltflib | `quality` | MIT | GLB introspection/verification |
| onnxruntime | `neural` | MIT | runs the optional ONNX models |
| pywebview | `desktop` | BSD-3-Clause | native window for the studio UI |
| pytest, pytest-timeout, httpx | `dev` | MIT | test suite |

## Optional local models (`recon3d models download`)

| Model | Licence | Deliberately excluded variants |
| --- | --- | --- |
| Depth Anything V2 **Small** (ONNX) | Apache-2.0 | Base/Large/Giant are CC-BY-NC-4.0 → not offered |
| U²-Net-p salient (ONNX) | Apache-2.0 | — |

## Optional external applications (never bundled, never required)

| Tool | Licence | Used when present |
| --- | --- | --- |
| COLMAP | BSD-3-Clause | structure-from-motion camera solve |
| Open3D | MIT | alternative decimation, Poisson meshing |
| Blender | GPL-2.0 | binary FBX export, high-quality renders |
| ONNX Runtime models you supply | yours | any stage that accepts a model |

The engine detects these at runtime (`recon3d doctor`) and falls back to its own
implementations when they are missing. Because they run as **separate processes**, their
copyleft terms do not extend to your assets or to this project.

## What this means for your output

- **Your meshes, textures and rigs are yours.** Nothing in the pipeline claims rights over
  generated assets; the licenses above cover code, not your geometry.
- **The permissive-only policy is deliberate.** Every default path uses MIT/BSD/Apache-2.0
  components, so commercial use of reconstructions is unencumbered.
- **Non-commercial traps are excluded by design.** The CC-BY-NC depth models and UniDepth
  are not offered by the model manager; the registry documents the exclusion so you can
  make an informed choice if you add them yourself.
- **Attribution.** Apache-2.0 and MIT components require preserving their notices if you
  redistribute *the library*. This project keeps a dependency table (this file) for that
  purpose; `pip` installs each package's own licence text alongside it.

## Verifying what you have

```bash
recon3d doctor --json       # which optional backends/models are present
pip list                    # installed versions
recon3d models list         # optional model licences and install state
```

If you redistribute Recon3D itself, include this file, `LICENSE`, and the licences of the
installed packages (they ship inside each wheel).

# Installation

Recon3D Engine is a pure-Python package with a small, permissively licensed dependency
set. It runs on CPU-only machines; a GPU is optional and only ever used as an accelerator.

## Requirements

| | Minimum | Recommended |
| --- | --- | --- |
| Python | 3.9 | 3.10 – 3.12 |
| RAM | 4 GB | 16 GB |
| Disk | 1 GB (+ your images and models) | 20 GB |
| OS | Windows 10/11, macOS 12+, Linux (glibc) | any |
| GPU | not required | NVIDIA CUDA for optional neural helpers / faster carving |

Verified in this repository with Python 3.11 on Linux, CPU-only, 2 cores / 4 GB RAM
(`recon3d doctor` prints the exact profile the engine will use).

## 1. Install the package

```bash
git clone -b arena/01a10ae3-software https://github.com/sher-45-ops/software.git recon3d-engine
cd recon3d-engine

# One command does all of the below and then verifies itself:
bash scripts/install.sh          # macOS/Linux
powershell scripts/install.ps1   # Windows

python -m venv .venv
# Windows (PowerShell):  .\.venv\Scripts\Activate.ps1
# Windows (cmd):         .venv\Scripts\activate.bat
# macOS/Linux:           . .venv/bin/activate

pip install -e ".[all]"      # core + API + MCP + quality extras
```

Or from a wheel/sdist, without a checkout:

```bash
pip install "recon3d[all]"
```

The data root defaults to `~/.recon3d` (Windows: `%USERPROFILE%\.recon3d`); override it with
`RECON3D_HOME` or `--data-root`. Nothing is installed outside the Python environment except
that one folder, so the checkout stays portable.

Extras are additive — the engine degrades gracefully when they are absent:

| Extra | Adds | Needed for |
| --- | --- | --- |
| *(core)* | numpy, scipy, opencv-python-headless, Pillow, scikit-image, trimesh, psutil, PyYAML, fast-simplification | reconstruction itself |
| `api` | fastapi, uvicorn, python-multipart | `recon3d serve` / `recon3d studio` |
| `mcp` | mcp | `python -m recon3d.mcpserver` |
| `quality` | xatlas, pygltflib | production UV unwrapping, GLB introspection |
| `neural` | onnxruntime | optional local depth/saliency models |
| `desktop` | pywebview | studio UI in a native window (browser is the default) |
| `dev` | pytest, pytest-timeout, httpx, build | running the test suite |

`pip install -e .` alone is enough for `doctor`, `create`, `add-images` and `reconstruct`.

### Offline / air-gapped installs

```bash
# on a connected machine
pip download -d ./wheels "recon3d[all]"
# copy ./wheels to the target machine, then
pip install --no-index --find-links ./wheels "recon3d[all]"
recon3d setup --offline        # never touch the network, even for optional models
```

## 2. Configure the data root

The data root holds projects, caches, logs and downloaded models. Default:
`~/.recon3d` (`%USERPROFILE%\.recon3d` on Windows).

```bash
recon3d setup                                   # create it with defaults
recon3d setup --data-root D:\recon3d-data       # explicit location
recon3d setup --performance-mode balanced       # auto|draft|balanced|quality|maximum
```

Resolution order for every setting is: CLI flag → environment variable → `config.json` →
built-in default. Useful variables:

| Variable | Meaning |
| --- | --- |
| `RECON3D_HOME` | data root (same as `--data-root`) |
| `RECON3D_OFFLINE=1` | forbid all network access, including model downloads |
| `RECON3D_PERFORMANCE_MODE` | `auto`/`draft`/`balanced`/`quality`/`maximum` |
| `RECON3D_MAX_WORKERS` | worker processes/threads for per-image stages |
| `RECON3D_LIMIT_MAX_TEXTURE_RESOLUTION` | hard cap on texture size, e.g. `2048` |

## 3. Verify

```bash
recon3d doctor              # human-readable
recon3d doctor --json       # machine-readable, for agents and CI
```

`doctor` checks Python and OS, CPU/RAM/disk, every core and optional dependency, the
optional external backends (COLMAP, Open3D, Blender, xatlas, onnxruntime, torch), the data
root layout, and prints concrete advice (for example: "only 3.8 GB RAM detected — use
`--preset draft`").

## 4. First reconstruction

```bash
python scripts/make_demo_dataset.py --out ./demo/refs --views 9 --masks
recon3d create demo --subject robot
recon3d add-images demo ./demo/refs
recon3d reconstruct demo --preset draft
```

Or with your own photographs:

```bash
recon3d create hero --subject character
recon3d add-images hero ./photos/front.png ./photos/right.png ./photos/back.png ./photos/left.png
recon3d reconstruct hero --preset standard --formats glb,obj --texture-resolution 2048
```

## Windows notes

- Everything runs from PowerShell/cmd; no POSIX shell is required and no path is hardcoded.
- Long paths: if your project root is deep, enable long-path support
  (`git config --system core.longpaths true`, or the LongPathsEnabled policy) — the engine
  keeps paths short by default, but deep data roots plus deep image names can still hit
  the 260-character limit on older builds.
- If `pip install` picks a Python without wheels for your version, install Python 3.11 or
  3.12 from python.org and recreate the venv.

## Uninstall

```bash
pip uninstall recon3d
rm -rf ~/.recon3d          # deletes projects, caches and downloaded models
```

Nothing else is written outside the data root, except the Python package itself.

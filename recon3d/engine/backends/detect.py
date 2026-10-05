"""Detection of optional local backends (spec #22, #33, #37).

Recon3D works with nothing but its core dependencies.  When more capable local
tools exist, the engine uses them and *records* that it did:

* **COLMAP** (BSD) - if a ``colmap`` binary is on PATH, the camera solve can use
  real SfM feature matching instead of the silhouette-based rig estimate.
* **Open3D** (MIT) - faster quadric decimation and Poisson meshing.
* **Blender** (GPL) - binary FBX export, rendering, extra post-processing.
* **ONNX Runtime** (MIT) - optional local neural depth/segmentation models from
  the model manager (never a hosted API).

Detection never downloads anything and never fails the pipeline.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class BackendInfo:
    name: str
    available: bool
    kind: str  # binary | python | none
    version: str = ""
    path: str = ""
    license: str = ""
    purpose: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "available": self.available, "kind": self.kind,
            "version": self.version, "path": self.path, "license": self.license,
            "purpose": self.purpose, "notes": self.notes,
        }


def _binary_version(command: List[str], timeout: int = 6) -> str:
    try:
        out = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        text = (out.stdout or "") + (out.stderr or "")
        for line in text.splitlines():
            line = line.strip()
            if line:
                return line[:120]
    except (OSError, subprocess.SubprocessError):
        pass
    return ""


def detect_colmap() -> BackendInfo:
    exe = shutil.which("colmap")
    info = BackendInfo("colmap", bool(exe), "binary" if exe else "none", path=exe or "",
                       license="BSD-3-Clause", purpose="structure-from-motion camera solve")
    if exe:
        info.version = _binary_version([exe, "--help"]) or "unknown"
        if "CUDA" in info.version or os.environ.get("COLMAP_CUDA"):
            info.notes = "CUDA build detected"
    else:
        info.notes = ("not installed; the engine falls back to its silhouette-based camera "
                      "solver (works for ring-style reference sets)")
    return info


def detect_blender() -> BackendInfo:
    exe = shutil.which("blender")
    info = BackendInfo("blender", bool(exe), "binary" if exe else "none", path=exe or "",
                       license="GPL-2.0-or-later",
                       purpose="binary FBX export, renders, optional post-processing")
    if exe:
        info.version = _binary_version([exe, "--version"]) or "unknown"
    else:
        info.notes = ("not installed; ASCII FBX is still exported by the engine itself "
                      "(importable by Blender/Unity/Unreal/Godot)")
    return info


def detect_open3d() -> BackendInfo:
    version = ""
    available = False
    try:
        import open3d  # type: ignore

        available = True
        version = getattr(open3d, "__version__", "unknown")
    except Exception:
        pass
    return BackendInfo("open3d", available, "python" if available else "none",
                       version=version, license="MIT",
                       purpose="fast decimation, Poisson meshing, point-cloud filters",
                       notes="" if available else "not installed (optional quality boost)")


def detect_onnxruntime() -> BackendInfo:
    version = ""
    available = False
    try:
        import onnxruntime  # type: ignore

        available = True
        version = getattr(onnxruntime, "__version__", "unknown")
    except Exception:
        pass
    return BackendInfo("onnxruntime", available, "python" if available else "none",
                       version=version, license="MIT",
                       purpose="local neural models (depth/segmentation) from the model manager",
                       notes="" if available else "not installed (classical algorithms are used)")


def detect_torch() -> BackendInfo:
    version = ""
    available = False
    cuda = False
    try:
        import torch  # type: ignore

        available = True
        version = getattr(torch, "__version__", "unknown")
        cuda = bool(torch.cuda.is_available())
    except Exception:
        pass
    return BackendInfo("torch", available, "python" if available else "none", version=version,
                       license="BSD-3-Clause", purpose="GPU acceleration for optional models",
                       notes=("CUDA available" if cuda else
                              "" if available else "not installed (CPU-only pipeline)"))


def detect_xatlas() -> BackendInfo:
    version = ""
    available = False
    try:
        import xatlas  # type: ignore

        available = True
        version = getattr(xatlas, "__version__", "bundled")
    except Exception:
        pass
    return BackendInfo("xatlas", available, "python" if available else "none", version=version,
                       license="MIT", purpose="production UV unwrapping",
                       notes="" if available else "not installed; the built-in unwrapper is used")


def detect_fast_simplification() -> BackendInfo:
    version = ""
    available = False
    try:
        import fast_simplification  # type: ignore

        available = True
        version = getattr(fast_simplification, "__version__", "unknown")
    except Exception:
        pass
    return BackendInfo("fast_simplification", available, "python" if available else "none",
                       version=version, license="MIT", purpose="fast quadric mesh decimation",
                       notes="" if available else "not installed; the built-in QEM decimator is used")


def detect_backends() -> Dict[str, Any]:
    """Detect every optional backend and summarise what the engine will use."""
    items = [detect_colmap(), detect_blender(), detect_open3d(), detect_onnxruntime(),
             detect_torch(), detect_xatlas(), detect_fast_simplification()]
    summary = {
        "colmap": "used for camera estimation" if items[0].available else "silhouette rig solver",
        "blender": "used for binary FBX/renders" if items[1].available else "not used (not required)",
        "decimation": "fast_simplification" if items[6].available else (
            "open3d" if items[2].available else "builtin QEM"),
        "uv": "xatlas" if items[5].available else "builtin unwrapper",
        "depth": "multi-view plan-sweep (classical, always available)",
        "segmentation": "classical threshold/saliency+grabcut (always available)",
        "gpu": "torch/CUDA available" if items[4].available else "CPU",
    }
    return {
        "backends": [i.to_dict() for i in items],
        "summary": summary,
        "available_count": sum(1 for i in items if i.available),
    }


def run_colmap(*args: str, timeout: int = 3600) -> Dict[str, Any]:
    """Run a whitelisted COLMAP subcommand.

    Only ``colmap`` is ever executed, always with an argument list (never a
    shell), and only when the binary exists (spec #35).
    """
    exe = shutil.which("colmap")
    if not exe:
        return {"ok": False, "error": "colmap binary not found on PATH"}
    allowed = {"feature_extractor", "exhaustive_matcher", "sequential_matcher", "mapper",
               "bundle_adjuster", "image_undistorter", "patch_match_stereo",
               "stereo_fusion", "poisson_mesher", "model_converter"}
    if not args or args[0] not in allowed:
        return {"ok": False, "error": f"subcommand not in the allow-list: {args[:1]}"}
    try:
        out = subprocess.run([exe, *args], capture_output=True, text=True,
                             timeout=timeout, check=False)
        return {
            "ok": out.returncode == 0,
            "returncode": out.returncode,
            "stdout_tail": out.stdout[-4000:] if out.stdout else "",
            "stderr_tail": out.stderr[-4000:] if out.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"colmap timed out after {timeout}s"}
    except OSError as exc:  # pragma: no cover
        return {"ok": False, "error": str(exc)}


def run_blender(script: str, *, timeout: int = 1800, extra: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run Blender in background mode with a Python expression (whitelisted)."""
    exe = shutil.which("blender")
    if not exe:
        return {"ok": False, "error": "blender binary not found on PATH"}
    command = [exe, "--background", "--factory-startup", "--python-expr", script]
    if extra:
        command.extend(extra)
    try:
        out = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        return {"ok": out.returncode == 0, "returncode": out.returncode,
                "stdout_tail": out.stdout[-4000:] if out.stdout else "",
                "stderr_tail": out.stderr[-4000:] if out.stderr else ""}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"blender timed out after {timeout}s"}
    except OSError as exc:  # pragma: no cover
        return {"ok": False, "error": str(exc)}

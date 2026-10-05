"""Recon3D Engine - local-first multi-view image -> 3D reconstruction.

The package is intentionally import-light: importing :mod:`recon3d` must not pull
in heavy optional dependencies (FastAPI, MCP, ONNX runtime, xatlas...).  Heavy
modules are imported lazily by the components that need them, so the CLI and the
engine work on a machine with nothing but numpy/scipy/opencv installed.
"""

from __future__ import annotations

__all__ = ["__version__", "VERSION", "get_version"]

VERSION = "1.0.0"
__version__ = VERSION


def get_version() -> str:
    """Return the engine version string."""
    return VERSION

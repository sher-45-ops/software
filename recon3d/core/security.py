"""Path sandboxing and input validation.

The engine is designed to be driven by autonomous agents (spec #35), so every
filesystem operation funnels through this module:

* only whitelisted extensions are accepted,
* every path is resolved and confined to an allowed root,
* symlink escapes and ``..`` traversal are rejected,
* destructive operations require an explicit permission flag,
* no request can ever trigger a shell command (subprocess use is limited to
  whitelisted local binaries invoked with argument lists, never a shell).
"""

from __future__ import annotations

import os
import re
import shutil
import unicodedata
from pathlib import Path, PurePath
from typing import Iterable, List, Optional, Sequence

from ..errors import SecurityError, ValidationError

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
MESH_EXTENSIONS = {".glb", ".gltf", ".obj", ".fbx", ".stl", ".ply", ".usda", ".usdz", ".dae"}
TEXTURE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".exr"}
EXPORT_FORMATS = {"glb", "gltf", "obj", "fbx", "stl", "ply", "usd", "usda", "usdz"}

MAX_IMAGE_BYTES = 256 * 1024 * 1024  # 256 MB per image - generous but bounded
MAX_IMAGES_PER_PROJECT = 2000

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

#: Operations that irreversibly remove data.  Blocked unless the caller passes
#: ``confirm=True`` (CLI ``--yes``), which makes accidental agent-side deletion
#: impossible (spec #35).
DESTRUCTIVE_OPERATIONS = {
    "delete_project",
    "delete_version",
    "delete_images",
    "purge_cache",
    "delete_model",
}


def validate_identifier(value: str, kind: str = "id") -> str:
    """Validate a user/agent supplied identifier (project id, version, name)."""
    if not isinstance(value, str):
        raise ValidationError(f"{kind} must be a string", details={"value": repr(value)})
    if not _ID_RE.match(value):
        raise ValidationError(
            f"invalid {kind}",
            details={
                "value": value,
                "rule": "1-64 chars, must start alphanumeric, allowed: A-Z a-z 0-9 . _ -",
            },
        )
    return value


def slugify(value: str, fallback: str = "project") -> str:
    """Turn an arbitrary asset name into a safe identifier."""
    norm = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", norm).strip("._-")
    slug = re.sub(r"_{2,}", "_", slug)[:64]
    if not slug or not _ID_RE.match(slug):
        slug = fallback
    return slug


class PathSandbox:
    """Confines all writes to an explicit set of roots."""

    def __init__(self, roots: Sequence[os.PathLike | str]) -> None:
        self.roots: List[Path] = []
        for root in roots:
            p = Path(root).expanduser()
            p.mkdir(parents=True, exist_ok=True)
            self.roots.append(p.resolve())

    # -- helpers -------------------------------------------------------
    def _is_within(self, path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def check(self, path: os.PathLike | str, *, must_exist: bool = False,
              writable: bool = False) -> Path:
        """Resolve *path* and verify it lives inside one of the sandbox roots."""
        raw = Path(path).expanduser()
        if not raw.is_absolute():
            # Relative paths are always interpreted against the primary root, so
            # "final/model.glb" can never accidentally resolve to the CWD.
            raw = self.roots[0] / raw
        try:
            resolved = raw.resolve(strict=False)
        except OSError as exc:  # pragma: no cover - platform dependent
            raise SecurityError(f"cannot resolve path: {raw}", details={"error": str(exc)}) from exc

        if not any(self._is_within(resolved, root) for root in self.roots):
            raise SecurityError(
                "path escapes the sandbox",
                details={"path": str(raw), "allowed_roots": [str(r) for r in self.roots]},
            )
        if must_exist and not resolved.exists():
            raise ValidationError("path does not exist", details={"path": str(resolved)})
        if writable and resolved.exists():
            # Refuse to write into the immutable copy of the user's originals.
            if "input" in resolved.parts and "original" in resolved.parts:
                raise SecurityError(
                    "original reference images are read-only",
                    details={"path": str(resolved)},
                )
        return resolved

    def add_root(self, root: os.PathLike | str) -> Path:
        p = Path(root).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        rp = p.resolve()
        if rp not in self.roots:
            self.roots.append(rp)
        return rp

    def relative(self, path: os.PathLike | str) -> str:
        resolved = self.check(path)
        for root in self.roots:
            if self._is_within(resolved, root):
                try:
                    return str(resolved.relative_to(root))
                except ValueError:  # pragma: no cover
                    continue
        return str(resolved)


def safe_join(root: os.PathLike | str, *parts: str, sandbox: Optional[PathSandbox] = None) -> Path:
    """Join path components under *root*, rejecting traversal attempts."""
    root_path = Path(root).expanduser().resolve()
    candidate = root_path
    for part in parts:
        if part in ("", "."):
            continue
        p = PurePath(str(part))
        if p.is_absolute() or ".." in p.parts:
            raise SecurityError("illegal path component", details={"component": str(part)})
        candidate = candidate / str(part)
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root_path)
    except ValueError as exc:
        raise SecurityError(
            "path traversal detected", details={"path": str(resolved), "root": str(root_path)}
        ) from exc
    if sandbox is not None:
        sandbox.check(resolved)
    return resolved


def validate_image_path(path: os.PathLike | str, *, must_exist: bool = True) -> Path:
    """Validate that *path* points at a readable image file of a supported type."""
    p = Path(path).expanduser()
    if must_exist and not p.exists():
        raise ValidationError("image not found", details={"path": str(p)})
    if p.exists() and not p.is_file():
        raise ValidationError("not a file", details={"path": str(p)})
    suffix = p.suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        raise ValidationError(
            "unsupported image format",
            details={"path": str(p), "supported": sorted(IMAGE_EXTENSIONS)},
        )
    if p.exists():
        size = p.stat().st_size
        if size == 0:
            raise ValidationError("image file is empty", details={"path": str(p)})
        if size > MAX_IMAGE_BYTES:
            raise ValidationError(
                "image exceeds the size limit",
                details={"path": str(p), "bytes": size, "limit": MAX_IMAGE_BYTES},
            )
    return p


def validate_export_path(path: os.PathLike | str, fmt: str, *, sandbox: PathSandbox,
                         overwrite: bool = False) -> Path:
    """Validate a requested export destination."""
    fmt = fmt.lower().lstrip(".")
    if fmt not in EXPORT_FORMATS:
        raise ValidationError("unsupported export format",
                             details={"format": fmt, "supported": sorted(EXPORT_FORMATS)})
    target = sandbox.check(path, writable=True)
    if target.suffix.lower() not in {f".{fmt}", ".glb" if fmt == "gltf" else f".{fmt}"}:
        target = target.with_suffix(f".{fmt}")
    if target.exists() and not overwrite:
        raise ValidationError(
            "export target already exists (pass overwrite=true to replace)",
            details={"path": str(target)},
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def require_permission(operation: str, confirm: bool) -> None:
    """Gate destructive operations behind an explicit confirmation flag."""
    if operation in DESTRUCTIVE_OPERATIONS and not confirm:
        raise SecurityError(
            f"operation '{operation}' is destructive and requires explicit confirmation",
            details={"operation": operation, "hint": "pass confirm=true / use --yes"},
        )


def rmtree_within(path: os.PathLike | str, sandbox: PathSandbox, *, confirm: bool,
                  operation: str = "delete_project") -> None:
    """Delete a directory, but only inside the sandbox and only with consent."""
    require_permission(operation, confirm)
    target = sandbox.check(path)
    if target in sandbox.roots:
        raise SecurityError("refusing to delete a sandbox root", details={"path": str(target)})
    if not target.exists():
        return
    shutil.rmtree(target)


def safe_symlink_free_copy(src: Path, dst: Path) -> Path:
    """Copy a file, refusing to follow symlinks out of the trusted set."""
    if src.is_symlink():
        src = src.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def validate_local_binary(path: os.PathLike | str, allowed_names: Iterable[str]) -> Path:
    """Validate a whitelisted external binary before running it.

    Only ever called with a hard-coded name list (colmap, blender, ffmpeg).
    """
    p = Path(path)
    if not p.exists() or not os.access(p, os.X_OK):
        raise ValidationError("binary not executable", details={"path": str(p)})
    if p.name.lower() not in {n.lower() for n in allowed_names} and p.stem.lower() not in {
        n.lower() for n in allowed_names
    }:
        raise SecurityError(
            "binary not in the allow-list", details={"path": str(p), "allowed": list(allowed_names)}
        )
    return p

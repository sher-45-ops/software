"""Global engine configuration.

Configuration resolution order (later wins):

1. Built-in defaults below.
2. ``<data-root>/config.json`` (created by ``recon3d setup`` / first run).
3. Environment variables prefixed ``RECON3D_`` (e.g. ``RECON3D_OFFLINE=1``).
4. Explicit overrides passed to :func:`load_config`.
"""

from __future__ import annotations

import json
import os
import platform
import tempfile
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

from .errors import SecurityError

ENV_PREFIX = "RECON3D_"

#: Directories the engine is ever allowed to write to.  Anything else is
#: rejected by :mod:`recon3d.core.security`.
_PROTECTED_NAMES = ("input/original",)


def default_data_root() -> Path:
    """Return the per-user data root (portable-install aware).

    If the environment variable ``RECON3D_HOME`` is set it wins.  Otherwise we
    use a ``.recon3d`` folder in the user's home directory - the same choice on
    Windows, Linux and macOS - so a project folder can be copied between
    machines (spec #56).
    """
    env = os.environ.get("RECON3D_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".recon3d"


def repo_root() -> Path:
    """Return the directory that contains the installed ``recon3d`` package."""
    return Path(__file__).resolve().parent.parent


@dataclass
class HardwareLimits:
    """Guard rails that keep the engine usable on consumer hardware."""

    max_workers: int = 0  # 0 => auto (cpu_count - 1, min 1)
    max_image_dimension: int = 4096  # long edge used for feature extraction
    max_dense_dimension: int = 2048  # long edge used for dense matching / carving
    max_texture_resolution: int = 8192
    max_voxel_grid: int = 512  # hard ceiling for volumetric grid resolution
    max_ram_fraction: float = 0.75  # share of system RAM the engine may target
    mem_budget_mb: int = 0  # 0 => derived from RAM and max_ram_fraction


@dataclass
class Config:
    """Runtime configuration of the engine."""

    data_root: str = field(default_factory=lambda: str(default_data_root()))
    projects_dir: str = ""  # defaults to <data_root>/projects
    cache_dir: str = ""  # defaults to <data_root>/cache
    models_dir: str = ""  # defaults to <data_root>/models
    log_dir: str = ""  # defaults to <data_root>/logs

    offline: bool = False
    telemetry: bool = False  # no telemetry exists; kept explicit for transparency
    allow_network_downloads: bool = True  # model manager honours this + offline

    performance_mode: str = "auto"  # auto|performance|balanced|quality|maximum
    device: str = "auto"  # auto|cpu|cuda|mps
    threads: int = 0  # 0 => auto

    backend_colmap: bool = True  # use COLMAP when installed
    backend_open3d: bool = True  # use Open3D when installed
    backend_blender: bool = True  # allow Blender bridge when installed
    backend_xatlas: bool = True  # use xatlas for UVs when installed
    backend_onnx: bool = True  # allow local ONNX models when present

    allow_overwrite: bool = False
    keep_intermediate: bool = True
    json_logs: bool = False

    limits: HardwareLimits = field(default_factory=HardwareLimits)
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- derived paths -------------------------------------------------
    def path(self, key: str) -> Path:
        raw = getattr(self, key, None)
        if not raw:
            raw = str(Path(self.data_root) / {
                "projects_dir": "projects",
                "cache_dir": "cache",
                "models_dir": "models",
                "log_dir": "logs",
            }[key])
        return Path(raw).expanduser()

    @property
    def projects_path(self) -> Path:
        return self.path("projects_dir")

    @property
    def cache_path(self) -> Path:
        return self.path("cache_dir")

    @property
    def models_path(self) -> Path:
        return self.path("models_dir")

    @property
    def logs_path(self) -> Path:
        return self.path("log_dir")

    def ensure_dirs(self) -> None:
        for p in (self.projects_path, self.cache_path, self.models_path, self.logs_path):
            p.mkdir(parents=True, exist_ok=True)

    # -- serialisation -------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["resolved"] = {
            "projects_dir": str(self.projects_path),
            "cache_dir": str(self.cache_path),
            "models_dir": str(self.models_path),
            "log_dir": str(self.logs_path),
            "platform": platform.platform(),
            "python": platform.python_version(),
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        known = {f.name for f in fields(cls)}
        limits_raw = data.get("limits") or {}
        extra = {k: v for k, v in data.items() if k not in known and k != "resolved"}
        kwargs: Dict[str, Any] = {k: v for k, v in data.items() if k in known and k != "limits"}
        cfg = cls(**kwargs)
        if limits_raw:
            limit_fields = {f.name for f in fields(HardwareLimits)}
            cfg.limits = HardwareLimits(**{k: v for k, v in limits_raw.items() if k in limit_fields})
        cfg.extra = extra
        return cfg

    def save(self, path: Optional[Path] = None) -> Path:
        target = Path(path) if path else (Path(self.data_root) / "config.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(target)
        return target

    def validate(self) -> None:
        if self.performance_mode not in {"auto", "performance", "balanced", "quality", "maximum", "draft"}:
            raise SecurityError(
                f"unknown performance_mode {self.performance_mode!r}",
                details={"allowed": ["auto", "performance", "balanced", "quality", "maximum", "draft"]},
            )
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise SecurityError(f"unknown device {self.device!r}")
        if self.limits.max_texture_resolution not in (512, 1024, 2048, 4096, 8192, 16384):
            raise SecurityError(
                f"unsupported texture resolution {self.limits.max_texture_resolution}",
                details={"allowed": [512, 1024, 2048, 4096, 8192, 16384]},
            )


_BOOL_KEYS = {"offline", "telemetry", "allow_network_downloads", "backend_colmap",
              "backend_open3d", "backend_blender", "backend_xatlas", "backend_onnx",
              "allow_overwrite", "keep_intermediate", "json_logs"}


def _coerce(key: str, value: str) -> Any:
    if key in _BOOL_KEYS:
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if key in {"threads"}:
        return int(value)
    return value


def load_config(
    overrides: Optional[Dict[str, Any]] = None,
    *,
    data_root: Optional[str] = None,
    config_file: Optional[Path] = None,
) -> Config:
    """Load configuration from disk + environment and apply overrides."""
    if data_root:
        os.environ.setdefault("RECON3D_HOME", data_root)
    cfg = Config()
    cfg.data_root = str(data_root) if data_root else cfg.data_root

    path = Path(config_file) if config_file else Path(cfg.data_root) / "config.json"
    if path.exists():
        try:
            cfg = Config.from_dict(json.loads(path.read_text(encoding="utf-8")))
            cfg.data_root = str(data_root) if data_root else cfg.data_root
        except json.JSONDecodeError:
            pass  # a broken config must never brick the engine

    for key in list(Config.__dataclass_fields__):  # type: ignore[attr-defined]
        env_key = ENV_PREFIX + key.upper()
        if key in os.environ:
            setattr(cfg, key, _coerce(key, os.environ[key]))
        elif env_key in os.environ:
            setattr(cfg, key, _coerce(key, os.environ[env_key]))

    # Nested limit overrides: RECON3D_LIMIT_MAX_VOXEL_GRID=256
    for key in HardwareLimits.__dataclass_fields__:  # type: ignore[attr-defined]
        env_key = ENV_PREFIX + "LIMIT_" + key.upper()
        if env_key in os.environ:
            raw = os.environ[env_key]
            cur = getattr(cfg.limits, key)
            setattr(cfg.limits, key, type(cur)(float(raw) if isinstance(cur, float) else int(raw)))

    for key, value in (overrides or {}).items():
        if key == "limits" and isinstance(value, dict):
            for lk, lv in value.items():
                if hasattr(cfg.limits, lk):
                    setattr(cfg.limits, lk, lv)
            continue
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            cfg.extra[key] = value

    if cfg.offline:
        cfg.allow_network_downloads = False
    cfg.validate()
    return cfg


def temp_dir(prefix: str = "recon3d-") -> Path:
    """Create a scratch directory that is *not* inside the project sandbox."""
    return Path(tempfile.mkdtemp(prefix=prefix))


def save_config(config: "Config", path: Optional[Path] = None) -> Path:
    """Persist *config* to ``<data_root>/config.json`` (or an explicit path).

    Only the user-facing fields are written; derived paths are recomputed at load
    time so a project folder stays portable between machines (spec #56).
    """
    target = Path(path) if path else Path(config.data_root).expanduser() / "config.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = config.to_dict(redact=False) if hasattr(config, "to_dict") else asdict(config)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target

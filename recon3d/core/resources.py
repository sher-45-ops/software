"""Hardware detection and automatic performance-mode selection (spec #32)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:  # psutil is a core dependency but never fatal
    import psutil
except Exception:  # pragma: no cover
    psutil = None  # type: ignore[assignment]


@dataclass
class GPUInfo:
    name: str
    vendor: str = "unknown"
    vram_mb: int = 0
    backend: str = "none"  # cuda|cuda-torch|mps|none
    compute_capability: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "vendor": self.vendor,
            "vram_mb": self.vram_mb,
            "backend": self.backend,
            "compute_capability": self.compute_capability,
        }


@dataclass
class HardwareProfile:
    os_name: str
    os_version: str
    python_version: str
    cpu_name: str
    physical_cores: int
    logical_cores: int
    ram_mb: int
    free_ram_mb: int
    disk_free_mb: int
    gpus: List[GPUInfo] = field(default_factory=list)

    # -- capability summary -------------------------------------------
    @property
    def has_gpu(self) -> bool:
        return any(g.backend != "none" for g in self.gpus)

    @property
    def max_vram_mb(self) -> int:
        return max([g.vram_mb for g in self.gpus] or [0])

    def suggested_workers(self) -> int:
        return max(1, min(self.logical_cores - 1 or 1, 16))

    def auto_performance_mode(self) -> str:
        """Pick a conservative default that still produces real geometry.

        The rules are intentionally pessimistic: an 8 GB laptop must not be
        pushed into a 512^3 voxel carve it cannot finish.
        """
        ram = self.ram_mb
        cores = self.logical_cores
        if ram >= 48000 and cores >= 12:
            return "quality"
        if ram >= 24000 and cores >= 8:
            return "balanced"
        if ram >= 14000 and cores >= 4:
            return "balanced"
        if ram >= 7000 and cores >= 2:
            return "performance"
        return "draft"

    def can_run(self, requirement: str) -> Tuple[bool, str]:
        """Check a named requirement, returning ``(ok, reason)``."""
        req = requirement.lower()
        if req in {"cuda", "gpu"}:
            ok = self.has_gpu
            return ok, "" if ok else "no CUDA/MPS capable GPU detected"
        if req == "cuda-8gb":
            ok = self.max_vram_mb >= 7600
            return ok, "" if ok else f"needs >=8GB VRAM, found {self.max_vram_mb}MB"
        if req == "ram-16gb":
            ok = self.ram_mb >= 15000
            return ok, "" if ok else f"needs >=16GB RAM, found {self.ram_mb}MB"
        if req == "ram-8gb":
            ok = self.ram_mb >= 7500
            return ok, "" if ok else f"needs >=8GB RAM, found {self.ram_mb}MB"
        if req == "cpu-4":
            ok = self.logical_cores >= 4
            return ok, "" if ok else f"needs >=4 cores, found {self.logical_cores}"
        if req.startswith("disk-"):
            need = int(req.split("-")[1])
            ok = self.disk_free_mb >= need
            return ok, "" if ok else f"needs {need}MB free disk, found {self.disk_free_mb}MB"
        return True, ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "os": f"{self.os_name} {self.os_version}",
            "python": self.python_version,
            "cpu": self.cpu_name,
            "cores": {"physical": self.physical_cores, "logical": self.logical_cores},
            "ram_mb": self.ram_mb,
            "free_ram_mb": self.free_ram_mb,
            "disk_free_mb": self.disk_free_mb,
            "gpus": [g.to_dict() for g in self.gpus],
            "gpu_available": self.has_gpu,
            "suggested_performance_mode": self.auto_performance_mode(),
            "suggested_workers": self.suggested_workers(),
        }


def _ram_mb() -> Tuple[int, int]:
    if psutil is not None:
        vm = psutil.virtual_memory()
        return int(vm.total / 1024 / 1024), int(vm.available / 1024 / 1024)
    try:  # POSIX fallback
        pages = os.sysconf("SC_PHYS_PAGES")
        avail = os.sysconf("SC_AVPHYS_PAGES")
        page = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page / 1024 / 1024), int(avail * page / 1024 / 1024)
    except (ValueError, OSError, AttributeError):  # pragma: no cover
        return 8192, 4096


def _cpu_name() -> str:
    if sys.platform.startswith("win"):
        return os.environ.get("PROCESSOR_IDENTIFIER", "unknown-windows-cpu")
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return os.environ.get("PROCESSOR_IDENTIFIER", "unknown-cpu")


def detect_gpus() -> List[GPUInfo]:
    """Detect GPUs without requiring torch/CUDA to be installed."""
    gpus: List[GPUInfo] = []

    # 1) torch (most informative when present)
    try:  # pragma: no cover - optional
        import torch  # type: ignore

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append(
                    GPUInfo(
                        name=props.name,
                        vendor="nvidia",
                        vram_mb=int(props.total_memory / 1024 / 1024),
                        backend="cuda-torch",
                        compute_capability=f"{props.major}.{props.minor}",
                    )
                )
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            gpus.append(GPUInfo(name="Apple Silicon (MPS)", vendor="apple", backend="mps"))
    except Exception:
        pass

    # 2) nvidia-smi (works with a plain CUDA driver, no python packages)
    if not gpus:
        exe = shutil.which("nvidia-smi")
        if exe:
            try:
                out = subprocess.run(
                    [exe, "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=8, check=False,
                )
                if out.returncode == 0:
                    for line in out.stdout.strip().splitlines():
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 2:
                            gpus.append(
                                GPUInfo(
                                    name=parts[0],
                                    vendor="nvidia",
                                    vram_mb=int(float(parts[1])) if parts[1] else 0,
                                    backend="cuda",
                                    compute_capability=parts[2] if len(parts) > 2 else "",
                                )
                            )
            except (OSError, subprocess.SubprocessError, ValueError):
                pass

    # 3) Windows wmic fallback
    if not gpus and sys.platform.startswith("win"):  # pragma: no cover
        try:
            out = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "name,AdapterRAM"],
                capture_output=True, text=True, timeout=10, check=False,
            )
            for line in out.stdout.splitlines()[1:]:
                if line.strip():
                    gpus.append(GPUInfo(name=line.strip()[:80], vendor="unknown", backend="none"))
        except (OSError, subprocess.SubprocessError):
            pass
    return gpus


def cuda_runtime_available() -> bool:
    """True when a CUDA driver/device is usable (torch or nvidia-smi detected)."""
    return any(g.backend.startswith("cuda") for g in detect_gpus())


def profile_hardware(disk_path: Optional[Path] = None) -> HardwareProfile:
    total_ram, free_ram = _ram_mb()
    disk_target = Path(disk_path) if disk_path else Path.home()
    try:
        usage = shutil.disk_usage(str(disk_target))
        disk_free = int(usage.free / 1024 / 1024)
    except OSError:  # pragma: no cover
        disk_free = 0
    try:
        import platform as _p

        os_name = _p.system()
        os_ver = _p.release()
    except Exception:  # pragma: no cover
        os_name, os_ver = sys.platform, ""
    return HardwareProfile(
        os_name=os_name,
        os_version=os_ver,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        cpu_name=_cpu_name(),
        physical_cores=psutil.cpu_count(logical=False) or os.cpu_count() or 1 if psutil else (os.cpu_count() or 1),
        logical_cores=(psutil.cpu_count(logical=True) if psutil else os.cpu_count()) or 1,
        ram_mb=total_ram,
        free_ram_mb=free_ram,
        disk_free_mb=disk_free,
        gpus=detect_gpus(),
    )


#: Per-mode processing budgets.  These are consumed by the reconstruction
#: stages so that a preset changes the *pipeline*, not just a number (spec #12).
PERFORMANCE_MODES: Dict[str, Dict[str, Any]] = {
    "draft": {
        "volume_sigma": 0.6, "iso_level": 0.45,
        "lens_search": False, "lens_refine_steps": 0, "lens_carve_resolution": 48,
        "feature_max_dim": 1024,
        "dense_max_dim": 512,
        "voxel_start": 48,
        "voxel_max": 112,
        "carve_passes": 1,
        "photo_consistency": False,
        "ba_iterations": 10,
        "mesh_smooth_iters": 4,
        "decimate_aggressiveness": 0.7,
        "texture_resolution": 1024,
        "ao_quality": "off",
        "compare_renders": 4,
        "refine_passes": 0,
        "lod_levels": 2,
    },
    "performance": {
        "volume_sigma": 0.55, "iso_level": 0.44,
        "feature_max_dim": 1600,
        "dense_max_dim": 768,
        "voxel_start": 48,
        "voxel_max": 160,
        "carve_passes": 2,
        "photo_consistency": True,
        "ba_iterations": 20,
        "mesh_smooth_iters": 3,
        "decimate_aggressiveness": 0.6,
        "texture_resolution": 2048,
        "ao_quality": "fast",
        "compare_renders": 6,
        "refine_passes": 1,
        "lod_levels": 3,
    },
    "balanced": {
        "volume_sigma": 0.5, "iso_level": 0.42,
        "lens_search": True, "lens_refine_steps": 1, "lens_carve_resolution": 56,
        "feature_max_dim": 2048,
        "dense_max_dim": 1024,
        "voxel_start": 64,
        "voxel_max": 256,
        "carve_passes": 2,
        "photo_consistency": True,
        "ba_iterations": 30,
        "mesh_smooth_iters": 3,
        "decimate_aggressiveness": 0.5,
        "texture_resolution": 4096,
        "ao_quality": "fast",
        "compare_renders": 8,
        "refine_passes": 2,
        "lod_levels": 4,
    },
    "quality": {
        "volume_sigma": 0.5, "iso_level": 0.42,
        "lens_search": True, "lens_refine_steps": 2, "lens_carve_resolution": 72,
        "feature_max_dim": 2600,
        "dense_max_dim": 1536,
        "voxel_start": 96,
        "voxel_max": 384,
        "carve_passes": 3,
        "photo_consistency": True,
        "ba_iterations": 45,
        "mesh_smooth_iters": 2,
        "decimate_aggressiveness": 0.4,
        "texture_resolution": 4096,
        "ao_quality": "balanced",
        "compare_renders": 12,
        "refine_passes": 3,
        "lod_levels": 4,
    },
    "maximum": {
        "volume_sigma": 0.45, "iso_level": 0.4,
        "lens_search": True, "lens_refine_steps": 2, "lens_carve_resolution": 96,
        "feature_max_dim": 3600,
        "dense_max_dim": 2048,
        "voxel_start": 128,
        "voxel_max": 512,
        "carve_passes": 4,
        "photo_consistency": True,
        "ba_iterations": 60,
        "mesh_smooth_iters": 2,
        "decimate_aggressiveness": 0.3,
        "texture_resolution": 8192,
        "ao_quality": "high",
        "compare_renders": 16,
        "refine_passes": 4,
        "lod_levels": 4,
    },
}


def mode_budget(mode: str, profile: Optional[HardwareProfile] = None) -> Dict[str, Any]:
    """Return the processing budget for a mode, clamped to the hardware."""
    key = (mode or "auto").lower()
    if key in {"auto", "draft"} and key == "auto":
        key = (profile or profile_hardware()).auto_performance_mode()
    if key not in PERFORMANCE_MODES:
        key = "balanced"
    budget = dict(PERFORMANCE_MODES[key])
    budget["mode"] = key
    if profile is None:
        try:
            profile = profile_hardware()
        except Exception:  # pragma: no cover
            profile = None
    if profile is not None:
        # Clamp to fit this machine instead of failing later.
        ram = profile.ram_mb
        if ram <= 4500:
            budget["voxel_max"] = min(budget["voxel_max"], 224)
            budget["dense_max_dim"] = min(budget["dense_max_dim"], 768)
            budget["texture_resolution"] = min(budget["texture_resolution"], 2048)
        elif ram <= 8500:
            budget["voxel_max"] = min(budget["voxel_max"], 288)
            budget["texture_resolution"] = min(budget["texture_resolution"], 4096)
        if profile.logical_cores <= 2:
            budget["carve_passes"] = min(budget["carve_passes"], 2)
        budget["workers"] = profile.suggested_workers()
        budget["clamped_for_ram_mb"] = ram
    return budget


def memory_guard(required_mb: int, profile: Optional[HardwareProfile] = None) -> Tuple[bool, str]:
    """Cheap pre-flight check used before allocating a big grid."""
    prof = profile or profile_hardware()
    avail = prof.free_ram_mb
    if required_mb > avail * 0.9:
        return False, (
            f"operation needs ~{required_mb}MB RAM but only {avail}MB is free; "
            "lower the quality preset or close other applications"
        )
    return True, ""


def voxel_grid_budget(profile: HardwareProfile, *, bytes_per_voxel: int = 24,
                      max_voxels: int = 0) -> int:
    """Largest cubic grid the machine can carve comfortably."""
    avail = max(512, profile.free_ram_mb)
    budget_mb = avail * 0.5
    n = int((budget_mb * 1024 * 1024 / max(1, bytes_per_voxel)) ** (1.0 / 3.0))
    if max_voxels:
        n = min(n, max_voxels)
    return max(48, min(n, 1024))


def hardware_report(disk_path: Optional[Path] = None) -> Dict[str, Any]:
    """Serialisable hardware description used by ``recon3d doctor`` and the API."""
    profile = profile_hardware(disk_path)
    data = profile.to_dict()
    data.update({
        "device": "cuda" if profile.has_gpu else "cpu",
        "cpu_count": profile.logical_cores,
        "physical_cores": profile.physical_cores,
        "ram_total_mb": profile.ram_mb,
        "ram_available_mb": profile.free_ram_mb,
        "gpu": data["gpus"][0] if data.get("gpus") else None,
    })
    return data

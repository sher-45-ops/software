"""Core runtime: projects, sandboxing, progress, jobs, pipeline orchestration."""

from __future__ import annotations

from .jobs import Job, JobManager, JobState, get_job_manager
from .project import (
    AssetFiles,
    Project,
    ProjectManager,
    ReferenceImage,
    Version,
    hash_inputs,
    sha256_file,
)
from .progress import (
    STAGES,
    CancellationToken,
    ProgressEvent,
    ProgressReporter,
    overall_progress,
    setup_logging,
)
from .resources import (
    PERFORMANCE_MODES,
    GPUInfo,
    HardwareProfile,
    detect_gpus,
    memory_guard,
    mode_budget,
    profile_hardware,
    voxel_grid_budget,
)
from .security import PathSandbox, require_permission, safe_join, slugify, validate_identifier

__all__ = [
    "AssetFiles",
    "CancellationToken",
    "GPUInfo",
    "HardwareProfile",
    "Job",
    "JobManager",
    "JobState",
    "PERFORMANCE_MODES",
    "PathSandbox",
    "Project",
    "ProjectManager",
    "ProgressEvent",
    "ProgressReporter",
    "ReferenceImage",
    "STAGES",
    "Version",
    "detect_gpus",
    "get_job_manager",
    "hash_inputs",
    "memory_guard",
    "mode_budget",
    "overall_progress",
    "profile_hardware",
    "require_permission",
    "safe_join",
    "setup_logging",
    "sha256_file",
    "slugify",
    "validate_identifier",
    "voxel_grid_budget",
]

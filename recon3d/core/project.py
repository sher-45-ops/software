"""Project manager: portable, self-describing project folders (spec #3, #56).

A Recon3D project is a plain directory.  Copying it to another machine (or
zipping it into a CI artifact) preserves everything: references, parameters,
intermediate stages, versions, reports and logs.  There is no database.

Layout::

    <data_root>/projects/<project_id>/
        project.json                 # metadata, params, image index, versions
        input/original/              # untouched copies of the user's images
        input/masks/                 # subject masks produced by segmentation
        input/processed/             # normalised images used by the pipeline
        intermediate/<stage>/        # cached stage outputs + checkpoints
        versions/v001/               # a complete, immutable asset version
            mesh/ textures/ rig/ lod/ previews/ renders/ reports/
        logs/events.jsonl
"""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..errors import ConflictError, NotFoundError, ValidationError
from .progress import STAGES
from .security import (
    PathSandbox,
    safe_join,
    validate_identifier,
    validate_image_path,
    rmtree_within,
    slugify,
)
from .store import dir_size_bytes, merge_dicts, read_json, utc_now, write_json

PROJECT_FILE = "project.json"
PROJECT_FORMAT_VERSION = 1

VIEW_LABELS = [
    "front", "front_right", "right", "back_right", "back", "back_left",
    "left", "front_left", "top", "bottom", "three_quarter", "detail", "unknown",
]

#: Canonical camera azimuth (degrees, clockwise from the subject's front) and
#: elevation for each view label - the skeleton of the coverage map.
VIEW_ANGLES: Dict[str, Tuple[float, float]] = {
    "front": (0.0, 0.0),
    "front_right": (45.0, 0.0),
    "right": (90.0, 0.0),
    "back_right": (135.0, 0.0),
    "back": (180.0, 0.0),
    "back_left": (225.0, 0.0),
    "left": (270.0, 0.0),
    "front_left": (315.0, 0.0),
    "top": (0.0, 80.0),
    "bottom": (0.0, -80.0),
    "three_quarter": (35.0, 15.0),
}


@dataclass
class ReferenceImage:
    """One reference image belonging to a project."""

    id: str
    filename: str  # name inside input/original
    path: str  # path relative to the project root
    view: str = "unknown"  # user/agent supplied hint
    detected_view: str = "unknown"
    azimuth: Optional[float] = None
    elevation: Optional[float] = None
    width: int = 0
    height: int = 0
    sha256: str = ""
    added_at: str = field(default_factory=utc_now)
    notes: str = ""
    quality: Dict[str, Any] = field(default_factory=dict)
    duplicate_of: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReferenceImage":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class AssetFiles:
    """Paths (relative to the version directory) of the generated artefacts."""

    mesh: Dict[str, str] = field(default_factory=dict)  # format -> relative path
    textures: Dict[str, str] = field(default_factory=dict)
    materials: str = ""
    rig: Dict[str, str] = field(default_factory=dict)
    lods: Dict[str, str] = field(default_factory=dict)
    previews: List[str] = field(default_factory=list)
    renders: List[str] = field(default_factory=list)
    reports: Dict[str, str] = field(default_factory=dict)
    pointcloud: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "AssetFiles":
        data = data or {}
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def all_paths(self) -> List[str]:
        out: List[str] = list(self.mesh.values()) + list(self.textures.values())
        out += list(self.rig.values()) + list(self.lods.values())
        out += list(self.previews) + list(self.renders) + list(self.reports.values())
        if self.materials:
            out.append(self.materials)
        if self.pointcloud:
            out.append(self.pointcloud)
        return out


@dataclass
class Version:
    """An immutable reconstruction result."""

    id: str  # v001
    created_at: str = field(default_factory=utc_now)
    label: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    quality: Dict[str, Any] = field(default_factory=dict)
    assets: AssetFiles = field(default_factory=AssetFiles)
    stages_completed: List[str] = field(default_factory=list)
    stages_failed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    duration_s: float = 0.0
    parent: Optional[str] = None
    notes: str = ""
    backend_report: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["assets"] = self.assets.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Version":
        payload = dict(data)
        payload["assets"] = AssetFiles.from_dict(payload.get("assets"))
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})


class Project:
    """A reconstruction project."""

    def __init__(self, root: Path, data: Optional[Dict[str, Any]] = None) -> None:
        self.root = Path(root).resolve()
        self._lock = threading.RLock()
        self.data: Dict[str, Any] = data or {}

    # ------------------------------------------------------------------
    # Construction / persistence
    # ------------------------------------------------------------------
    @classmethod
    def create(cls, root: Path, *, name: str, project_id: Optional[str] = None,
               subject_type: str = "auto", style: str = "realistic",
               description: str = "", params: Optional[Dict[str, Any]] = None,
               overwrite: bool = False) -> "Project":
        root = Path(root)
        if root.exists() and (root / PROJECT_FILE).exists() and not overwrite:
            raise ConflictError(
                "project already exists", details={"path": str(root), "hint": "open it instead"}
            )
        pid = validate_identifier(project_id or slugify(name), "project_id")
        root.mkdir(parents=True, exist_ok=True)
        project = cls(root)
        project.data = {
            "format_version": PROJECT_FORMAT_VERSION,
            "id": pid,
            "name": name or pid,
            "description": description,
            "subject_type": subject_type,
            "style": style,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "images": [],
            "versions": [],
            "current_version": None,
            "params": params or {},
            "tags": [],
            "engine": {"name": "recon3d", "format": PROJECT_FORMAT_VERSION},
            "notes": "",
        }
        project._ensure_structure()
        project.save()
        return project

    @classmethod
    def open(cls, root: Path) -> "Project":
        root = Path(root)
        path = root / PROJECT_FILE
        if not path.exists():
            raise NotFoundError("project.json not found", details={"path": str(root)})
        data = read_json(path)
        if not isinstance(data, dict):
            raise ValidationError("project.json is corrupted", details={"path": str(path)})
        project = cls(root, data)
        project._ensure_structure()
        return project

    def save(self) -> Path:
        with self._lock:
            self.data["updated_at"] = utc_now()
            return write_json(self.root / PROJECT_FILE, self.data)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def id(self) -> str:
        return str(self.data.get("id", self.root.name))

    @property
    def name(self) -> str:
        return str(self.data.get("name", self.id))

    @name.setter
    def name(self, value: str) -> None:
        self.data["name"] = str(value)

    @property
    def subject_type(self) -> str:
        return str(self.data.get("subject_type", "auto"))

    @subject_type.setter
    def subject_type(self, value: str) -> None:
        self.data["subject_type"] = str(value)

    @property
    def style(self) -> str:
        return str(self.data.get("style", "realistic"))

    @style.setter
    def style(self, value: str) -> None:
        self.data["style"] = str(value)

    @property
    def params(self) -> Dict[str, Any]:
        return dict(self.data.get("params") or {})

    @params.setter
    def params(self, value: Dict[str, Any]) -> None:
        self.data["params"] = merge_dicts(self.params, value)

    @property
    def sandbox(self) -> PathSandbox:
        return PathSandbox([self.root])

    # -- directories ----------------------------------------------------
    @property
    def input_dir(self) -> Path:
        return self.root / "input"

    @property
    def originals_dir(self) -> Path:
        return self.input_dir / "original"

    @property
    def masks_dir(self) -> Path:
        return self.input_dir / "masks"

    @property
    def processed_dir(self) -> Path:
        return self.input_dir / "processed"

    @property
    def intermediate_dir(self) -> Path:
        return self.root / "intermediate"

    @property
    def versions_dir(self) -> Path:
        return self.root / "versions"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def events_path(self) -> Path:
        return self.logs_dir / "events.jsonl"

    @property
    def log_file(self) -> Path:
        return self.logs_dir / "recon3d.log"

    def _ensure_structure(self) -> None:
        for path in (self.originals_dir, self.masks_dir, self.processed_dir,
                     self.intermediate_dir, self.versions_dir, self.logs_dir):
            path.mkdir(parents=True, exist_ok=True)

    def stage_dir(self, stage: str, *, create: bool = True) -> Path:
        if stage not in STAGES and stage not in {"features", "sparse", "dense"}:
            # Unknown stages are allowed but must stay filesystem-safe.
            stage = slugify(stage, "stage")
        path = self.intermediate_dir / stage
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------
    @property
    def images(self) -> List[ReferenceImage]:
        return [ReferenceImage.from_dict(d) for d in self.data.get("images", [])]

    def image(self, image_id: str) -> ReferenceImage:
        for img in self.images:
            if img.id == image_id or img.filename == image_id:
                return img
        raise NotFoundError("image not found", details={"image": image_id, "project": self.id})

    def image_path(self, img: ReferenceImage) -> Path:
        return self.root / img.path

    def add_images(
        self,
        paths: Sequence[Path],
        *,
        views: Optional[Sequence[Optional[str]]] = None,
        auto_view_detection: bool = True,
        compute_hash: bool = True,
        notes: str = "",
    ) -> List[ReferenceImage]:
        """Copy reference images into the project (originals are never modified)."""
        from ..engine.analysis.views import detect_view_for_file  # local import: avoids cycle

        added: List[ReferenceImage] = []
        existing_hashes = {img.sha256: img.id for img in self.images if img.sha256}
        with self._lock:
            if len(self.images) + len(paths) > 2000:
                raise ValidationError("too many images for one project", details={"limit": 2000})
            used_names = {img.filename for img in self.images}
            for index, raw in enumerate(paths):
                src = validate_image_path(raw)
                digest = ""
                if compute_hash:
                    digest = sha256_file(src)
                if digest and digest in existing_hashes:
                    # Duplicate content: keep the file out of the project but
                    # report the relationship (spec #5 asks for duplicate detection).
                    dup_of = existing_hashes[digest]
                    img = ReferenceImage(
                        id=f"img{len(self.images) + len(added) + 1:04d}",
                        filename=src.name,
                        path="",
                        view="duplicate",
                        sha256=digest,
                        duplicate_of=dup_of,
                        notes="duplicate of an already added image; not copied",
                    )
                    self.data.setdefault("duplicates", []).append(img.to_dict())
                    continue

                filename = unique_name(src.name, used_names)
                used_names.add(filename)
                dest = self.originals_dir / filename
                if not _same_file(src, dest):
                    shutil.copy2(src, dest)
                hint = (views[index] if views and index < len(views) else None) or "unknown"
                image = ReferenceImage(
                    id=f"img{len(self.images) + len(added) + 1:04d}",
                    filename=filename,
                    path=str(dest.relative_to(self.root)).replace("\\", "/"),
                    view=hint,
                    sha256=digest,
                    notes=notes,
                )
                if auto_view_detection:
                    try:
                        probe = detect_view_for_file(dest)
                        image.detected_view = str(probe.get("view", "unknown"))
                        image.azimuth = probe.get("azimuth")
                        image.elevation = probe.get("elevation")
                        image.width = int(probe.get("width") or 0)
                        image.height = int(probe.get("height") or 0)
                    except Exception:  # pragma: no cover - detection is best effort
                        pass
                self.data.setdefault("images", []).append(image.to_dict())
                existing_hashes.setdefault(digest, image.id)
                added.append(image)
        self.save()
        return added

    def update_image(self, image_id: str, **fields: Any) -> ReferenceImage:
        with self._lock:
            for entry in self.data.get("images", []):
                if entry.get("id") == image_id or entry.get("filename") == image_id:
                    entry.update(fields)
                    entry["updated_at"] = utc_now()
                    self.save()
                    return ReferenceImage.from_dict(entry)
        raise NotFoundError("image not found", details={"image": image_id})

    def remove_images(self, image_ids: Iterable[str], *, confirm: bool = False,
                      delete_files: bool = False) -> int:
        from .security import require_permission

        require_permission("delete_images", confirm)
        ids = set(image_ids)
        removed = 0
        with self._lock:
            keep = []
            for entry in self.data.get("images", []):
                if entry.get("id") in ids or entry.get("filename") in ids:
                    removed += 1
                    if delete_files and entry.get("path"):
                        target = safe_join(self.root, entry["path"])
                        if target.exists():
                            target.unlink()
                    continue
                keep.append(entry)
            self.data["images"] = keep
        self.save()
        return removed

    def clear_derived(self) -> None:
        """Remove masks/processed/intermediate data but keep the originals."""
        for path in (self.masks_dir, self.processed_dir, self.intermediate_dir):
            if path.exists():
                shutil.rmtree(path)
        self._ensure_structure()

    # ------------------------------------------------------------------
    # Versions
    # ------------------------------------------------------------------
    def next_version_id(self) -> str:
        existing = {v.get("id") for v in self.data.get("versions", [])}
        n = 1
        while f"v{n:03d}" in existing:
            n += 1
        return f"v{n:03d}"

    def create_version(self, version_id: Optional[str] = None, *, label: str = "",
                       params: Optional[Dict[str, Any]] = None,
                       parent: Optional[str] = None) -> Version:
        vid = validate_identifier(version_id or self.next_version_id(), "version_id")
        path = self.versions_dir / vid
        if path.exists():
            raise ConflictError("version already exists", details={"version": vid})
        path.mkdir(parents=True, exist_ok=True)
        # The output tree is part of the public contract:
        #   final/       the deliverable meshes (glb/obj/fbx/stl/ply/usd/...)
        #   textures/    PBR maps (basecolor, normal, roughness, metallic, ao, orm)
        #   lod/         LOD0..LODn meshes
        #   rig/         skeleton, skin weights, deformation report
        #   previews/    turntable + still previews for humans and agents
        #   renders/     reference-angle renders used by the comparison loop
        #   reports/     machine-readable JSON reports
        #   intermediate/ side artefacts (point cloud, per-stage dumps)
        for sub in ("final", "textures", "rig", "lod", "previews", "renders",
                    "reports", "intermediate"):
            (path / sub).mkdir(exist_ok=True)
        version = Version(id=vid, label=label, params=dict(params or {}), parent=parent)
        with self._lock:
            self.data.setdefault("versions", []).append(version.to_dict())
            self.data["current_version"] = vid
        self.save()
        return version

    def version_path(self, version_id: str) -> Path:
        return self.versions_dir / validate_identifier(version_id, "version_id")

    def versions(self) -> List[Version]:
        return [Version.from_dict(v) for v in self.data.get("versions", [])]

    @property
    def current_version(self) -> Optional[str]:
        """Id of the current version (the newest one, unless changed explicitly)."""
        return self.data.get("current_version")

    def get_version(self, version_id: Optional[str] = None) -> Version:
        vid = version_id or self.data.get("current_version")
        if not vid:
            raise NotFoundError("project has no versions yet", details={"project": self.id})
        for version in self.versions():
            if version.id == vid:
                return version
        raise NotFoundError("version not found", details={"version": vid, "project": self.id})

    def update_version(self, version: Version, *, save: bool = True) -> Version:
        with self._lock:
            entries = self.data.setdefault("versions", [])
            payload = version.to_dict()
            for i, entry in enumerate(entries):
                if entry.get("id") == version.id:
                    entries[i] = payload
                    break
            else:
                entries.append(payload)
        if save:
            self.save()
        write_json(self.versions_dir / version.id / "version.json", version.to_dict())
        return version

    def set_current_version(self, version_id: str) -> Version:
        version = self.get_version(version_id)
        self.data["current_version"] = version.id
        self.save()
        return version

    def delete_version(self, version_id: str, *, confirm: bool = False) -> None:
        version = self.get_version(version_id)
        rmtree_within(self.version_path(version.id), self.sandbox, confirm=confirm,
                      operation="delete_version")
        with self._lock:
            self.data["versions"] = [v for v in self.data.get("versions", [])
                                     if v.get("id") != version.id]
            if self.data.get("current_version") == version.id:
                remaining = self.data["versions"]
                self.data["current_version"] = remaining[-1]["id"] if remaining else None
        self.save()

    def version_dir(self, version: Version, kind: str) -> Path:
        path = self.version_path(version.id) / kind
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def disk_usage(self) -> Dict[str, int]:
        return {
            "input": dir_size_bytes(self.input_dir),
            "intermediate": dir_size_bytes(self.intermediate_dir),
            "versions": dir_size_bytes(self.versions_dir),
            "logs": dir_size_bytes(self.logs_dir),
            "total": dir_size_bytes(self.root),
        }

    def to_dict(self, *, include_images: bool = True) -> Dict[str, Any]:
        data = dict(self.data)
        if not include_images:
            data.pop("images", None)
        data["root"] = str(self.root)
        try:
            data["disk_usage"] = self.disk_usage()
        except OSError:  # pragma: no cover
            pass
        return data


class ProjectManager:
    """Create, find and delete projects on disk."""

    def __init__(self, projects_dir: Path) -> None:
        self.projects_dir = Path(projects_dir)
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, project_id: str, *, must_exist: bool = False) -> Path:
        pid = validate_identifier(project_id, "project_id")
        path = safe_join(self.projects_dir, pid)
        if must_exist and not (path / PROJECT_FILE).exists():
            raise NotFoundError("project not found", details={"project": pid,
                                                              "projects_dir": str(self.projects_dir)})
        return path

    def create(self, name: str, **kwargs: Any) -> Project:
        pid = validate_identifier(kwargs.pop("project_id", None) or slugify(name), "project_id")
        return Project.create(self.path_for(pid), name=name, project_id=pid, **kwargs)

    def open(self, project_id: str) -> Project:
        return Project.open(self.path_for(project_id, must_exist=True))

    def exists(self, project_id: str) -> bool:
        try:
            return (self.path_for(project_id) / PROJECT_FILE).exists()
        except ValidationError:  # pragma: no cover
            return False

    def list(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for entry in sorted(self.projects_dir.iterdir()):
            if not entry.is_dir():
                continue
            data = read_json(entry / PROJECT_FILE)
            if not isinstance(data, dict):
                continue
            out.append(
                {
                    "id": data.get("id", entry.name),
                    "name": data.get("name", entry.name),
                    "subject_type": data.get("subject_type", "auto"),
                    "style": data.get("style", "realistic"),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                    "images": len(data.get("images") or []),
                    "versions": [v.get("id") for v in (data.get("versions") or [])],
                    "current_version": data.get("current_version"),
                    "path": str(entry),
                    "disk_usage": dir_size_bytes(entry),
                }
            )
        return out

    def delete(self, project_id: str, *, confirm: bool = False) -> None:
        path = self.path_for(project_id, must_exist=True)
        rmtree_within(path, PathSandbox([self.projects_dir]), confirm=confirm,
                      operation="delete_project")

    def resolve(self, identifier: str) -> Project:
        """Open a project by id, by directory path, or by fuzzy name."""
        candidate = Path(identifier).expanduser()
        if candidate.is_dir() and (candidate / PROJECT_FILE).exists():
            return Project.open(candidate)
        if self.exists(identifier):
            return self.open(identifier)
        # fuzzy: match by name field, case-insensitive
        lowered = identifier.lower()
        for info in self.list():
            if str(info["name"]).lower() == lowered or str(info["id"]).lower() == lowered:
                return self.open(str(info["id"]))
        for info in self.list():
            if lowered in str(info["name"]).lower() or lowered in str(info["id"]).lower():
                return self.open(str(info["id"]))
        raise NotFoundError(
            "project not found",
            details={"query": identifier, "projects_dir": str(self.projects_dir),
                     "available": [i["id"] for i in self.list()]},
        )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_inputs(items: Sequence[Any]) -> str:
    """Stable hash of a list of strings/numbers used for stage checkpoints."""
    h = hashlib.sha256()
    for item in items:
        h.update(json.dumps(item, sort_keys=True, default=str).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:32]


def _same_file(source: Path, destination: Path) -> bool:
    """True when the source already *is* the destination file.

    Re-adding an image that already lives in ``input/original`` (or an upload
    staged inside the project) must be a no-op rather than a crash:
    ``shutil.copy2`` raises ``SameFileError`` when source and destination are the
    same path.
    """
    try:
        if source.resolve() == destination.resolve():
            return True
    except OSError:  # pragma: no cover - unresolvable paths fall through
        return False
    try:
        return destination.exists() and source.samefile(destination)
    except OSError:  # pragma: no cover
        return False


def unique_name(filename: str, used: Iterable[str]) -> str:
    used = set(used)
    if filename not in used:
        return filename
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    i = 2
    while f"{stem}_{i}{suffix}" in used:
        i += 1
    return f"{stem}_{i}{suffix}"

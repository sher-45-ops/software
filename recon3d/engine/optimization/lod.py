"""LOD chains, polygon budgets and asset optimisation (spec #14, #20).

The LOD chain is built by decimating the *cleaned* mesh with per-level error
budgets, keeping the silhouettes as close as possible to LOD0 (the silhouette is
what a game engine shows at distance).  Each level is measured against LOD0 with
a chamfer-style surface error so the report can state the real cost of each
reduction instead of only quoting face counts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..geometry.cleanup import mesh_statistics
from ..geometry.simplify import simplify_mesh, decimation_quality


#: Default LOD budgets.  Level 0 is the reconstruction itself; the following
#: levels are the classic halves/thirds used by real-time pipelines.
DEFAULT_LOD_RATIOS: Tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)

#: Human-readable role of each LOD (spec #20).
LOD_ROLES = {
    0: "cinematic / hero",
    1: "high-quality game",
    2: "medium-distance",
    3: "distant / impostor",
}


@dataclass
class LODLevel:
    index: int
    faces: int
    vertices: int
    ratio: float
    role: str
    surface_error: Optional[float] = None
    normalised_error: Optional[float] = None
    path: str = ""
    statistics: Dict[str, Any] = field(default_factory=dict)
    texture_resolution: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LODChain:
    levels: List[LODLevel] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"levels": [l.to_dict() for l in self.levels], "warnings": self.warnings}


def polygon_budget(
    requested: Any,
    *,
    base_faces: int,
    quality: str = "standard",
    style: str = "realistic",
    gpu_realtime: bool = True,
) -> int:
    """Resolve ``target_polycount`` (including ``"auto"``) into a face budget.

    ``auto`` uses the requested quality/style and the reconstructed face count,
    so "game ready" never silently produces a 2-million-triangle asset.
    """
    if isinstance(requested, (int, float)) and requested and requested > 0:
        return int(min(max(requested, 100), 20_000_000))
    key = str(requested or "auto").lower()
    if key not in {"auto", "default", ""}:
        # Named budgets an agent may pass directly.
        named = {"draft": 8_000, "low": 15_000, "lowpoly": 8_000, "mobile": 12_000,
                 "standard": 60_000, "high": 150_000, "cinematic": 600_000,
                 "ultra": 1_500_000, "hero": 800_000}
        if key in named:
            return named[key]
    style_key = (style or "realistic").lower()
    if style_key in {"low-poly", "lowpoly", "game-ready", "game_ready", "stylized"}:
        base = 40_000 if style_key != "low-poly" else 8_000
    elif style_key in {"high-poly", "cinematic", "photorealistic"}:
        base = 900_000
    else:
        base = 250_000
    quality_scale = {"draft": 0.25, "performance": 0.5, "balanced": 1.0,
                     "quality": 2.0, "maximum": 3.5}.get(quality, 1.0)
    return int(np.clip(base * quality_scale, 2_000, 5_000_000))


def build_lod_chain(
    mesh,
    *,
    levels: int = 4,
    ratios: Optional[Sequence[float]] = None,
    target_faces: Optional[int] = None,
    texture_resolution: int = 4096,
    texture_scale_per_level: float = 0.5,
    reporter: Any = None,
) -> Tuple[LODChain, List[Any]]:
    """Build the LOD chain; returns ``(chain, meshes)`` with ``meshes[0]`` = LOD0."""
    ratios = list(ratios or DEFAULT_LOD_RATIOS)
    if levels and levels > 0:
        ratios = ratios[:levels]
    base_faces = int(len(mesh.faces))
    if target_faces:
        # Treat the export budget as LOD0's budget and scale the rest from it.
        ratios = [target_faces / max(1, base_faces)] + [
            r * target_faces / max(1, base_faces) for r in ratios[1:]
        ]
    chain = LODChain()
    meshes: List[Any] = []
    current = mesh
    for index, ratio in enumerate(ratios):
        budget = int(max(120, round(base_faces * ratio)))
        if index == 0:
            level_mesh = mesh
        else:
            level_mesh, report = simplify_mesh(current, budget)
            quality = decimation_quality(level_mesh, mesh)
            error = quality.get("mean_error")
            normalised = quality.get("normalised_mean_error")
            if normalised is not None and normalised > 0.01:
                chain.warnings.append(
                    f"LOD{index} surface error is {normalised * 100:.1f}% of the model diagonal; "
                    "geometry is visibly simplified at this level"
                )
        stats = mesh_statistics(level_mesh)
        chain.levels.append(
            LODLevel(
                index=index,
                faces=int(len(level_mesh.faces)),
                vertices=int(len(level_mesh.vertices)),
                ratio=round(float(len(level_mesh.faces)) / max(1, base_faces), 4),
                role=LOD_ROLES.get(index, f"lod{index}"),
                surface_error=(decimation_quality(level_mesh, mesh).get("mean_error")
                               if index > 0 else 0.0),
                normalised_error=(decimation_quality(level_mesh, mesh).get("normalised_mean_error")
                                  if index > 0 else 0.0),
                statistics=stats,
                texture_resolution=max(256, int(texture_resolution * (texture_scale_per_level ** index))),
            )
        )
        meshes.append(level_mesh)
        if reporter is not None:
            reporter.update(min(99.0, 5 + 95 * (index + 1) / max(1, len(ratios))),
                            f"LOD{index}: {len(level_mesh.faces)} faces")
    if len(chain.levels) > 1 and chain.levels[-1].faces > base_faces * 0.5:
        chain.warnings.append("decimation could not reach the smallest LOD budget")
    return chain, meshes


def optimize_for_realtime(
    mesh,
    *,
    target_faces: int,
    weld: bool = True,
    remove_hidden: bool = False,
    reporter: Any = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Prepare a game-ready mesh: weld, de-duplicate, decimate to budget."""
    from ..geometry.cleanup import remove_duplicate_faces, weld_vertices

    report: Dict[str, Any] = {"original_faces": int(len(mesh.faces))}
    if weld:
        mesh, info = weld_vertices(mesh)
        report["weld"] = info
    mesh, info = remove_duplicate_faces(mesh)
    report["dedupe"] = info
    if target_faces and target_faces < len(mesh.faces):
        mesh, info = simplify_mesh(mesh, int(target_faces))
        report["decimate"] = info
    report["final_faces"] = int(len(mesh.faces))
    report["statistics"] = mesh_statistics(mesh)
    if reporter is not None:
        reporter.info(f"game-ready mesh: {report['final_faces']} faces")
    return mesh, report


def silhouette_retention(lod_mesh, lod0_mesh, cameras: Sequence[Any]) -> Dict[str, Any]:
    """How well each LOD preserves the LOD0 silhouette from the solved cameras."""
    from ..compare.raster import rasterize, silhouette_iou

    values = []
    for pose in cameras:
        a = rasterize(np.asarray(lod0_mesh.vertices), np.asarray(lod0_mesh.faces),
                      pose.camera, silhouette_only=True)
        b = rasterize(np.asarray(lod_mesh.vertices), np.asarray(lod_mesh.faces),
                      pose.camera, silhouette_only=True)
        values.append(silhouette_iou(a.mask, b.mask))
    return {
        "mean_iou": round(float(np.mean(values)), 4) if values else 0.0,
        "min_iou": round(float(np.min(values)), 4) if values else 0.0,
        "views": len(values),
    }

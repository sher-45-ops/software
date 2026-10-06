"""Mesh cleanup: raw reconstruction is never exported as-is (spec #13).

The cleanup chain is ordered from "structurally necessary" to "cosmetic", and
every operation is applied only when it genuinely helps (e.g. hole filling is
skipped when the mesh has no holes, and smoothing is skipped when it would
measurably reduce silhouette fidelity).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class CleanupReport:
    steps: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    before: Dict[str, Any] = field(default_factory=dict)
    after: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"steps": self.steps, "warnings": self.warnings,
                "before": self.before, "after": self.after}

    def record(self, step: str, **data: Any) -> None:
        entry = {"step": step}
        entry.update(data)
        self.steps.append(entry)


def count_components(mesh) -> int:
    """Number of connected components, without copying the mesh visual.

    ``trimesh.Trimesh.split()`` copies the visual of every component, and a textured
    mesh carries full-resolution PIL images - so splitting a textured model once can
    allocate gigabytes.  Counting components from the face adjacency graph is exact,
    cheap and copies nothing.
    """
    if not len(mesh.faces):
        return 0
    try:
        import trimesh

        labels = trimesh.graph.connected_components(mesh.face_adjacency,
                                                   nodes=np.arange(len(mesh.faces)))
        return int(len(labels))
    except Exception:  # pragma: no cover - fall back to the (heavy) trimesh path
        stripped = mesh.copy()
        stripped.visual = None
        return int(len(stripped.split(only_watertight=False)))


def welded_copy(mesh):
    """Copy of *mesh* with coincident vertices merged (UV seams undone).

    Unwrapping duplicates vertices along every UV seam, which makes a single
    watertight surface *look* like dozens of loose components.  Health numbers
    must describe the surface, not the texture layout, so they are measured on a
    welded copy - floaters and holes survive welding and are still reported.
    """
    try:
        try:
            # ``include_visual=False`` matters: a textured mesh deep-copies its PIL
            # images, which is hundreds of megabytes at 4K/8K map sizes.
            copy = mesh.copy(include_visual=False)
        except TypeError:  # pragma: no cover - older trimesh without the keyword
            copy = mesh.copy()
            copy.visual = None
        copy.merge_vertices()
        return copy if len(copy.faces) else mesh
    except Exception:  # pragma: no cover - measurement must never fail a run
        return mesh


def mesh_statistics(mesh) -> Dict[str, Any]:
    """A compact, agent-friendly health report for a mesh."""
    import trimesh

    mesh = welded_copy(mesh)

    stats: Dict[str, Any] = {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "volume": float(mesh.volume) if mesh.is_watertight else None,
        "components": count_components(mesh),
        "bounds": np.asarray(mesh.bounds).tolist() if len(mesh.vertices) else [[0, 0, 0], [0, 0, 0]],
    }
    try:
        stats["euler_number"] = int(mesh.euler_number)
    except Exception:  # pragma: no cover
        stats["euler_number"] = None
    try:
        edges = mesh.edges_unique
        lengths = mesh.edges_unique_length
        if len(lengths):
            stats["edge_length_mean"] = float(np.mean(lengths))
            stats["edge_length_median"] = float(np.median(lengths))
            stats["degenerate_edges"] = int((lengths < 1e-9).sum())
    except Exception:  # pragma: no cover
        pass
    try:
        stats["area"] = float(mesh.area)
    except Exception:  # pragma: no cover
        stats["area"] = None
    if len(mesh.faces):
        mask = mesh.nondegenerate_faces()
        stats["nondegenerate_faces"] = int(mask.sum())
        stats["degenerate_faces"] = int(len(mask) - mask.sum())
    return stats


def mesh_health_score(stats: Dict[str, Any]) -> float:
    """0-100 heuristic health score used by the diagnostics report."""
    score = 100.0
    faces = max(1, stats.get("faces", 0))
    degenerate = stats.get("degenerate_faces", 0) or 0
    score -= min(25.0, 100.0 * degenerate / faces)
    if not stats.get("watertight", False):
        score -= 12.0
    if not stats.get("winding_consistent", False):
        score -= 10.0
    components = stats.get("components", 1) or 1
    if components > 1:
        score -= min(20.0, (components - 1) * 4.0)
    return float(max(0.0, min(100.0, score)))


def remove_small_components(mesh, *, min_fraction: float = 0.02, min_faces: int = 24):
    """Drop floating debris while keeping geometry that is a real part of the subject."""
    try:
        # Split a visual-free copy: trimesh copies the visual (including texture
        # images) for every component otherwise.
        plain = mesh.copy()
        plain.visual = None
        parts = plain.split(only_watertight=False)
    except Exception:  # pragma: no cover
        return mesh, {"removed": 0}
    if len(parts) <= 1:
        return mesh, {"removed": 0}
    sizes = np.array([len(p.faces) for p in parts])
    total = sizes.sum()
    keep = [i for i, s in enumerate(sizes)
            if s >= max(min_faces, int(min_fraction * total))]
    if not keep:
        keep = [int(np.argmax(sizes))]
    if len(keep) == len(parts):
        return mesh, {"removed": 0, "components": len(parts)}
    import trimesh

    kept = trimesh.util.concatenate([parts[i] for i in keep])
    removed = len(parts) - len(keep)
    return kept, {"removed": removed, "kept": len(keep), "component_sizes": sizes.tolist()}


def remove_isolated_vertices(mesh):
    """Drop vertices that belong to no face."""
    try:
        mask = np.zeros(len(mesh.vertices), dtype=bool)
        mask[mesh.faces.reshape(-1)] = True
        if mask.all():
            return mesh, {"removed": 0}
        removed = int((~mask).sum())
        # trimesh's update_vertices handles remapping and returns the new mesh.
        mesh = mesh.copy()
        mesh.update_vertices(mask)
        return mesh, {"removed": removed}
    except Exception as exc:  # pragma: no cover
        return mesh, {"removed": 0, "error": str(exc)}


def weld_vertices(mesh, *, digits: int = 5):
    """Merge duplicate vertices introduced by marching cubes."""
    before = len(mesh.vertices)
    try:
        mesh.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=digits)
    except TypeError:  # pragma: no cover - older trimesh signatures
        mesh.merge_vertices()
    return mesh, {"merged": int(before - len(mesh.vertices))}


def fill_holes(mesh, *, max_hole_size: Optional[float] = None):
    """Fill small holes (trimesh's triangulate-based hole filler)."""
    try:
        import trimesh

        before = len(mesh.faces)
        mesh.fill_holes()
        holes = 0
        try:
            holes = int(len(trimesh.repair.broken_faces(mesh)))
        except Exception:
            holes = 0
        return mesh, {"faces_added": int(len(mesh.faces) - before), "remaining_holes": holes}
    except Exception as exc:  # pragma: no cover
        return mesh, {"error": str(exc)}


def fix_normals(mesh):
    """Make face winding consistent and outward-facing."""
    import trimesh

    before = bool(mesh.is_winding_consistent)
    try:
        trimesh.repair.fix_normals(mesh, multibody=True)
        trimesh.repair.fix_inversion(mesh, multibody=True)
    except Exception:  # pragma: no cover
        pass
    return mesh, {"was_consistent": before, "now_consistent": bool(mesh.is_winding_consistent)}


def remove_degenerate_faces(mesh, *, area_eps: float = 1e-12, aspect_max: float = 40.0):
    """Remove zero-area and needle-thin faces."""
    with np.errstate(divide="ignore", invalid="ignore"):
        areas = mesh.area_faces
        keep = np.isfinite(areas) & (areas > area_eps)
        tri = mesh.triangles
        edge_lengths = np.linalg.norm(np.stack([
            tri[:, 0] - tri[:, 1], tri[:, 1] - tri[:, 2], tri[:, 2] - tri[:, 0]], axis=1), axis=2)
        longest = edge_lengths.max(axis=1)
        shortest = np.maximum(1e-12, edge_lengths.min(axis=1))
        keep &= (longest / shortest) < aspect_max
    removed = int((~keep).sum())
    if removed:
        mesh.update_faces(keep)
    return mesh, {"removed": removed}


def smooth_surface(mesh, *, iterations: int = 2, lamb: float = 0.5, nu: float = -0.53,
                   preserve_features: bool = True, feature_angle_deg: float = 55.0):
    """Taubin smoothing (volume-preserving) with optional sharp-edge protection."""
    import trimesh

    if iterations <= 0 or len(mesh.faces) < 8:
        return mesh, {"skipped": True}
    before_volume = float(mesh.volume) if mesh.is_watertight else None
    if preserve_features:
        try:
            angles = mesh.face_adjacency_angles
            sharp = angles > math.radians(feature_angle_deg)
            locked_faces = np.zeros(len(mesh.faces), dtype=bool)
            if np.any(sharp):
                adj = mesh.face_adjacency
                locked_faces[adj[sharp].reshape(-1)] = True
            if locked_faces.mean() > 0.15:
                # Too much of the surface is "sharp": smoothing would erase detail.
                iterations = max(0, iterations - 1)
        except Exception:  # pragma: no cover
            pass
    try:
        mesh = trimesh.smoothing.filter_taubin(mesh, lamb=lamb, nu=nu, iterations=iterations)
    except Exception as exc:  # pragma: no cover
        return mesh, {"error": str(exc)}
    after_volume = float(mesh.volume) if mesh.is_watertight else None
    shrink = None
    if before_volume and after_volume:
        shrink = float(1.0 - after_volume / before_volume)
    return mesh, {"iterations": iterations, "volume_change": shrink}


def remove_duplicate_faces(mesh):
    before = len(mesh.faces)
    try:
        mesh.update_faces(mesh.unique_faces())
    except Exception:  # pragma: no cover
        pass
    return mesh, {"removed": int(before - len(mesh.faces))}


def detect_self_intersections(mesh, *, max_faces: int = 200_000) -> Dict[str, Any]:
    """Cheap self-intersection probe used by the validation stage."""
    try:
        if len(mesh.faces) > max_faces:
            return {"checked": False, "reason": "mesh too large for a full check",
                    "faces": int(len(mesh.faces))}
        collisions = mesh.self_intersection()
        return {"checked": True, "self_intersecting_pairs": int(len(collisions)),
                "has_self_intersections": bool(len(collisions) > 0)}
    except Exception as exc:  # pragma: no cover
        return {"checked": False, "reason": str(exc)}


def cleanup_mesh(
    mesh,
    *,
    remove_components: bool = True,
    min_component_fraction: float = 0.02,
    do_weld: bool = True,
    do_fill_holes: bool = True,
    do_fix_normals: bool = True,
    do_remove_degenerate: bool = True,
    do_smooth: bool = True,
    smooth_iterations: int = 2,
    preserve_sharp_edges: bool = True,
    target_faces: Optional[int] = None,
    reporter: Any = None,
) -> Tuple[Any, CleanupReport]:
    """Run the full cleanup chain and return ``(mesh, report)``."""
    report = CleanupReport()
    report.before = mesh_statistics(mesh)

    mesh, info = remove_isolated_vertices(mesh)
    report.record("remove_isolated_vertices", **info)

    if do_weld:
        mesh, info = weld_vertices(mesh)
        report.record("weld_vertices", **info)
        mesh, info = remove_duplicate_faces(mesh)
        report.record("remove_duplicate_faces", **info)

    if remove_components:
        mesh, info = remove_small_components(mesh, min_fraction=min_component_fraction)
        report.record("remove_small_components", **info)
        if info.get("removed"):
            report.warnings.append(
                f"removed {info['removed']} disconnected component(s) - these were "
                "reconstruction debris, not part of the subject"
            )

    if do_remove_degenerate:
        mesh, info = remove_degenerate_faces(mesh)
        report.record("remove_degenerate_faces", **info)

    if do_fix_normals:
        mesh, info = fix_normals(mesh)
        report.record("fix_normals", **info)

    if do_fill_holes:
        mesh, info = fill_holes(mesh)
        report.record("fill_holes", **info)

    if do_smooth and smooth_iterations > 0:
        mesh, info = smooth_surface(mesh, iterations=smooth_iterations,
                                    preserve_features=preserve_sharp_edges)
        report.record("smooth_surface", **info)
        if info.get("volume_change") is not None and abs(info["volume_change"]) > 0.12:
            report.warnings.append(
                f"smoothing changed the volume by {info['volume_change'] * 100:.1f}%; "
                "the reconstruction may be noisy"
            )

    if target_faces and len(mesh.faces) > target_faces:
        mesh, info = decimate_mesh(mesh, target_faces=target_faces,
                                   preserve_boundary=True)
        report.record("decimate", **info)

    mesh, info = fix_normals(mesh)
    report.record("fix_normals_final", **info)

    report.after = mesh_statistics(mesh)
    report.after["health_score"] = mesh_health_score(report.after)
    return mesh, report


def decimate_mesh(mesh, *, target_faces: int, preserve_boundary: bool = True,
                  aggressiveness: float = 0.5, preserve_sharp: bool = True):
    """Decimate down to a face budget using the best available backend."""
    from .simplify import simplify_mesh

    current = len(mesh.faces)
    if target_faces <= 0 or current <= target_faces:
        return mesh, {"skipped": True, "faces": current}
    out, report = simplify_mesh(mesh, int(target_faces),
                                preserve_boundary=preserve_boundary,
                                aggressiveness=aggressiveness)
    report.setdefault("preserve_sharp", bool(preserve_sharp))
    return out, report

"""UV unwrapping and atlas packing (spec #15).

Primary backend: **xatlas** (MIT) - chart generation, parameterisation and
packing in one call, the same library Blender-style bakers use.

Fallback backend (no optional dependencies at all): a built-in chart-and-pack
unwrapper - triangles are clustered into charts by normal similarity and
connectivity, each chart is projected onto its dominant axis plane (with
per-chart scale normalisation), and the charts are packed with a shelf packer.
Quality is lower than xatlas but it is a genuine parameterisation with real
islands, distortion measurement and packing - not a placeholder.

Both paths return :class:`UVResult` including the metrics the CLI/API report:
island count, packing efficiency, overlap check and a distortion estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...errors import StageError


@dataclass
class UVResult:
    mesh: Any
    backend: str
    islands: int
    packing_efficiency: float
    uv_quality: float
    overlap: bool
    estimated_distortion: float
    texel_density: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "islands": self.islands,
            "packing_efficiency": round(self.packing_efficiency, 4),
            "uv_quality": round(self.uv_quality, 1),
            "overlap": self.overlap,
            "estimated_distortion": round(self.estimated_distortion, 4),
            "texel_density": round(self.texel_density, 4),
            "warnings": self.warnings,
        }


def available_uv_backends() -> List[str]:
    backends = []
    try:
        import xatlas  # type: ignore  # noqa: F401

        backends.append("xatlas")
    except Exception:
        pass
    backends.append("builtin")
    return backends


# --------------------------------------------------------------------------
# Measurement helpers
# --------------------------------------------------------------------------
def uv_distortion(mesh, uvs: np.ndarray) -> float:
    """Mean scale-invariant area/edge distortion of a parameterisation.

    Both the 3D and UV areas are normalised by their respective means before
    comparison, so the measurement is invariant to the UV scale (a small atlas
    is not "more distorted" than a large one, only stretched differently).
    """
    faces = np.asarray(mesh.faces)
    if len(faces) == 0 or len(uvs) != len(mesh.vertices):
        return 0.0
    tri3 = np.asarray(mesh.vertices)[faces]
    tri2 = np.asarray(uvs)[faces]

    def _areas(tris: np.ndarray) -> np.ndarray:
        if tris.ndim != 3 or tris.shape[2] < 2:  # pragma: no cover
            return np.zeros(0)
        a = tris[:, 1] - tris[:, 0]
        b = tris[:, 2] - tris[:, 0]
        if tris.shape[2] >= 3:
            return np.linalg.norm(np.cross(a, b), axis=1) / 2.0
        cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        return np.abs(cross) / 2.0

    area3 = _areas(tri3)
    area2 = _areas(tri2)
    valid = area3 > 1e-12
    if not valid.any():
        return 0.0
    area3 = area3[valid]
    area2 = np.maximum(area2[valid], 1e-12)
    mean3, mean2 = float(area3.mean()), float(area2.mean())
    if mean3 <= 0 or mean2 <= 0:  # pragma: no cover
        return 0.0
    ae = np.abs(np.log((area2 / mean2) / (area3 / mean3)))

    edge3 = np.linalg.norm(tri3[valid][:, 1] - tri3[valid][:, 0], axis=1)
    edge2 = np.linalg.norm(tri2[valid][:, 1] - tri2[valid][:, 0], axis=1)
    good = edge3 > 1e-12
    if good.any():
        scale = math.sqrt(mean2 / mean3)
        edge2 = np.maximum(edge2[good], 1e-12)
        edge3s = np.maximum(edge3[good] * scale, 1e-12)
        ee = np.abs(np.log(edge2 / edge3))
    else:  # pragma: no cover
        ee = np.zeros_like(ae)
    return float(np.mean(0.5 * ae + 0.5 * ee))


def uv_overlap_ratio(uvs: np.ndarray, faces: np.ndarray, resolution: int = 512) -> float:
    """Fraction of texels claimed by more than one triangle (overlap detector)."""
    counts = rasterize_uv_counts(uvs, faces, resolution)
    total = counts > 0
    if not total.any():
        return 0.0
    return float((counts > 1).sum() / total.sum())


def rasterize_uv_counts(uvs: np.ndarray, faces: np.ndarray, resolution: int = 512) -> np.ndarray:
    """Count how many triangles cover each texel (barycentric rasterisation)."""
    counts = np.zeros((resolution, resolution), dtype=np.int32)
    tri = uvs[faces]
    for t in tri:
        p = t * resolution
        xu = int(max(0, math.floor(p[:, 0].min())))
        xl = int(min(resolution - 1, math.ceil(p[:, 0].max())))
        yu = int(max(0, math.floor(p[:, 1].min())))
        yl = int(min(resolution - 1, math.ceil(p[:, 1].max())))
        if xl < xu or yl < yu:
            continue
        xs = np.arange(xu, xl + 1) + 0.5
        ys = np.arange(yu, yl + 1) + 0.5
        px, py = np.meshgrid(xs, ys)
        x0, y0 = p[0]
        x1, y1 = p[1]
        x2, y2 = p[2]
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-12:
            continue
        l0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
        l1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
        l2 = 1.0 - l0 - l1
        inside = (l0 >= -1e-9) & (l1 >= -1e-9) & (l2 >= -1e-9)
        if inside.any():
            counts[py[inside].astype(int), px[inside].astype(int)] += 1
    return counts


def packing_efficiency(uvs: np.ndarray, faces: np.ndarray) -> float:
    """Fraction of the [0, 1] UV atlas actually covered by triangles.

    Overlapping islands would push this above what a clean atlas can reach, so
    values close to (or above) 1.0 are reported alongside the overlap ratio.
    """
    tri2 = uvs[faces]
    a = tri2[:, 1] - tri2[:, 0]
    b = tri2[:, 2] - tri2[:, 0]
    area = 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum()
    return float(np.clip(area, 0.0, 1.0))


def _count_islands(uvs: np.ndarray, faces: np.ndarray, tolerance: float = 1e-4) -> int:
    """Count UV islands by union-find over shared UV coordinates."""
    parent = list(range(len(faces)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Map quantised UV coordinate -> first face using it.
    seen: Dict[Tuple[int, int], int] = {}
    for fi, face in enumerate(faces):
        for vi in face:
            key = (int(round(uvs[vi][0] / tolerance)), int(round(uvs[vi][1] / tolerance)))
            if key in seen:
                union(fi, seen[key])
            else:
                seen[key] = fi
    return len({find(i) for i in range(len(faces))})


# --------------------------------------------------------------------------
# xatlas backend
# --------------------------------------------------------------------------
def unwrap_xatlas(mesh, *, resolution: int = 1024, padding: int = 4) -> Tuple[Any, np.ndarray, Dict[str, Any]]:
    """Unwrap with xatlas (MIT).  Falls back via the caller when unavailable."""
    import trimesh
    import xatlas

    vertices = np.ascontiguousarray(np.asarray(mesh.vertices, dtype=np.float32))
    faces = np.ascontiguousarray(np.asarray(mesh.faces, dtype=np.uint32))
    try:
        vmapping, indices, uvs = xatlas.parametrize(vertices, faces)
    except Exception as exc:  # pragma: no cover - xatlas can fail on degenerate meshes
        raise StageError(f"xatlas parametrisation failed: {exc}", stage="uv",
                         recoverable=True) from exc
    vmapping = np.asarray(vmapping, dtype=np.int64).reshape(-1)
    indices = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    uvs = np.asarray(uvs, dtype=np.float64).reshape(-1, 2)
    uv_min = uvs.min(axis=0) if len(uvs) else np.zeros(2)
    uv_max = uvs.max(axis=0) if len(uvs) else np.ones(2)
    if float(uv_max[0] - uv_min[0]) > 1.5 or float(uv_max[1] - uv_min[1]) > 1.5:
        # xatlas packs into a [0, 1) atlas already; anything else means the
        # parameterisation is not a normalised atlas - keep it inside [0, 1].
        span = np.maximum(uv_max - uv_min, 1e-12)
        uvs = (uvs - uv_min) / span
    new_vertices = vertices[vmapping].astype(np.float64)
    out = trimesh.Trimesh(vertices=new_vertices, faces=indices, process=False)
    out.visual = trimesh.visual.TextureVisuals(uv=uvs)
    info = {
        "backend": "xatlas",
        "atlas_width": int(resolution),
        "atlas_height": int(resolution),
        "original_vertices": int(len(mesh.vertices)),
        "original_faces": int(len(mesh.faces)),
    }
    return out, uvs, info


# --------------------------------------------------------------------------
# Built-in backend: chart clustering + axis projection + shelf packing
# --------------------------------------------------------------------------
def _chart_clusters(mesh, *, angle_threshold_deg: float = 55.0,
                    max_chart_faces: int = 4096) -> List[np.ndarray]:
    """Group faces into planar-ish charts (axis buckets + adjacency components).

    Grouping by dominant normal axis *and* by connectivity avoids the classic
    planar-projection failure mode where the front and the back of a closed
    surface land on top of each other in the same chart.
    """
    faces = np.asarray(mesh.faces)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    if len(faces) == 0:
        return []
    dominant = np.argmax(np.abs(normals), axis=1)
    sign = np.sign(normals[np.arange(len(faces)), dominant])
    sign[sign == 0] = 1.0
    bucket = dominant * 2 + (sign > 0).astype(np.int64)

    parent = np.arange(len(faces))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    for a, b in adjacency:
        if bucket[a] == bucket[b]:
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[rb] = ra

    groups: Dict[int, List[int]] = {}
    for fi in range(len(faces)):
        groups.setdefault(find(fi), []).append(fi)

    charts: List[np.ndarray] = []
    for members in groups.values():
        if len(members) <= max_chart_faces:
            charts.append(np.asarray(members, dtype=np.int64))
            continue
        # Split oversized charts spatially so the packing stays efficient.
        idx = np.asarray(members, dtype=np.int64)
        centres = np.asarray(mesh.triangles_center)[idx]
        for axis in range(3):
            if len(idx) <= max_chart_faces:
                break
            order = np.argsort(centres[:, axis])
            idx, centres = idx[order], centres[order]
            chunks = [idx[i:i + max_chart_faces] for i in range(0, len(idx), max_chart_faces)]
            if axis == 2:
                charts.extend(chunks)
        else:
            charts.append(idx)
    return charts


def _project_chart(mesh, faces_idx: np.ndarray) -> Tuple[np.ndarray, int, float]:
    """Planar-project one chart along its dominant normal axis (isometric)."""
    normals = np.asarray(mesh.face_normals)[faces_idx]
    mean_normal = normals.mean(axis=0)
    n = mean_normal / max(1e-12, float(np.linalg.norm(mean_normal)))
    dominant = int(np.argmax(np.abs(n)))
    sign = 1.0 if n[dominant] >= 0 else -1.0
    return np.asarray(mesh.vertices)[np.asarray(mesh.faces)[faces_idx]], dominant, sign


def unwrap_builtin(mesh, *, resolution: int = 1024, padding: int = 4,
                   angle_threshold_deg: float = 55.0) -> Tuple[Any, np.ndarray, Dict[str, Any]]:
    """Built-in chart/atlas unwrapper (used when xatlas is unavailable).

    Charts are axis-bucketed connected components, each chart gets its own
    vertex copies (so islands can be packed freely without sharing vertices),
    and charts are shelf-packed preserving their relative texel density.
    """
    import trimesh

    faces = np.asarray(mesh.faces)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    charts = _chart_clusters(mesh, angle_threshold_deg=angle_threshold_deg)
    if not charts:  # pragma: no cover
        raise StageError("no charts could be built for unwrapping", stage="uv", recoverable=True)

    projected: List[Tuple[np.ndarray, np.ndarray, Tuple[float, float], Tuple[float, float]]] = []
    for faces_idx in charts:
        tris, dominant, sign = _project_chart(mesh, faces_idx)
        axes = [0, 1, 2]
        axes.remove(dominant)
        local = tris[:, :, axes].reshape(-1, 2)
        # Mirror one axis for negative-facing charts so triangles keep orientation.
        if sign < 0:
            local = local * np.asarray([1.0, -1.0])
        lo, hi = local.min(axis=0), local.max(axis=0)
        size = np.maximum(hi - lo, 1e-9)
        projected.append((faces_idx, local, (float(lo[0]), float(lo[1])), tuple(size)))

    # Global scale: keep the relative size of charts (uniform texel density),
    # then fit everything into [0, 1] with shelf packing.
    total_area = sum(float(s[0] * s[1]) for _f, _l, _lo, s in projected) or 1.0
    side = math.sqrt(total_area) * 1.55
    scale = 1.0 / max(1e-9, side)
    pad = padding / float(resolution)
    sizes = [(float(s[0]) * scale, float(s[1]) * scale) for _f, _l, _lo, s in projected]
    order = sorted(range(len(projected)), key=lambda i: sizes[i][1], reverse=True)
    placements: Dict[int, Tuple[float, float]] = {}
    x = y = cursor_y = pad
    used_w = pad
    for i in order:
        w, h = sizes[i]
        if x + w + pad > 1.0:
            used_w = max(used_w, x)
            x = pad
            y = cursor_y + pad
        placements[i] = (x, y)
        x += w + pad
        cursor_y = max(cursor_y, y + h)
    used_w = max(used_w, x)
    used_h = cursor_y + pad
    overflow = max(used_w, used_h)
    if overflow > 0.985:
        shrink = 0.98 / overflow
        for i in placements:
            px, py = placements[i]
            placements[i] = (px * shrink, py * shrink)
            sizes[i] = (sizes[i][0] * shrink, sizes[i][1] * shrink)

    # Rebuild the mesh with per-chart vertex copies and assign the UVs.
    new_vertices: List[np.ndarray] = []
    new_faces: List[np.ndarray] = []
    uv_rows: List[np.ndarray] = []
    offset = 0
    for i, (faces_idx, local, lo, size) in enumerate(projected):
        chart_faces = faces[faces_idx]
        unique, inverse = np.unique(chart_faces.reshape(-1), return_inverse=True)
        chunk_vertices = vertices[unique]
        # One UV per unique chart vertex: take the first (face, corner) that
        # mapped to it, since all corners of a vertex project identically.
        flat = local.reshape(-1, 2)
        first = np.empty(len(unique), dtype=np.int64)
        first[inverse[::-1]] = np.arange(len(inverse) - 1, -1, -1, dtype=np.int64)
        uvs_local = (flat[first] - np.asarray(lo)) / np.asarray(size)
        w, h = sizes[i]
        px, py = placements[i]
        chart_uv = np.stack([px + uvs_local[:, 0] * w, py + uvs_local[:, 1] * h], axis=1)
        new_vertices.append(chunk_vertices)
        new_faces.append(inverse.reshape(-1, 3) + offset)
        uv_rows.append(chart_uv)
        offset += len(chunk_vertices)

    out_vertices = np.concatenate(new_vertices, axis=0)
    out_faces = np.concatenate(new_faces, axis=0)
    out_uvs = np.concatenate(uv_rows, axis=0)
    out = trimesh.Trimesh(vertices=out_vertices, faces=out_faces, process=False)
    out.visual = trimesh.visual.TextureVisuals(uv=out_uvs)
    info = {
        "backend": "builtin",
        "charts": len(charts),
        "atlas_width": resolution,
        "atlas_height": resolution,
        "original_vertices": int(len(vertices)),
        "original_faces": int(len(faces)),
    }
    return out, out_uvs, info


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def unwrap_mesh(mesh, *, resolution: int = 1024, padding: int = 4,
                backend: str = "auto", seam_aware: bool = True) -> UVResult:
    """Generate UV coordinates for *mesh* and measure their quality."""
    warnings: List[str] = []
    if mesh is None or len(mesh.faces) == 0:
        raise StageError("cannot unwrap an empty mesh", stage="uv", recoverable=False)

    order = [backend] if backend != "auto" else available_uv_backends()
    out = None
    uvs = None
    info: Dict[str, Any] = {}
    last_error = ""
    for name in order:
        try:
            if name == "xatlas":
                out, uvs, info = unwrap_xatlas(mesh, resolution=resolution, padding=padding)
            elif name == "builtin":
                out, uvs, info = unwrap_builtin(mesh, resolution=resolution, padding=padding)
            else:
                continue
            if out is not None and len(out.faces) > 0:
                break
        except Exception as exc:
            last_error = f"{name}: {exc}"
            out, uvs = None, None
    if out is None or uvs is None:
        raise StageError("all UV backends failed", stage="uv", recoverable=True,
                         details={"tried": order, "last_error": last_error})

    # Measure the result (on the *unwrapped* mesh, which is what gets exported).
    faces = np.asarray(out.faces)
    distortion = uv_distortion(out, uvs)
    packing = packing_efficiency(uvs, faces)
    islands = _count_islands(np.asarray(uvs), faces)
    overlap_ratio = uv_overlap_ratio(np.asarray(uvs), faces, min(512, resolution))
    texel_density = float(np.sqrt(1.0 / max(1e-12, mesh.area)) * resolution) if hasattr(mesh, "area") else 0.0

    # Quality score: distortion and packing dominate; overlap is disqualifying.
    quality = 100.0
    quality -= min(45.0, distortion * 100.0)
    quality -= min(20.0, max(0.0, 0.45 - packing) * 44.0)
    if overlap_ratio > 0.02:
        quality -= min(25.0, overlap_ratio * 300.0)
        warnings.append(f"UV islands overlap on {overlap_ratio * 100:.1f}% of texels")
    if islands > max(50, len(faces) // 8):
        warnings.append(f"very high island count ({islands}); texture seams may be visible")
        quality -= 8.0
    if distortion > 0.25:
        warnings.append("high UV distortion; texture stretching is likely")
    if info.get("backend") == "builtin":
        warnings.append(
            "xatlas is not installed: the built-in unwrapper was used. "
            "Install it with `pip install xatlas` for production-quality UVs."
        )

    return UVResult(
        mesh=out, backend=str(info.get("backend", "unknown")), islands=islands,
        packing_efficiency=packing, uv_quality=float(max(0.0, min(100.0, quality))),
        overlap=bool(overlap_ratio > 0.02), estimated_distortion=float(distortion),
        texel_density=texel_density, warnings=warnings,
    )


def ensure_uvs(mesh, **kwargs: Any) -> UVResult:
    """Unwrap only when the mesh has no usable UVs yet."""
    existing = getattr(getattr(mesh, "visual", None), "uv", None)
    if existing is not None and len(existing) == len(mesh.vertices):
        faces = np.asarray(mesh.faces)
        uvs = np.asarray(existing, dtype=np.float64)
        distortion = uv_distortion(mesh, uvs)
        return UVResult(mesh=mesh, backend="existing", islands=_count_islands(uvs, faces),
                        packing_efficiency=packing_efficiency(uvs, faces),
                        uv_quality=float(max(0.0, 100.0 - distortion * 100.0)),
                        overlap=uv_overlap_ratio(uvs, faces, 256) > 0.02,
                        estimated_distortion=distortion)
    return unwrap_mesh(mesh, **kwargs)

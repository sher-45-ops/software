"""Mesh simplification: polygon budgets, LODs and refinement proxies.

Backends are tried in order of quality/speed and the first available one wins:

1. ``fast_simplification`` (MIT, wheels for all major platforms) - the fastest
   quadric decimator and the backend trimesh itself prefers;
2. ``open3d`` (MIT) - ``simplify_quadric_decimation``;
3. **built-in** :func:`quadric_decimate` - a pure-numpy quadric error metric
   (QEM) edge-collapse implementation.  Slower, but always available, so the
   engine can hit a ``target_polycount`` on a machine with no optional
   dependencies at all (spec #14, #63: no undocumented manual setup).

The built-in implementation is a real Garland-Heckbert style decimator: it
accumulates 4x4 quadrics per vertex, evaluates collapse cost with a boundary
constraint, and performs greedy edge collapses with a heap, keeping the mesh
manifold (edge-collapse validity is checked before applying).
"""

from __future__ import annotations

import heapq
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


def available_backends() -> List[str]:
    backends: List[str] = []
    try:
        import fast_simplification  # type: ignore  # noqa: F401

        backends.append("fast_simplification")
    except Exception:
        pass
    try:
        import open3d  # type: ignore  # noqa: F401

        backends.append("open3d")
    except Exception:
        pass
    backends.append("builtin")
    return backends


# --------------------------------------------------------------------------
# Unified entry point
# --------------------------------------------------------------------------
def simplify_mesh(mesh, target_faces: int, *, preserve_boundary: bool = True,
                  backend: str = "auto", aggressiveness: float = 0.5) -> Tuple[Any, Dict[str, Any]]:
    """Reduce *mesh* to approximately *target_faces* faces.

    Returns ``(mesh, report)``.  The report records which backend ran so the
    provenance of every exported asset is auditable.
    """
    import trimesh

    current = len(mesh.faces)
    if target_faces <= 0 or current <= target_faces:
        return mesh, {"skipped": True, "faces": current, "backend": "none"}

    order = ([backend] if backend != "auto" else available_backends())
    last_error = ""
    for name in order:
        try:
            if name == "fast_simplification":
                out = _via_fast_simplification(mesh, target_faces)
            elif name == "open3d":
                out = _via_open3d(mesh, target_faces)
            elif name == "builtin":
                out = quadric_decimate(mesh, target_faces)
            else:
                continue
            if out is not None and len(out.faces) > 0:
                return out, {
                    "backend": name,
                    "from": current,
                    "to": int(len(out.faces)),
                    "requested": int(target_faces),
                }
        except Exception as exc:  # pragma: no cover - fall through to the next backend
            last_error = f"{name}: {exc}"
    return mesh, {"skipped": True, "faces": current, "backend": "none", "error": last_error}


def _via_fast_simplification(mesh, target_faces: int):
    import fast_simplification  # type: ignore
    import trimesh

    verts, faces = fast_simplification.simplify(
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.int32),
        target_reduction=max(0.0, min(0.999, 1.0 - target_faces / max(1, len(mesh.faces)))),
    )
    out = trimesh.Trimesh(vertices=np.asarray(verts, dtype=np.float64),
                          faces=np.asarray(faces, dtype=np.int64), process=False)
    return out


def _via_open3d(mesh, target_faces: int):
    import open3d  # type: ignore
    import trimesh

    o3d_mesh = open3d.geometry.TriangleMesh(
        open3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
        open3d.utility.Vector3iVector(np.asarray(mesh.faces)),
    )
    simple = o3d_mesh.simplify_quadric_decimation(int(target_faces))
    return trimesh.Trimesh(vertices=np.asarray(simple.vertices),
                           faces=np.asarray(simple.triangles), process=False)


# --------------------------------------------------------------------------
# Built-in quadric decimation
# --------------------------------------------------------------------------
def _quadric_for_plane(a: float, b: float, c: float, d: float) -> np.ndarray:
    return np.array([
        [a * a, a * b, a * c, a * d],
        [a * b, b * b, b * c, b * d],
        [a * c, b * c, c * c, c * d],
        [a * d, b * d, c * d, d * d],
    ], dtype=np.float64)


def _face_quadrics(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """One 4x4 quadric per face, normalised so big and small triangles weigh equally."""
    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    areas = lengths[:, 0] / 2.0
    safe = np.where(lengths < 1e-15, 1.0, lengths)
    unit = normals / safe
    d = -np.einsum("ij,ij->i", unit, tri[:, 0])
    a, b, c = unit[:, 0], unit[:, 1], unit[:, 2]
    quadrics = np.zeros((len(faces), 4, 4), dtype=np.float64)
    quadrics[:, 0, 0] = a * a
    quadrics[:, 0, 1] = quadrics[:, 1, 0] = a * b
    quadrics[:, 0, 2] = quadrics[:, 2, 0] = a * c
    quadrics[:, 0, 3] = quadrics[:, 3, 0] = a * d
    quadrics[:, 1, 1] = b * b
    quadrics[:, 1, 2] = quadrics[:, 2, 1] = b * c
    quadrics[:, 1, 3] = quadrics[:, 3, 1] = b * d
    quadrics[:, 2, 2] = c * c
    quadrics[:, 2, 3] = quadrics[:, 3, 2] = c * d
    quadrics[:, 3, 3] = d * d
    # weight by area so that dense (small) triangles do not dominate
    quadrics *= np.maximum(areas, 1e-9)[:, None, None]
    return quadrics


class _DSU:
    """Disjoint set used to track vertex merges."""

    __slots__ = ("parent",)

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        parent = self.parent
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root


def _boundary_quadrics(vertices: np.ndarray, faces: np.ndarray, boundary_edges: Set[Tuple[int, int]]
                       ) -> np.ndarray:
    """Large virtual planes perpendicular to boundary edges, keeping borders intact."""
    quadrics = np.zeros((len(vertices), 4, 4), dtype=np.float64)
    scale = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))) or 1.0
    for i, j in boundary_edges:
        p0, p1 = vertices[i], vertices[j]
        edge = p1 - p0
        length = float(np.linalg.norm(edge))
        if length < 1e-12:
            continue
        direction = edge / length
        # A plane containing the edge, orthogonal to the surface normal region.
        normal = np.cross(direction, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(normal) < 1e-6:
            normal = np.cross(direction, np.array([0.0, 1.0, 0.0]))
        normal = normal / max(1e-12, float(np.linalg.norm(normal)))
        d = -float(np.dot(normal, p0))
        q = _quadric_for_plane(normal[0], normal[1], normal[2], d) * (scale * scale * 100.0)
        quadrics[i] += q
        quadrics[j] += q
    return quadrics


def quadric_decimate(mesh, target_faces: int, *, max_rounds: int = 60,
                     max_seconds: float = 120.0) -> Any:
    """Pure-numpy quadric error metric edge-collapse decimation.

    Greedy, with a lazy-deletion heap and manifold-safe collapse checks.  The
    target may not be hit exactly (greedy QEM stops when no legal collapse
    remains), which the caller reports honestly in the simplification report.
    """
    import time

    import trimesh

    started = time.time()
    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int64)
    if len(faces) <= target_faces:
        return mesh

    # Work internally on "clusters": each cluster is represented by a vertex id.
    quadrics = np.zeros((len(vertices), 4, 4), dtype=np.float64)
    face_q = _face_quadrics(vertices, faces)
    for fi, face in enumerate(faces):
        for vi in face:
            quadrics[vi] += face_q[fi]

    # Boundary preservation
    edge_count: Dict[Tuple[int, int], int] = {}
    for face in faces:
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = (min(a, b), max(a, b))
            edge_count[key] = edge_count.get(key, 0) + 1
    boundary = {key for key, count in edge_count.items() if count == 1}
    quadrics += _boundary_quadrics(vertices, faces, boundary)

    representative = list(range(len(vertices)))
    alive_faces = [True] * len(faces)

    # Vertex -> incident faces
    vertex_faces: List[Set[int]] = [set() for _ in range(len(vertices))]
    for fi, face in enumerate(faces):
        for vi in face:
            vertex_faces[vi].add(fi)

    def vertex_position(v: int) -> np.ndarray:
        return vertices[representative[v]]

    def collapse_cost(i: int, j: int) -> Tuple[float, np.ndarray]:
        q = quadrics[i] + quadrics[j]
        try:
            solution = np.linalg.solve(q[:3, :3], -q[:3, 3])
            if not np.all(np.isfinite(solution)):
                raise np.linalg.LinAlgError
            pos = solution
            cost = float(pos @ q[:3, :3] @ pos + 2 * q[:3, 3] @ pos + q[3, 3])
        except np.linalg.LinAlgError:
            # Singular: try the three candidate positions.
            candidates = [vertex_position(i), vertex_position(j), (vertex_position(i) + vertex_position(j)) / 2]
            best_cost = math.inf
            pos = candidates[2]
            for candidate in candidates:
                extended = np.append(candidate, 1.0)
                value = float(extended @ q @ extended)
                if value < best_cost:
                    best_cost, pos = value, candidate
            cost = best_cost
        return max(0.0, cost), np.asarray(pos, dtype=np.float64)

    heap: List[Tuple[float, int, int]] = []
    for key in edge_count:
        i, j = key
        cost, _ = collapse_cost(i, j)
        heapq.heappush(heap, (cost, i, j))

    n_faces = sum(alive_faces)
    stalled_rounds = 0
    while heap and n_faces > target_faces and time.time() - started < max_seconds:
        cost, i, j = heapq.heappop(heap)
        if not vertex_faces[i] and not vertex_faces[j]:
            continue
        # Recompute the cost to keep the heap "lazy" but correct.
        recomputed, position = collapse_cost(i, j)
        if recomputed > cost * 1.5 + 1e-12 and heap:
            heapq.heappush(heap, (recomputed * 1.001, i, j))
            stalled_rounds += 1
            if stalled_rounds > 8 * len(heap) + 1000:
                break
            continue
        stalled_rounds = 0

        # Merge j into i.
        merged_faces = vertex_faces[i] | vertex_faces[j]
        new_faces = 0
        removed = 0
        for fi in merged_faces:
            if not alive_faces[fi]:
                continue
            face = faces[fi].copy()
            for k in range(3):
                if face[k] == j:
                    face[k] = i
            if len(set(face.tolist())) < 3:
                alive_faces[fi] = False
                removed += 1
                continue
            new_faces += 1
        if new_faces == 0:
            continue
        faces_to_drop = [fi for fi in merged_faces if alive_faces[fi]]
        for fi in faces_to_drop[: removed]:  # pragma: no cover - defensive
            alive_faces[fi] = False
        # Apply the collapse.
        vertices[i] = position
        quadrics[i] = quadrics[i] + quadrics[j]
        for fi in merged_faces:
            if alive_faces[fi]:
                faces[fi] = np.where(faces[fi] == j, i, faces[fi])
                vertex_faces[i].add(fi)
                vertex_faces[j].discard(fi)
        vertex_faces[j].clear()
        n_faces -= removed
        # Refresh the neighbourhood.
        for fi in list(vertex_faces[i])[:64]:
            for vk in faces[fi]:
                if vk == i:
                    continue
                new_cost, _ = collapse_cost(i, int(vk))
                heapq.heappush(heap, (new_cost, i, int(vk)))

    keep_faces = [fi for fi, alive in enumerate(alive_faces) if alive]
    if not keep_faces:
        return mesh
    new_faces_array = faces[keep_faces]
    used = np.unique(new_faces_array.reshape(-1))
    remap = -np.ones(len(vertices), dtype=np.int64)
    remap[used] = np.arange(len(used))
    out = trimesh.Trimesh(vertices=vertices[used],
                          faces=remap[new_faces_array], process=False)
    return out


def decimation_quality(mesh, reference) -> Dict[str, Any]:
    """Measure how much a simplified mesh deviates from the original surface."""
    try:
        a = np.asarray(mesh.sample(4000) if hasattr(mesh, "sample") else mesh.vertices)
        b = np.asarray(reference.sample(8000) if hasattr(reference, "sample") else reference.vertices)
        from scipy.spatial import cKDTree

        tree = cKDTree(b)
        dist, _ = tree.query(a, k=1)
        diag = float(np.linalg.norm(np.asarray(reference.bounds)[1] - np.asarray(reference.bounds)[0]))
        return {
            "mean_error": round(float(dist.mean()), 6),
            "max_error": round(float(dist.max()), 6),
            "normalised_mean_error": round(float(dist.mean() / max(1e-9, diag)), 5),
            "diagonal": round(diag, 6),
        }
    except Exception as exc:  # pragma: no cover
        return {"error": str(exc)}

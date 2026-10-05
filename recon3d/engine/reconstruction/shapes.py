"""Procedural shape builders used for tests, demos and self-checks.

These are *not* the reconstruction output.  They are ground-truth subjects used
by the test suite (so the pipeline can be verified against a known shape) and by
``recon3d demo`` to produce a runnable example dataset.  The reconstruction
itself never uses them.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np


def _mesh_from_arrays(vertices: np.ndarray, faces: np.ndarray):
    import trimesh

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def icosphere(radius: float = 1.0, subdivisions: int = 3, center=(0, 0, 0)):
    """UV-free icosphere with vertex colours optional."""
    import trimesh

    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    mesh.apply_translation(center)
    return mesh


def ellipsoid(radii: Tuple[float, float, float] = (1.0, 0.6, 0.8), subdivisions: int = 4,
              center=(0, 0, 0), transform: np.ndarray | None = None):
    mesh = icosphere(1.0, subdivisions, center=(0, 0, 0))
    mesh.vertices = mesh.vertices * np.asarray(radii, dtype=np.float64)
    if transform is not None:
        mesh.apply_transform(transform)
    mesh.apply_translation(center)
    return mesh


def capsule_between(p0, p1, radius: float, sections: int = 12, height_segments: int = 6):
    """A capsule from p0 to p1 (used to build limb-like test subjects)."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length < 1e-9:
        return ellipsoid((radius, radius, radius), 3, center=p0)
    import trimesh

    body = trimesh.creation.capsule(height=max(1e-6, length), radius=radius,
                                    count=[height_segments, sections])
    # capsule is built along +Z centred on the origin; align it with the axis
    z = axis / length
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(z, ref)) > 0.9999:
        rot = np.eye(4)
        if np.dot(z, ref) < 0:
            rot = trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0])
    else:
        v = np.cross(ref, z)
        c = float(np.dot(ref, z))
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rot3 = np.eye(3) + kmat + kmat.dot(kmat) * (1.0 / (1.0 + c))
        rot = np.eye(4)
        rot[:3, :3] = rot3
    body.apply_transform(rot)
    body.apply_translation((p0 + p1) / 2.0)
    return body


def make_test_subject(kind: str = "robot", *, scale: float = 1.0):
    """Return a ``trimesh.Trimesh`` used as reconstruction ground truth.

    Supported kinds: ``robot`` (humanoid-ish, asymmetric details), ``vase``
    (rotationally symmetric organic), ``crate`` (hard-surface box with bevels),
    ``suzanne_ish`` (blob with two protrusions).
    """
    import trimesh

    parts: List[trimesh.Trimesh] = []
    if kind == "robot":
        torso = ellipsoid((0.30, 0.20, 0.42), 3, center=(0, 0, 1.15))
        head = ellipsoid((0.16, 0.16, 0.17), 3, center=(0, 0, 1.62))
        pelvis = ellipsoid((0.24, 0.17, 0.16), 3, center=(0, 0, 0.80))
        parts += [torso, head, pelvis]
        for side in (-1, 1):
            parts.append(capsule_between((0.34 * side, 0, 1.30), (0.62 * side, 0.05, 1.02), 0.075))
            parts.append(capsule_between((0.62 * side, 0.05, 1.02), (0.72 * side, 0.08, 0.72), 0.062))
            parts.append(capsule_between((0.13 * side, 0, 0.74), (0.16 * side, 0.0, 0.40), 0.095))
            parts.append(capsule_between((0.16 * side, 0, 0.40), (0.18 * side, 0.03, 0.05), 0.075))
            foot = ellipsoid((0.10, 0.20, 0.05), 2, center=(0.18 * side, 0.06, 0.05))
            parts.append(foot)
        # asymmetric detail so orientation is measurable
        parts.append(capsule_between((0.30, -0.16, 1.45), (0.62, -0.22, 1.62), 0.05))
        subject = trimesh.util.concatenate(parts)
    elif kind == "vase":
        profile = []
        heights = np.linspace(0.0, 1.0, 28)
        for t in heights:
            r = 0.12 + 0.22 * math.sin(math.pi * (0.15 + 0.75 * t)) + 0.06 * math.sin(6 * math.pi * t) * t
            profile.append((r, t * 0.9))
        verts = []
        faces = []
        segments = 48
        for ri, (r, y) in enumerate(profile):
            for si in range(segments):
                a = 2 * math.pi * si / segments
                verts.append((r * math.cos(a), r * math.sin(a), y))
        for ri in range(len(profile) - 1):
            for si in range(segments):
                a = ri * segments + si
                b = ri * segments + (si + 1) % segments
                c = (ri + 1) * segments + (si + 1) % segments
                d = (ri + 1) * segments + si
                faces.append((a, b, c))
                faces.append((a, c, d))
        subject = _mesh_from_arrays(np.array(verts), np.array(faces))
    elif kind == "crate":
        box = trimesh.creation.box(extents=(0.5, 0.4, 0.35))
        lid = trimesh.creation.box(extents=(0.54, 0.44, 0.05))
        lid.apply_translation((0, 0, 0.20))
        handle = capsule_between((-0.15, 0, 0.25), (0.15, 0, 0.25), 0.025)
        parts += [box, lid, handle]
        parts.append(trimesh.creation.box(extents=(0.06, 0.44, 0.06)).apply_translation((0.26, 0, 0.0)))
        subject = trimesh.util.concatenate(parts)
    else:  # suzanne_ish blob
        base = ellipsoid((0.3, 0.3, 0.28), 4, center=(0, 0, 1.0))
        parts.append(base)
        parts.append(capsule_between((0.22, 0, 1.18), (0.5, 0, 1.30), 0.07))
        parts.append(capsule_between((-0.22, 0, 1.18), (-0.5, 0, 1.30), 0.07))
        parts.append(ellipsoid((0.12, 0.16, 0.10), 3, center=(0, -0.24, 1.02)))
        subject = trimesh.util.concatenate(parts)

    if scale != 1.0:
        subject.apply_scale(scale)
    subject.process(validate=True)
    return subject


def make_ground_truth_mesh(kind: str = "robot", *, scale: float = 1.0):
    """Alias used by the demo/test tooling."""
    return make_test_subject(kind, scale=scale)


def vertex_color_from_position(mesh, palette: Dict[str, List[int]] | None = None):
    """Give a test subject a simple position-based colouring (for texture tests)."""
    import trimesh

    verts = np.asarray(mesh.vertices)
    z = verts[:, 2]
    zn = (z - z.min()) / max(1e-9, z.max() - z.min())
    colors = np.zeros((len(verts), 4), dtype=np.uint8)
    colors[:, 0] = (60 + 180 * zn).astype(np.uint8)
    colors[:, 1] = (110 + 60 * (1 - zn)).astype(np.uint8)
    colors[:, 2] = (200 - 120 * zn).astype(np.uint8)
    colors[:, 3] = 255
    # a distinctive red patch so orientation is observable
    mask = (verts[:, 1] < -0.15) & (zn > 0.6)
    colors[mask] = (220, 40, 40, 255)
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)
    return mesh

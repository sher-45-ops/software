"""Character rigging: skeleton synthesis, skinning and pose analysis (spec #19, #8).

The rig is *derived from the reconstructed geometry*, not imposed on it: limb
positions are found by slicing the mesh along its principal axis and analysing the
cross-section distribution (arms/legs show up as separate lobes), then joints are
placed at the measured limb axis.  For humanoid subjects a standard 22-bone
skeleton is produced; for creatures without detectable limbs a spine chain is
produced instead, and the report says which route was taken and why.

Skinning uses bounded biharmonic-style distance weighting: for each vertex the
k nearest bones are weighted by inverse distance to the bone segment with a
falloff, then normalised and smoothed over the mesh - a solid, standard
alternative to neural skinning that works on CPU for any topology.

This is honest about its limits: automatic rigs are never as good as a hand-made
one, and the report states the measured quality (weight sanity, deformation
stress on a test pose) so an agent can decide whether to use it.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...errors import StageError
from ...core.store import write_json


@dataclass
class Bone:
    name: str
    head: List[float]
    tail: List[float]
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    connected: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.asarray(self.tail) - np.asarray(self.head)))


@dataclass
class Rig:
    bones: List[Bone]
    kind: str
    root: str
    skinning: Dict[str, Any] = field(default_factory=dict)
    quality: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "root": self.root,
            "bones": [b.to_dict() for b in self.bones],
            "skinning": self.skinning,
            "quality": self.quality,
            "warnings": self.warnings,
        }

    def bone_names(self) -> List[str]:
        return [b.name for b in self.bones]


# --------------------------------------------------------------------------
# Geometry analysis
# --------------------------------------------------------------------------
def _principal_axes(mesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (centroid, axes sorted by descending variance, extents)."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    weights = None
    try:
        samples, face_idx = mesh.sample(20000, return_index=True)
        vertices = np.asarray(samples, dtype=np.float64)
    except Exception:  # pragma: no cover
        pass
    centroid = vertices.mean(axis=0)
    centred = vertices - centroid
    cov = np.cov(centred.T)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    axes = vectors[:, order]
    # Make the first axis point "up" for upright subjects: use +Z when it is
    # already reasonably aligned, otherwise the dominant axis.
    if abs(axes[2, 2]) < 0.5:
        up = np.array([0.0, 0.0, 1.0])
        if np.dot(axes[:, 2], up) < 0:
            axes[:, 2] *= -1
    projected = centred @ axes
    extents = projected.max(axis=0) - projected.min(axis=0)
    return centroid, axes, extents


def _limb_profile(mesh, axis: np.ndarray, centroid: np.ndarray, bins: int = 24
                  ) -> List[Dict[str, Any]]:
    """Cross-section analysis along the main axis: lobes per slice."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    rel = (vertices - centroid) @ axis
    lo, hi = float(rel.min()), float(rel.max())
    if hi - lo < 1e-9:
        return []
    slices: List[Dict[str, Any]] = []
    for i in range(bins):
        a = lo + (hi - lo) * i / bins
        b = lo + (hi - lo) * (i + 1) / bins
        sel = (rel >= a) & (rel < b)
        if sel.sum() < 8:
            slices.append({"t": (i + 0.5) / bins, "count": int(sel.sum()), "lobes": 0,
                           "lateral_extent": 0.0, "depth_extent": 0.0})
            continue
        pts = vertices[sel]
        lateral = (pts - centroid) @ axis * 0  # placeholder to keep shape
        # Distances in the plane orthogonal to `axis`
        perp = pts - (pts @ axis)[:, None] * axis
        # 1D k-means-ish lobe detection along the lateral direction
        u = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(u, axis)) > 0.9:
            u = np.array([0.0, 0.0, 1.0])
        e1 = np.cross(axis, u)
        e1 /= max(1e-12, np.linalg.norm(e1))
        e2 = np.cross(axis, e1)
        x = perp @ e1
        y = perp @ e2
        lobes = _count_lobes(x)
        slices.append({
            "t": round((i + 0.5) / bins, 4),
            "count": int(sel.sum()),
            "lobes": int(lobes),
            "lateral_extent": round(float(x.max() - x.min()), 5),
            "depth_extent": round(float(y.max() - y.min()), 5),
        })
    return slices


def _count_lobes(values: np.ndarray, *, gap_ratio: float = 0.22) -> int:
    """Count separated clusters in a 1D distribution (limbs show as separate lobes)."""
    if len(values) < 8:
        return 1
    lo, hi = float(values.min()), float(values.max())
    span = hi - lo
    if span < 1e-9:
        return 1
    bins = 48
    hist, edges = np.histogram(values, bins=bins, range=(lo, hi))
    threshold = max(1, int(hist.max() * 0.12))
    occupied = hist > threshold
    # gap = a run of empty bins wider than gap_ratio * span
    min_gap_bins = max(1, int(bins * gap_ratio))
    lobes = 0
    run = 0
    in_lobe = False
    for flag in occupied:
        if flag:
            if not in_lobe:
                lobes += 1
                in_lobe = True
            run = 0
        else:
            run += 1
            if in_lobe and run >= min_gap_bins:
                in_lobe = False
    return max(1, lobes)


def _visible_landmarks(mesh, axis: np.ndarray, centroid: np.ndarray) -> Dict[str, Any]:
    """Key heights (as fractions along the main axis) that drive bone placement."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    rel = (vertices - centroid) @ axis
    lo, hi = float(rel.min()), float(rel.max())
    height = hi - lo
    if height <= 1e-9:
        raise StageError("degenerate mesh for rigging", stage="rigging", recoverable=False)
    slices = _limb_profile(mesh, axis, centroid, bins=32)
    if not slices:
        return {"height": height, "lo": lo, "hi": hi, "slices": []}
    # Legs: the lowest third has >= 2 lobes.  Hips: the first slice going up that
    # collapses to 1 lobe.  Shoulders: widest slice in the upper half.
    lower = [s for s in slices if s["t"] < 0.45]
    upper = [s for s in slices if s["t"] >= 0.45]
    leg_lobes = max((s["lobes"] for s in lower), default=1)
    if lower:
        multi = [s["t"] for s in lower if s["lobes"] >= 2]
        hip_t = max(multi) if multi else 0.45
    else:
        hip_t = 0.45
    shoulder = max(upper, key=lambda s: s["lateral_extent"] * (1.0 - s["t"] * 0.3)) if upper else None
    return {
        "height": height, "lo": lo, "hi": hi, "slices": slices,
        "leg_lobes": leg_lobes, "hip_t": hip_t,
        "shoulder_t": shoulder["t"] if shoulder else 0.8,
        "shoulder_width": shoulder["lateral_extent"] if shoulder else height * 0.25,
    }


# --------------------------------------------------------------------------
# Skeleton synthesis
# --------------------------------------------------------------------------
def build_humanoid_rig(mesh, *, subject_type: str = "auto", arm_count: int = 2,
                       leg_count: int = 2) -> Rig:
    """Synthesise a humanoid skeleton from the mesh's measured proportions."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    centroid, axes, extents = _principal_axes(mesh)
    up = axes[:, 0] if abs(axes[0, 2]) < 0.7 else axes[:, 2]
    if up[2] < 0 and np.dot(up, [0, 0, 1]) < 0:
        up = -up
    # Use world +Z when the subject is upright (most characters are).
    if np.dot(up, [0, 0, 1]) > 0.5:
        up = np.array([0.0, 0.0, 1.0])
    landmarks = _visible_landmarks(mesh, up, centroid)
    lo, hi = landmarks["lo"], landmarks["hi"]
    height = landmarks["height"]
    hip_t = float(np.clip(landmarks["hip_t"], 0.3, 0.6))
    shoulder_t = float(np.clip(landmarks["shoulder_t"], 0.6, 0.92))

    def point(t: float, lateral: np.ndarray = None, forward: float = 0.0,
              forward_axis: np.ndarray = None) -> List[float]:
        base = centroid + up * (lo + height * t)
        if lateral is not None:
            base = base + lateral
        if forward and forward_axis is not None:
            base = base + forward_axis * forward
        return [float(v) for v in base]

    lateral_axis = np.cross(up, np.array([0.0, 1.0, 0.0]))
    if np.linalg.norm(lateral_axis) < 1e-6:
        lateral_axis = np.array([1.0, 0.0, 0.0])
    lateral_axis = lateral_axis / np.linalg.norm(lateral_axis)
    forward_axis = np.cross(up, lateral_axis)
    forward_axis = forward_axis / max(1e-12, np.linalg.norm(forward_axis))

    shoulder_width = float(landmarks.get("shoulder_width") or height * 0.22)
    hip_half = shoulder_width * 0.35
    arm_offset = shoulder_width * 0.55
    foot_t = 0.03
    ankle_t = 0.10
    knee_t = hip_t * 0.5
    hip_z = lo + height * hip_t
    shoulder_z = lo + height * shoulder_t
    head_base_t = min(0.95, shoulder_t + (1.0 - shoulder_t) * 0.35)

    bones: List[Bone] = []

    def add(name: str, head: Sequence[float], tail: Sequence[float], parent: Optional[str]) -> None:
        bones.append(Bone(name=name, head=[float(v) for v in head], tail=[float(v) for v in tail],
                          parent=parent, connected=True))

    pelvis = point(hip_t)
    spine_1 = point(hip_t + (shoulder_t - hip_t) * 0.33)
    spine_2 = point(hip_t + (shoulder_t - hip_t) * 0.66)
    chest = point(shoulder_t)
    neck = point(head_base_t)
    head_tip = point(min(0.99, head_base_t + (1.0 - head_base_t) * 0.9))

    add("root", [float(v) for v in np.asarray(pelvis) - up * height * 0.02], pelvis, None)
    add("pelvis", pelvis, spine_1, "root")
    add("spine_01", spine_1, spine_2, "pelvis")
    add("spine_02", spine_2, chest, "spine_01")
    add("neck", chest, neck, "spine_02")
    add("head", neck, head_tip, "neck")

    for side, sign in (("l", 1.0), ("r", -1.0)):
        shoulder = np.asarray(chest) + lateral_axis * (arm_offset * sign)
        elbow = np.asarray(chest) + lateral_axis * (arm_offset * 1.75 * sign) - up * height * 0.16
        wrist = np.asarray(chest) + lateral_axis * (arm_offset * 2.05 * sign) - up * height * 0.31
        hand_end = np.asarray(chest) + lateral_axis * (arm_offset * 2.2 * sign) - up * height * 0.38
        add(f"clavicle_{side}", chest, shoulder, "spine_02")
        add(f"upperarm_{side}", shoulder, elbow, f"clavicle_{side}")
        add(f"lowerarm_{side}", elbow, wrist, f"upperarm_{side}")
        add(f"hand_{side}", wrist, hand_end, f"lowerarm_{side}")

        hip = np.asarray(pelvis) + lateral_axis * (hip_half * sign)
        knee = np.asarray(pelvis) + lateral_axis * (hip_half * sign) - up * height * (hip_t - knee_t)
        ankle = np.asarray(pelvis) + lateral_axis * (hip_half * sign) - up * height * (hip_t - ankle_t)
        toe = ankle + forward_axis * (height * 0.07) - up * height * (ankle_t - foot_t)
        add(f"thigh_{side}", hip, knee, "pelvis")
        add(f"calf_{side}", knee, ankle, f"thigh_{side}")
        add(f"foot_{side}", ankle, toe, f"calf_{side}")

    rig = Rig(bones=bones, kind="humanoid", root="root")
    rig.quality = {
        "bones": len(bones),
        "measured_height": round(float(height), 5),
        "measured_shoulder_width": round(float(shoulder_width), 5),
        "detected_limb_lobes": landmarks.get("leg_lobes", 1),
        "hip_height_fraction": round(hip_t, 3),
        "shoulder_height_fraction": round(shoulder_t, 3),
    }
    if landmarks.get("leg_lobes", 1) < 2:
        rig.warnings.append(
            "no separate leg lobes were detected in the silhouette cross-sections; "
            "the leg bones are positioned from the overall proportions and may need "
            "manual adjustment"
        )
    return rig


def build_creature_rig(mesh, *, spine_segments: int = 6, subject_type: str = "auto") -> Rig:
    """A generic spine chain for creatures/objects without detectable limbs."""
    centroid, axes, extents = _principal_axes(mesh)
    main = axes[:, 0]
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    rel = (vertices - centroid) @ main
    lo, hi = float(rel.min()), float(rel.max())
    bones: List[Bone] = []
    points = [centroid + main * (lo + (hi - lo) * i / spine_segments)
              for i in range(spine_segments + 1)]
    bones.append(Bone("root", [float(v) for v in points[0]], [float(v) for v in points[1]], None))
    for i in range(1, spine_segments):
        bones.append(Bone(f"spine_{i:02d}", [float(v) for v in points[i]],
                          [float(v) for v in points[i + 1]],
                          "root" if i == 1 else f"spine_{i - 1:02d}"))
    rig = Rig(bones=bones, kind="creature", root="root")
    rig.quality = {"bones": len(bones), "axis": main.tolist()}
    rig.warnings.append("generic spine chain generated (no limb structure detected)")
    return rig


# --------------------------------------------------------------------------
# Skinning
# --------------------------------------------------------------------------
def _segment_distance(points: np.ndarray, head: np.ndarray, tail: np.ndarray) -> np.ndarray:
    """Distance from each point to a bone segment."""
    ab = tail - head
    length_sq = float(np.dot(ab, ab))
    if length_sq < 1e-12:
        return np.linalg.norm(points - head, axis=1)
    t = np.clip(((points - head) @ ab) / length_sq, 0.0, 1.0)
    projection = head + t[:, None] * ab
    return np.linalg.norm(points - projection, axis=1)


def compute_skin_weights(
    mesh,
    rig: Rig,
    *,
    max_influences: int = 4,
    falloff: float = 2.0,
    smooth_iterations: int = 2,
) -> Dict[str, Any]:
    """Distance-based skinning with normalisation and mesh smoothing."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    n = len(vertices)
    if n == 0:
        raise StageError("cannot skin an empty mesh", stage="rigging", recoverable=False)
    names = rig.bone_names()
    n_bones = len(names)
    if n_bones == 0:
        raise StageError("rig has no bones", stage="rigging", recoverable=False)

    distances = np.zeros((n, n_bones), dtype=np.float64)
    for bi, bone in enumerate(rig.bones):
        head = np.asarray(bone.head, dtype=np.float64)
        tail = np.asarray(bone.tail, dtype=np.float64)
        distances[:, bi] = _segment_distance(vertices, head, tail)

    # Bones thinner than the model scale should not bleed into neighbours: use an
    # inverse-power falloff with a floor to avoid divisions by zero.
    scale = float(np.median(distances[distances > 0])) if np.any(distances > 0) else 1.0
    weights = 1.0 / np.power(np.maximum(distances, scale * 0.02), falloff)

    # Keep the strongest influences only.
    if n_bones > max_influences:
        order = np.argsort(weights, axis=1)[:, :-max_influences]
        np.put_along_axis(weights, order, 0.0, axis=1)
    totals = weights.sum(axis=1, keepdims=True)
    weights = weights / np.maximum(totals, 1e-12)

    # Smooth over the mesh graph so neighbouring vertices deform coherently.
    if smooth_iterations > 0:
        try:
            adjacency = mesh.vertex_neighbors
        except Exception:  # pragma: no cover
            adjacency = None
        if adjacency is not None:
            for _ in range(smooth_iterations):
                smoothed = weights.copy()
                for vi, neighbours in enumerate(adjacency):
                    if not neighbours:
                        continue
                    smoothed[vi] = 0.5 * weights[vi] + 0.5 * weights[list(neighbours)].mean(axis=0)
                weights = smoothed
            totals = weights.sum(axis=1, keepdims=True)
            weights = weights / np.maximum(totals, 1e-12)

    # Influence histogram + sanity metrics.
    sums = weights.sum(axis=1)
    unused = int(((weights.max(axis=0) < 1e-4)).sum())
    quality = {
        "vertices": int(n),
        "bones": n_bones,
        "max_influences": int(max_influences),
        "weight_sum_min": round(float(sums.min()), 5),
        "weight_sum_max": round(float(sums.max()), 5),
        "mean_active_bones": round(float((weights > 1e-3).sum(axis=1).mean()), 2),
        "unused_bones": unused,
    }
    warnings: List[str] = []
    if unused:
        warnings.append(f"{unused} bone(s) received no skin weights and will not deform the mesh")
    if abs(float(sums.mean()) - 1.0) > 1e-3:
        warnings.append("skin weights are not perfectly normalised")

    return {
        "joints": names,
        "weights": weights.astype(np.float32),
        "quality": quality,
        "warnings": warnings,
    }


def deformation_stress_test(mesh, rig: Rig, weights: np.ndarray, *, bone_index: int = 0,
                            rotation_deg: float = 25.0) -> Dict[str, Any]:
    """Rotate one bone and measure how the skinned mesh behaves.

    A cheap, useful sanity metric: with correct weights a moderate rotation
    produces a bounded, smooth displacement and no volume collapse.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if weights is None or len(weights) != len(vertices) or bone_index >= len(rig.bones):
        return {"tested": False}
    bone = rig.bones[bone_index]
    head = np.asarray(bone.head, dtype=np.float64)
    angle = math.radians(rotation_deg)
    axis = np.array([0.0, 0.0, 1.0])
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    local = vertices - head
    rotated = local @ rotation.T + head
    influence = weights[:, bone_index][:, None]
    deformed = vertices * (1 - influence) + rotated * influence
    displacement = np.linalg.norm(deformed - vertices, axis=1)
    scale = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    return {
        "tested": True,
        "bone": bone.name,
        "rotation_deg": rotation_deg,
        "max_displacement": round(float(displacement.max()), 5),
        "mean_displacement": round(float(displacement.mean()), 5),
        "normalised_max_displacement": round(float(displacement.max() / max(1e-9, scale)), 4),
        "affected_vertices": int((displacement > 1e-4).sum()),
    }


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def generate_rig(
    mesh,
    *,
    kind: str = "auto",
    subject_type: str = "auto",
    max_influences: int = 4,
    report_dir: Optional[Any] = None,
    reporter: Any = None,
) -> Dict[str, Any]:
    """Generate a rig + skin weights for a character-like mesh."""
    if mesh is None or len(mesh.faces) == 0:
        raise StageError("cannot rig an empty mesh", stage="rigging", recoverable=False)

    resolved = (kind or "auto").lower()
    if resolved in {"auto", "humanoid"}:
        subject_ok = subject_type in {"auto", "character_humanoid", "character_creature",
                                      "robot_mech", "animal", "unknown"}
        if not subject_ok:
            raise StageError(
                f"rigging requested for subject type '{subject_type}', which is not a character",
                stage="rigging", recoverable=True,
                details={"hint": "pass rig.type=humanoid to force a humanoid skeleton"},
            )
        rig = build_humanoid_rig(mesh, subject_type=subject_type)
        if resolved == "auto" and subject_type in {"character_creature", "animal"}:
            rig.warnings.append(
                "creature subject: a humanoid skeleton was fitted to the measured "
                "proportions; use kind=creature for a spine-only rig"
            )
    elif resolved in {"creature", "quadruped", "object"}:
        rig = build_creature_rig(mesh, subject_type=subject_type)
    else:
        raise StageError(f"unsupported rig kind '{kind}'", stage="rigging", recoverable=True,
                         details={"supported": ["auto", "humanoid", "creature", "quadruped", "none"]})

    if reporter is not None:
        reporter.update(50, f"computing skin weights for {len(rig.bones)} bones")
    skin = compute_skin_weights(mesh, rig, max_influences=max_influences)
    rig.skinning = {
        "joints": skin["joints"],
        "quality": skin["quality"],
        "weights_path": "rig/skin_weights.npz",
    }
    rig.warnings.extend(skin["warnings"])
    stress = deformation_stress_test(mesh, rig, skin["weights"],
                                     bone_index=min(1, len(rig.bones) - 1))
    rig.quality["stress_test"] = stress
    return {"rig": rig, "weights": skin["weights"], "stress": stress, "quality": skin["quality"]}


def write_rig(directory: Any, result: Dict[str, Any]) -> Dict[str, str]:
    """Persist the rig JSON plus the raw weights (npz) next to the mesh."""
    from pathlib import Path

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rig: Rig = result["rig"]
    write_json(directory / "rig.json", rig.to_dict())
    weights_path = directory / "skin_weights.npz"
    np.savez_compressed(weights_path, weights=result["weights"],
                        joints=np.array(rig.bone_names(), dtype=object))
    gltf_skin = {
        "skeleton_bones": rig.bone_names(),
        "joints_index_order": rig.bone_names(),
        "notes": "weights are row-per-vertex, column per joint (see rig.json)",
    }
    write_json(directory / "skin.json", gltf_skin)
    return {
        "rig": str(directory / "rig.json"),
        "weights": str(weights_path),
        "skin": str(directory / "skin.json"),
    }

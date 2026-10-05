"""Asset intelligence: subject classification, symmetry and pose analysis.

All features are measurable image statistics - silhouette proportions, limb
counting, wheel detection, skin-tone fraction, face detection, hard-surface
statistics - combined by explicit rules.  Nothing here is generative, and every
verdict carries the evidence that produced it so an agent can audit or override
it (spec #3 "Asset Intelligence", #8, #9).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...core.store import utc_now, write_json
from ..ingestion.loader import LoadedImage, to_gray

try:
    import cv2

    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False


SUBJECT_TYPES = [
    "character_humanoid", "character_creature", "animal", "robot_mech",
    "vehicle_wheeled", "vehicle_aircraft", "weapon", "furniture", "building",
    "plant", "tool_machine", "sculpture", "object_generic", "auto",
]

#: Typical real-world size in metres, used only for an *optional* unit scale
#: (exposed in the report; never baked into geometry silently).
TYPICAL_SCALE_M: Dict[str, float] = {
    "character_humanoid": 1.75, "character_creature": 1.8, "animal": 0.8,
    "robot_mech": 2.2, "vehicle_wheeled": 4.5, "vehicle_aircraft": 12.0,
    "weapon": 0.9, "furniture": 1.0, "building": 10.0, "plant": 1.0,
    "tool_machine": 0.5, "sculpture": 1.5, "object_generic": 0.4, "auto": 1.0,
}


@dataclass
class SubjectAnalysis:
    subject_type: str = "auto"
    confidence: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    has_face: bool = False
    has_limbs: bool = False
    limb_count: int = 0
    has_wheels: bool = False
    wheel_count: int = 0
    bilateral_symmetry: float = 0.0
    dominant_colors: List[List[int]] = field(default_factory=list)
    hard_surface_score: float = 0.0
    skin_fraction: float = 0.0
    aspect_ratio: float = 1.0
    recommended_rig: str = "none"
    symmetry_plane: str = "x"
    suggested_scale_m: float = 1.0
    style_hints: List[str] = field(default_factory=list)
    pose: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Low level features
# --------------------------------------------------------------------------
def skin_fraction(rgb: np.ndarray) -> float:
    """Fraction of pixels in common skin-tone ranges (YCrCb + HSV)."""
    if rgb.size == 0:
        return 0.0
    if _HAS_CV2:
        ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        m1 = ((ycrcb[..., 1] > 133) & (ycrcb[..., 1] < 183) &
              (ycrcb[..., 2] > 77) & (ycrcb[..., 2] < 145))
        m2 = ((hsv[..., 0] < 25) & (hsv[..., 1] > 30) & (hsv[..., 2] > 60))
        return float(np.logical_and(m1, m2).mean())
    return 0.0  # pragma: no cover


def _silhouette_profile(mask: np.ndarray, bands: int = 16) -> np.ndarray:
    """Normalised width of the silhouette in each horizontal band (top->bottom)."""
    m = mask > 0
    if m.sum() == 0:
        return np.zeros(bands)
    ys = np.nonzero(m.any(axis=1))[0]
    y0, y1 = ys.min(), ys.max()
    profile = []
    for i in range(bands):
        ya = y0 + int((y1 - y0) * i / bands)
        yb = max(ya + 1, y0 + int((y1 - y0) * (i + 1) / bands))
        strip = m[ya:yb]
        widths = strip.sum(axis=1)
        profile.append(float(widths.mean() / max(1, m.shape[1])))
    return np.array(profile)


def _count_legs(mask: np.ndarray, bands: int = 4) -> int:
    """Count separate vertical structures in the lowest band (legs/wheels/feet)."""
    m = (mask > 0)
    if m.sum() == 0:
        return 0
    ys = np.nonzero(m.any(axis=1))[0]
    y0, y1 = ys.min(), ys.max()
    band = m[y1 - max(2, (y1 - y0) // bands): y1 + 1]
    column = band.any(axis=0).astype(np.uint8)
    if column.sum() == 0:
        return 0
    # count runs of True separated by a decent gap
    runs = 0
    in_run = False
    gap = 0
    min_gap = max(2, int(0.02 * m.shape[1]))
    for value in column:
        if value:
            gap = 0
            if not in_run:
                runs += 1
                in_run = True
        else:
            if in_run:
                gap += 1
                if gap >= min_gap:
                    in_run = False
    return int(runs)


def _bilateral_symmetry(mask: np.ndarray) -> float:
    m = (mask > 0).astype(np.float32)
    if m.sum() == 0:
        return 0.0
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    crop = m[y0:y1 + 1, x0:x1 + 1]
    flipped = crop[:, ::-1]
    inter = np.logical_and(crop > 0.5, flipped > 0.5).sum()
    union = np.logical_or(crop > 0.5, flipped > 0.5).sum()
    return float(inter / union) if union else 0.0


def detect_wheels(gray: np.ndarray, mask: Optional[np.ndarray] = None,
                  region: str = "lower") -> Tuple[int, float]:
    """Hough-circle based wheel detection in the lower part of the frame."""
    if not _HAS_CV2:
        return 0, 0.0
    try:
        h, w = gray.shape
        y0 = int(0.45 * h) if region == "lower" else 0
        sub = gray[y0:h]
        if mask is not None:
            subm = (mask[y0:h] > 0).astype(np.uint8) * 255
            sub = cv2.bitwise_and(sub, subm)
        sub = cv2.medianBlur(sub, 5)
        min_r = max(6, int(0.05 * min(sub.shape)))
        max_r = max(min_r + 4, int(0.35 * min(sub.shape)))
        circles = cv2.HoughCircles(sub, cv2.HOUGH_GRADIENT, dp=1.4, minDist=min_r * 1.6,
                                   param1=110, param2=32, minRadius=min_r, maxRadius=max_r)
        if circles is None:
            return 0, 0.0
        found = circles[0]
        # Require left-right pairing for a wheeled-vehicle verdict.
        xs = sorted(float(c[0]) for c in found)
        paired = 0
        for i in range(1, len(xs)):
            if abs(xs[i] - xs[i - 1]) > 0.1 * w:
                paired += 1
        conf = float(np.clip(len(found) / 3.0, 0, 1) * (0.6 + 0.4 * min(1.0, paired / 2.0)))
        return int(len(found)), conf
    except Exception:  # pragma: no cover
        return 0, 0.0


def hard_surface_score(gray: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Fraction of strong, straight edges - high for mechanical/architectural subjects."""
    if not _HAS_CV2:
        return 0.0
    try:
        edges = cv2.Canny(gray, 80, 200)
        if mask is not None:
            edges = cv2.bitwise_and(edges, edges, mask=(mask > 0).astype(np.uint8) * 255)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                                minLineLength=max(20, gray.shape[1] // 12), maxLineGap=6)
        if lines is None:
            return 0.0
        lengths = [math.hypot(float(x2 - x1), float(y2 - y1)) for x1, y1, x2, y2 in lines[:, 0]]
        edge_px = float((edges > 0).sum()) + 1e-6
        return float(np.clip(sum(lengths) / edge_px * 0.35, 0.0, 1.0))
    except Exception:  # pragma: no cover
        return 0.0


def dominant_colors(rgb: np.ndarray, mask: Optional[np.ndarray] = None, k: int = 4) -> List[List[int]]:
    try:
        pixels = rgb.reshape(-1, 3)
        if mask is not None:
            sel = (mask.reshape(-1) > 0)
            if sel.sum() > 32:
                pixels = pixels[sel]
        if len(pixels) > 40000:
            idx = np.random.default_rng(0).choice(len(pixels), 40000, replace=False)
            pixels = pixels[idx]
        if _HAS_CV2:
            data = np.float32(pixels)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
            _compact, labels, centers = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
            counts = np.bincount(labels.reshape(-1), minlength=k)
            order = np.argsort(counts)[::-1]
            return [[int(v) for v in centers[i]] for i in order]
    except Exception:  # pragma: no cover
        pass
    mean = rgb.reshape(-1, 3).mean(axis=0)
    return [[int(v) for v in mean]]


def pose_estimate(mask: np.ndarray, profile: np.ndarray) -> str:
    """Classify the pose very coarsely from the silhouette width profile."""
    if profile.size == 0 or profile.max() <= 0.02:
        return "unknown"
    arms_out = float(profile[3:6].mean())
    torso = float(profile[6:9].mean())
    if arms_out > torso * 1.35:
        return "t_pose_or_arms_out"
    if arms_out > torso * 1.12:
        return "a_pose_or_relaxed"
    return "standing_or_neutral"


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def analyze_subject(images: Sequence[LoadedImage], masks: Sequence[Optional[np.ndarray]],
                    *, write_to: Optional[Path] = None) -> SubjectAnalysis:
    """Classify the subject and measure symmetry/pose/scale evidence."""
    if not images:
        return SubjectAnalysis()

    # Prefer ring views (a front view is the most informative single image).
    order = list(range(min(len(images), 8)))
    profiles = []
    symmetries = []
    aspects = []
    skin_scores = []
    wheel_counts = []
    wheel_confs = []
    hard_scores = []
    leg_counts = []
    face_flags = []

    from .views import detect_face

    for idx in order:
        image = images[idx]
        mask = masks[idx] if idx < len(masks) else None
        if mask is None and image.alpha is not None:
            mask = (image.alpha > 127).astype(np.uint8)
        if mask is None:
            mask = np.ones((image.height, image.width), np.uint8)
        if mask.shape != (image.height, image.width):  # pragma: no cover
            mask = np.ones((image.height, image.width), np.uint8)
        profiles.append(_silhouette_profile(mask))
        symmetries.append(_bilateral_symmetry(mask))
        ys, xs = np.nonzero(mask)
        if len(ys):
            aspects.append(float((ys.max() - ys.min() + 1) / max(1, xs.max() - xs.min() + 1)))
        sel = mask > 0
        pixels = image.array[sel] if sel.any() else image.array.reshape(-1, 3)
        if len(pixels) > 40000:
            pixels = pixels[np.random.default_rng(1).choice(len(pixels), 40000, replace=False)]
        skin_scores.append(skin_fraction(pixels.reshape(-1, 1, 3)))
        counts, conf = detect_wheels(to_gray(image.array), mask)
        wheel_counts.append(counts)
        wheel_confs.append(conf)
        hard_scores.append(hard_surface_score(to_gray(image.array), mask))
        leg_counts.append(_count_legs(mask))
        found, _conf = detect_face(image)
        face_flags.append(found)

    mean_profile = np.mean(profiles, axis=0) if profiles else np.zeros(16)
    symmetry = float(np.mean(symmetries)) if symmetries else 0.0
    aspect = float(np.median(aspects)) if aspects else 1.0
    skin = float(np.mean(skin_scores)) if skin_scores else 0.0
    wheels = int(max(wheel_counts or [0]))
    wheel_conf = float(max(wheel_confs or [0.0]))
    hard = float(np.mean(hard_scores)) if hard_scores else 0.0
    legs = int(np.median(leg_counts)) if leg_counts else 0
    has_face = bool(any(face_flags))

    evidence = {
        "aspect_ratio_h_over_w": round(aspect, 3),
        "bilateral_symmetry": round(symmetry, 3),
        "skin_fraction": round(skin, 4),
        "wheel_like_circles": wheels,
        "wheel_confidence": round(wheel_conf, 3),
        "hard_surface_score": round(hard, 3),
        "leg_like_columns": legs,
        "face_detected": has_face,
        "silhouette_profile": [round(float(v), 3) for v in mean_profile.tolist()],
        "viewed_images": len(profiles),
    }

    # -- rule based classification -------------------------------------
    scores: Dict[str, float] = {k: 0.0 for k in SUBJECT_TYPES if k != "auto"}
    if has_face:
        scores["character_humanoid"] += 4.0
    if skin > 0.06:
        scores["character_humanoid"] += 2.5
        scores["character_creature"] += 0.8
    if aspect > 1.5:  # clearly taller than wide
        scores["character_humanoid"] += 1.2
        scores["character_creature"] += 0.8
        scores["animal"] += 0.4
        scores["building"] += 1.0
    if aspect < 0.8:
        scores["object_generic"] += 0.6
        scores["furniture"] += 0.8
        scores["vehicle_wheeled"] += 0.6
    if legs >= 2 and aspect > 1.3:
        scores["character_humanoid"] += 1.6
        scores["character_creature"] += 0.7
    if legs >= 4 and aspect < 1.4:
        scores["animal"] += 2.0
    if legs >= 4 and wheel_conf > 0.4:
        scores["vehicle_wheeled"] += 1.5
    if wheels >= 2 and wheel_conf > 0.45:
        scores["vehicle_wheeled"] += 3.0
        scores["tool_machine"] += 0.6
    if wheels >= 2 and hard > 0.35:
        scores["robot_mech"] += 0.8
    if hard > 0.45:
        scores["vehicle_wheeled"] += 0.8
        scores["weapon"] += 0.7
        scores["tool_machine"] += 1.0
        scores["building"] += 0.8
        scores["robot_mech"] += 0.9
        scores["object_generic"] += 0.3
    if hard < 0.18 and symmetry > 0.8:
        scores["sculpture"] += 0.6
        scores["character_creature"] += 0.3
    if symmetry > 0.9 and aspect > 1.2:
        scores["character_humanoid"] += 0.7
        scores["furniture"] += 0.4
    if aspect > 2.2:
        scores["weapon"] += 0.8
        scores["building"] += 0.6
    if not scores or max(scores.values()) < 1.0:
        scores["object_generic"] += 1.0

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    total = sum(max(0.0, v) for _k, v in ranked) or 1.0
    best, best_score = ranked[0]
    confidence = float(np.clip(best_score / total * (best_score / 4.0), 0.05, 0.98))

    rig = "none"
    if best in {"character_humanoid", "character_creature"}:
        rig = "humanoid"
    elif best == "animal":
        rig = "quadruped"
    elif best == "robot_mech":
        rig = "humanoid" if aspect > 1.3 else "none"

    style_hints: List[str] = []
    if hard > 0.4:
        style_hints.append("hard_surface")
    if symmetry > 0.85 and hard < 0.3:
        style_hints.append("organic")
    if skin > 0.05:
        style_hints.append("skin_visible")

    analysis = SubjectAnalysis(
        subject_type=best,
        confidence=round(confidence, 3),
        evidence=evidence,
        alternatives=[{"type": k, "score": round(v, 2)} for k, v in ranked[1:4] if v > 0],
        has_face=has_face,
        has_limbs=legs >= 2,
        limb_count=legs,
        has_wheels=bool(wheels >= 2 and wheel_conf > 0.45),
        wheel_count=wheels,
        bilateral_symmetry=round(symmetry, 3),
        dominant_colors=dominant_colors(images[0].array,
                                        masks[0] if masks and masks[0] is not None else None),
        hard_surface_score=round(hard, 3),
        skin_fraction=round(skin, 4),
        aspect_ratio=round(aspect, 3),
        recommended_rig=rig,
        symmetry_plane="x",
        suggested_scale_m=TYPICAL_SCALE_M.get(best, 1.0),
        style_hints=style_hints,
        pose=pose_estimate(masks[0] if masks and masks[0] is not None else
                           np.ones((images[0].height, images[0].width), np.uint8),
                           profiles[0] if profiles else np.zeros(16)),
    )
    if write_to is not None:
        write_json(Path(write_to) / "subject_analysis.json", analysis.to_dict())
    return analysis


def symmetry_report(mesh, *, axis: str = "x", samples: int = 20000) -> Dict[str, Any]:
    """Measure mirror symmetry of a reconstructed mesh (evidence, not a guess)."""
    try:
        import trimesh

        points = mesh.sample(samples) if hasattr(mesh, "sample") else np.asarray(mesh.vertices)
        pts = np.asarray(points, dtype=np.float64)
        axis_index = {"x": 0, "y": 1, "z": 2}.get(axis, 0)
        mirrored = pts.copy()
        mirrored[:, axis_index] *= -1
        # Nearest-neighbour distance from mirrored points to the original cloud.
        from scipy.spatial import cKDTree

        tree = cKDTree(pts)
        dist, _idx = tree.query(mirrored, k=1)
        scale = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        normalised = float(dist.mean() / max(1e-9, scale))
        score = float(np.clip(1.0 - normalised * 8.0, 0.0, 1.0))
        return {
            "axis": axis,
            "mean_mirror_error": round(normalised, 5),
            "symmetry_score": round(score, 3),
            "scale_m": round(scale, 5),
        }
    except Exception as exc:  # pragma: no cover
        return {"axis": axis, "symmetry_score": 0.0, "error": str(exc)}

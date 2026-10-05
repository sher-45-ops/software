"""View detection and camera-ring ordering (spec #4, #5, #42).

Two independent mechanisms cooperate:

* **filename/provenance hints** - ``front.png``, ``3-4_left.jpg`` ... are parsed
  into canonical view labels;
* **silhouette geometry** - for unlabelled images we compute normalised subject
  silhouettes, run classical MDS on the IoU-distance matrix (rotating subjects
  trace a circle in appearance space) and read the ring order off the 2D
  embedding.  Relative angles recovered this way are sufficient for
  reconstruction: the absolute orientation of the ring is a gauge freedom, so a
  mislabelled "front" never degrades the geometry - it only affects reporting.

Everything returns a confidence so that the diagnostics can be honest about
what was measured versus assumed.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...core.project import VIEW_ANGLES, Project
from ...core.store import utc_now, write_json
from ..ingestion.loader import LoadedImage, load_image, to_gray, resize_array

try:
    import cv2

    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False


# --------------------------------------------------------------------------
# Filename hints
# --------------------------------------------------------------------------
_VIEW_PATTERNS: List[Tuple[str, str]] = [
    (r"(^|[^a-z])back[_\-\s]?(left|lt|l)([^a-z]|$)", "back_left"),
    (r"(^|[^a-z])back[_\-\s]?(right|rt|r)([^a-z]|$)", "back_right"),
    (r"(^|[^a-z])(front|fore)[_\-\s]?(left|lt|l)([^a-z]|$)", "front_left"),
    (r"(^|[^a-z])(front|fore)[_\-\s]?(right|rt|r)([^a-z]|$)", "front_right"),
    (r"(^|[^a-z])(three[_\-\s]?quarter|3[_\-\s]?4|3q|34view|threequarter)([^a-z]|$)", "three_quarter"),
    (r"(^|[^a-z])(bottom|under|below|underside)([^a-z]|$)", "bottom"),
    (r"(^|[^a-z])(top|above|overhead|birdseye|plan)([^a-z]|$)", "top"),
    (r"(^|[^a-z])(front|face|forward|fwd)([^a-z]|$)", "front"),
    (r"(^|[^a-z])(back|rear|behind)([^a-z]|$)", "back"),
    (r"(^|[^a-z])(left|side[_\-\s]?l|lt)([^a-z]|$)", "left"),
    (r"(^|[^a-z])(right|side[_\-\s]?r|rt)([^a-z]|$)", "right"),
    (r"(^|[^a-z])(side|profile)([^a-z]|$)", "left"),
    (r"(^|[^a-z])(detail|closeup|close[_\-\s]?up|macro|zoom)([^a-z]|$)", "detail"),
]

_AZIMUTH_TOKENS = {
    "front": 0.0, "front_right": 45.0, "right": 90.0, "back_right": 135.0,
    "back": 180.0, "back_left": 225.0, "left": 270.0, "front_left": 315.0,
    "top": 0.0, "bottom": 0.0, "three_quarter": 35.0, "detail": None, "unknown": None,
}


def view_from_filename(name: str) -> Tuple[str, float]:
    """Parse a filename into ``(view, confidence)``."""
    stem = Path(name).stem.lower().replace("-", "_")
    stem_norm = re.sub(r"[_]+", "_", stem)
    for pattern, view in _VIEW_PATTERNS:
        if re.search(pattern, stem_norm):
            specificity = 0.85 if view in {"front_left", "front_right", "back_left",
                                           "back_right", "three_quarter"} else 0.7
            return view, specificity
    return "unknown", 0.0


def view_angle(view: str) -> Tuple[Optional[float], Optional[float]]:
    return VIEW_ANGLES.get(view, (None, None))


# --------------------------------------------------------------------------
# Silhouette descriptors
# --------------------------------------------------------------------------
def silhouette_descriptor(image: LoadedImage, mask: Optional[np.ndarray], *, size: int = 96,
                          align: bool = True) -> Optional[np.ndarray]:
    """Return a normalised binary silhouette (scale/translation invariant)."""
    h, w = image.height, image.width
    if mask is None:
        if image.alpha is not None:
            mask = (image.alpha > 127).astype(np.uint8)
        else:
            return None
    m = mask.astype(np.uint8)
    if m.size == 0 or m.sum() < 16:
        return None
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    crop = m[y0:y1 + 1, x0:x1 + 1]
    if crop.size == 0:
        return None
    if align:
        # Preserve aspect ratio so tall/wide silhouettes stay distinguishable.
        target_h = size
        target_w = max(8, int(round(size * crop.shape[1] / max(1, crop.shape[0]))))
        target_w = min(target_w, size * 3)
    else:  # pragma: no cover
        target_h = target_w = size
    if _HAS_CV2:
        resized = cv2.resize(crop.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_AREA)
    else:  # pragma: no cover
        idx_y = np.linspace(0, crop.shape[0] - 1, target_h).astype(int)
        idx_x = np.linspace(0, crop.shape[1] - 1, target_w).astype(int)
        resized = crop[np.ix_(idx_y, idx_x)].astype(np.float32)
    canvas = np.zeros((size, max(size, target_w)), np.float32)
    off = max(0, (canvas.shape[1] - target_w) // 2)
    canvas[:, off:off + target_w] = resized
    return (canvas > 0.5).astype(np.float32)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    aa = a[:h, :w] > 0.5
    bb = b[:h, :w] > 0.5
    inter = np.logical_and(aa, bb).sum()
    union = np.logical_or(aa, bb).sum()
    return float(inter / union) if union else 0.0


def distance_matrix(descriptors: Sequence[np.ndarray]) -> np.ndarray:
    n = len(descriptors)
    d = np.zeros((n, n), np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            value = 1.0 - _iou(descriptors[i], descriptors[j])
            d[i, j] = d[j, i] = value
    return d


def classical_mds(distance: np.ndarray, dims: int = 2) -> np.ndarray:
    """Torgerson classical MDS - no sklearn needed."""
    n = distance.shape[0]
    if n < 2:
        return np.zeros((n, dims))
    d2 = distance ** 2
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j.dot(d2).dot(j)
    b = (b + b.T) / 2.0
    vals, vecs = np.linalg.eigh(b)
    order = np.argsort(vals)[::-1][:dims]
    vals = np.clip(vals[order], 0, None)
    coords = vecs[:, order] * np.sqrt(vals)
    if coords.shape[1] < dims:  # pragma: no cover
        coords = np.pad(coords, ((0, 0), (0, dims - coords.shape[1])))
    return coords


def ring_order(distance: np.ndarray) -> Tuple[List[int], float]:
    """Order images around the camera ring; returns ``(order, quality 0-1)``."""
    n = distance.shape[0]
    if n <= 2:
        return list(range(n)), 0.3
    coords = classical_mds(distance, 2)
    # Handle a degenerate embedding (all points coincident).
    if np.allclose(coords, 0):
        return list(range(n)), 0.0
    centre = coords.mean(axis=0)
    angles = np.arctan2(coords[:, 1] - centre[1], coords[:, 0] - centre[0])
    order = list(np.argsort(angles))
    # Quality: how well does a circle explain the distances?  Compare the
    # observed distances with the chord lengths implied by the recovered angles.
    quality = _embedding_circularity(coords, distance, order)
    return order, quality


def _embedding_circularity(coords: np.ndarray, distance: np.ndarray, order: Sequence[int]) -> float:
    n = coords.shape[0]
    radii = np.linalg.norm(coords - coords.mean(axis=0), axis=1)
    if radii.mean() <= 1e-9:
        return 0.0
    radial_consistency = float(1.0 - np.clip(radii.std() / max(1e-9, radii.mean()), 0, 1))
    # adjacent distances should be smaller than the global mean
    adj = []
    for i in range(1, n):
        adj.append(distance[order[i - 1], order[i]])
    if not adj:  # pragma: no cover
        return radial_consistency
    mean_all = float(distance[np.triu_indices(n, 1)].mean())
    separation = float(np.clip(1.0 - (np.mean(adj) / max(1e-9, mean_all)), 0, 1))
    return round(0.6 * radial_consistency + 0.4 * separation, 3)


# --------------------------------------------------------------------------
# Face / front anchoring
# --------------------------------------------------------------------------
def detect_face(image: LoadedImage) -> Tuple[bool, float]:
    """Detect a frontal human face (helps anchor the ring's front direction).

    Uses the OpenCV haar cascade that ships with opencv-python.  Failure to
    detect is not an error - it only lowers the anchoring confidence.
    """
    if not _HAS_CV2:
        return False, 0.0
    try:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(str(cascade_path))
        if cascade.empty():  # pragma: no cover
            return False, 0.0
        gray = to_gray(image.array)
        small_scale = 640.0 / max(gray.shape)
        if small_scale < 1.0:
            gray = cv2.resize(gray, (int(gray.shape[1] * small_scale), int(gray.shape[0] * small_scale)))
        faces = cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5,
                                         minSize=(max(24, gray.shape[0] // 12),) * 2)
        if len(faces) == 0:
            return False, 0.0
        # At least one face should be reasonably central for it to be a front view.
        h, w = gray.shape
        for (x, y, fw, fh) in faces:
            cx, cy = (x + fw / 2) / w, (y + fh / 2) / h
            if 0.25 < cx < 0.75 and 0.05 < cy < 0.7:
                return True, float(min(1.0, len(faces) / 2.0))
        return False, 0.0
    except Exception:  # pragma: no cover
        return False, 0.0


def _looks_like_top_down(descriptor: Optional[np.ndarray], distance_row: np.ndarray) -> bool:
    if descriptor is None:
        return False
    fill = float((descriptor > 0.5).mean())
    if fill < 0.45:
        return False
    finite = distance_row[distance_row > 0]
    if finite.size < 2:
        return False
    # Top/bottom views sit far from everything on the ring.
    return bool(distance_row.mean() > finite.mean() + 1.2 * finite.std())


# --------------------------------------------------------------------------
# Assignment
# --------------------------------------------------------------------------
@dataclass
class ViewAssignment:
    image_id: str
    filename: str
    view: str
    azimuth: Optional[float]
    elevation: float
    confidence: float
    method: str
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def assign_views(project: Project, *, use_masks: bool = True,
                 max_dim: int = 768) -> Dict[str, Any]:
    """Determine the view label + camera azimuth/elevation for every image."""
    from ..ingestion.loader import ImageSet, load_masks

    image_set = ImageSet(project)
    if len(image_set) == 0:
        return {"assignments": [], "warnings": ["no images"], "ring_quality": 0.0}

    loaded_images = [image_set.load(e["id"]).with_max_dim(max_dim) for e in image_set.entries]
    masks = load_masks(project, loaded_images) if use_masks else {}

    assignments: List[ViewAssignment] = []
    warnings: List[str] = []
    descriptors: List[Optional[np.ndarray]] = []
    has_explicit: List[bool] = []

    for entry, image in zip(image_set.entries, loaded_images):
        filename_view, filename_conf = view_from_filename(image.path.name)
        project_hint = entry.get("view", "unknown")
        mask = masks.get(image.path.name)
        if mask is not None and mask.shape != (image.height, image.width):
            mask = None
        desc = silhouette_descriptor(image, mask)
        descriptors.append(desc)

        view = project_hint if project_hint not in (None, "", "unknown") else filename_view
        conf = 0.95 if project_hint not in (None, "", "unknown") else filename_conf
        method = "metadata" if conf >= 0.9 else ("filename" if conf > 0 else "unknown")
        azimuth, elevation = view_angle(view) if view != "unknown" else (None, None)
        if elevation is None:
            elevation = 0.0
        assignments.append(
            ViewAssignment(
                image_id=entry["id"],
                filename=image.path.name,
                view=view,
                azimuth=azimuth,
                elevation=float(elevation or 0.0),
                confidence=round(conf, 3),
                method=method,
            )
        )
        has_explicit.append(conf > 0)

    # -- ring ordering for unlabelled images ---------------------------
    unlabelled = [i for i, ok in enumerate(has_explicit) if not ok and descriptors[i] is not None]
    ring_quality = 0.0
    if len(unlabelled) >= 3:
        descs = [descriptors[i] for i in unlabelled]
        distances = distance_matrix(descs)  # type: ignore[arg-type]
        order, ring_quality = ring_order(distances)
        # Top/bottom outliers are pulled out of the ring before assigning angles.
        outliers = []
        for pos, idx in enumerate(order):
            if _looks_like_top_down(descriptors[unlabelled[idx]], distances[idx]):
                outliers.append(idx)
        ring_members = [i for i in order if i not in outliers]
        n = len(ring_members)
        for pos, idx in enumerate(ring_members):
            azimuth = 360.0 * pos / max(1, n)
            target = assignments[unlabelled[idx]]
            target.azimuth = azimuth
            target.elevation = 0.0
            target.view = _label_from_azimuth(azimuth, len(ring_members))
            target.method = "silhouette_ring"
            target.confidence = round(0.4 + 0.5 * ring_quality, 3)
            target.notes = "angle derived from silhouette ordering (absolute rotation is arbitrary)"
        for idx in outliers:
            target = assignments[unlabelled[idx]]
            target.view = "top"
            target.azimuth = None
            target.elevation = 80.0
            target.method = "silhouette_outlier"
            target.confidence = 0.35
            target.notes = "appears to be a top/bottom view (outlier in the appearance ring)"
        # Anchor the ring using a detected face when possible.
        anchor = _find_face_anchor(loaded_images, unlabelled)
        if anchor is not None:
            anchor_angle = assignments[anchor].azimuth or 0.0
            for idx in unlabelled:
                a = assignments[idx]
                if a.azimuth is not None:
                    a.azimuth = (a.azimuth - anchor_angle) % 360.0
                    a.view = _label_from_azimuth(a.azimuth, max(1, len(ring_members)))
                    a.confidence = min(1.0, a.confidence + 0.25)
                    a.notes += "; anchored by face detection"
            warnings.append("view labels were inferred from silhouette ordering and anchored by face detection")
        else:
            warnings.append(
                "absolute orientation of the camera ring is estimated; "
                "geometry is unaffected (rotation is a gauge freedom)"
            )
    elif len(unlabelled) > 0 and len(has_explicit) >= 3:
        warnings.append("some images have no view label and too few siblings to order them automatically")

    existing_views = [a.view for a in assignments if a.view not in ("unknown", "detail")]
    if len(set(existing_views)) < 3 and len(assignments) >= 3:
        warnings.append("fewer than three distinct view directions were identified; "
                        "consider naming files front/back/left/right")

    report = {
        "assignments": [a.to_dict() for a in assignments],
        "ring_quality": ring_quality,
        "warnings": warnings,
        "assigned_at": utc_now(),
        "labelled_images": sum(1 for a in assignments if a.view not in ("unknown", "detail")),
        "estimated_azimuths": sum(1 for a in assignments if a.method == "silhouette_ring"),
    }
    write_json(project.stage_dir("camera_estimation") / "view_assignment.json", report)
    for a in assignments:
        try:
            project.update_image(a.image_id, detected_view=a.view, azimuth=a.azimuth,
                                 elevation=a.elevation)
        except Exception:  # pragma: no cover
            pass
    return report


def _find_face_anchor(images: Sequence[LoadedImage], indices: Sequence[int]) -> Optional[int]:
    for idx in indices:
        found, conf = detect_face(images[idx])
        if found:
            return idx
    return None


def _label_from_azimuth(azimuth: float, count: int) -> str:
    if count <= 4:
        table = ["front", "right", "back", "left"]
        return table[int(round(azimuth / 90.0)) % 4]
    a = azimuth % 360.0
    for label, (ang, _elev) in VIEW_ANGLES.items():
        if label in {"top", "bottom", "detail", "three_quarter"}:
            continue
        delta = abs(((a - ang + 180) % 360) - 180)
        if delta <= 22.5:
            return label
    return "unknown"


def detect_view_for_file(path: Path) -> Dict[str, Any]:
    """Quick, cheap view probe used at ingestion time."""
    path = Path(path)
    view, conf = view_from_filename(path.name)
    info: Dict[str, Any] = {"view": view, "confidence": conf, "method": "filename"}
    try:
        image = load_image(path, max_dim=512)
        info["width"] = image.width
        info["height"] = image.height
        info["source_size"] = list(image.source_size)
        if view == "unknown":
            orientation = "portrait" if image.height > image.width * 1.05 else (
                "landscape" if image.width > image.height * 1.05 else "square")
            info["orientation"] = orientation
            # A very square, very filled silhouette hints at a top view; that
            # decision is refined later with the full ring ordering.
            if orientation == "square":
                info["view"] = "top"
                info["confidence"] = 0.2
                info["method"] = "aspect_ratio"
    except Exception:  # pragma: no cover
        info.setdefault("width", 0)
        info.setdefault("height", 0)
    if view in VIEW_ANGLES:
        azimuth, elevation = VIEW_ANGLES[view]
        info["azimuth"], info["elevation"] = azimuth, elevation
    return info


def elevation_for_view(view: str) -> float:
    _az, elev = VIEW_ANGLES.get(view, (None, 0.0))
    return float(elev or 0.0)


def angular_gap_degrees(a: Optional[float], b: Optional[float]) -> float:
    if a is None or b is None:
        return 180.0
    return abs(((a - b + 180) % 360) - 180)

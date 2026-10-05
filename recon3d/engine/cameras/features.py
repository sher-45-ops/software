"""Classical feature extraction and cross-view matching.

Used for three things (all optional, all local):

1. **Ring ordering** - the number of geometrically consistent matches between
   two views is a much stronger similarity signal than silhouette IoU, so when
   the images are textured we order the camera ring from a feature-based
   distance matrix.
2. **Camera azimuth refinement** - matched correspondences constrain relative
   rotation, which lets us refine estimated azimuths.
3. **Consistency diagnostics** - a low match count between neighbouring views is
   real evidence that the references disagree (occlusion, mirroring, different
   lighting or a different subject), which the quality report should surface
   instead of silently producing a bad model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..ingestion.loader import LoadedImage, to_gray

try:
    import cv2

    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False


@dataclass
class FeatureSet:
    image: str
    keypoints: np.ndarray  # Nx2 (x, y)
    descriptors: Optional[np.ndarray]  # Nx128 float32
    method: str = "sift"
    scale: float = 1.0  # coordinates are in the (possibly rescaled) image

    @property
    def count(self) -> int:
        return int(len(self.keypoints))


@dataclass
class MatchResult:
    a: str
    b: str
    matches: int
    inliers: int
    inlier_ratio: float
    mean_parallax: float = 0.0
    essential_ok: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "a": self.a, "b": self.b, "matches": self.matches, "inliers": self.inliers,
            "inlier_ratio": round(self.inlier_ratio, 3), "mean_parallax": round(self.mean_parallax, 3),
            "essential_ok": self.essential_ok,
        }


def extract_features(image: LoadedImage, mask: Optional[np.ndarray] = None, *,
                     max_features: int = 3000, max_dim: int = 1200) -> FeatureSet:
    """Extract SIFT (or ORB fallback) keypoints outside the mask's excluded area."""
    if not _HAS_CV2:
        return FeatureSet(image=image.path.name, keypoints=np.zeros((0, 2)), descriptors=None,
                          method="none")
    img = image
    if max(img.size) > max_dim:
        img = image.with_max_dim(max_dim)
    gray = to_gray(img.array)
    detector = None
    method = "sift"
    try:
        detector = cv2.SIFT_create(nfeatures=max_features)
    except Exception:  # pragma: no cover - SIFT unavailable (rare)
        detector = cv2.ORB_create(nfeatures=max_features)
        method = "orb"
    train_mask = None
    if mask is not None:
        m = mask
        if m.shape != gray.shape:
            m = cv2.resize(m.astype(np.uint8), (gray.shape[1], gray.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        # Ignore background features: they would match between unrelated views.
        train_mask = (m > 0).astype(np.uint8) * 255
        if train_mask.sum() < 32 * 255:  # tiny subject - fall back to all pixels
            train_mask = None
    keypoints, descriptors = detector.detectAndCompute(gray, train_mask)
    if keypoints is None or len(keypoints) == 0:
        return FeatureSet(image=image.path.name, keypoints=np.zeros((0, 2)), descriptors=None,
                          method=method, scale=img.scale)
    pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    desc = descriptors.astype(np.float32) if descriptors is not None else None
    return FeatureSet(image=image.path.name, keypoints=pts, descriptors=desc, method=method,
                      scale=img.scale)


def match_features(a: FeatureSet, b: FeatureSet, *, ratio: float = 0.78,
                   ransac_threshold: float = 3.0, max_matches: int = 4000) -> MatchResult:
    """Ratio-test matching plus RANSAC fundamental-matrix inlier counting."""
    if a.descriptors is None or b.descriptors is None or len(a.keypoints) < 8 or len(b.keypoints) < 8:
        return MatchResult(a.image, b.image, 0, 0, 0.0)
    if not _HAS_CV2:
        return MatchResult(a.image, b.image, 0, 0, 0.0)  # pragma: no cover
    norm = cv2.NORM_L2 if a.method == "sift" else cv2.NORM_HAMMING
    try:
        matcher = cv2.BFMatcher(norm)
        raw = matcher.knnMatch(a.descriptors, b.descriptors, k=2)
    except Exception:  # pragma: no cover
        return MatchResult(a.image, b.image, 0, 0, 0.0)
    good = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
        if len(good) >= max_matches:
            break
    if len(good) < 8:
        return MatchResult(a.image, b.image, len(good), 0, 0.0)
    src = np.float32([a.keypoints[m.queryIdx] for m in good]).reshape(-1, 1, 2)
    dst = np.float32([b.keypoints[m.trainIdx] for m in good]).reshape(-1, 1, 2)
    try:
        F, inlier_mask = cv2.findFundamentalMat(src, dst, cv2.FM_RANSAC, ransac_threshold, 0.999, 2000)
        inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    except Exception:  # pragma: no cover
        inliers = 0
    parallax = float(np.abs(src[dst.reshape(-1, 2)[:, 0].argsort()][:, 0, 0] -
                              dst[dst.reshape(-1, 2)[:, 0].argsort()][:, 0, 0]).mean()) if len(src) > 2 else 0.0
    return MatchResult(a.image, b.image, len(good), inliers,
                       inliers / max(1, len(good)), parallax, essential_ok=inliers > 30)


def pairwise_distance_matrix(features: Sequence[FeatureSet], *, ratio: float = 0.78,
                             progress=None) -> Tuple[np.ndarray, List[MatchResult]]:
    """Distance = 1 - normalised inlier count; also returns the raw matches."""
    n = len(features)
    distances = np.ones((n, n), dtype=np.float64)
    results: List[MatchResult] = []
    for i in range(n):
        distances[i, i] = 0.0
        for j in range(i + 1, n):
            r = match_features(features[i], features[j], ratio=ratio)
            results.append(r)
            # Normalise by the smaller feature set so partial occlusion is fair.
            denom = max(1, min(features[i].count, features[j].count))
            score = r.inliers / denom
            distances[i, j] = distances[j, i] = float(np.clip(1.0 - min(1.0, score * 3.0), 0, 1))
            if progress is not None:
                progress()
    return distances, results


def feature_consistency_report(features: Sequence[FeatureSet], matches: Sequence[MatchResult],
                               ring_order: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """Summarise cross-view consistency for the diagnostics report."""
    if not matches:
        return {"pairs": 0, "mean_inlier_ratio": 0.0, "warnings": ["no feature matches computed"]}
    ratios = [m.inlier_ratio for m in matches]
    counts = [m.inliers for m in matches]
    warnings: List[str] = []
    if float(np.mean(counts)) < 25:
        warnings.append(
            "very few geometrically consistent feature matches between views: "
            "textures may be textureless/repetitive, or the views may not show the same subject"
        )
    low = [m for m in matches if m.inliers < 12]
    if len(low) > len(matches) * 0.6:
        warnings.append("most image pairs share almost no features; "
                        "cross-view appearance consistency is low")
    return {
        "pairs": len(matches),
        "mean_inlier_ratio": round(float(np.mean(ratios)), 3),
        "mean_inliers": round(float(np.mean(counts)), 1),
        "max_inliers": int(max(counts)),
        "warnings": warnings,
        "pairs_detail": [m.to_dict() for m in matches[:200]],
    }


def feature_ring_order(distances: np.ndarray) -> Tuple[List[int], float]:
    """Order images around the ring using feature-based distances."""
    from ..analysis.views import ring_order

    return ring_order(distances)

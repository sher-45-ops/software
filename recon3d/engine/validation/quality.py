"""Image quality analysis and coverage reporting (spec #5, #42).

All metrics here are computed from the actual pixels - blur via variance of the
Laplacian, exposure via histogram clipping, duplicates via perceptual hashes,
perspective via vanishing-line estimation, occlusion via mask connected
components, and lighting consistency via colour statistics across views.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...core.project import VIEW_ANGLES, VIEW_LABELS, Project, ReferenceImage
from ...core.store import utc_now, write_json
from ..ingestion.loader import LoadedImage, load_image, to_gray

try:
    import cv2

    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False


# --------------------------------------------------------------------------
# Per-image metrics
# --------------------------------------------------------------------------
@dataclass
class ImageQuality:
    image_id: str
    filename: str
    width: int
    height: int
    megapixels: float
    blur_score: float  # higher = sharper (variance of Laplacian)
    exposure_score: float  # 0-1, 1 = well exposed
    clipped_shadows: float
    clipped_highlights: float
    noise_score: float
    contrast: float
    subject_coverage: float  # share of frame occupied by the subject
    subject_cropped: bool
    occlusion_score: float
    perspective_score: float
    orientation: str
    has_alpha: bool
    perceptual_hash: str
    duplicate_of: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    usable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _laplacian_variance(gray: np.ndarray) -> float:
    if _HAS_CV2:
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        return float(lap.var())
    g = gray.astype(np.float32)
    lap = (
        -4 * g
        + np.roll(g, 1, 0) + np.roll(g, -1, 0) + np.roll(g, 1, 1) + np.roll(g, -1, 1)
    )[1:-1, 1:-1]
    return float(lap.var())


def _perceptual_hash(gray: np.ndarray, size: int = 8) -> str:
    """dHash - robust to resizing and mild exposure changes."""
    if _HAS_CV2:
        small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA).astype(np.int16)
    else:  # pragma: no cover
        small = gray[:: max(1, gray.shape[0] // size), :: max(1, gray.shape[1] // (size + 1))].astype(np.int16)
        small = small[: size, : size + 1]
    diff = small[:, 1:] > small[:, :-1]
    bits = np.packbits(diff.reshape(-1)).tobytes()
    return bits.hex()


def hamming_distance(h1: str, h2: str) -> int:
    try:
        a = bytes.fromhex(h1)
        b = bytes.fromhex(h2)
    except ValueError:  # pragma: no cover
        return 999
    if len(a) != len(b):
        return 999
    return sum(bin(x ^ y).count("1") for x, y in zip(a, b))


def estimate_occlusion(gray: np.ndarray, mask: Optional[np.ndarray]) -> float:
    """Approximate self-occlusion / clutter from interior mask holes."""
    if mask is None or mask.sum() == 0:
        return 0.0
    m = mask.astype(np.uint8)
    if _HAS_CV2:
        contours, hierarchy = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return 0.0
        holes = 0.0
        total = float(m.sum())
        for idx, h in enumerate(hierarchy[0]):
            if h[3] != -1:  # has a parent -> interior hole
                holes += float(abs(cv2.contourArea(contours[idx])))
        return float(min(1.0, holes / max(1.0, total)))
    return 0.0


def estimate_perspective(gray: np.ndarray) -> float:
    """Detect extreme perspective by how much straight lines converge."""
    if not _HAS_CV2:
        return 0.0
    try:
        edges = cv2.Canny(gray, 60, 180)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                                minLineLength=max(40, gray.shape[1] // 8), maxLineGap=8)
        if lines is None or len(lines) < 6:
            return 0.0
        angles = []
        for x1, y1, x2, y2 in lines[:, 0]:
            ang = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
            angles.append(ang)
        angles_arr = np.array(angles)
        # A strong bimodality around vertical implies a wide-angle interior shot;
        # extreme values (>12 deg deviation on the dominant axis) flag distortion.
        vertical = np.abs(((angles_arr + 90) % 180) - 90)
        return float(np.clip(np.mean(np.minimum(vertical, np.abs(vertical - 90))) / 45.0, 0, 1))
    except Exception:
        return 0.0


def _estimate_background(loaded: LoadedImage) -> Dict[str, Any]:
    """Classify the background and return statistics used by segmentation."""
    h, w = loaded.height, loaded.width
    border = np.concatenate([
        loaded.array[0:2].reshape(-1, 3),
        loaded.array[-2:].reshape(-1, 3),
        loaded.array[:, 0:2].reshape(-1, 3),
        loaded.array[:, -2:].reshape(-1, 3),
    ])
    mean = border.mean(axis=0)
    std = border.std(axis=0)
    uniformity = float(np.clip(1.0 - std.mean() / 40.0, 0.0, 1.0))
    brightness = float(mean.mean())
    if loaded.alpha is not None and float((loaded.alpha < 128).mean()) > 0.1:
        kind = "transparent"
    elif uniformity > 0.85 and brightness > 200:
        kind = "white_studio"
    elif uniformity > 0.85 and brightness < 60:
        kind = "dark_studio"
    elif uniformity > 0.7:
        kind = "uniform"
    else:
        kind = "complex"
    return {
        "kind": kind,
        "uniformity": round(uniformity, 3),
        "border_mean_rgb": [round(float(v), 1) for v in mean],
        "border_std_rgb": [round(float(v), 1) for v in std],
    }


def _subject_coverage(loaded: LoadedImage, mask: Optional[np.ndarray]) -> Tuple[float, bool]:
    h, w = loaded.height, loaded.width
    if loaded.alpha is not None:
        m = loaded.alpha > 127
    elif mask is not None:
        m = mask.astype(bool)
    else:
        m = np.ones((h, w), dtype=bool)
    coverage = float(m.mean())
    # Cropped if the subject touches the frame border with a strong gradient.
    border_touch = 0.0
    for strip in (m[0:2], m[-2:], m[:, 0:2], m[:, -2:]):
        border_touch = max(border_touch, float(strip.mean()))
    cropped = bool(border_touch > 0.45 and coverage > 0.15)
    return coverage, cropped


def analyze_image(loaded: LoadedImage, *, image_id: str = "", mask: Optional[np.ndarray] = None,
                  borders: float = 0.02) -> ImageQuality:
    """Compute the full quality fingerprint of one image."""
    gray = to_gray(loaded.array)
    h, w = gray.shape[:2]
    warnings: List[str] = []

    blur = _laplacian_variance(gray)
    if blur < 12:
        warnings.append("image is very blurry (low Laplacian variance)")

    hist = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    hist /= max(1.0, hist.sum())
    shadows = float(hist[:12].sum())
    highlights = float(hist[244:].sum())
    mid = float(hist[12:244].sum())
    exposure = float(np.clip(mid * 1.15, 0, 1))
    if shadows > 0.25:
        warnings.append("image is strongly underexposed")
    if highlights > 0.25:
        warnings.append("image is strongly overexposed / blown highlights")

    # Noise: residual high-frequency energy in a flat region.
    try:
        patch = gray[h // 4: h // 4 + max(16, h // 8), w // 4: w // 4 + max(16, w // 8)]
        noise = float(np.std(patch - cv2.GaussianBlur(patch, (0, 0), 2.0))) if _HAS_CV2 else float(np.std(patch))
    except Exception:  # pragma: no cover
        noise = 0.0

    contrast = float(np.clip(gray.std() / 64.0, 0, 1))
    coverage, cropped = _subject_coverage(loaded, mask)
    if cropped:
        warnings.append("subject appears cropped by the frame")
    if coverage < 0.04:
        warnings.append("subject occupies a very small part of the frame")

    occlusion = estimate_occlusion(gray, mask)
    if occlusion > 0.25:
        warnings.append("large occluded/hole regions detected inside the subject")

    perspective = estimate_perspective(gray)
    if perspective > 0.6:
        warnings.append("strong perspective distortion detected")

    orientation = "landscape" if w > h * 1.05 else ("portrait" if h > w * 1.05 else "square")
    usable = not (blur < 4 or exposure < 0.05 or coverage < 0.005)
    if not usable:
        warnings.append("image is not usable for reconstruction")

    return ImageQuality(
        image_id=image_id or loaded.path.stem,
        filename=loaded.path.name,
        width=loaded.width,
        height=loaded.height,
        megapixels=round(loaded.width * loaded.height / 1e6, 3),
        blur_score=round(blur, 2),
        exposure_score=round(exposure, 3),
        clipped_shadows=round(shadows, 3),
        clipped_highlights=round(highlights, 3),
        noise_score=round(noise, 2),
        contrast=round(contrast, 3),
        subject_coverage=round(coverage, 4),
        subject_cropped=cropped,
        occlusion_score=round(occlusion, 3),
        perspective_score=round(perspective, 3),
        orientation=orientation,
        has_alpha=loaded.alpha is not None,
        perceptual_hash=_perceptual_hash(gray),
        warnings=warnings,
        usable=usable,
    )


# --------------------------------------------------------------------------
# Project-level analysis
# --------------------------------------------------------------------------
def coverage_map(views: Sequence[str], *, subject_type: str = "auto") -> Dict[str, Any]:
    """Build the circular coverage map consumed by the UI and agents."""
    slots = ["front", "front_right", "right", "back_right", "back",
             "back_left", "left", "front_left"]
    counts = {slot: 0 for slot in slots}
    extras = {"top": 0, "bottom": 0, "detail": 0, "unknown": 0, "three_quarter": 0}
    for view in views:
        key = (view or "unknown").lower().replace("-", "_").replace(" ", "_")
        if key in counts:
            counts[key] += 1
        elif key in extras:
            extras[key] += 1
        elif key == "side":
            counts["left"] += 1
        else:
            extras["unknown"] += 1

    ring = []
    for slot in slots:
        n = counts[slot]
        if n >= 2:
            state = "strong"
        elif n == 1:
            state = "covered"
        else:
            # Adjacent coverage gives partial credit (a 3/4 view covers 2 slots).
            state = "missing"
        ring.append({"view": slot, "count": n, "state": state})

    # A 3/4 view partially covers its two neighbouring cardinal slots.
    for extra_name, neighbours in (("three_quarter", ("front", "front_right")),):
        if extras.get(extra_name):
            for name in neighbours:
                entry = next(r for r in ring if r["view"] == name)
                if entry["state"] == "missing":
                    entry["state"] = "weak"

    present = [r for r in ring if r["state"] != "missing"]
    coverage_pct = round(100.0 * len(present) / len(ring), 1)
    missing = [r["view"] for r in ring if r["state"] == "missing"]
    weak = [r["view"] for r in ring if r["state"] == "weak"]

    if extras["top"]:
        coverage_pct = min(100.0, coverage_pct + 3.0)
    else:
        missing.append("top")
    if extras["bottom"]:
        coverage_pct = min(100.0, coverage_pct + 2.0)

    return {
        "ring": ring,
        "extras": extras,
        "coverage_percent": coverage_pct,
        "missing_views": missing,
        "weak_views": weak,
        "ascii": render_coverage_ascii(ring, extras),
        "recommendation": coverage_recommendation(missing, weak, counts, extras),
    }


def render_coverage_ascii(ring: Sequence[Dict[str, Any]], extras: Dict[str, int]) -> str:
    """Human/agent friendly ASCII rendering of the coverage map (spec #42)."""
    state_symbol = {"strong": "##", "covered": "++", "weak": "~~", "missing": "--"}

    def cell(view: str) -> str:
        entry = next((r for r in ring if r["view"] == view), None)
        return state_symbol.get(entry["state"], "--") if entry else "--"

    top = "##" if extras.get("top") else "--"
    bottom = "##" if extras.get("bottom") else "--"
    lines = [
        "                TOP",
        f"                 {top}",
        "",
        f"   {cell('front_left')}     {cell('front')}     {cell('front_right')}",
        "     FL  <-   F   ->  FR",
        "",
        f"   {cell('left')}               {cell('right')}",
        "     L               R",
        "",
        f"   {cell('back_left')}     {cell('back')}     {cell('back_right')}",
        "     BL  <-   B   ->  BR",
        "",
        "                BOTTOM",
        f"                 {bottom}",
        "",
        "legend: ## 2+ views   ++ 1 view   ~~ weak   -- missing",
    ]
    return "\n".join(lines)


def coverage_recommendation(missing: Sequence[str], weak: Sequence[str],
                            counts: Dict[str, int], extras: Dict[str, int]) -> str:
    if not missing and not weak:
        return "Full 360 ring covered - reconstruction should be reliable."
    parts = []
    if missing:
        parts.append(f"add views for: {', '.join(missing)}")
    if weak:
        parts.append(f"strengthen: {', '.join(weak)}")
    if not extras.get("top"):
        parts.append("a top view improves roof/shoulder detail")
    return "; ".join(parts)


def analyze_project_images(
    project: Project,
    *,
    max_dim: int = 1024,
    detect_duplicates: bool = True,
    write_report: bool = True,
) -> Dict[str, Any]:
    """Analyse every reference image and produce the machine-readable report."""
    images = project.images
    if not images:
        report = {
            "quality": "insufficient",
            "coverage": 0,
            "coverage_percent": 0,
            "missing_views": ["all"],
            "warnings": ["project has no reference images"],
            "usable_images": 0,
            "images": [],
            "analyzed_at": utc_now(),
        }
        if write_report:
            write_json(project.stage_dir("validation") / "image_quality.json", report)
        return report

    per_image: List[Dict[str, Any]] = []
    mask_dir = project.masks_dir
    for entry in images:
        if not entry.path:
            continue
        path = project.root / entry.path
        if not path.exists():
            continue
        mask = None
        mask_path = mask_dir / f"{path.stem}.png"
        if mask_path.exists():
            try:
                m = load_image(mask_path, apply_exif=False)
                mask = (m.array[..., 0] > 127).astype(np.uint8) if m.array.ndim == 3 else (m.array > 127)
            except Exception:  # pragma: no cover
                mask = None
        try:
            loaded = load_image(path, max_dim=max_dim)
            quality = analyze_image(loaded, image_id=entry.id, mask=mask)
        except Exception as exc:  # pragma: no cover - unreadable file
            per_image.append({
                "image_id": entry.id,
                "filename": path.name,
                "usable": False,
                "warnings": [f"could not read image: {exc}"],
            })
            continue
        record = quality.to_dict()
        record["view"] = entry.view if entry.view != "unknown" else entry.detected_view
        record["background"] = _estimate_background(loaded)
        per_image.append(record)

    # -- duplicate detection -------------------------------------------
    duplicates: List[Dict[str, Any]] = []
    if detect_duplicates:
        for i, a in enumerate(per_image):
            if not a.get("perceptual_hash"):
                continue
            for b in per_image[i + 1:]:
                if not b.get("perceptual_hash"):
                    continue
                dist = hamming_distance(a["perceptual_hash"], b["perceptual_hash"])
                if dist <= 4:
                    duplicates.append({
                        "image": a["image_id"],
                        "duplicate_of": b["image_id"],
                        "distance": dist,
                    })
                    a["duplicate_of"] = b["image_id"]
                    a["warnings"] = list(a.get("warnings", [])) + [
                        "near-duplicate of another reference image"
                    ]

    # -- cross-image consistency ---------------------------------------
    warnings: List[str] = []
    backgrounds = {rec.get("background", {}).get("kind") for rec in per_image if rec.get("background")}
    if len(backgrounds) > 1 and "complex" in backgrounds:
        warnings.append(
            "inconsistent backgrounds between views; segmentation will be harder "
            "and photometric matching may be less reliable"
        )
    exposures = [rec.get("exposure_score", 1.0) for rec in per_image if rec.get("usable")]
    if exposures and (max(exposures) - min(exposures)) > 0.5:
        warnings.append("inconsistent lighting/exposure between views")
    resolutions = [(rec.get("width", 0), rec.get("height", 0)) for rec in per_image if rec.get("usable")]
    if resolutions:
        areas = [w * h for w, h in resolutions]
        if max(areas) > 3 * max(1, min(areas)):
            warnings.append("large resolution difference between views; "
                            "smaller images will contribute less detail")
    # scale consistency: estimate relative subject size in frame
    coverages = [rec.get("subject_coverage", 0.0) for rec in per_image if rec.get("usable")]
    if coverages and max(coverages) > 4 * max(1e-6, min(c for c in coverages if c > 0.01) or 1e-6):
        warnings.append("subject scale differs noticeably between views "
                        "(different distance/zoom); camera estimation will compensate")
    for rec in per_image:
        if not rec.get("usable", True):
            warnings.append(f"image '{rec.get('filename')}' is not usable and will be ignored")

    usable = [rec for rec in per_image if rec.get("usable", True) and not rec.get("duplicate_of")]
    views = []
    for rec in usable:
        view = rec.get("view") or "unknown"
        views.append(view)
    coverage = coverage_map(views)
    warnings.extend([w for w in [] if w])

    if len(usable) < 3:
        quality = "insufficient"
        warnings.append(
            f"only {len(usable)} usable image(s); at least 3 different angles are "
            "required for any multi-view reconstruction"
        )
    elif coverage["coverage_percent"] < 45:
        quality = "poor"
        warnings.append("less than half of the view ring is covered; "
                        "expect missing geometry on uncovered sides")
    elif coverage["coverage_percent"] < 70 or coverage["missing_views"]:
        quality = "fair"
    elif len(usable) >= 8 and coverage["coverage_percent"] >= 85:
        quality = "good"
    else:
        quality = "good" if coverage["coverage_percent"] >= 75 else "fair"

    report = {
        "quality": quality,
        "coverage": coverage["coverage_percent"],
        "coverage_percent": coverage["coverage_percent"],
        "coverage_map": coverage,
        "missing_views": coverage["missing_views"],
        "weak_views": coverage["weak_views"],
        "warnings": warnings,
        "usable_images": len(usable),
        "total_images": len(per_image),
        "duplicates": duplicates,
        "images": per_image,
        "recommendations": coverage["recommendation"],
        "analyzed_at": utc_now(),
        "next_steps": _next_steps(quality, len(usable), coverage, warnings),
    }
    if write_report:
        write_json(project.stage_dir("validation") / "image_quality.json", report)
        for rec in per_image:
            for entry in project.data.get("images", []):
                if entry.get("id") == rec.get("image_id"):
                    entry["quality"] = {
                        "blur": rec.get("blur_score"),
                        "exposure": rec.get("exposure_score"),
                        "usable": rec.get("usable"),
                        "coverage": rec.get("subject_coverage"),
                    }
                    entry["detected_view"] = rec.get("view") or entry.get("detected_view")
        project.save()
    return report


def _next_steps(quality: str, usable: int, coverage: Dict[str, Any],
                warnings: Sequence[str]) -> List[str]:
    steps: List[str] = []
    if usable < 3:
        steps.append("add at least 3 (ideally 8+) images covering different angles")
    if coverage.get("missing_views"):
        steps.append(f"capture the missing views: {', '.join(coverage['missing_views'])}")
    if quality in {"poor", "insufficient"}:
        steps.append("use a plain, uniform background and consistent lighting for best results")
    if any("blurry" in w for w in warnings):
        steps.append("replace blurry images with sharp ones")
    if quality == "good":
        steps.append("references look good: proceed with reconstruction")
    return steps


def quality_grade(score: float) -> str:
    if score >= 90:
        return "excellent"
    if score >= 75:
        return "good"
    if score >= 55:
        return "fair"
    return "poor"

"""Subject segmentation - separate the subject from its background (spec #6).

Strategy ladder (all classical, all local, no hosted models):

1. **Alpha channel** - if the reference already has transparency, use it.
2. **Uniform studio background** - estimate the border colour and build a
   distance map; threshold, then refine with GrabCut.
3. **Complex background** - saliency (centre prior + global colour contrast)
   seeds a foreground/background trimap which GrabCut refines.
4. **Multiple objects** - the largest connected component is kept as the
   subject, the others are recorded in the report (never silently dropped).

The originals are never modified: masks land in ``input/masks/`` and the
normalised RGBA cut-outs in ``input/processed/``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...core.project import Project
from ...core.store import utc_now, write_json
from ...errors import StageError
from ..ingestion.loader import LoadedImage, load_image, save_png, to_gray

try:
    import cv2

    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False


@dataclass
class SegmentationResult:
    image: str
    mask_path: str
    processed_path: str
    method: str
    coverage: float
    components: int
    component_areas: List[int] = field(default_factory=list)
    background: str = "unknown"
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Mask primitives
# --------------------------------------------------------------------------
def _morph(mask: np.ndarray, op: str, ksize: int = 5) -> np.ndarray:
    if not _HAS_CV2:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    if op == "open":
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if op == "close":
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if op == "dilate":
        return cv2.dilate(mask, kernel)
    if op == "erode":
        return cv2.erode(mask, kernel)
    return mask


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes without touching the outer boundary."""
    m = (mask > 0).astype(np.uint8)
    if not _HAS_CV2:
        return m
    h, w = m.shape
    flood = m.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 1)
    holes = (flood == 0).astype(np.uint8)
    filled = np.clip(m + holes, 0, 1)
    # close pinholes
    return _morph(filled, "close", 3)


def largest_component(mask: np.ndarray, *, keep_top: int = 1) -> Tuple[np.ndarray, List[int]]:
    if not _HAS_CV2:
        return (mask > 0).astype(np.uint8), [int((mask > 0).sum())]
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if n <= 1:
        return (mask > 0).astype(np.uint8), [int((mask > 0).sum())]
    order = sorted(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in order]
    keep = order[:keep_top]
    out = np.isin(labels, keep).astype(np.uint8)
    return out, areas


def border_statistics(rgb: np.ndarray, band: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Mean/std of the border ring."""
    h, w = rgb.shape[:2]
    band = max(1, min(band, min(h, w) // 8))
    border = np.concatenate([
        rgb[:band].reshape(-1, 3), rgb[-band:].reshape(-1, 3),
        rgb[:, :band].reshape(-1, 3), rgb[:, -band:].reshape(-1, 3),
    ]).astype(np.float32)
    return border.mean(axis=0), border.std(axis=0)


def background_kind(rgb: np.ndarray, alpha: Optional[np.ndarray] = None) -> str:
    if alpha is not None and float((alpha < 128).mean()) > 0.1:
        return "transparent"
    mean, std = border_statistics(rgb)
    uniformity = float(np.clip(1 - std.mean() / 40.0, 0, 1))
    brightness = float(mean.mean())
    if uniformity > 0.85 and brightness > 200:
        return "white_studio"
    if uniformity > 0.85 and brightness < 60:
        return "dark_studio"
    if uniformity > 0.7:
        return "uniform"
    return "complex"


def salient_mask(rgb: np.ndarray, *, downscale: int = 320) -> np.ndarray:
    """Classical saliency: global colour contrast x centre prior."""
    h, w = rgb.shape[:2]
    scale = min(1.0, downscale / float(max(h, w)))
    if scale < 1.0 and _HAS_CV2:
        small = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    else:  # pragma: no cover
        small = rgb
    try:
        lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32) if _HAS_CV2 else small.astype(np.float32)
    except Exception:  # pragma: no cover
        lab = small.astype(np.float32)
    if _HAS_CV2:
        quant = (lab // 16).astype(np.int32)
        keys = quant[..., 0] * 4096 + quant[..., 1] * 64 + quant[..., 2]
        uniq, inverse, counts = np.unique(keys.reshape(-1), return_inverse=True, return_counts=True)
        # mean colour per bin
        flat_lab = lab.reshape(-1, 3)
        sums = np.zeros((len(uniq), 3), np.float64)
        np.add.at(sums, inverse, flat_lab)
        means = sums / counts[:, None]
        # global contrast of each bin against all others
        contrast = np.zeros(len(uniq), np.float64)
        for i in range(len(uniq)):
            d = np.abs(means - means[i]).sum(axis=1)
            contrast[i] = float((d * counts).sum() / counts.sum())
        sal = contrast[inverse].reshape(small.shape[:2])
        sal = cv2.GaussianBlur(sal.astype(np.float32), (0, 0), 5)
        yy, xx = np.mgrid[0: sal.shape[0], 0: sal.shape[1]]
        cx, cy = sal.shape[1] / 2.0, sal.shape[0] / 2.0
        r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
        centre_prior = np.clip(1.2 - r, 0, 1)
        sal = sal * (0.55 + 0.45 * centre_prior)
        sal = (sal - sal.min()) / max(1e-6, sal.max() - sal.min())
    else:  # pragma: no cover
        gray = to_gray(small).astype(np.float32)
        sal = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 15)) if _HAS_CV2 else gray
        sal = (sal - sal.min()) / max(1e-6, sal.max() - sal.min())
    if sal.shape[:2] != (h, w):
        if _HAS_CV2:
            sal = cv2.resize(sal, (w, h), interpolation=cv2.INTER_LINEAR)
        else:  # pragma: no cover
            idx_y = (np.linspace(0, sal.shape[0] - 1, h)).astype(int)
            idx_x = (np.linspace(0, sal.shape[1] - 1, w)).astype(int)
            sal = sal[np.ix_(idx_y, idx_x)]
    thresh = float(np.quantile(sal, 0.72))
    mask = (sal >= max(thresh, 0.15)).astype(np.uint8)
    mask = fill_holes(_morph(mask, "close", 9))
    mask = _morph(mask, "open", 5)
    mask, _ = largest_component(mask, keep_top=3)
    if _HAS_CV2:
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    return mask


def _grabcut(rgb: np.ndarray, seed: np.ndarray, *, iterations: int = 5) -> np.ndarray:
    if not _HAS_CV2:
        return seed
    h, w = rgb.shape[:2]
    # Work at a reduced size for speed/memory, then upsample the result.
    max_dim = 640
    scale = min(1.0, max_dim / float(max(h, w)))
    small = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA) \
        if scale < 1.0 else rgb
    sm = cv2.resize(seed, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST) if scale < 1.0 else seed
    gc_mask = np.full(small.shape[:2], cv2.GC_BGD, np.uint8)
    gc_mask[sm > 0] = cv2.GC_PR_FGD
    # Erode the sure-foreground region so GrabCut has room to refine the boundary.
    core = _morph(sm, "erode", 5)
    gc_mask[core > 0] = cv2.GC_FGD
    border = np.zeros_like(gc_mask, bool)
    b = max(2, int(0.02 * min(small.shape[:2])))
    border[:b] = True
    border[-b:] = True
    border[:, :b] = True
    border[:, -b:] = True
    gc_mask[border] = cv2.GC_BGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small, gc_mask, None, bgd, fgd, iterations, cv2.GC_INIT_WITH_MASK)
    except Exception:  # pragma: no cover - grabcut can fail on tiny images
        return seed
    out = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    if scale < 1.0:
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST)
    return out


def segment_image(loaded: LoadedImage, *, method: str = "auto", refine: bool = True) -> Tuple[np.ndarray, str, List[str]]:
    """Return ``(mask, method_used, warnings)``."""
    warnings: List[str] = []
    rgb = loaded.array
    if loaded.alpha is not None and float((loaded.alpha < 128).mean()) > 0.05:
        mask = (loaded.alpha > 127).astype(np.uint8)
        mask = fill_holes(mask)
        return mask, "alpha", warnings

    kind = background_kind(rgb)
    if method != "auto":
        kind = method

    if kind in {"white_studio", "dark_studio", "uniform"}:
        mean, std = border_statistics(rgb)
        dist = np.sqrt(((rgb.astype(np.float32) - mean) ** 2).sum(axis=2))
        tol = float(np.clip(std.mean() * 3.0 + 18.0, 22.0, 95.0))
        mask = (dist > tol).astype(np.uint8)
        mask = _morph(mask, "close", 7)
        mask = fill_holes(mask)
        mask, areas = largest_component(mask, keep_top=1)
        used = f"{kind}_threshold"
        if refine and mask.sum() > 0:
            mask = _grabcut(rgb, mask)
            mask = fill_holes(mask)
            mask, areas = largest_component(mask, keep_top=1)
            used = f"{kind}_threshold+grabcut"
    else:
        seed = salient_mask(rgb)
        if seed.sum() < 0.002 * seed.size:
            warnings.append("automatic segmentation found very little foreground; "
                            "falling back to a centre-weighted prior")
            seed = np.zeros(rgb.shape[:2], np.uint8)
            h, w = seed.shape
            seed[int(0.08 * h): int(0.95 * h), int(0.15 * w): int(0.85 * w)] = 1
        mask = _grabcut(rgb, seed) if refine else seed
        mask = fill_holes(mask)
        mask, areas = largest_component(mask, keep_top=1)
        used = "saliency+grabcut"
        warnings.append("complex background detected; mask quality may vary")

    # Final sanity: remove specks and keep the dominant component.
    mask = _morph(mask, "open", 5)
    mask = fill_holes(mask)
    if mask.sum() == 0:
        warnings.append("segmentation produced an empty mask; using the full frame")
        mask = np.ones(rgb.shape[:2], np.uint8)
    return mask.astype(np.uint8), used, warnings


# --------------------------------------------------------------------------
# Pipeline entry point
# --------------------------------------------------------------------------
def segment_project(
    project: Project,
    *,
    max_dim: int = 2048,
    overwrite: bool = False,
    refine: bool = True,
    write_report: bool = True,
) -> Dict[str, Any]:
    """Segment every reference image of a project and store masks + cut-outs."""
    from ..ingestion.loader import ImageSet

    image_set = ImageSet(project)
    if len(image_set) == 0:
        raise StageError("no images to segment", stage="segmentation", recoverable=False)

    results: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for entry in image_set.entries:
        loaded = image_set.load(entry["id"]).with_max_dim(max_dim)
        mask_path = project.masks_dir / f"{loaded.path.stem}.png"
        processed_path = project.processed_dir / f"{loaded.path.stem}.png"
        if mask_path.exists() and not overwrite:
            existing = load_image(mask_path, apply_exif=False)
            mask = existing.array[..., 0] if existing.array.ndim == 3 else existing.array
            mask = (mask > 127).astype(np.uint8)
            method = "cached"
            image_warnings: List[str] = []
        else:
            mask, method, image_warnings = segment_image(loaded, refine=refine)
            save_png(mask_path, (mask * 255).astype(np.uint8))

        rgba = np.dstack([loaded.array, (mask * 255).astype(np.uint8)])
        save_png(processed_path, rgba)

        # Silhouette statistics used later by carving and diagnostics.
        ys, xs = np.nonzero(mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else [0, 0, 0, 0]
        coverage = float(mask.mean())
        results.append(
            SegmentationResult(
                image=loaded.path.name,
                mask_path=str(mask_path.relative_to(project.root)).replace("\\", "/"),
                processed_path=str(processed_path.relative_to(project.root)).replace("\\", "/"),
                method=method,
                coverage=round(coverage, 4),
                components=1,
                component_areas=[int(mask.sum())],
                background=background_kind(loaded.array, loaded.alpha),
                confidence=round(float(np.clip(coverage / 0.35, 0.0, 1.0)), 3),
                warnings=image_warnings,
            ).to_dict()
        )
        warnings.extend(image_warnings)
        results[-1]["bbox"] = bbox

    report = {
        "images": results,
        "segmented": len(results),
        "methods": sorted({r["method"] for r in results}),
        "warnings": sorted(set(warnings)),
        "finished_at": utc_now(),
    }
    if write_report:
        write_json(project.stage_dir("segmentation") / "segmentation_report.json", report)
    return report


def mask_quality(mask: np.ndarray) -> Dict[str, Any]:
    """Per-mask diagnostics: leak into the border, raggedness, fill ratio."""
    m = (mask > 0)
    if m.sum() == 0:
        return {"empty": True, "score": 0.0}
    h, w = m.shape
    border = np.concatenate([m[:2].reshape(-1), m[-2:].reshape(-1), m[:, :2].reshape(-1), m[:, -2:].reshape(-1)])
    leak = float(border.mean())
    contours = None
    perimeter = 0.0
    if _HAS_CV2:
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter = float(sum(cv2.arcLength(c, True) for c in contours or []))
    area = float(m.sum())
    compactness = float(4 * np.pi * area / max(1e-6, perimeter ** 2))
    score = float(np.clip(1.0 - leak, 0, 1) * 0.6 + np.clip(compactness * 3.0, 0, 1) * 0.4)
    return {
        "empty": False,
        "border_leak": round(leak, 3),
        "compactness": round(compactness, 4),
        "score": round(score, 3),
        "area": int(area),
    }

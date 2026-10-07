"""Reference-vs-reconstruction comparison, scoring and refinement (spec #24, #48).

The comparison is the engine's own feedback loop:

1. render the reconstructed asset from every solved camera,
2. compare each render with its reference image (silhouette IoU, colour error),
3. aggregate into the machine-readable quality block an agent reads,
4. when the mismatch is systematic, correct it: a global similarity
   (scale + translation) search that maximises multi-view silhouette agreement,
   followed by an optional re-carve with the corrected parameters.

Steps 1-3 are always run; step 4 is bounded by the requested quality preset, so
"draft" still reports honest numbers without spending the refinement budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...core.store import utc_now, write_json
from ..compare.raster import Camera, rasterize, silhouette_iou, image_rmse


@dataclass
class ViewComparison:
    filename: str
    view: str
    iou: float
    color_rmse: float
    rendered_coverage: float
    reference_coverage: float
    coverage_delta: float
    missing_pixels: int
    extra_pixels: int
    masks_used: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "view": self.view,
            "silhouette_iou": round(self.iou, 4),
            "color_rmse": round(self.color_rmse, 4),
            "rendered_coverage": round(self.rendered_coverage, 4),
            "reference_coverage": round(self.reference_coverage, 4),
            "coverage_delta": round(self.coverage_delta, 4),
            "missing_pixels": int(self.missing_pixels),
            "extra_pixels": int(self.extra_pixels),
        }


def render_reference_views(
    rig,
    mesh,
    *,
    textured: bool = True,
    light_dir: Sequence[float] = (0.45, -0.65, -0.62),
    ambient: float = 0.35,
    background: Sequence[int] = (245, 245, 245),
    supersample: int = 2,
) -> Dict[str, np.ndarray]:
    """Render the mesh from every solved camera; returns ``{filename: rgb}``."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    uvs = None
    texture = None
    if textured:
        visual = getattr(mesh, "visual", None)
        uv = getattr(visual, "uv", None) if visual is not None else None
        if uv is not None:
            uvs = np.asarray(uv, dtype=np.float64)
            material = getattr(visual, "material", None)
            image = None
            if material is not None:
                image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
            if image is not None:
                try:
                    texture = np.asarray(image.convert("RGB"), dtype=np.uint8)
                except Exception:  # pragma: no cover
                    texture = None
            if texture is None:
                uvs = None
    renders: Dict[str, np.ndarray] = {}
    for pose in rig.cameras:
        result = rasterize(verts, faces, pose.camera, texture=texture, uvs=uvs,
                           light_dir=light_dir, ambient=ambient, background=background,
                           supersample=supersample, cull_backfaces=False)
        renders[pose.filename] = result.color
    return renders


def compare_to_references(
    rig,
    mesh,
    images: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    *,
    renders: Optional[Dict[str, np.ndarray]] = None,
    supersample: int = 2,
) -> Dict[str, Any]:
    """Compare renders with references and aggregate the diagnostics."""
    if renders is None:
        renders = render_reference_views(rig, mesh, supersample=supersample)

    per_view: List[ViewComparison] = []
    for pose in rig.cameras:
        render = renders.get(pose.filename)
        reference = images.get(pose.filename)
        if render is None or reference is None:
            continue
        mask = masks.get(pose.filename)
        if mask is None:
            mask_arr = np.ones(reference.shape[:2], dtype=bool)
        else:
            mask_arr = mask > 0
        render_mask = render.mean(axis=2) < 250  # background assumed light
        if render.shape[:2] != mask_arr.shape:
            from ..ingestion.loader import resize_mask_to

            mask_arr = resize_mask_to(mask_arr.astype(np.uint8), render.shape[1], render.shape[0]) > 0
        iou = silhouette_iou(render_mask, mask_arr)
        # Colour error only where both agree the surface exists.
        both = render_mask & mask_arr
        rmse = image_rmse(render, reference, both) if both.any() else 1.0
        missing = int(np.logical_and(mask_arr, ~render_mask).sum())
        extra = int(np.logical_and(render_mask, ~mask_arr).sum())
        per_view.append(ViewComparison(
            filename=pose.filename, view=pose.view, iou=iou, color_rmse=rmse,
            rendered_coverage=float(render_mask.mean()), reference_coverage=float(mask_arr.mean()),
            coverage_delta=float(render_mask.mean() - mask_arr.mean()),
            missing_pixels=missing, extra_pixels=extra,
        ))

    if not per_view:
        return {"views": [], "mean_iou": 0.0, "warnings": ["no comparable views"]}

    ious = np.array([v.iou for v in per_view])
    rmses = np.array([v.color_rmse for v in per_view])
    deltas = np.array([v.coverage_delta for v in per_view])

    missing_regions: List[str] = []
    warnings: List[str] = []
    for v in per_view:
        if v.iou < 0.55:
            missing_regions.append(f"{v.view} view ({v.filename})")
        elif v.coverage_delta < -0.02:
            missing_regions.append(f"{v.view} view is thinner than the reference")
    if len(missing_regions) > len(per_view) * 0.5:
        warnings.append(
            "the reconstruction disagrees with more than half of the reference views; "
            "this usually means inconsistent references (different subject, mirrored "
            "images) or a failed camera solve"
        )
    if float(np.mean(rmses)) > 0.28:
        warnings.append("high colour error versus the references; texture projection may be "
                        "misaligned or the references have very different lighting")
    if float(np.mean(deltas)) < -0.03:
        warnings.append("the reconstruction is systematically smaller than the references "
                        "(silhouette under-coverage)")
    if float(np.mean(deltas)) > 0.03:
        warnings.append("the reconstruction is systematically larger than the references "
                        "(visual hull bloat - concavities cannot be represented)")

    return {
        "views": [v.to_dict() for v in per_view],
        "mean_iou": round(float(ious.mean()), 4),
        "min_iou": round(float(ious.min()), 4),
        "mean_color_rmse": round(float(rmses.mean()), 4),
        "mean_coverage_delta": round(float(deltas.mean()), 4),
        "weak_views": [v.filename for v in per_view if v.iou < 0.6],
        "missing_regions": sorted(set(missing_regions)),
        "warnings": warnings,
        "compared_at": utc_now(),
    }


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def score_quality(
    *,
    comparison: Dict[str, Any],
    mesh_stats: Dict[str, Any],
    uv_metrics: Optional[Dict[str, Any]] = None,
    texture_stats: Optional[Dict[str, Any]] = None,
    image_report: Optional[Dict[str, Any]] = None,
    camera_quality: Optional[Dict[str, Any]] = None,
    inferred: Optional[Sequence[Dict[str, Any]]] = None,
    budgets: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build the agent-facing quality block (spec #48).

    ``inferred`` is the honesty channel: every surface or texture region the
    engine *filled in* (symmetry completion, texture diffusion) rather than
    observed is listed here with its confidence **and** echoed into
    ``missing_regions``, so a caller that only reads that list still sees it.
    ``budgets`` carries the measured target-vs-actual result of the preset's
    named asset budgets (e.g. ``game_ready``'s mobile budget).
    """
    mean_iou = float(comparison.get("mean_iou", 0.0))
    reference_similarity = float(np.clip(mean_iou * 100.0 * 1.05, 0.0, 100.0))

    health = mesh_stats.get("health_score")
    if health is None:
        from ..geometry.cleanup import mesh_health_score

        health = mesh_health_score(mesh_stats)

    geometry_quality = 100.0
    geometry_quality -= (1.0 - min(1.0, mean_iou / 0.9)) * 55.0
    geometry_quality -= min(15.0, 100.0 * (mesh_stats.get("degenerate_faces", 0) /
                                           max(1, mesh_stats.get("faces", 1))))
    components = mesh_stats.get("components", 1) or 1
    if components > 1:
        geometry_quality -= min(10.0, (components - 1) * 2.5)
    if not mesh_stats.get("watertight", False):
        geometry_quality -= 4.0
    cam_quality = camera_quality or {}
    if cam_quality.get("mean_reprojection_error", 0.0) > 0.02:
        geometry_quality -= 6.0
    geometry_quality = float(np.clip(geometry_quality, 0.0, 100.0))

    texture_quality = 60.0
    if texture_stats:
        coverage = float(texture_stats.get("coverage_ratio", 0.0))
        texture_quality = 30.0 + 65.0 * coverage
        if texture_stats.get("resolution", 0) >= 4096:
            texture_quality += 5.0
    elif uv_metrics is None:
        texture_quality = 0.0
    texture_quality = float(np.clip(texture_quality, 0.0, 100.0))

    missing: List[str] = list(comparison.get("missing_regions", []))
    if image_report:
        missing.extend(image_report.get("missing_views", []) or [])
        missing.append("bottom of object" if "bottom" in (image_report.get("missing_views") or [])
                       else "")

    inferred_regions: List[Dict[str, Any]] = []
    for entry in (inferred or []):
        if not isinstance(entry, dict):
            continue
        item = dict(entry)
        try:
            item["confidence"] = round(float(item.get("confidence", 0.0)), 3)
        except (TypeError, ValueError):
            item["confidence"] = 0.0
        item.setdefault("kind", "inferred")
        inferred_regions.append(item)
        missing.append(f"inferred: {item.get('region', 'unnamed region')} "
                       f"(confidence {item['confidence']:.2f}, {item['kind']})")

    warnings: List[str] = []
    warnings.extend(comparison.get("warnings", []) or [])
    warnings.extend((image_report or {}).get("warnings", []) or [])
    warnings.extend(cam_quality.get("warnings", []) or [])
    if inferred_regions:
        kinds = sorted({str(e.get("kind")) for e in inferred_regions})
        warnings.append(
            f"{len(inferred_regions)} region(s) were *filled in* rather than observed "
            f"({', '.join(kinds)}); each is listed with a confidence under 'inferred' "
            "and cannot be verified from the supplied references"
        )
    warnings = [w for w in dict.fromkeys(warnings) if w]

    overall = float(np.clip(0.45 * geometry_quality + 0.30 * reference_similarity +
                            0.15 * health + 0.10 * texture_quality, 0.0, 100.0))
    grade = ("excellent" if overall >= 88 else "good" if overall >= 74 else
             "fair" if overall >= 58 else "poor")

    budget_report = [dict(b) for b in (budgets or []) if isinstance(b, dict)]
    return {
        "overall": round(overall, 1),
        "grade": grade,
        "geometry_quality": round(geometry_quality, 1),
        "reference_similarity": round(reference_similarity, 1),
        "mesh_health": round(float(health), 1),
        "texture_quality": round(texture_quality, 1),
        "missing_regions": sorted(set(m for m in missing if m)),
        "inferred": inferred_regions,
        "inferred_summary": {
            "regions": len(inferred_regions),
            "kinds": sorted({str(e.get("kind")) for e in inferred_regions}),
            "mean_confidence": (round(float(np.mean([e["confidence"] for e in inferred_regions])), 3)
                                if inferred_regions else None),
        },
        "budgets": budget_report,
        "warnings": warnings,
        "metrics": {
            "mean_silhouette_iou": comparison.get("mean_iou"),
            "mean_color_rmse": comparison.get("mean_color_rmse"),
            "mean_coverage_delta": comparison.get("mean_coverage_delta"),
            "views_compared": len(comparison.get("views", []) or []),
        },
        "scored_at": utc_now(),
    }


# --------------------------------------------------------------------------
# Refinement: global similarity correction + optional re-carve
# --------------------------------------------------------------------------
def refine_against_references(
    rig,
    mesh,
    images: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    *,
    iterations: int = 2,
    resolution: int = 160,
    reporter: Any = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Search a global scale/offset that best explains every reference silhouette.

    The visual hull cannot represent concavities, so its silhouette is always a
    superset of the reference - but the *scale* and *centring* can still be off,
    and this pass fixes exactly that, measuring the improvement objectively.
    """
    from ..ingestion.loader import resize_mask_to

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    centre = np.asarray(rig.center, dtype=np.float64)
    scale = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))

    # Targets at a fixed working resolution for a stable, fast objective.
    targets: Dict[str, np.ndarray] = {}
    cameras: Dict[str, Camera] = {}
    for pose in rig.cameras:
        mask = masks.get(pose.filename)
        if mask is None:
            continue
        aspect = pose.width / max(1, pose.height)
        h = int(resolution)
        w = max(48, int(round(h * aspect)))
        targets[pose.filename] = resize_mask_to(mask.astype(np.uint8), w, h) > 0
        cameras[pose.filename] = pose.camera

    def objective(scale_f: float, offset: np.ndarray) -> float:
        transformed = (verts - centre) * scale_f + centre + offset
        total = 0.0
        count = 0
        for name, target in targets.items():
            pose = next(p for p in rig.cameras if p.filename == name)
            h, w = target.shape
            cam = Camera(R=pose.camera.R, t=pose.camera.t,
                         fx=pose.camera.fx * (w / max(1, pose.width)),
                         fy=pose.camera.fy * (h / max(1, pose.height)),
                         cx=(pose.camera.cx + 0.5) * (w / max(1, pose.width)) - 0.5,
                         cy=(pose.camera.cy + 0.5) * (h / max(1, pose.height)) - 0.5,
                         width=w, height=h)
            result = rasterize(transformed, faces, cam, silhouette_only=True)
            total += silhouette_iou(result.mask, target)
            count += 1
        return total / max(1, count)

    best_scale = 1.0
    best_offset = np.zeros(3)
    best = objective(best_scale, best_offset)
    baseline = best

    scale_steps = [0.04, -0.04, 0.08, -0.08, 0.15, -0.15]
    offset_steps = [0.03 * scale, -0.03 * scale]
    history = [{"scale": 1.0, "iou": round(best, 4)}]

    for _ in range(max(1, iterations)):
        moved = False
        for step in scale_steps:
            trial = best_scale + step
            if trial <= 0.2 or trial >= 3.0:
                continue
            value = objective(trial, best_offset)
            if value > best + 1e-4:
                best, best_scale, moved = value, trial, True
        for axis in range(3):
            for step in offset_steps:
                trial = best_offset.copy()
                trial[axis] += step
                value = objective(best_scale, trial)
                if value > best + 1e-4:
                    best, best_offset, moved = value, trial, True
        history.append({"scale": round(best_scale, 4),
                        "offset": [round(v, 5) for v in best_offset],
                        "iou": round(best, 4)})
        scale_steps = [s * 0.5 for s in scale_steps]
        offset_steps = [s * 0.5 for s in offset_steps]
        if not moved:
            break
        if reporter is not None:
            reporter.update(60, f"alignment pass: IoU {best:.3f} at scale {best_scale:.3f}")

    report = {
        "baseline_iou": round(baseline, 4),
        "refined_iou": round(best, 4),
        "scale": round(best_scale, 5),
        "offset": [round(float(v), 6) for v in best_offset],
        "improvement": round(best - baseline, 4),
        "history": history,
        "applied": bool(abs(best_scale - 1.0) > 1e-3 or np.linalg.norm(best_offset) > 1e-6),
    }
    if report["applied"] and best > baseline + 1e-3:
        out = mesh.copy()
        out.vertices = (np.asarray(out.vertices) - centre) * best_scale + centre + best_offset
        return out, report
    report["applied"] = False
    return mesh, report


def write_comparison(directory: Path, comparison: Dict[str, Any],
                     quality: Optional[Dict[str, Any]] = None,
                     refinement: Optional[Dict[str, Any]] = None) -> str:
    payload = dict(comparison)
    if quality:
        payload["quality"] = quality
    if refinement:
        payload["refinement"] = refinement
    path = Path(directory) / "reference_comparison.json"
    write_json(path, payload)
    return str(path)

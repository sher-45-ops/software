"""The reconstruction pipeline (spec #3, #12, #31).

One orchestrator drives every stage, so the CLI, the REST API, the MCP server
and the studio UI all execute exactly the same code path.  Key properties:

* **Stage graph with dependencies.**  Stages declare what they need, so a
  request that only changes textures re-uses the cached mesh (spec #45).
* **Checkpoints.**  Every completed stage writes a checkpoint keyed by a hash of
  its inputs; re-running with the same inputs re-uses the work (spec #31).
* **Quality presets change the pipeline**, not just numbers: voxel resolution,
  carve levels, photo-consistency, refinement passes, texture size and LOD count
  all come from the preset budget (spec #12).
* **Failure isolation.**  Optional stages (rigging, AO, previews) degrade
  gracefully: their failure is recorded and the job still produces a usable
  asset (spec #31).
"""

from __future__ import annotations

import gc
import json
import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..engine.geometry.cleanup import cleanup_mesh, mesh_health_score, mesh_statistics
from ..engine.reconstruction.dataset import silhouette_iou_report
from ..engine.reconstruction.depth import (depth_statistics_consistency, fuse_depth_maps,
                                           save_point_cloud, statistical_outlier_removal)
from ..engine.reconstruction.hull import (HullResult, fuse_depth_points, hull_to_mesh,
                                          reconstruct_hull, voxel_surface_points)
from ..engine.analysis.classify import analyze_subject, symmetry_report
from ..engine.cameras.features import (extract_features, feature_consistency_report,
                                       pairwise_distance_matrix)
from ..engine.cameras.rig import (calibrate_lens_and_rig, calibrate_rig_height,
                                  estimate_rig, load_rig, refine_rig_silhouette)
from ..engine.compare.quality import (compare_to_references, refine_against_references,
                                      render_reference_views, score_quality, write_comparison)
from ..engine.compare.raster import Camera, rasterize
from ..engine.export.writers import export_asset, verify_export
from ..engine.ingestion.loader import ImageSet, load_masks, resize_mask_to
from ..engine.materials.library import (apply_material_request, estimate_materials,
                                        write_materials)
from ..engine.optimization.lod import build_lod_chain, optimize_for_realtime, polygon_budget
from ..engine.rigging.skeleton import generate_rig, write_rig
from ..engine.segmentation.subject import segment_project
from ..engine.textures.project import (derive_pbr_maps, pack_orm, project_texture,
                                       save_texture_maps)
from ..engine.textures.uv import unwrap_mesh
from ..engine.validation.assets import validate_version
from ..engine.validation.quality import analyze_project_images
from .jobs import (Job, checkpoint_valid, invalidate_stage, read_checkpoint,
                   write_checkpoint)
from .project import AssetFiles, Project, Version, hash_inputs
from .progress import ProgressReporter, STAGES, overall_progress
from .store import merge_dicts, read_json, utc_now, write_json
from .resources import mode_budget, profile_hardware, voxel_grid_budget
from ..errors import (CancelledError, ConfigError, ConflictError, InsufficientDataError,
                      NotFoundError, SecurityError, StageError, ValidationError)


#: Errors that a retry cannot fix: they describe the *request* or the sandbox,
#: not a transient hardware/IO condition.  Everything else is retried (bounded by
#: ``stage_retries``), because a local pipeline can fail on a busy disk or a
#: momentary allocation failure and the same call often succeeds straight away.
NON_RETRYABLE_ERRORS = (CancelledError, ValidationError, SecurityError, ConfigError,
                        NotFoundError, ConflictError, InsufficientDataError)


def _may_retry(error: BaseException) -> bool:
    """False for failures a retry cannot fix (bad input, sandbox, cancellation)."""
    if isinstance(error, NON_RETRYABLE_ERRORS):
        return False
    cause = error.__cause__
    return not isinstance(cause, NON_RETRYABLE_ERRORS)


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
DEFAULT_PARAMS: Dict[str, Any] = {
    "quality": "standard",          # draft | standard | high | ultra | game-ready | cinematic
    "preset": "",                   # explicit preset name (overrides quality)
    "style": "realistic",
    "geometry": "auto",             # low | medium | high | auto
    "texture_resolution": 0,        # 0 -> from the quality preset
    "target_polycount": "auto",
    "preserve_sharp_edges": True,
    "symmetry": "auto",             # auto | on | off | x | y | z
    "generate_uvs": True,
    "generate_pbr": True,
    "generate_rig": False,
    "generate_lods": True,
    "lod_levels": 0,                # 0 -> from preset
    "generate_previews": True,
    "generate_pointcloud": True,
    "photo_consistency": True,
    "material_request": "",
    "material": "",
    "subject_type": "auto",
    "refine_passes": -1,            # -1 -> from preset
    "compare_renders": 0,           # 0 -> from preset
    "export_formats": ["glb", "obj"],
    "units": "normalized",          # normalized | meters | centimeters
    "subject_height_m": 0.0,        # 0 -> from subject analysis
    "seed": 0,
    "backend": "auto",
    "stage_retries": 1,             # extra attempts for a stage that fails non-deterministically
    "force_stages": [],             # stage names to re-run even if cached
    "skip_stages": [],              # stage names to skip entirely
    "label": "",
    "notes": "",
}


#: Which preset corresponds to each user-facing ``quality`` value.
QUALITY_TO_PRESET = {
    "draft": "draft",
    "performance": "draft",
    "standard": "standard",
    "balanced": "standard",
    "high": "high",
    "quality": "high",
    "ultra": "ultra",
    "maximum": "ultra",
    "game-ready": "game_ready",
    "game_ready": "game_ready",
    "gameready": "game_ready",
    "cinematic": "cinematic",
    "photoreal": "cinematic",
}


def load_preset(name: str) -> Dict[str, Any]:
    """Load a preset from ``recon3d/presets`` (falls back to built-ins)."""
    resolved = QUALITY_TO_PRESET.get(str(name).lower().replace(" ", "-"), str(name).lower())
    path = Path(__file__).resolve().parent.parent / "presets" / f"{resolved}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:  # pragma: no cover
            pass
    performance_key = {"draft": "draft", "standard": "balanced", "high": "quality",
                       "ultra": "maximum", "game_ready": "balanced",
                       "cinematic": "maximum"}.get(resolved, "balanced")
    return {"name": resolved, "performance_mode": performance_key,
            "params": {}, "_builtin": True}


def resolve_params(raw: Optional[Dict[str, Any]] = None, *,
                   project_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Merge defaults + project params + request params + preset into a plan."""
    from .store import merge_dicts

    params = dict(DEFAULT_PARAMS)
    params = merge_dicts(params, project_params or {})
    params = merge_dicts(params, raw or {})
    preset_name = params.get("preset") or params.get("quality") or "standard"
    preset = load_preset(str(preset_name))
    params["_preset"] = preset.get("name", str(preset_name))
    params["_performance_mode"] = preset.get("performance_mode", "balanced")
    preset_params = dict(preset.get("params") or {})
    # Explicit user values win over the preset; preset fills "auto"/0 entries.
    for key, value in preset_params.items():
        current = params.get(key)
        if current in (None, 0, "", "auto", -1) or key not in params:
            params[key] = value
    for key, value in preset_params.items():
        policy = (preset.get("policy") or {}).get(key)
        if policy in {"force", "clamp"}:
            params[key] = value
    return params


# --------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------
@dataclass
class StageSpec:
    name: str
    fn: Callable[["PipelineContext"], Dict[str, Any]]
    requires: Tuple[str, ...] = ()
    optional: bool = False
    weight: float = 1.0
    #: Stages whose outputs are named after the version (the final deliverables).
    #: They are cheap to write and must describe *this* version, so a resumed run
    #: regenerates them instead of copying the previous version's files.
    always_run: bool = False


@dataclass
class PipelineContext:
    project: Project
    params: Dict[str, Any]
    version: Version
    budget: Dict[str, Any]
    reporter: ProgressReporter
    profile: Any
    state: Dict[str, Any] = field(default_factory=dict)
    stage_reports: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    cache: Dict[str, Any] = field(default_factory=dict)
    force: bool = False
    #: Name of the stage currently executing - lets helpers such as
    #: :func:`_persist_mesh` write into the right ``intermediate/<stage>/`` folder.
    current_stage: str = ""

    # -- convenience accessors -----------------------------------------
    @property
    def version_dir(self) -> Path:
        return self.project.version_path(self.version.id)

    def dir_for(self, kind: str) -> Path:
        path = self.version_dir / kind
        path.mkdir(parents=True, exist_ok=True)
        return path

    def request(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    def images(self) -> List[Any]:
        if "images" not in self.cache:
            image_set = ImageSet(self.project, max_dim=self.budget.get("dense_max_dim", 1024))
            self.cache["images"] = [image_set.load(e["id"]) for e in image_set.entries]
        return self.cache["images"]

    def image_arrays(self) -> Dict[str, np.ndarray]:
        return {img.path.name: img.array for img in self.images()}

    def masks(self) -> Dict[str, np.ndarray]:
        if "masks" not in self.cache:
            self.cache["masks"] = load_masks(self.project, self.images())
        return self.cache["masks"]

    def mask_array(self, name: str) -> np.ndarray:
        mask = self.masks().get(name)
        return mask if mask is not None else np.ones((1, 1), dtype=np.uint8)

    def report(self, stage: str, payload: Dict[str, Any]) -> None:
        self.stage_reports[stage] = payload
        try:
            from .store import write_json

            write_json(self.project.stage_dir(stage) / "stage_report.json", payload)
        except OSError:  # pragma: no cover
            pass


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------
def _stage_ingestion(ctx: PipelineContext) -> Dict[str, Any]:
    images = ctx.images()
    if not images:
        raise InsufficientDataError(
            "this project has no reference images",
            details={"hint": "add images with `recon3d add-images` or POST /images"},
        )
    report = {
        "images": [img.to_dict() for img in images],
        "count": len(images),
        "total_pixels": int(sum(img.width * img.height for img in images)),
    }
    ctx.reporter.update(100, f"loaded {len(images)} reference images")
    return report


def _stage_validation(ctx: PipelineContext) -> Dict[str, Any]:
    report = analyze_project_images(
        ctx.project,
        max_dim=min(1024, ctx.budget.get("dense_max_dim", 1024)),
        write_report=True,
    )
    ctx.cache["image_report"] = report
    usable = int(report.get("usable_images", 0))
    if usable < 3:
        raise InsufficientDataError(
            "at least three usable reference images covering different angles are required",
            details={
                "usable_images": usable,
                "total_images": report.get("total_images", 0),
                "missing_views": report.get("missing_views", []),
                "warnings": report.get("warnings", []),
                "next_steps": report.get("next_steps", []),
            },
        )
    for warning in report.get("warnings", []):
        ctx.reporter.warning(warning)
    ctx.reporter.update(100, f"quality={report['quality']}, coverage={report['coverage']}%")
    return report


def _stage_segmentation(ctx: PipelineContext) -> Dict[str, Any]:
    report = segment_project(ctx.project, max_dim=ctx.budget.get("dense_max_dim", 1024))
    ctx.cache.pop("masks", None)
    if report.get("warnings"):
        for warning in report["warnings"][:4]:
            ctx.reporter.warning(warning)
    ctx.reporter.update(100, f"segmented {report['segmented']} images")
    return report


def _stage_analysis(ctx: PipelineContext) -> Dict[str, Any]:
    images = ctx.images()
    masks = [ctx.mask_array(img.path.name) for img in images]
    analysis = analyze_subject(images, masks,
                               write_to=ctx.project.stage_dir("camera_estimation"))
    ctx.cache["subject"] = analysis
    if ctx.request("subject_type", "auto") in (None, "", "auto"):
        ctx.params["subject_type"] = analysis.subject_type
    ctx.reporter.update(100, f"subject type: {analysis.subject_type} "
                             f"(confidence {analysis.confidence:.2f})")
    return analysis.to_dict()


def _stage_features(ctx: PipelineContext) -> Dict[str, Any]:
    """Optional cross-view feature matching used for ordering and diagnostics."""
    images = ctx.images()
    step = max(1, len(images) // 8)
    selected = images[::step][:8]
    if len(selected) < 3:
        return {"skipped": True, "reason": "fewer than 3 images"}
    features = []
    for img in selected:
        mask = ctx.mask_array(img.path.name)
        features.append(extract_features(img, mask,
                                         max_dim=ctx.budget.get("feature_max_dim", 1200)))
    if all(f.count < 20 for f in features):
        report = {"skipped": True,
                  "reason": "too few detectable features (textureless or very smooth surfaces)"}
        ctx.reporter.info("feature matching skipped: the references have too few features")
        return report
    counter = {"n": 0}
    total_pairs = len(features) * (len(features) - 1) // 2 or 1

    def progress() -> None:
        counter["n"] += 1
        ctx.reporter.update(min(99.0, 100.0 * counter["n"] / total_pairs),
                            f"matching view pairs ({counter['n']}/{total_pairs})")

    _distances, matches = pairwise_distance_matrix(features, progress=progress)
    report = feature_consistency_report(features, matches)
    for warning in report.get("warnings", []):
        ctx.reporter.warning(warning)
    ctx.cache["feature_report"] = report
    return report


def _stage_cameras(ctx: PipelineContext) -> Dict[str, Any]:
    """Solve the cameras: ring geometry, lens (FOV) and the scale gauge.

    Three things have to be settled before carving, and all three are estimated
    from the reference set itself rather than assumed:

    * the ring geometry (azimuth/elevation/distance per image),
    * the **lens** - reference images rarely carry usable EXIF data, and the
      assumed field of view decides how much perspective the capture is believed
      to contain, which directly controls the reconstructed depth,
    * the **scale gauge** - the reconstruction is defined up to a similarity, so
      one distance is fixed to make the subject measure exactly
      ``subject_height`` engine units (which is what the export scaling and the
      ground-truth comparisons rely on).
    """
    subject = ctx.cache.get("subject")
    height_m = float(ctx.request("subject_height_m", 0.0) or 0.0)
    if height_m <= 0 and subject is not None:
        height_m = float(getattr(subject, "suggested_scale_m", 1.0) or 1.0)
    images = ctx.images()
    masks = ctx.masks()
    payload: Dict[str, Any] = {}
    if ctx.budget.get("lens_search", True) and len(images) >= 4:
        candidates = tuple(float(v) for v in ctx.budget.get(
            "lens_fov_candidates", (26.0, 30.0, 34.0, 38.0, 43.0, 48.0, 55.0)))
        rig, lens_report = calibrate_lens_and_rig(
            ctx.project, images=images, masks=masks, arrays=ctx.image_arrays(),
            subject_height_m=height_m, fov_candidates=candidates,
            refine_steps=int(ctx.budget.get("lens_refine_steps", 1)),
            coarse_resolution=int(ctx.budget.get("lens_carve_resolution", 64)),
            reporter=ctx.reporter,
        )
        payload["lens"] = lens_report
        ctx.reporter.info(
            f"camera solve: {lens_report['fov_deg']:.0f} deg lens assumed "
            f"(silhouette agreement {lens_report['mean_iou']:.3f})"
        )
    else:
        rig = estimate_rig(ctx.project, subject_height_m=height_m, reporter=ctx.reporter)
        gauge = calibrate_rig_height(rig, masks, resolution=80, reporter=ctx.reporter)
        rig = gauge["rig"]
        payload["gauge"] = {k: v for k, v in gauge.items() if k != "rig"}
    for warning in rig.warnings:
        ctx.reporter.warning(warning)
    ctx.cache["rig"] = rig
    write_json(ctx.project.stage_dir("camera_estimation") / "cameras.json", rig.to_dict())
    payload.update({
        "method": rig.method,
        "cameras": [p.to_dict() for p in rig.cameras],
        "center": rig.center.tolist(),
        "warnings": rig.warnings,
        "quality": rig.quality,
        "subject_height_units": rig.subject_height,
        "scale_hint_m": rig.scale_hint_m,
    })
    ctx.report("camera_estimation", payload)
    ctx.reporter.update(100, f"solved {len(rig.cameras)} cameras "
                             f"(ring quality {rig.ring_quality:.2f})")
    return payload


def _carve_and_extract(ctx: "PipelineContext", rig, *, reporter,
                       progress_range=(3.0, 72.0)):
    """Carve the visual hull for ``rig`` and extract its surface.

    Factored out of :func:`_stage_reconstruct` so the silhouette-refinement
    stage can honestly re-run the *same* reconstruction against an updated
    camera rig before deciding whether to keep the refinement.
    """
    from ..engine.reconstruction.hull import (reconstruct_hull, hull_to_mesh,
                                              fuse_depth_points)

    masks = ctx.masks()
    arrays = ctx.image_arrays()
    mode = ctx.budget
    levels = int(mode.get("carve_levels", 2))
    voxel_start = int(mode.get("voxel_start", 64))
    voxel_max = min(int(mode.get("voxel_max", 256)),
                    voxel_grid_budget(ctx.profile, max_voxels=512))
    reporter.update(progress_range[0],
                    f"carving silhouettes ({levels} level(s), up to {voxel_max}^3)")
    hull = reconstruct_hull(
        rig, masks, arrays,
        voxel_start=voxel_start, voxel_max=voxel_max, levels=levels,
        photo_consistency=bool(ctx.request("photo_consistency", True)),
        reporter=reporter, progress_range=progress_range,
    )
    mesh = _extract_surface(ctx, hull, reporter=reporter)
    # Depth evidence must earn its place: keep the fused volume only when the
    # extracted surface actually agrees with the references better than the
    # silhouette hull it replaces.
    hull, mesh, info = _validate_depth_fusion(ctx, rig, hull, mesh, reporter=reporter)
    if info:
        hull.statistics["depth_fusion"] = info
    return hull, mesh


def _extract_surface(ctx: "PipelineContext", hull, *, reporter=None):
    """Marching cubes with the preset's field blur / iso level."""
    mode = ctx.budget
    return hull_to_mesh(hull, smooth=True,
                        level=float(mode.get("iso_level", 0.42)),
                        volume_sigma=float(mode.get("volume_sigma", 0.5)))


def _validate_depth_fusion(ctx: "PipelineContext", rig, hull, mesh, *, reporter):
    """Carve the hull with depth evidence and keep it only if it measures better.

    The fused volume is re-extracted and scored against the references; whichever
    surface wins is the one the pipeline continues with.  Returns
    ``(hull, mesh, report)``.
    """
    points = ctx.cache.get("depth_points")
    if points is None or len(points) <= 200:
        return hull, mesh, {"skipped": True, "reason": "no depth evidence"}
    try:
        fused = fuse_depth_points(hull, points, carve_thickness=1.5, reporter=reporter)
    except Exception as exc:  # pragma: no cover - depth support is optional
        reporter.warning(f"depth fusion failed and was skipped: {exc}")
        return hull, mesh, {"skipped": True, "reason": str(exc)}
    info = dict(fused.statistics.get("depth_fusion") or {})
    if fused is hull or info.get("skipped"):
        return hull, mesh, info
    fused_mesh = _extract_surface(ctx, fused, reporter=reporter)
    base_iou = float(silhouette_iou_report(rig, mesh, ctx.masks()).get("mean_iou", 0.0))
    fused_iou = float(silhouette_iou_report(rig, fused_mesh, ctx.masks()).get("mean_iou", 0.0))
    info["validation"] = {"hull_iou": round(base_iou, 4), "fused_iou": round(fused_iou, 4),
                          "accepted": bool(fused_iou >= base_iou + 1e-4)}
    if fused_iou >= base_iou + 1e-4:
        reporter.info(f"depth fusion accepted: silhouette IoU {base_iou:.3f} -> {fused_iou:.3f}")
        return fused, fused_mesh, info
    reporter.info(f"depth fusion rejected: silhouette IoU {fused_iou:.3f} < {base_iou:.3f}; "
                  "keeping the silhouette hull")
    return hull, mesh, info


def _canonical_frame(ctx: "PipelineContext", rig, mesh, hull, *, reporter):
    """Move + scale (mesh, rig, hull) into the documented engine frame."""
    framing: Dict[str, Any] = {}
    centre = np.asarray(mesh.bounds, dtype=np.float64).mean(axis=0)
    delta = np.asarray(rig.center, dtype=np.float64) - centre
    if float(np.linalg.norm(delta)) > 1e-6:
        mesh.apply_translation(delta)
        rig = rig.translated(delta)
        hull.bounds_min = np.asarray(hull.bounds_min) + delta
        hull.bounds_max = np.asarray(hull.bounds_max) + delta
        framing["recentred_by"] = np.round(delta, 6).tolist()
    target_height = float(rig.subject_height or 1.0)
    height = float(mesh.extents[2])
    if target_height > 0 and height > 1e-9 and abs(height / target_height - 1.0) > 0.005:
        factor = target_height / height
        mesh.apply_translation(-np.asarray(rig.center))
        mesh.apply_scale(factor)
        mesh.apply_translation(np.asarray(rig.center))
        rig = rig.scaled(factor)
        framing["scaled_by"] = round(float(factor), 6)
    if framing:
        reporter.info(f"canonical frame: {framing}")
    return rig, mesh, hull, framing


def _stage_reconstruct(ctx: PipelineContext) -> Dict[str, Any]:
    rig = ctx.cache["rig"]
    images = ctx.images()
    arrays = ctx.image_arrays()
    masks = ctx.masks()
    mode = ctx.budget
    reporter = ctx.reporter
    hull, mesh = _carve_and_extract(ctx, rig, reporter=reporter, progress_range=(3.0, 72.0))
    ctx.cache["hull"] = hull

    # -- dense depth support (the MVS half of the pipeline) --------------
    depth_report: Dict[str, Any] = {"skipped": True}
    if mode.get("dense_max_dim", 0) and len(rig.cameras) >= 3:
        try:
            reporter.update(74, "estimating multi-view depth")
            fused = fuse_depth_maps(rig, arrays, masks,
                                    steps=int(mode.get("depth_steps", 24)),
                                    pixel_step=max(2, int(mode.get("dense_max_dim", 1024) // 220)),
                                    reporter=reporter, progress_range=(74.0, 88.0),
                                    max_views=min(8, len(rig.cameras)))
            depth_report = fused["statistics"]
            ctx.cache["depth_points"] = fused["points"]
            ctx.cache["depth_colors"] = fused.get("colors")
            consistency = depth_statistics_consistency(fused["depth_maps"])
            depth_report["consistency"] = consistency
            for warning in consistency.get("warnings", []):
                reporter.warning(warning)
            if len(fused["points"]) > 200:
                # The sweep runs *after* the carve, so this is the first moment the
                # hull can be re-judged against depth evidence: fuse, re-extract,
                # and keep the better surface.
                ctx.cache["depth_points"] = fused["points"]
                ctx.cache["depth_colors"] = fused.get("colors")
                hull, mesh, fusion = _validate_depth_fusion(ctx, rig, hull, mesh,
                                                            reporter=reporter)
                ctx.cache["hull"] = hull
                ctx.cache["mesh"] = mesh
                # Record the final verdict on the hull too: the pre-depth carve ran
                # the same check with no evidence available, and leaving that "skipped"
                # note in place would misrepresent the decision in reports/stages.json.
                hull.statistics["depth_fusion"] = fusion
                depth_report["volume_fusion"] = fusion
                depth_report["fused_into_volume"] = bool(fusion.get("validation", {}).get("accepted"))
                reporter.update(90, "validated depth evidence against the silhouette hull")
        except Exception as exc:  # pragma: no cover - depth is optional support
            depth_report = {"error": str(exc), "skipped": True}
            reporter.warning(f"depth estimation failed and was skipped: {exc}")

    # -- point cloud artefact -------------------------------------------
    pointcloud_path = ""
    if ctx.request("generate_pointcloud", True):
        try:
            points = ctx.cache.get("depth_points")
            colors = ctx.cache.get("depth_colors")
            if points is None or len(points) < 100:
                points = voxel_surface_points(hull, stride=2)
                colors = None
            if len(points):
                points = statistical_outlier_removal(points, k=10, std_ratio=2.5)
                pointcloud_path = save_point_cloud(
                    ctx.dir_for("intermediate") / "pointcloud.ply", points, colors
                )
        except Exception as exc:  # pragma: no cover
            reporter.warning(f"point cloud export failed: {exc}")

    # -- canonical frame -------------------------------------------------
    # The engine's frame is documented as "subject bounding-box centre at the
    # rig centre, subject height = subject_height units".  Applying that as a
    # *similarity* (mesh + cameras together) leaves every projection unchanged,
    # so the accuracy of the carve is untouched while the exported asset gets a
    # predictable origin and scale.
    rig, mesh, hull, framing = _canonical_frame(ctx, rig, mesh, hull, reporter=reporter)
    if framing:
        ctx.cache["rig"] = rig
        write_json(ctx.project.stage_dir("camera_estimation") / "cameras.json", rig.to_dict())
    stats = mesh_statistics(mesh)
    reporter.update(95, f"marching cubes produced {stats['faces']} faces")
    ctx.cache["mesh"] = mesh
    _persist_mesh(ctx, mesh)
    ctx.version.assets.pointcloud = pointcloud_path
    report = {
        "hull": hull.statistics,
        "depth": depth_report,
        "mesh": stats,
        "bounds": hull.bounds_min.tolist() + hull.bounds_max.tolist(),
        "pointcloud": pointcloud_path,
    }
    ctx.report("mesh_reconstruction", report)
    return report


def _stage_silhouette_refine(ctx: PipelineContext) -> Dict[str, Any]:
    """Refine the cameras against the reconstruction - and prove it.

    Evaluating a camera perturbation against the very mesh it was tuned on is
    circular: walking every camera closer always raises silhouette IoU without
    the model being any more correct (and leaves the mesh at the old scale,
    which then costs real accuracy in every downstream comparison).  This stage
    therefore

    1. searches only the non-degenerate rotation axes (``allow_distance=False``),
    2. re-carves and re-extracts at the *same* resolution with the refined rig,
    3. keeps the new (rig, mesh) pair only when the freshly carved geometry
       matches the references better than the pair it replaces.

    Everything is measured, nothing is assumed: the report carries both numbers
    and the reason for the decision.
    """
    passes = int(ctx.budget.get("refine_passes", 0))
    if passes <= 0:
        return {"skipped": True, "reason": "preset disables silhouette refinement"}
    rig = ctx.cache["rig"]
    mesh = ctx.cache["mesh"]
    masks = ctx.masks()
    reporter = ctx.reporter

    before = silhouette_iou_report(rig, mesh, masks)
    baseline_iou = float(before.get("mean_iou", 0.0))
    reporter.update(2, f"refining cameras against the reconstruction (baseline IoU {baseline_iou:.3f})")
    refined_rig, report = refine_rig_silhouette(rig, mesh, masks, allow_distance=False,
                                                reporter=reporter)
    stale = silhouette_iou_report(refined_rig, mesh, masks)
    report["iou_before_full_res"] = round(baseline_iou, 4)
    report["iou_stale_mesh"] = round(float(stale.get("mean_iou", 0.0)), 4)

    accepted = False
    after_iou = baseline_iou
    if int(report.get("refined", 0)) > 0:
        try:
            hull, candidate = _carve_and_extract(ctx, refined_rig, reporter=reporter,
                                                 progress_range=(20.0, 80.0))
            candidate_rig, candidate, hull, framing = _canonical_frame(
                ctx, refined_rig, candidate, hull, reporter=reporter)
            after = silhouette_iou_report(candidate_rig, candidate, masks)
            after_iou = float(after.get("mean_iou", 0.0))
            report["candidate_iou"] = round(after_iou, 4)
            if after_iou >= baseline_iou + 1e-4:
                ctx.cache["rig"] = candidate_rig
                ctx.cache["mesh"] = candidate
                _persist_mesh(ctx, candidate, "silhouette_refine")
                ctx.cache["hull"] = hull
                # The exported asset and the solved cameras must travel together:
                # always persist the rig that belongs to the kept mesh, so agents
                # can re-project or re-measure an exported model reproducibly.
                write_json(ctx.project.stage_dir("camera_estimation") / "cameras.json",
                           candidate_rig.to_dict())
                accepted = True
        except Exception as exc:  # pragma: no cover - refinement is optional
            report["error"] = str(exc)
            reporter.warning(f"refinement re-carve failed; keeping the original solve: {exc}")
    else:
        report["reason"] = "no camera perturbation improved the proxy silhouette"
    report["iou_after_full_res"] = round(after_iou, 4)
    report["accepted"] = accepted
    if not accepted:
        ctx.cache["rig"] = rig
        ctx.cache["mesh"] = mesh
        ctx.reporter.info("refinement rejected: the re-carved model matched no better than "
                          "the original solve, so the original cameras and mesh were kept")
    else:
        ctx.reporter.info(f"refinement accepted: re-carved silhouette IoU "
                          f"{baseline_iou:.3f} -> {after_iou:.3f}")
    ctx.report("camera_estimation", report)
    return report


def _stage_cleanup(ctx: PipelineContext) -> Dict[str, Any]:
    mesh = ctx.cache["mesh"]
    target_faces = _resolve_polycount(ctx)
    mesh, report = cleanup_mesh(
        mesh,
        remove_components=True,
        min_component_fraction=0.015,
        do_smooth=True,
        smooth_iterations=int(ctx.budget.get("mesh_smooth_iters", 2)),
        preserve_sharp_edges=bool(ctx.request("preserve_sharp_edges", True)),
        target_faces=int(target_faces * 1.6) if target_faces else None,
        reporter=ctx.reporter,
    )
    ctx.cache["mesh"] = mesh
    _persist_mesh(ctx, mesh)
    ctx.cache["mesh_stats"] = mesh_statistics(mesh)
    for warning in report.warnings:
        ctx.reporter.warning(warning)
    payload = report.to_dict()
    payload["target_polycount"] = target_faces
    ctx.report("mesh_cleanup", payload)
    ctx.reporter.update(100, f"cleanup: {payload['after'].get('faces', 0)} faces, "
                             f"health {payload['after'].get('health_score', 0):.0f}")
    return payload


def _resolve_polycount(ctx: PipelineContext) -> Optional[int]:
    requested = ctx.request("target_polycount", "auto")
    base = len(ctx.cache["mesh"].faces)
    quality = ctx.params.get("_performance_mode", "balanced")
    style = str(ctx.request("style", "realistic"))
    return polygon_budget(requested, base_faces=base, quality=quality, style=style)


def _stage_topology(ctx: PipelineContext) -> Dict[str, Any]:
    """Produce the export topology (game-ready / target polycount)."""
    mesh = ctx.cache["mesh"]
    target = _resolve_polycount(ctx)
    report: Dict[str, Any] = {"requested": ctx.request("target_polycount", "auto"),
                              "resolved": target}
    if target and target < len(mesh.faces):
        mesh, decimate_report = optimize_for_realtime(mesh, target_faces=int(target),
                                                      reporter=ctx.reporter)
        report["decimation"] = decimate_report
    report["statistics"] = mesh_statistics(mesh)
    ctx.cache["mesh"] = mesh
    _persist_mesh(ctx, mesh, "optimization")
    ctx.cache["mesh_stats"] = report["statistics"]
    ctx.report("optimization", report)
    ctx.reporter.update(100, f"topology: {report['statistics']['faces']} faces")
    return report


def _stage_uv(ctx: PipelineContext) -> Dict[str, Any]:
    if not ctx.request("generate_uvs", True):
        return {"skipped": True, "reason": "uv generation disabled"}
    mesh = ctx.cache["mesh"]
    resolution = _texture_resolution(ctx)
    result = unwrap_mesh(mesh, resolution=max(512, min(2048, resolution)),
                         backend="auto")
    for warning in result.warnings:
        ctx.reporter.warning(warning)
    ctx.cache["mesh"] = result.mesh
    ctx.cache["uvs"] = np.asarray(result.mesh.visual.uv)
    # The unwrap duplicates vertices along UV seams.  Persist exactly that mesh: a
    # resumed run has to export the same topology (and with its UVs) as the run
    # that produced the checkpoint.
    _persist_mesh(ctx, result.mesh, "uv")
    ctx.report("uv", result.to_dict())
    ctx.reporter.update(100, f"UV: {result.islands} islands, distortion "
                             f"{result.estimated_distortion:.3f}")
    return result.to_dict()


def _texture_resolution(ctx: PipelineContext) -> int:
    requested = int(ctx.request("texture_resolution", 0) or 0)
    if requested <= 0:
        requested = int(ctx.budget.get("texture_resolution", 2048))
    cap = int(ctx.params.get("_limits", {}).get("max_texture_resolution", 8192)) \
        if isinstance(ctx.params.get("_limits"), dict) else 8192
    if ctx.profile.ram_mb < 8000:
        cap = min(cap, 4096)
    if ctx.profile.ram_mb < 4500:
        cap = min(cap, 2048)
    allowed = [512, 1024, 2048, 4096, 8192]
    requested = min(requested, cap)
    # Snap to a supported resolution (never silently exceed the request).
    return max(512, min(allowed, key=lambda v: abs(v - requested)))


def _stage_texture(ctx: PipelineContext) -> Dict[str, Any]:
    mesh = ctx.cache["mesh"]
    uvs = ctx.cache.get("uvs")
    if uvs is None:
        return {"skipped": True, "reason": "mesh has no UVs (uv stage skipped or failed)"}
    rig = ctx.cache["rig"]
    resolution = _texture_resolution(ctx)
    texture = project_texture(rig, mesh, uvs, ctx.image_arrays(), ctx.masks(),
                              resolution=resolution, reporter=ctx.reporter)
    for warning in texture.warnings:
        ctx.reporter.warning(warning)

    # Material estimation drives the metallic/roughness derivation.
    assignment = estimate_materials(
        (np.clip(texture.base_color, 0, 1) * 255).astype(np.uint8),
        subject_type=str(ctx.request("subject_type", "auto")),
        style=str(ctx.request("style", "realistic")),
    )
    request_text = str(ctx.request("material_request", "") or ctx.request("material", ""))
    if request_text:
        assignment = apply_material_request(assignment, request_text)
    texture = derive_pbr_maps(
        texture,
        normal_strength=float(assignment.material.normal_strength),
        roughness_range=(max(0.05, assignment.material.roughness - 0.25),
                         min(0.98, assignment.material.roughness + 0.25)),
        metallic_hint=assignment.material.metallic,
        material_profile=assignment.material.name,
    )
    texture.covered_mask = texture.base_color.sum(axis=2) > 1e-6

    texture_dir = ctx.dir_for("textures")
    maps = save_texture_maps(texture, texture_dir, resolution=None, reporter=ctx.reporter)
    orm = pack_orm(texture)
    from PIL import Image

    orm_path = texture_dir / "orm.png"
    Image.fromarray(orm).save(orm_path)
    maps["orm"] = str(orm_path)

    ctx.cache["texture"] = texture
    ctx.cache["material_assignment"] = assignment
    ctx.version.assets.textures = {k: str(Path(v).relative_to(ctx.version_dir))
                                   for k, v in maps.items()}

    # Attach the UVs only.  Attaching the texture images to the in-memory mesh
    # would make every later mesh copy (``split()``, decimation, component
    # counting) duplicate megabytes of PIL data per component - the exporters get
    # the maps explicitly through ``_texture_payload`` instead.
    try:
        import trimesh

        mesh.visual = trimesh.visual.TextureVisuals(uv=np.asarray(uvs))
    except Exception as exc:  # pragma: no cover - visual assignment is best effort
        ctx.reporter.warning(f"could not attach UVs to the mesh object: {exc}")

    report = {
        "texture": texture.statistics,
        "maps": {k: str(Path(v).name) for k, v in maps.items()},
        "resolution": resolution,
        "material": assignment.to_dict(),
        "warnings": texture.warnings,
    }
    ctx.report("texture", report)
    ctx.report("materials", {"assignment": assignment.to_dict()})
    return report


def _stage_materials(ctx: PipelineContext) -> Dict[str, Any]:
    """Write the material library entry + region analysis next to the asset."""
    assignment = ctx.cache.get("material_assignment")
    if assignment is None:
        from ..engine.materials.library import MaterialAssignment, get_material

        material = get_material(str(ctx.request("material", "plastic") or "plastic"))
        assignment = MaterialAssignment(material=material, confidence=0.2,
                                        evidence={"reason": "no texture stage ran"})
    directory = ctx.dir_for("materials")
    path = write_materials(directory, assignment)
    ctx.version.assets.materials = str(Path(path).relative_to(ctx.version_dir))
    report = {"assignment": assignment.to_dict(), "path": path}
    ctx.report("materials", report)
    return report


def _stage_rig(ctx: PipelineContext) -> Dict[str, Any]:
    if not ctx.request("generate_rig", False):
        return {"skipped": True, "reason": "rigging not requested"}
    subject_type = str(ctx.request("subject_type", "auto"))
    if subject_type not in {"auto", "character_humanoid", "character_creature", "robot_mech",
                            "animal", "unknown"}:
        raise StageError(
            f"rigging was requested but the subject was classified as '{subject_type}'",
            stage="rigging", recoverable=True,
            details={"hint": "pass subject_type=character_humanoid to force a humanoid rig"},
        )
    mesh = ctx.cache["mesh"]
    result = generate_rig(mesh, kind="auto", subject_type=subject_type,
                          report_dir=ctx.dir_for("rig"), reporter=ctx.reporter)
    paths = write_rig(ctx.dir_for("rig"), result)
    ctx.cache["rig_data"] = {
        "weights": result["weights"],
        "joints": result["rig"].bone_names(),
        "bones": [b.to_dict() for b in result["rig"].bones],
        "root": result["rig"].root,
    }
    ctx.version.assets.rig = {k: str(Path(v).relative_to(ctx.version_dir)) for k, v in paths.items()}
    for warning in result["rig"].warnings:
        ctx.reporter.warning(warning)
    report = {"rig": result["rig"].to_dict(), "stress": result["stress"], "paths": paths}
    ctx.report("rigging", report)
    return report


def _stage_lod(ctx: PipelineContext) -> Dict[str, Any]:
    if not ctx.request("generate_lods", True):
        return {"skipped": True, "reason": "LOD generation disabled"}
    mesh = ctx.cache["mesh"]
    levels = int(ctx.request("lod_levels", 0) or 0) or int(ctx.budget.get("lod_levels", 4))
    chain, meshes = build_lod_chain(mesh, levels=levels,
                                    texture_resolution=_texture_resolution(ctx),
                                    reporter=ctx.reporter)
    lod_dir = ctx.dir_for("lod")
    paths: Dict[str, str] = {}
    for level, level_mesh in zip(chain.levels, meshes):
        target = lod_dir / f"lod{level.index}.glb"
        # Each LOD carries its own (smaller) texture set: shipping the full-resolution
        # maps on every level would blow up both memory and file size for no gain.
        export_asset(level_mesh, target, "glb",
                     textures=_texture_payload(ctx, resolution=level.texture_resolution),
                     material=_material_payload(ctx),
                     rig=ctx.cache.get("rig_data") if level.index == 0 else None)
        level.path = str(Path(target).relative_to(ctx.version_dir))
        paths[f"lod{level.index}"] = level.path
        # LOD0 *is* the pipeline mesh (the chain deliberately reuses it rather than
        # copying), so only drop our extra references here - never mutate the mesh.
        del level_mesh
        gc.collect()
    ctx.version.assets.lods = paths
    payload = chain.to_dict()
    ctx.report("lod", payload)
    ctx.reporter.update(100, f"built {len(chain.levels)} LOD levels")
    return payload


def _texture_payload(ctx: PipelineContext, *, resolution: Optional[int] = None
                    ) -> Optional[Dict[str, Any]]:
    """Texture maps for export, optionally resampled to ``resolution``.

    LOD levels pass their own (smaller) resolution so a 4-level chain costs a
    fraction of the memory and disk of four full-resolution copies.
    """
    texture = ctx.cache.get("texture")
    if texture is None:
        return None
    payload = {
        "basecolor": (np.clip(texture.base_color, 0, 1) * 255).astype(np.uint8),
        "normal": texture.normal,
        "roughness": texture.roughness,
        "metallic": texture.metallic,
        "ao": (np.clip(texture.ao, 0, 1) * 255).astype(np.uint8) if texture.ao.dtype != np.uint8 else texture.ao,
        "orm": pack_orm(texture),
    }
    if resolution is None:
        return payload
    return {key: _resample_texture(value, int(resolution)) for key, value in payload.items()
            if value is not None}


def _resample_texture(image: np.ndarray, resolution: int) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim < 2 or int(image.shape[0]) == int(resolution):
        return image
    try:
        import cv2

        interpolation = cv2.INTER_AREA if resolution < image.shape[0] else cv2.INTER_LINEAR
        resized = cv2.resize(image, (int(resolution), int(resolution)),
                             interpolation=interpolation)
        return resized if resized.ndim == image.ndim else resized[..., None]
    except Exception:  # pragma: no cover - cv2 is a core dependency, but never fatal
        step = max(1, int(round(image.shape[0] / max(1, resolution))))
        return image[::step, ::step]


def _material_payload(ctx: PipelineContext) -> Dict[str, Any]:
    assignment = ctx.cache.get("material_assignment")
    if assignment is None:
        return {"name": "recon3d_material", "base_color": (200, 200, 200),
                "metallic": 0.0, "roughness": 0.7}
    return assignment.material.to_dict()


def _stage_preview(ctx: PipelineContext) -> Dict[str, Any]:
    if not ctx.request("generate_previews", True):
        return {"skipped": True, "reason": "preview rendering disabled"}
    from ..engine.preview.renderer import render_turntable  # local import to keep startup light

    mesh = ctx.cache["mesh"]
    rig = ctx.cache.get("rig")
    preview_dir = ctx.dir_for("previews")
    result = render_turntable(mesh, preview_dir, rig=rig,
                              resolution=int(ctx.budget.get("preview_resolution", 640)),
                              frames=int(ctx.budget.get("preview_frames", 12)),
                              reporter=ctx.reporter)
    ctx.version.assets.previews = [str(Path(p).relative_to(ctx.version_dir))
                                   for p in result.get("stills", [])]
    report = {k: v for k, v in result.items() if k != "stills"}
    report["stills"] = [str(Path(p).name) for p in result.get("stills", [])]
    ctx.report("preview", report)
    return report


def _stage_comparison(ctx: PipelineContext) -> Dict[str, Any]:
    rig = ctx.cache.get("rig")
    mesh = ctx.cache["mesh"]
    images = ctx.image_arrays()
    masks = ctx.masks()
    comparison = compare_to_references(rig, mesh, images, masks)
    for warning in comparison.get("warnings", []):
        ctx.reporter.warning(warning)

    refinement_report: Dict[str, Any] = {}
    passes = int(ctx.budget.get("refine_passes", 0) if ctx.request("refine_passes", -1) in (None, -1)
                 else ctx.request("refine_passes", 0))
    if passes > 0:
        ctx.reporter.update(50, "refining the model against the references")
        mesh, refinement_report = refine_against_references(rig, mesh, images, masks,
                                                            iterations=min(2, passes),
                                                            reporter=ctx.reporter)
        if refinement_report.get("applied"):
            ctx.cache["mesh"] = mesh
            comparison = compare_to_references(rig, mesh, images, masks)
            for warning in comparison.get("warnings", []):
                ctx.reporter.warning(warning)
            # Re-export is only needed when geometry changed materially.
            ctx.cache["geometry_changed_after_comparison"] = True
        ctx.report("refinement", refinement_report)

    ctx.cache["comparison"] = comparison
    ctx.cache["refinement"] = refinement_report

    # Comparison renders (agent inspection artefacts).
    render_dir = ctx.dir_for("renders")
    renders = render_reference_views(rig, mesh)
    from PIL import Image

    stills: List[str] = []
    for index, (name, array) in enumerate(sorted(renders.items())):
        target = render_dir / f"compare_{Path(name).stem}.png"
        Image.fromarray(array).save(target)
        stills.append(str(Path(target).relative_to(ctx.version_dir)))
    ctx.version.assets.renders = stills
    report = {"comparison": comparison, "refinement": refinement_report,
              "renders": [Path(s).name for s in stills]}
    ctx.report("comparison", report)
    return report


def _stage_export(ctx: PipelineContext) -> Dict[str, Any]:
    mesh = ctx.cache["mesh"]
    formats = ctx.request("export_formats", ["glb", "obj"]) or ["glb"]
    final_dir = ctx.dir_for("final")
    assets = ctx.version.assets
    scale = _export_scale(ctx)
    results: Dict[str, Any] = {}
    for fmt in formats:
        fmt = str(fmt).lower().lstrip(".")
        target = final_dir / f"{ctx.project.id}_{ctx.version.id}.{fmt}"
        try:
            result = export_asset(mesh, target, fmt, textures=_texture_payload(ctx),
                                  material=_material_payload(ctx),
                                  rig=ctx.cache.get("rig_data"))
            verification = verify_export(Path(result.path), fmt)
            results[fmt] = {**result.to_dict(), "verification": verification}
            assets.mesh[fmt] = str(Path(result.path).relative_to(ctx.version_dir))
            if not verification.get("ok"):
                ctx.reporter.warning(f"{fmt} export verification failed: "
                                     f"{verification.get('error', 'unknown')}")
        except Exception as exc:
            ctx.reporter.error(f"{fmt} export failed: {exc}")
            results[fmt] = {"error": str(exc), "format": fmt}
    ctx.report("export", results)
    ctx.reporter.update(100, f"exported {len([r for r in results.values() if 'path' in r])} formats")
    return results


def _export_scale(ctx: PipelineContext) -> float:
    """Scale factor from the normalised reconstruction frame to the requested units."""
    units = str(ctx.request("units", "normalized") or "normalized").lower()
    if units in {"normalized", "none", "engine"}:
        return 1.0
    subject = ctx.cache.get("subject")
    height_m = float(ctx.request("subject_height_m", 0.0) or 0.0)
    if height_m <= 0:
        height_m = float(getattr(subject, "suggested_scale_m", 1.0) or 1.0)
    rig = ctx.cache.get("rig")
    # The reconstruction frame normalises the subject height to 1.0 world unit.
    factor = height_m
    if units in {"centimeters", "cm"}:
        factor *= 100.0
    elif units in {"millimeters", "mm"}:
        factor *= 1000.0
    return float(factor)


# --------------------------------------------------------------------------
# Stage table
# --------------------------------------------------------------------------
STAGE_TABLE: List[StageSpec] = [
    StageSpec("ingestion", _stage_ingestion),
    StageSpec("validation", _stage_validation, ("ingestion",)),
    StageSpec("segmentation", _stage_segmentation, ("validation",)),
    StageSpec("analysis", _stage_analysis, ("segmentation",), optional=True),
    StageSpec("features", _stage_features, ("segmentation",), optional=True),
    StageSpec("camera_estimation", _stage_cameras, ("analysis", "features")),
    StageSpec("mesh_reconstruction", _stage_reconstruct, ("camera_estimation",)),
    StageSpec("silhouette_refine", _stage_silhouette_refine, ("mesh_reconstruction",),
              optional=True),
    StageSpec("mesh_cleanup", _stage_cleanup, ("mesh_reconstruction",)),
    StageSpec("optimization", _stage_topology, ("mesh_cleanup",), optional=True),
    StageSpec("uv", _stage_uv, ("optimization",), optional=True),
    StageSpec("texture", _stage_texture, ("uv",), optional=True),
    StageSpec("materials", _stage_materials, ("texture",), optional=True),
    StageSpec("rigging", _stage_rig, ("optimization",), optional=True),
    StageSpec("lod", _stage_lod, ("texture", "optimization"), optional=True),
    StageSpec("preview", _stage_preview, ("texture", "optimization"), optional=True),
    StageSpec("comparison", _stage_comparison, ("mesh_cleanup",), optional=True),
    StageSpec("export", _stage_export, ("comparison",), optional=True, always_run=True),
]

#: Stage name -> public progress stage used in events (spec #30).
STAGE_PROGRESS_NAMES: Dict[str, str] = {
    "ingestion": "ingestion",
    "validation": "validation",
    "segmentation": "segmentation",
    "analysis": "validation",
    "features": "camera_estimation",
    "camera_estimation": "camera_estimation",
    "mesh_reconstruction": "mesh_reconstruction",
    "silhouette_refine": "camera_estimation",
    "mesh_cleanup": "mesh_cleanup",
    "optimization": "optimization",
    "uv": "uv",
    "texture": "texture",
    "materials": "materials",
    "rigging": "rigging",
    "lod": "lod",
    "preview": "preview",
    "comparison": "comparison",
    "export": "export",
}


def stage_names() -> List[str]:
    return [spec.name for spec in STAGE_TABLE]


def select_stages(only: Optional[Sequence[str]] = None,
                  skip: Optional[Sequence[str]] = None) -> List[StageSpec]:
    """Resolve a stage subset, pulling in the dependencies it needs."""
    if not only:
        specs = list(STAGE_TABLE)
    else:
        wanted = {s.strip().lower() for s in only}
        by_name = {spec.name: spec for spec in STAGE_TABLE}
        chosen: Dict[str, StageSpec] = {}

        def add(name: str) -> None:
            if name in chosen or name not in by_name:
                return
            for dependency in by_name[name].requires:
                add(dependency)
            chosen[name] = by_name[name]

        for name in wanted:
            add(name)
        specs = [spec for spec in STAGE_TABLE if spec.name in chosen]
    if skip:
        skip_set = {s.strip().lower() for s in skip}
        specs = [spec for spec in specs if spec.name not in skip_set]
    return specs


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------
def _inputs_hash(ctx: PipelineContext, spec: StageSpec) -> str:
    """Hash of everything a stage depends on (drives checkpoint reuse)."""
    project = ctx.project
    images = []
    for img in project.data.get("images", []):
        images.append([img.get("id"), img.get("path"), img.get("sha256", ""), img.get("view")])
    # Two things must *not* take part in this hash:
    #   * the version id - checkpoints live in the project's
    #     ``intermediate/<stage>/``, so one run must be able to resume another,
    #     and every run generates a new version id;
    #   * anything the pipeline derives from the images (analysed subject type,
    #     solved scale hint, measured scores) - a stage would otherwise never
    #     match its own checkpoint once a later stage wrote those values back.
    # What identifies a stage's inputs is the *requested* parameters, the image
    # set, and the input hashes of the stages it depends on.
    requested = ctx.state.get("requested_params") or ctx.params
    dependency_hashes = ctx.state.get("stage_input_hashes") or {}
    payload = {
        "stage": spec.name,
        "params": {k: v for k, v in requested.items() if not k.startswith("_")},
        "images": images,
        "dependencies": {name: dependency_hashes.get(name, "") for name in spec.requires},
    }
    return hash_inputs([payload])


def run_pipeline(
    project: Project,
    job: Job,
    *,
    params: Optional[Dict[str, Any]] = None,
    stages: Optional[Sequence[str]] = None,
    label: str = "",
    resume: bool = True,
) -> Dict[str, Any]:
    """Execute the pipeline for one job; returns the result payload."""
    reporter = job.reporter
    assert reporter is not None
    started = time.time()
    resolved = resolve_params(params, project_params=project.params)
    resolved["_limits"] = {
        "max_texture_resolution": 8192,
    }

    profile = profile_hardware()
    budget = mode_budget(str(resolved.get("_performance_mode", "balanced")), profile)
    budget.setdefault("carve_levels", 2)
    budget["carve_levels"] = 3 if budget["mode"] in {"quality", "maximum"} else 2
    if budget["mode"] == "draft":
        budget["carve_levels"] = 1
    budget.setdefault("depth_steps", 20 if budget["mode"] == "draft" else 28)
    budget.setdefault("preview_resolution", 640)
    budget.setdefault("preview_frames", 12)

    version = project.create_version(
        params={k: v for k, v in resolved.items() if not k.startswith("_")},
        label=label or str(resolved.get("quality", "standard")),
        parent=project.data.get("current_version"),
    )
    ctx = PipelineContext(project=project, params=resolved, version=version, budget=budget,
                          reporter=reporter, profile=profile)
    ctx.force = bool(resolved.get("force_stages"))
    # Snapshot the request before any stage runs: stages may write back what they
    # derived (the analysed subject type, for instance), and a checkpoint that
    # hashed those values could never be re-used by a later run.
    ctx.state["requested_params"] = json.loads(json.dumps(resolved, default=str))

    specs = select_stages(stages, resolved.get("skip_stages"))
    force_stages = {str(s).lower() for s in (resolved.get("force_stages") or [])}
    total_weight = sum(spec.weight for spec in specs) or 1.0
    completed = 0.0
    failed: List[str] = []
    stage_results: Dict[str, Any] = {}

    reporter.info(
        f"pipeline start: preset={resolved['_preset']} mode={budget['mode']} "
        f"stages={[s.name for s in specs]}"
    )

    try:
        for spec in specs:
            reporter.check_cancelled()
            progress_name = STAGE_PROGRESS_NAMES.get(spec.name, spec.name)
            reporter.stage_start(progress_name, f"stage: {spec.name}")

            stage_dir = project.stage_dir(spec.name)
            stage_dir.mkdir(parents=True, exist_ok=True)
            inputs_hash = _inputs_hash(ctx, spec)
            # Downstream stages chain on this value, so it must be known before
            # the stage runs (and it is identical when the stage is resumed).
            ctx.state.setdefault("stage_input_hashes", {})[spec.name] = inputs_hash
            cached = read_checkpoint(project, spec.name)
            # Validate the outputs whenever the checkpoint recorded any, so a
            # stage whose artefacts were deleted is rebuilt instead of re-used.
            outputs_expected = bool((cached or {}).get("outputs"))
            if (resume and not spec.always_run and spec.name not in force_stages
                    and checkpoint_valid(project, spec.name, inputs_hash,
                                         require_outputs=outputs_expected)):
                report = (cached or {}).get("statistics") or {}
                checkpoint_data = cached or {}
                stage_results[spec.name] = report
                ctx.stage_reports[spec.name] = report
                reporter.info(f"stage '{spec.name}' re-used cached results")
                completed += spec.weight
                job.progress = overall_progress(progress_name, 1.0)
                try:
                    _restore_cache_from_disk(ctx, spec, checkpoint_data)
                except StageError as exc:
                    # The checkpoint is stale or damaged: rebuild this stage.
                    reporter.warning(f"cached results for '{spec.name}' are unusable "
                                     f"({exc.message}); rebuilding the stage")
                    invalidate_stage(project, spec.name)
                else:
                    _materialise_version_outputs(ctx, checkpoint_data)
                    continue

            reporter.update(1, f"running {spec.name}")
            ctx.current_stage = spec.name
            version_before = _version_files(project, version)
            attempts = max(1, int(resolved.get("stage_retries", 1) or 0) + 1)
            stage_error: Optional[StageError] = None
            last_exc: Optional[BaseException] = None
            report: Any = {}
            attempt = 1
            for attempt in range(1, attempts + 1):
                try:
                    report = spec.fn(ctx) or {}
                except CancelledError:
                    raise
                except Exception as exc:
                    last_exc = exc
                    stage_error = exc if isinstance(exc, StageError) else StageError(
                        str(exc), stage=spec.name, recoverable=spec.optional)
                    if attempt < attempts and _may_retry(exc):
                        delay = min(2.0, 0.5 * attempt)
                        reporter.warning(
                            f"stage '{spec.name}' failed on attempt {attempt}/{attempts} "
                            f"({stage_error.message}); retrying in {delay:.1f}s")
                        time.sleep(delay)
                        continue
                    break
                else:
                    stage_error = None
                    break
            if stage_error is not None:
                stage_info = job.stages.setdefault(spec.name, {})
                stage_info["status"] = "failed"
                stage_info["attempts"] = attempt
                stage_results[spec.name] = {"error": stage_error.message,
                                            "code": stage_error.code,
                                            "attempts": attempt}
                failed.append(spec.name)
                if not spec.optional:
                    write_checkpoint(project, spec.name, inputs_hash=inputs_hash,
                                     status="failed",
                                     statistics={"error": stage_error.message,
                                                 "attempts": attempt})
                    raise stage_error from last_exc
                reporter.error(f"optional stage '{spec.name}' failed and was skipped after "
                               f"{attempt} attempt(s): {stage_error.message}")
                completed += spec.weight
                continue
            if attempt > 1:
                reporter.info(f"stage '{spec.name}' recovered on attempt {attempt}")
                job.stages.setdefault(spec.name, {})["attempts"] = attempt

            stage_results[spec.name] = report
            ctx.stage_reports[spec.name] = report
            job.stages.setdefault(spec.name, {})["status"] = "completed"
            outputs = stage_artefacts(
                project, spec.name,
                produced=_version_files(project, version) - version_before)
            outputs.update(_inputs_from_dependencies(project, spec))
            write_checkpoint(project, spec.name, inputs_hash=inputs_hash,
                             outputs=outputs,
                             statistics=_json_safe(report))
            completed += spec.weight
            job.progress = overall_progress(progress_name, 1.0)
            reporter.update(100, f"{spec.name} complete",
                            stage=progress_name, overall=job.progress)

        # -- finalise -----------------------------------------------------
        _refresh_version_assets(project, version)
        ctx.cache["mesh_stats"] = mesh_statistics(ctx.cache["mesh"]) if "mesh" in ctx.cache else {}
        comparison = ctx.cache.get("comparison") or {}
        quality = score_quality(
            comparison=comparison,
            mesh_stats=ctx.cache.get("mesh_stats", {}),
            uv_metrics=ctx.stage_reports.get("uv"),
            texture_stats=(ctx.stage_reports.get("texture") or {}).get("texture"),
            image_report=ctx.cache.get("image_report"),
            camera_quality=(ctx.cache.get("rig").quality if ctx.cache.get("rig") is not None else {}),
        )
        ctx.cache["quality"] = quality
        version.quality = quality
        version.statistics = _build_statistics(ctx)
        version.stages_completed = [name for name, r in stage_results.items()
                                    if not (isinstance(r, dict) and r.get("error"))]
        version.stages_failed = failed
        version.warnings = list(dict.fromkeys(
            [w for w in (quality.get("warnings") or [])] + list(job.warnings)))
        version.backend_report = _backend_report(ctx)
        version.duration_s = time.time() - started
        version.assets.mesh = {k: v for k, v in version.assets.mesh.items()}

        reports_dir = ctx.dir_for("reports")
        from .store import write_json

        write_json(reports_dir / "quality.json", quality)
        write_json(reports_dir / "statistics.json", version.statistics)
        write_json(reports_dir / "stages.json", _json_safe(stage_results))
        if comparison:
            write_comparison(reports_dir, comparison, quality,
                             ctx.cache.get("refinement") or None)
        if ctx.cache.get("image_report"):
            write_json(reports_dir / "image_quality.json", ctx.cache["image_report"])

        validation = validate_version(project, version)
        version.assets.reports["validation"] = "reports/validation_report.json"
        write_json(reports_dir / "validation_report.json", validation)
        if not validation.get("ok"):
            for issue in validation.get("errors", []):
                reporter.warning(f"validation: {issue}")

        project.update_version(version)
        result = {
            "version": version.id,
            "project": project.id,
            "status": "completed" if not failed else "completed_with_warnings",
            "quality": quality,
            "statistics": version.statistics,
            "outputs": _output_paths(project, version),
            "stages_failed": failed,
            "validation": {"ok": validation.get("ok"), "errors": validation.get("errors", []),
                           "warnings": validation.get("warnings", [])},
            "duration_s": round(version.duration_s, 2),
        }
        reporter.update(100, "pipeline complete", stage="done", overall=100.0)
        return result
    except Exception:
        try:
            project.delete_version(version.id, confirm=True)
        except Exception:  # pragma: no cover
            pass
        raise


#: Stages whose geometry is persisted so a later run can resume from them.
MESH_STAGE_FILES = ("mesh_reconstruction", "silhouette_refine", "mesh_cleanup",
                    "optimization", "uv", "topology")


def _inputs_from_dependencies(project: Any, spec: StageSpec) -> Dict[str, str]:
    """Artefacts a stage needs from the stages it depends on.

    They are recorded in its own checkpoint: a stage cannot be resumed without its
    inputs (a materials stage cannot restore itself if the texture maps it read were
    deleted), so a vanished *input* has to invalidate the dependent stage too.
    """
    needed: Dict[str, str] = {}
    for dependency in spec.requires:
        try:
            checkpoint = read_checkpoint(project, dependency)
        except Exception:  # pragma: no cover - defensive
            continue
        if checkpoint:
            needed.update(checkpoint.get("outputs") or {})
    return needed


def _version_files(project: Any, version: Any) -> set:
    """Project-relative paths of every file currently in a version directory."""
    if version is None:
        return set()
    try:
        root = project.version_path(version.id)
    except Exception:  # pragma: no cover - defensive
        return set()
    if not root.exists():
        return set()
    return {str(path.relative_to(project.root)) for path in root.rglob("*") if path.is_file()}


def stage_artefacts(project: Any, stage: str, *, produced: Optional[set] = None) -> Dict[str, str]:
    """Files a stage produced, relative to the project root.

    Recorded in the stage's checkpoint so :func:`checkpoint_valid` can detect a
    stage whose outputs were deleted or truncated (spec: corrupted-stage
    detection) instead of silently re-using a broken checkpoint.
    """
    directory = project.stage_dir(stage, create=False)
    if not directory.exists():
        return {}
    found: Dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "_checkpoint.json":
            relative = str(path.relative_to(project.root))
            found[relative] = relative
    # Version outputs (textures, rig, LODs, previews, final files) are recorded too:
    # a follow-up run resumes a stage by reading the *previous* version's files, and
    # a checkpoint must not claim outputs that were deleted in the meantime.
    for relative in sorted(produced or ()):
        found[relative] = relative
    return found


def _persist_mesh(ctx: PipelineContext, mesh: Any, name: Optional[str] = None) -> Optional[str]:
    """Write a stage's mesh next to its checkpoint so it can be resumed."""
    if mesh is None or len(getattr(mesh, "faces", [])) == 0:
        return None
    stage = name or ctx.current_stage
    if stage not in MESH_STAGE_FILES:
        return None
    target = ctx.project.stage_dir(stage) / "mesh.ply"
    try:
        with target.open("wb") as handle:
            mesh.export(handle, file_type="ply")
    except Exception as exc:  # pragma: no cover - persistence must never break a run
        ctx.reporter.warning(f"could not persist the mesh for stage '{stage}': {exc}")
        return None
    return str(target.relative_to(ctx.project.root))


def _mesh_stage_order() -> List[str]:
    """Geometry stages in pipeline order (used to pick the mesh to reload)."""
    return [spec.name for spec in STAGE_TABLE if spec.name in MESH_STAGE_FILES]


def _load_stage_mesh(ctx: PipelineContext) -> Any:
    """Persisted mesh for the stage being resumed (crash recovery).

    The stage's own surface is preferred; when it is missing (or the stage never
    persisted one) the closest earlier geometry stage is used, then the newest one.
    """
    order = _mesh_stage_order()
    current = getattr(ctx, "current_stage", "")
    if current in order:
        index = order.index(current)
        candidates = ([current] + list(reversed(order[:index]))
                      + list(reversed(order[index + 1:])))
    else:
        candidates = list(reversed(order))
    for stage in candidates:
        candidate = ctx.project.stage_dir(stage, create=False) / "mesh.ply"
        if candidate.exists():
            try:
                import trimesh

                mesh = trimesh.load(str(candidate), force="mesh", process=False)
                if mesh is not None and len(getattr(mesh, "faces", [])) > 0:
                    ctx.reporter.info(f"resumed geometry from '{stage}/mesh.ply' "
                                      f"({len(mesh.faces)} faces)")
                    return mesh
            except Exception as exc:  # pragma: no cover - fall through to the next stage
                ctx.reporter.warning(f"could not reload '{candidate.name}': {exc}")
    return None


def _recorded_version_dir(ctx: PipelineContext, checkpoint: Dict[str, Any],
                          kind: str, filename: str) -> Optional[Path]:
    """Directory of ``kind`` (textures/ rig/ materials/) recorded by a checkpoint.

    A resumed run creates a *new* version, so the artefacts to reload live in the
    version that produced the checkpoint - its paths are recorded in ``outputs``
    (the intermediates of the same stage are recorded next to them, so the lookup
    insists on the directory that really holds the wanted file).
    """
    root = Path(ctx.project.root)
    candidates: List[Path] = []
    for relative in (checkpoint.get("outputs") or {}):
        path = root / relative
        if path.exists() and path.parent.name == kind and path.name == filename:
            candidates.append(path.parent)
    # Fall back to the run's own version (a stage that produced its outputs twice,
    # or a checkpoint written before this bookkeeping existed).
    local = Path(ctx.version_dir) / kind
    if (local / filename).exists():
        candidates.append(local)
    for candidate in candidates:
        if (candidate / filename).exists():
            return candidate
    return None


def _restore_cache_from_disk(ctx: PipelineContext, spec: StageSpec,
                             checkpoint: Dict[str, Any]) -> None:
    """Reload the artefacts a cached stage produced into the in-memory cache.

    Re-using a checkpoint is only honest when the stage's outputs are actually
    back in the cache - otherwise the next stage would fail (or, worse, work on a
    stale object).  Anything that cannot be restored invalidates the checkpoint.
    """
    name = spec.name
    if name in {"camera_estimation", "silhouette_refine", "mesh_reconstruction"}:
        try:
            ctx.cache["rig"] = load_rig(ctx.project)
        except Exception as exc:  # pragma: no cover
            raise StageError(f"cached cameras could not be reloaded: {exc}", stage=name,
                             details={"hint": "delete intermediate/camera_estimation and retry"})
    if name in MESH_STAGE_FILES:
        mesh = _load_stage_mesh(ctx)
        if mesh is None:
            raise StageError(
                "the cached geometry for this stage is missing, so the run cannot resume "
                "from it", stage=name,
                details={"hint": "delete the stage's intermediate directory to force a rebuild"})
        ctx.cache["mesh"] = mesh
    if name == "validation":
        # The image-quality report feeds the final quality score: without it a routed
        # run would *report* a worse asset than the one it actually produced.
        stats = checkpoint.get("statistics") or {}
        if stats:
            ctx.cache["image_report"] = stats
    if name == "comparison":
        stats = checkpoint.get("statistics") or {}
        comparison = stats.get("comparison")
        if not isinstance(comparison, dict) or not comparison:
            raise StageError("the cached reference comparison is missing, so the run "
                             "cannot report honest quality numbers for the resumed model",
                             stage=name)
        ctx.cache["comparison"] = comparison
        ctx.cache["refinement"] = stats.get("refinement") or {}
    if name in {"uv", "texture", "materials"}:
        uvs_path = ctx.project.stage_dir("uv", create=False) / "uv.npz"
        if uvs_path.exists():
            try:
                ctx.cache["uvs"] = np.load(uvs_path)["uvs"]
            except Exception as exc:  # pragma: no cover
                raise StageError(f"cached UVs could not be reloaded: {exc}", stage=name)
    if name in {"texture", "materials"}:
        loaded = _restore_texture(ctx, checkpoint)
        if loaded is None:
            raise StageError("the cached texture maps are missing, so the run cannot resume",
                             stage=name)
        if name == "materials":
            if not _restore_material_assignment(ctx, checkpoint):
                raise StageError("the cached material assignment is missing, so the run "
                                 "cannot resume", stage=name)
    if name == "rigging":
        if not _restore_rig_data(ctx, checkpoint):
            raise StageError("the cached rig data is missing, so the run cannot resume",
                             stage=name)
    _attach_uvs(ctx)


def _materialise_version_outputs(ctx: PipelineContext, checkpoint: Dict[str, Any]) -> List[str]:
    """Copy a re-used stage's version files into this run's version directory.

    A new version is a complete snapshot of the asset, so re-using a cached stage
    must bring its outputs along (textures, rig, LODs, previews, final exports) -
    otherwise a fully cached re-run would produce a version containing nothing but
    reports, while still claiming every stage completed.
    """
    root = Path(ctx.project.root)
    target_root = Path(ctx.version_dir)
    copied: List[str] = []
    for relative in (checkpoint.get("outputs") or {}):
        parts = Path(relative).parts
        if len(parts) < 3 or parts[0] != "versions":
            continue
        source = root / relative
        if not source.is_file():
            continue
        destination = target_root.joinpath(*parts[2:])
        if destination.exists() and destination.stat().st_size == source.stat().st_size:
            continue
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        except OSError as exc:  # pragma: no cover - keep the run alive, report it
            ctx.reporter.warning(f"could not copy '{relative}' into the new version: {exc}")
            continue
        copied.append(str(destination.relative_to(target_root)))
    return copied


#: version sub-directory -> the asset pointer it restores on a resumed run.
_ASSET_DIRS = ("textures", "rig", "lod", "previews", "renders", "final", "reports",
               "materials", "pointcloud")


def _refresh_version_assets(project: Project, version: Version) -> None:
    """Point a version's asset list at the files that are actually in its directory.

    Stages write their own asset entries while they run; a version assembled from
    re-used checkpoints (or a version whose files were copied over) has to have its
    pointers rebuilt from disk, so ``version.json`` never advertises a missing file
    and never hides a present one.
    """
    base = project.version_path(version.id)
    if not base.is_dir():
        return
    assets = version.assets
    files = [path for path in base.rglob("*") if path.is_file()]
    known = set(assets.all_paths())

    def add(target: List[str], path: Path) -> None:
        relative = str(path.relative_to(base))
        if relative not in target:
            target.append(relative)

    for path in sorted(files):
        relative = str(path.relative_to(base))
        if relative in known or relative == "version.json":
            continue
        top = path.relative_to(base).parts[0]
        if top == "textures":
            assets.textures.setdefault(path.stem, relative)
        elif top == "rig":
            assets.rig.setdefault(path.stem, relative)
        elif top == "lod":
            assets.lods.setdefault(path.stem, relative)
        elif top == "previews":
            add(assets.previews, path)
        elif top == "renders":
            add(assets.renders, path)
        elif top == "final":
            assets.mesh.setdefault(path.suffix.lstrip(".") or path.stem, relative)
        elif top == "reports":
            assets.reports.setdefault(path.stem, relative)
        elif top == "materials" and path.name == "materials.json":
            assets.materials = assets.materials or relative
        elif top == "pointcloud":
            assets.pointcloud = assets.pointcloud or relative


def _attach_uvs(ctx: PipelineContext) -> None:
    """Give a restored mesh its UVs back.

    The exporters read texture coordinates from ``mesh.visual.uv``; a mesh loaded
    from disk has none, so a resumed run would export a GLB without TEXCOORD_0
    while still shipping texture maps - a silently wrong asset.
    """
    mesh = ctx.cache.get("mesh")
    uvs = ctx.cache.get("uvs")
    if mesh is None or uvs is None:
        return
    try:
        import trimesh

        uvs = np.asarray(uvs, dtype=np.float64)
        if len(uvs) != len(mesh.vertices):
            return
        existing = getattr(getattr(mesh, "visual", None), "uv", None)
        if existing is not None and len(np.asarray(existing)) == len(uvs):
            return
        mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
    except Exception as exc:  # pragma: no cover - UV attachment is best effort
        ctx.reporter.warning(f"could not re-attach UVs to the resumed mesh: {exc}")


def _restore_texture(ctx: PipelineContext, checkpoint: Dict[str, Any]) -> Any:
    """Rebuild the texture cache from the PNG maps a previous run wrote."""
    from ..engine.textures.project import TextureResult

    directory = _recorded_version_dir(ctx, checkpoint, "textures", "basecolor.png")
    if directory is None:
        return None
    wanted = {"basecolor": "base_color", "normal": "normal", "roughness": "roughness",
              "metallic": "metallic", "ao": "ao"}
    arrays: Dict[str, np.ndarray] = {}
    for filename, attribute in wanted.items():
        path = directory / f"{filename}.png"
        if not path.exists():
            return None
        from PIL import Image

        arrays[attribute] = np.asarray(Image.open(path))
    ctx.cache["texture"] = TextureResult(
        base_color=arrays["base_color"], normal=arrays["normal"],
        roughness=arrays["roughness"], metallic=arrays["metallic"], ao=arrays["ao"],
        covered_mask=arrays["base_color"].sum(axis=2) > 1e-6,
        statistics={"resumed": True},
    )
    ctx.reporter.info("resumed texture maps from the version's textures/ directory")
    return ctx.cache["texture"]


def _restore_material_assignment(ctx: PipelineContext, checkpoint: Dict[str, Any]) -> bool:
    """Reload the material assignment a previous run wrote; False when it is gone."""
    from ..engine.materials.library import MaterialAssignment, get_material

    directory = _recorded_version_dir(ctx, checkpoint, "materials", "materials.json")
    path = (directory / "materials.json") if directory is not None else None
    data = read_json(path) if path is not None and path.exists() else None
    if not isinstance(data, dict):
        return False
    entry = (data.get("assignment") or {}).get("material") or {}
    name = entry.get("name") or data.get("library_entry") or "plastic"
    material = get_material(name)
    ctx.cache["material_assignment"] = MaterialAssignment(
        material=material, confidence=float((data.get("assignment") or {}).get("confidence", 0.5)),
        evidence={"reason": "resumed from materials.json"})
    ctx.reporter.info(f"resumed material assignment '{material.name}'")
    return True


def _restore_rig_data(ctx: PipelineContext, checkpoint: Dict[str, Any]) -> bool:
    """Reload skeleton + skin weights a previous run wrote; False when they are gone."""
    directory = _recorded_version_dir(ctx, checkpoint, "rig", "rig.json")
    if directory is None:
        return False
    rig_path = directory / "rig.json"
    weights_path = directory / "skin_weights.npz"
    if not rig_path.exists() or not weights_path.exists():
        return False
    data = read_json(rig_path) or {}
    with np.load(weights_path, allow_pickle=True) as handle:
        weights = handle["weights"]
        joints = [str(j) for j in handle["joints"]]
    ctx.cache["rig_data"] = {"weights": weights, "joints": joints, "rig": data}
    ctx.reporter.info(f"resumed rig ({len(joints)} bones)")
    return True


def _json_safe(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {str(k): _json_safe(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_json_safe(v) for v in payload]
    if isinstance(payload, (np.integer, np.floating)):
        return payload.item()
    if isinstance(payload, np.ndarray):
        return payload.tolist()
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, (str, int, float, bool)) or payload is None:
        return payload
    return str(payload)


def _build_statistics(ctx: PipelineContext) -> Dict[str, Any]:
    stats = dict(ctx.cache.get("mesh_stats") or {})
    lods = ctx.stage_reports.get("lod") or {}
    texture = (ctx.stage_reports.get("texture") or {})
    uv = ctx.stage_reports.get("uv") or {}
    rig = ctx.stage_reports.get("rigging") or {}
    stats.update({
        "vertices": stats.get("vertices", 0),
        "triangles": stats.get("faces", 0),
        "faces": stats.get("faces", 0),
        "mesh_health": (stats.get("health_score")
                        if stats.get("health_score") is not None
                        else mesh_health_score(stats)),
        "texture_resolution": texture.get("resolution"),
        "uv_islands": uv.get("islands"),
        "uv_distortion": uv.get("estimated_distortion"),
        "lod_levels": len(lods.get("levels", []) or []),
        "lod_faces": [lvl.get("faces") for lvl in (lods.get("levels") or [])],
        "rig_bones": len((rig.get("rig") or {}).get("bones", []) or []),
        "watertight": stats.get("watertight"),
        "components": stats.get("components"),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    return _json_safe(stats)


def _backend_report(ctx: PipelineContext) -> Dict[str, Any]:
    from ..engine.geometry.simplify import available_backends
    from ..engine.textures.uv import available_uv_backends
    from ..engine.backends.detect import detect_backends

    return {
        "decimation": available_backends(),
        "uv": available_uv_backends(),
        "external": detect_backends(),
    }


def _output_paths(project: Project, version: Version) -> Dict[str, Any]:
    base = project.version_path(version.id)

    def resolve(rel: str) -> str:
        return str(Path(rel)) if Path(rel).is_absolute() else str(base / rel)

    outputs: Dict[str, Any] = {
        "version_dir": str(base),
        "mesh": {fmt: resolve(rel) for fmt, rel in version.assets.mesh.items()},
        "textures": {name: resolve(rel) for name, rel in version.assets.textures.items()},
        "lods": {name: resolve(rel) for name, rel in version.assets.lods.items()},
        "rig": {name: resolve(rel) for name, rel in version.assets.rig.items()},
        "previews": [resolve(p) for p in version.assets.previews],
        "renders": [resolve(p) for p in version.assets.renders],
        "reports": {name: resolve(rel) for name, rel in version.assets.reports.items()},
    }
    if version.assets.materials:
        outputs["materials"] = resolve(version.assets.materials)
    if version.assets.pointcloud:
        outputs["pointcloud"] = resolve(version.assets.pointcloud)
    return outputs


# --------------------------------------------------------------------------
# Introspection helpers (used by the CLI, the API and the agent manifest)
# --------------------------------------------------------------------------
def available_presets() -> List[Dict[str, Any]]:
    """Return the shipped presets with their descriptions and budgets."""
    from .resources import mode_budget, profile_hardware

    profile = profile_hardware()
    presets: List[Dict[str, Any]] = []
    for name in ("draft", "standard", "high", "ultra", "game_ready", "cinematic"):
        try:
            data = load_preset(name)
        except Exception:  # pragma: no cover - a missing preset must not break the CLI
            data = {"name": name, "params": {}}
        mode = str(data.get("performance_mode", "balanced"))
        presets.append({
            "name": name,
            "description": data.get("description", ""),
            "performance_mode": mode,
            "params": data.get("params", {}),
            "policy": data.get("policy", {}),
            "budget": mode_budget(mode, profile),
        })
    return presets


def describe_stages() -> List[Dict[str, Any]]:
    """List the pipeline stages with their dependencies and whether they are optional."""
    from ..agent.manifest import STAGE_DESCRIPTIONS

    return [{"name": spec.name, "description": STAGE_DESCRIPTIONS.get(spec.name, ""),
             "requires": list(spec.requires), "optional": bool(spec.optional)}
            for spec in STAGE_TABLE]

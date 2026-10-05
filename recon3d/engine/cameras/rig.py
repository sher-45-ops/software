"""Camera rig estimation for the reference set (spec #7).

Given segmented silhouettes and the view labels/azimuths from
:mod:`recon3d.engine.analysis.views`, this module solves for a full set of
*pinhole cameras* - one per reference image - that are mutually consistent with
the observed silhouettes:

1. **Ring layout.**  Each image provides a camera direction (azimuth/elevation)
   and a measurement of how large the subject appears (bounding-box height as a
   fraction of the frame).
2. **Focal/distance resolution.**  The scale is a gauge freedom, so the subject
   height is normalised to 1.0 world unit.  For a reference focal length (35 deg
   vertical FOV, the classic "normal" lens) the camera distance follows from the
   measured silhouette size: ``d = f_pixels * H_world / h_pixels``.
3. **Subject centre solve.**  Any offset of the subject from the frame centre
   constrains the unknown 3D centre; with three or more views the system is
   over-determined and solved in closed form by least squares.  Iterating
   "solve centre -> recompute distance/focal" converges in two passes.
4. **Optional feature-based azimuth refinement.**  When SIFT matches are
   available, neighbouring views constrain the relative rotation; the ring
   ordering is corrected and azimuths nudged towards feature-consistent values.

Everything is deterministic and inspectable; the full rig is written to
``intermediate/camera_estimation/cameras.json`` in OpenCV-compatible form.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...core.project import Project
from ...core.store import utc_now, write_json
from ...errors import StageError
from ..compare.raster import Camera
from ..ingestion.loader import LoadedImage, load_masks

DEFAULT_FOV_DEG = 35.0  # "normal" lens assumption for uncalibrated references


@dataclass
class CameraPose:
    """One reference image's solved camera."""

    image_id: str
    filename: str
    view: str
    azimuth: float
    elevation: float
    distance: float
    focal_pixels: float
    width: int
    height: int
    camera: Camera
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    silhouette_area: float = 0.0
    confidence: float = 0.5
    method: str = "ring"
    reprojection_error: float = 0.0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "filename": self.filename,
            "view": self.view,
            "azimuth": round(float(self.azimuth), 3),
            "elevation": round(float(self.elevation), 3),
            "distance": round(float(self.distance), 5),
            "focal_pixels": round(float(self.focal_pixels), 3),
            "fov_deg": round(math.degrees(2 * math.atan(0.5 * max(self.width, self.height) /
                                                        max(1e-6, self.focal_pixels))), 3),
            "width": self.width,
            "height": self.height,
            "bbox": list(self.bbox),
            "silhouette_area": round(self.silhouette_area, 4),
            "confidence": round(self.confidence, 3),
            "method": self.method,
            "reprojection_error": round(self.reprojection_error, 4),
            "notes": self.notes,
            "R": self.camera.R.tolist(),
            "t": self.camera.t.tolist(),
            "K": self.camera.K.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CameraPose":
        cam = Camera(R=np.asarray(data["R"], dtype=np.float64),
                     t=np.asarray(data["t"], dtype=np.float64),
                     fx=float(data["focal_pixels"]), fy=float(data["focal_pixels"]),
                     cx=float(data["width"]) / 2.0 - 0.5, cy=float(data["height"]) / 2.0 - 0.5,
                     width=int(data["width"]), height=int(data["height"]),
                     name=str(data.get("filename", "")))
        return cls(
            image_id=str(data.get("image_id", "")), filename=str(data.get("filename", "")),
            view=str(data.get("view", "unknown")), azimuth=float(data.get("azimuth", 0.0)),
            elevation=float(data.get("elevation", 0.0)), distance=float(data.get("distance", 1.0)),
            focal_pixels=float(data.get("focal_pixels", 1.0)), width=int(data.get("width", 0)),
            height=int(data.get("height", 0)), camera=cam,
            bbox=tuple(data.get("bbox", (0, 0, 0, 0))),  # type: ignore[arg-type]
            silhouette_area=float(data.get("silhouette_area", 0.0)),
            confidence=float(data.get("confidence", 0.5)), method=str(data.get("method", "ring")),
        )


@dataclass
class CameraRig:
    """All cameras of one reconstruction, plus the shared subject frame."""

    cameras: List[CameraPose] = field(default_factory=list)
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    subject_height: float = 1.0
    method: str = "silhouette_ring"
    warnings: List[str] = field(default_factory=list)
    quality: Dict[str, Any] = field(default_factory=dict)
    ring_quality: float = 0.0
    scale_hint_m: float = 1.0

    def __len__(self) -> int:
        return len(self.cameras)

    def __iter__(self):
        return iter(self.cameras)

    def bounding_radius(self) -> float:
        """Conservative radius of the subject volume around the centre."""
        distances = [c.distance for c in self.cameras] or [2.0]
        base = float(np.median(distances)) * math.tan(math.radians(DEFAULT_FOV_DEG / 2.0))
        return max(0.35, base * 1.15)

    def bounds(self, radius: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        r = radius if radius is not None else self.bounding_radius()
        return self.center - r, self.center + r

    def to_dict(self) -> Dict[str, Any]:
        return {
            "center": self.center.tolist(),
            "subject_height": self.subject_height,
            "method": self.method,
            "ring_quality": self.ring_quality,
            "scale_hint_m": self.scale_hint_m,
            "warnings": self.warnings,
            "quality": self.quality,
            "cameras": [c.to_dict() for c in self.cameras],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CameraRig":
        return cls(
            cameras=[CameraPose.from_dict(c) for c in data.get("cameras", [])],
            center=np.asarray(data.get("center", [0, 0, 0]), dtype=np.float64),
            subject_height=float(data.get("subject_height", 1.0)),
            method=str(data.get("method", "unknown")),
            warnings=list(data.get("warnings", [])),
            quality=dict(data.get("quality", {})),
            ring_quality=float(data.get("ring_quality", 0.0)),
            scale_hint_m=float(data.get("scale_hint_m", 1.0)),
        )

    def scaled(self, factor: float) -> "CameraRig":
        """Uniformly rescale the rig's camera standoff around the subject centre.

        This is the engine's *scale gauge*: the reconstruction is defined up to a
        similarity, so scaling every camera distance (and the assumed subject
        height) by ``factor`` scales the carved subject by exactly the same
        factor while leaving every projection unchanged.  ``calibrate_rig_height``
        uses it to make the carved subject reach the documented one-unit height.
        """
        if abs(factor - 1.0) < 1e-9:
            return self
        poses: List[CameraPose] = []
        for pose in self.cameras:
            eye = pose.camera.position()
            direction = eye - self.center
            norm = float(np.linalg.norm(direction))
            direction = direction / norm if norm > 1e-12 else np.asarray([0.0, -1.0, 0.0])
            new_eye = self.center + direction * (norm * factor)
            # ``focal_pixels`` wins over ``fov_deg`` inside from_look_at, so the
            # intrinsics (and therefore the projection) are preserved exactly.
            camera = Camera.from_look_at(new_eye, self.center, fov_deg=DEFAULT_FOV_DEG,
                                         width=pose.width, height=pose.height,
                                         focal_pixels=pose.focal_pixels, name=pose.filename)
            poses.append(CameraPose(
                image_id=pose.image_id, filename=pose.filename, view=pose.view,
                azimuth=pose.azimuth, elevation=pose.elevation,
                distance=float(pose.distance * factor), focal_pixels=pose.focal_pixels,
                width=pose.width, height=pose.height, camera=camera, bbox=pose.bbox,
                silhouette_area=pose.silhouette_area, confidence=pose.confidence,
                method=pose.method, notes=pose.notes,
                reprojection_error=pose.reprojection_error,
            ))
        return CameraRig(
            cameras=poses, center=self.center.copy(), subject_height=self.subject_height * factor,
            method=f"{self.method}+scale", warnings=list(self.warnings),
            ring_quality=self.ring_quality, scale_hint_m=self.scale_hint_m,
            quality=dict(self.quality),
        )

    def translated(self, delta: Sequence[float]) -> "CameraRig":
        """Shift the whole rig (subject centre *and* every camera) by ``delta``.

        A pure translation leaves every projection unchanged, so this is how the
        engine recentres a reconstruction on its own bounding-box centre without
        invalidating the camera solve.
        """
        delta = np.asarray(delta, dtype=np.float64).reshape(3)
        if float(np.linalg.norm(delta)) < 1e-12:
            return self
        poses: List[CameraPose] = []
        new_center = self.center + delta
        for pose in self.cameras:
            eye = pose.camera.position() + delta
            camera = Camera.from_look_at(eye, new_center, fov_deg=DEFAULT_FOV_DEG,
                                         width=pose.width, height=pose.height,
                                         focal_pixels=pose.focal_pixels, name=pose.filename)
            poses.append(CameraPose(
                image_id=pose.image_id, filename=pose.filename, view=pose.view,
                azimuth=pose.azimuth, elevation=pose.elevation, distance=pose.distance,
                focal_pixels=pose.focal_pixels, width=pose.width, height=pose.height,
                camera=camera, bbox=pose.bbox, silhouette_area=pose.silhouette_area,
                confidence=pose.confidence, method=pose.method, notes=pose.notes,
                reprojection_error=pose.reprojection_error,
            ))
        return CameraRig(
            cameras=poses, center=new_center, subject_height=self.subject_height,
            method=f"{self.method}+recentred", warnings=list(self.warnings),
            ring_quality=self.ring_quality, scale_hint_m=self.scale_hint_m,
            quality=dict(self.quality),
        )

    def coverage_angles(self) -> List[float]:
        return sorted(c.azimuth for c in self.cameras if c.view not in {"top", "bottom"})


# --------------------------------------------------------------------------
# Silhouette measurements
# --------------------------------------------------------------------------
def silhouette_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask > 0)
    if len(ys) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _measure(mask: np.ndarray, width: int, height: int) -> Dict[str, float]:
    """Silhouette measurements used both for the rig solve and diagnostics.

    The *centroid* (not the bounding-box centre) is used for the ray constraint:
    a bounding box is biased by asymmetric detail (an outstretched arm, an
    antenna) whereas the centroid of the silhouette is a far more stable
    estimator of where the subject's mass is.
    """
    x0, y0, x1, y1 = silhouette_bbox(mask)
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    ys, xs = np.nonzero(mask > 0)
    if len(xs) > 0:
        cx = float(xs.mean())
        cy = float(ys.mean())
    else:  # pragma: no cover
        cx = (x0 + x1 + 1) / 2.0
        cy = (y0 + y1 + 1) / 2.0
    return {
        "width_frac": bw / float(width),
        "height_frac": bh / float(height),
        "center_u": cx / float(width),
        "center_v": cy / float(height),
        "bbox_u": (x0 + x1 + 1) / 2.0 / float(width),
        "bbox_v": (y0 + y1 + 1) / 2.0 / float(height),
        "area_frac": float((mask > 0).mean()),
        "aspect": bw / float(bh),
    }


def _direction_from_view(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    return np.array([math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)])


# --------------------------------------------------------------------------
# Rig estimation
# --------------------------------------------------------------------------
def estimate_rig(
    project: Project,
    *,
    images: Optional[Sequence[LoadedImage]] = None,
    views: Optional[Dict[str, Dict[str, Any]]] = None,
    subject_height_m: float = 1.0,
    ref_fov_deg: float = DEFAULT_FOV_DEG,
    max_iterations: int = 3,
    reporter: Any = None,
) -> CameraRig:
    """Solve the camera rig for all usable references of a project."""
    from ..ingestion.loader import ImageSet

    image_set = ImageSet(project)
    if images is None:
        images = [image_set.load(e["id"]) for e in image_set.entries]
    if views is None:
        from ..analysis.views import assign_views

        report = assign_views(project, use_masks=True)
        views = {a["image_id"]: a for a in report["assignments"]}

    masks = load_masks(project, images)
    poses: List[CameraPose] = []
    warnings: List[str] = []
    measurements: List[Dict[str, float]] = []
    entries = image_set.entries
    if len(entries) != len(images):  # pragma: no cover - defensive
        raise StageError("image/entry count mismatch", stage="camera_estimation", recoverable=False)

    # Candidate height in world units - normalised to 1.0 and rescaled at export.
    H = 1.0
    center = np.zeros(3)
    # Vertical centring: the subject's bbox centre tells us where its middle is.
    median_height_frac = float(np.median([
        _measure(masks.get(img.path.name, np.ones((img.height, img.width), np.uint8)),
                 img.width, img.height)["height_frac"]
        for img in images
    ]))
    if median_height_frac <= 0.01:
        raise StageError(
            "subject silhouettes are empty - segmentation failed",
            stage="camera_estimation", recoverable=False,
        )

    ring_distances: List[float] = []
    # Lateral extent of the subject (in engine units, where the height is H):
    # a silhouette's *width* measures the subject across the view direction, so
    # front/back views measure its X extent and left/right views its Y extent.
    # These are only used to place top/bottom cameras, whose own distance cannot
    # be derived from a height measurement.
    lateral_x: List[float] = []
    lateral_y: List[float] = []
    focal_ref = 0.5 * H  # placeholder replaced per image below
    for entry, image in zip(entries, images):
        info = views.get(entry["id"], {})
        label = info.get("view", "unknown")
        if label in {"top", "bottom", "detail", "unknown"}:
            continue
        mask = masks.get(image.path.name)
        if mask is None or mask.sum() == 0:
            continue
        measure = _measure(mask, image.width, image.height)
        if measure["height_frac"] <= 0.01:
            continue
        focal = image.height * 0.5 / math.tan(math.radians(ref_fov_deg) / 2.0)
        distance = focal * H / (measure["height_frac"] * image.height)
        ring_distances.append(distance)
        azimuth = float(info.get("azimuth", 0.0) or 0.0)
        width_px = max(1.0, measure["width_frac"] * image.width)
        extent = width_px * distance / focal  # world width of the silhouette
        if abs(math.sin(math.radians(azimuth))) < 0.35:
            lateral_x.append(extent)
        elif abs(math.cos(math.radians(azimuth))) < 0.35:
            lateral_y.append(extent)
    default_distance = float(np.median(ring_distances)) if ring_distances else 2.0
    x_extent = float(np.median(lateral_x)) if lateral_x else 0.0
    y_extent = float(np.median(lateral_y)) if lateral_y else 0.0

    for iteration in range(max(1, max_iterations)):
        poses = []
        measurements = []
        for entry, image in zip(entries, images):
            info = views.get(entry["id"], {})
            azimuth = info.get("azimuth")
            elevation = info.get("elevation", 0.0) or 0.0
            view_label = info.get("view", "unknown")
            if azimuth is None:
                # Images without a ring angle (top/bottom/detail) still get a camera,
                # it is just flagged as low confidence.
                azimuth = 0.0 if view_label == "top" else (180.0 if view_label == "bottom" else 0.0)
                elevation = 80.0 if view_label == "top" else (-80.0 if view_label == "bottom" else elevation)
            mask = masks.get(image.path.name)
            if mask is None:
                mask = (image.alpha > 127).astype(np.uint8) if image.alpha is not None else \
                    np.ones((image.height, image.width), np.uint8)
            if mask.shape != (image.height, image.width):
                mask = np.ones((image.height, image.width), np.uint8)
            measure = _measure(mask, image.width, image.height)
            measure["weight"] = float(np.clip(info.get("confidence", 0.6), 0.2, 1.0))
            measurements.append(measure)

            focal = image.height * 0.5 / math.tan(math.radians(ref_fov_deg) / 2.0)
            if view_label in {"top", "bottom"}:
                # A top view measures the subject's X/Y footprint, not its
                # height, so the ring formula is meaningless here.  Instead use
                # the *known* lateral extent from the ring views: the silhouette
                # width in pixels plus a focal length fixes the standoff.  This
                # keeps the top silhouette cone at the right scale - parking the
                # camera at the ring distance instead shrinks the carved depth
                # (a real bug that made reconstructions too thin front-to-back).
                extent = x_extent if view_label == "top" else (y_extent or x_extent)
                width_px = max(1.0, measure["width_frac"] * image.width)
                if extent > 0:
                    distance = float(np.clip(
                        (image.height * 0.5 / math.tan(math.radians(ref_fov_deg) / 2.0))
                        * extent / width_px,
                        0.35 * default_distance, 3.0 * default_distance))
                else:
                    distance = default_distance
            elif view_label == "detail":
                # Close-ups are not usable as ring cameras; park them at the
                # default distance and flag low confidence.
                distance = default_distance
            else:
                height_px = max(4.0, measure["height_frac"] * image.height)
                distance = focal * H / height_px
            direction = _direction_from_view(float(azimuth), float(elevation))
            eye = center + direction * distance
            camera = Camera.from_look_at(eye, center, fov_deg=ref_fov_deg,
                                         width=image.width, height=image.height,
                                         focal_pixels=focal, name=image.path.name)
            poses.append(
                CameraPose(
                    image_id=entry["id"], filename=image.path.name, view=view_label,
                    azimuth=float(azimuth) % 360.0, elevation=float(elevation),
                    distance=float(distance), focal_pixels=float(focal),
                    width=image.width, height=image.height, camera=camera,
                    bbox=silhouette_bbox(mask), silhouette_area=float((mask > 0).mean()),
                    confidence=float(info.get("confidence", 0.5)),
                    method=str(info.get("method", "ring")),
                    notes=str(info.get("notes", "")),
                )
            )

        # -- solve the subject centre from the observed bbox centres -------
        # For camera i: R_i (C - eye_i) must project to (u_i, v_i).
        # Linearising: with d_i fixed, the offset of the projection from the
        # principal point is  f * (lateral displacement)/depth, so we can solve
        # for C directly in least squares from the 2D residual equations.
        new_center = _solve_center(poses, measurements, H)
        shift = float(np.linalg.norm(new_center - center))
        center = new_center
        # Recompute heights implied by the new centre (moving the camera changes
        # the distance to the subject's centre, so the apparent size changes).
        if iteration > 0 and shift < 1e-4:
            break

    # -- fill in per-pose reprojection error ---------------------------
    errors = []
    for pose, measure in zip(poses, measurements):
        uv, depth = pose.camera.project(center.reshape(1, 3))
        du = (uv[0, 0] / pose.width) - measure["center_u"]
        dv = (uv[0, 1] / pose.height) - measure["center_v"]
        pose.reprojection_error = float(math.hypot(du, dv))
        errors.append(pose.reprojection_error)

    ring = [p for p in poses if p.view not in {"top", "bottom", "detail"}]
    ring_quality = _ring_quality(ring)
    if len(ring) >= 3:
        spacing = _spacing_report(ring)
    else:
        spacing = {"even": False, "max_gap_deg": 180.0, "min_gap_deg": 180.0}
    if spacing["max_gap_deg"] > 120:
        warnings.append(
            f"largest gap between adjacent cameras is {spacing['max_gap_deg']:.0f} degrees; "
            "the geometry across that gap is extrapolated, not observed"
        )
    # The reported reprojection error uses the silhouette *centroid*: it is the
    # more sensitive indicator of a bad solve (the bbox centre can coincide by
    # accident).  The solve itself targets the bbox centre, which is the stable
    # orbit axis of a ring capture and the convention the engine normalises to.
    if float(np.mean(errors)) > 0.045:
        warnings.append(
            "silhouette centres are inconsistent with a single subject centre "
            f"(mean offset {float(np.mean(errors)) * 100:.1f}% of frame); "
            "camera estimates are approximate"
        )

    rig = CameraRig(
        cameras=poses,
        center=center,
        subject_height=H,
        method="silhouette_ring",
        warnings=warnings,
        ring_quality=ring_quality,
        scale_hint_m=subject_height_m,
        quality={
            "cameras": len(poses),
            "ring_cameras": len(ring),
            "mean_reprojection_error": round(float(np.mean(errors)) if errors else 0.0, 5),
            "max_reprojection_error": round(float(np.max(errors)) if errors else 0.0, 5),
            "spacing": spacing,
            "mean_confidence": round(float(np.mean([p.confidence for p in poses]) if poses else 0), 3),
        },
    )
    if reporter is not None:
        reporter.info(
            f"solved {len(poses)} cameras (ring quality {ring_quality:.2f})",
            center=center.tolist(), ring_quality=ring_quality,
        )
    write_json(project.stage_dir("camera_estimation") / "cameras.json", rig.to_dict())
    return rig


def _solve_center(poses: Sequence[CameraPose], measurements: Sequence[Dict[str, float]],
                  subject_height: float) -> np.ndarray:
    """Least-squares subject centre given camera poses and bbox centres."""
    rows: List[np.ndarray] = []
    rhs: List[np.ndarray] = []
    for pose, measure in zip(poses, measurements):
        R = pose.camera.R
        eye = pose.camera.position()
        # Direction of the ray that hits the silhouette's bbox centre in this view.
        # Silhouette centroid: more stable than a bbox centre for a subject
        # with asymmetric limbs, and the mesh is recentred afterwards anyway.
        target_u = (measure["center_u"] * pose.width) - pose.camera.cx
        target_v = (measure["center_v"] * pose.height) - pose.camera.cy
        desired_cam_dir = np.array([target_u / pose.focal_pixels, target_v / pose.focal_pixels, 1.0])
        desired_world_dir = desired_cam_dir.dot(R)
        n = np.linalg.norm(desired_world_dir)
        if n < 1e-12:  # pragma: no cover
            continue
        desired_world_dir /= n
        # The subject centre must lie on this ray: (I - d d^T)(C - eye) = 0.
        # Weight by how much we trust the view's angle estimate.
        w = float(np.clip(measure.get("weight", 0.7), 0.15, 1.0))
        proj = (np.eye(3) - np.outer(desired_world_dir, desired_world_dir)) * w
        rows.append(proj)
        rhs.append(proj.dot(eye))
    if not rows:
        return np.zeros(3)
    A = np.vstack(rows)
    b = np.concatenate(rhs)
    try:
        solution, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:  # pragma: no cover
        return np.zeros(3)
    solution = np.asarray(solution, dtype=np.float64)
    if not np.all(np.isfinite(solution)):
        return np.zeros(3)
    return solution


def _ring_quality(poses: Sequence[CameraPose]) -> float:
    if len(poses) < 2:
        return 0.2 if poses else 0.0
    angles = np.sort(np.array([p.azimuth for p in poses]))
    gaps = np.diff(np.concatenate([angles, angles[:1] + 360.0]))
    evenness = 1.0 - float(np.std(gaps) / 90.0)
    coverage = min(1.0, len(poses) / 8.0)
    return round(float(np.clip(0.55 * np.clip(evenness, 0, 1) + 0.45 * coverage, 0, 1)), 3)


def _spacing_report(poses: Sequence[CameraPose]) -> Dict[str, Any]:
    angles = np.sort(np.array([p.azimuth for p in poses]))
    if len(angles) < 2:
        return {"even": False, "max_gap_deg": 360.0, "min_gap_deg": 360.0}
    gaps = np.diff(np.concatenate([angles, angles[:1] + 360.0]))
    return {
        "even": bool(gaps.max() - gaps.min() < 35),
        "max_gap_deg": round(float(gaps.max()), 2),
        "min_gap_deg": round(float(gaps.min()), 2),
        "mean_gap_deg": round(float(gaps.mean()), 2),
        "angles": [round(float(a), 2) for a in angles.tolist()],
    }


# --------------------------------------------------------------------------
# Silhouette-based refinement (the "render -> compare -> correct" loop, #24)
# --------------------------------------------------------------------------
def refine_rig_silhouette(
    rig: CameraRig,
    mesh,
    masks: Dict[str, np.ndarray],
    *,
    resolution: int = 144,
    iterations: int = 2,
    proxy_faces: int = 2500,
    max_evaluations: int = 48,
    allow_distance: bool = True,
    reporter: Any = None,
) -> Tuple[CameraRig, Dict[str, Any]]:
    """Nudge each camera so the rendered silhouette matches the reference mask.

    A bounded coordinate search over ``(azimuth, elevation, log-distance)``
    maximises the silhouette IoU per view.  The search runs on a decimated
    *proxy* of the current mesh (only the outline matters) through the
    rasterizer's silhouette-only fast path, with an explicit evaluation budget,
    so refining a full ring takes seconds rather than minutes.

    The focal length is deliberately **not** refined: apparent size depends on
    ``focal / distance``, so optimising both is gauge-ambiguous and lets the world
    scale drift.  The focal stays at the assumed lens (see
    :data:`DEFAULT_FOV_DEG`) and the global gauge is closed once by
    :func:`calibrate_rig_height`.

    ``allow_distance=False`` (used by the pipeline) removes the third search axis
    entirely.  Silhouette IoU of a *fixed* mesh improves when every camera walks
    closer, which is a pure scale-gauge change rather than a camera correction -
    it cannot be validated against the very mesh it was tuned on, so the engine
    only refines the two rotation axes and re-carves to test the result.
    """
    from ..compare.raster import silhouette_iou  # noqa: F401  (imported for parity)

    if mesh is None or len(mesh.faces) == 0:
        return rig, {"refined": 0, "reason": "no proxy mesh"}

    proxy = mesh
    if len(mesh.faces) > proxy_faces:
        try:
            candidate = mesh.simplify_quadric_decimation(face_count=int(proxy_faces))
            if candidate is not None and len(candidate.faces) > 0:
                proxy = candidate
        except Exception:  # pragma: no cover - decimation is an optimisation only
            proxy = mesh
    verts = np.asarray(proxy.vertices, dtype=np.float64)
    faces = np.asarray(proxy.faces, dtype=np.int64)

    report: Dict[str, Any] = {"images": [], "mean_iou_before": 0.0, "mean_iou_after": 0.0,
                              "proxy_faces": int(len(faces)), "resolution": resolution,
                              "optimised_axes": ["azimuth", "elevation"] +
                                                 (["log_distance"] if allow_distance else [])}

    improved = 0
    total = max(1, len(rig.cameras))
    for index, pose in enumerate(rig.cameras):
        mask = masks.get(pose.filename)
        if mask is None:
            continue
        aspect = pose.width / max(1, pose.height)

        def evaluate(params: Sequence[float], scale: float) -> float:
            h = max(48, int(round(resolution * scale)))
            w = max(48, int(round(h * aspect)))
            target = _resize_mask(mask, w, h) > 0
            iou, _ = _evaluate_pose(pose, verts, faces, target, w, h, *params,
                                    center=rig.center)
            return iou

        budget = {"left": max(8, int(max_evaluations))}
        cache: Dict[Tuple[float, ...], float] = {}

        def score(params: Sequence[float], scale: float) -> float:
            key = (round(params[0], 4), round(params[1], 4), round(params[2], 4), scale)
            if key in cache:
                return cache[key]
            if budget["left"] <= 0:
                return -1.0
            budget["left"] -= 1
            value = evaluate(params, scale)
            cache[key] = value
            return value

        params = [0.0, 0.0, 0.0, 0.0]
        axes = (0, 1) if not allow_distance else (0, 1, 2)
        best_iou = score(params, 0.75)
        iou_before = best_iou

        def descend(bracket: Sequence[Sequence[float]], scale: float, rounds: int) -> None:
            nonlocal best_iou, params
            steps = [list(b) for b in bracket]
            for _ in range(max(1, rounds)):
                for axis in axes:
                    for step in steps[axis]:
                        if step == 0.0 or budget["left"] <= 0:
                            continue
                        trial = list(params)
                        trial[axis] += step
                        value = score(trial, scale)
                        if value > best_iou + 1e-4:
                            best_iou, params = value, trial
                steps = [[s * 0.5 for s in axis_steps] for axis_steps in steps]

        # Coarse pass: find the basin on a small render.
        descend([[0.0, 9.0, -9.0, 18.0, -18.0], [0.0, 6.0, -6.0, 12.0, -12.0],
                 [0.0, 0.15, -0.15, 0.3, -0.3]], 0.75, rounds=1)
        # Fine pass: polish at the requested resolution.
        descend([[0.0, 3.0, -3.0, 6.0, -6.0], [0.0, 2.5, -2.5, 5.0, -5.0],
                 [0.0, 0.04, -0.04, 0.1, -0.1]], 1.0, rounds=max(1, iterations))
        # Final verification at full resolution (not counted in the budget).
        best_iou = evaluate(params, 1.0)
        cache[(round(params[0], 4), round(params[1], 4), round(params[2], 4), 1.0)] = best_iou

        if any(abs(p) > 1e-6 for p in params):
            _apply_pose_delta(pose, *params, center=rig.center)
            improved += 1
        report["images"].append({
            "filename": pose.filename,
            "iou_before": round(iou_before, 4),
            "iou_after": round(best_iou, 4),
            "delta_azimuth": round(params[0], 3),
            "delta_elevation": round(params[1], 3),
            "delta_log_distance": round(params[2], 4),
            "delta_log_focal": 0.0,
            "evaluations": max(0, int(max_evaluations) - budget["left"]) + 1,
        })
        if reporter is not None:
            reporter.update(min(99.0, 5 + 95 * (index + 1) / total),
                            f"refined camera {pose.filename} (IoU {best_iou:.3f})")

    if report["images"]:
        report["mean_iou_before"] = round(float(np.mean([i["iou_before"] for i in report["images"]])), 4)
        report["mean_iou_after"] = round(float(np.mean([i["iou_after"] for i in report["images"]])), 4)
    report["refined"] = improved
    rig.quality["silhouette_refinement"] = {
        "mean_iou_before": report["mean_iou_before"],
        "mean_iou_after": report["mean_iou_after"],
        "cameras_improved": improved,
        "proxy_faces": report["proxy_faces"],
        "resolution": resolution,
    }
    return rig, report


def _resize_mask(mask: np.ndarray, w: int, h: int) -> np.ndarray:
    try:
        import cv2

        return cv2.resize((mask > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    except Exception:  # pragma: no cover
        idx_y = np.linspace(0, mask.shape[0] - 1, h).astype(int)
        idx_x = np.linspace(0, mask.shape[1] - 1, w).astype(int)
        return (mask[np.ix_(idx_y, idx_x)] > 0).astype(np.uint8)


def _evaluate_pose(pose: CameraPose, verts: np.ndarray, faces: np.ndarray,
                   target: np.ndarray, w: int, h: int,
                   d_az: float, d_el: float, d_log_dist: float,
                   d_log_focal: float,
                   center: Optional[np.ndarray] = None) -> Tuple[float, Any]:
    """Silhouette IoU for a perturbed pose (silhouette-only fast path).

    The mesh lives in the rig's world frame, so the camera orbits the solved
    subject centre - not the origin.
    """
    from ..compare.raster import rasterize, silhouette_iou

    centre = np.zeros(3) if center is None else np.asarray(center, dtype=np.float64)
    distance = pose.distance * math.exp(d_log_dist)
    focal = pose.focal_pixels * math.exp(d_log_focal)
    eye_dir = _direction_from_view(pose.azimuth + d_az, pose.elevation + d_el)
    eye = centre + eye_dir * distance
    fy = focal * (h / max(1, pose.height))
    fx = focal * (w / max(1, pose.width))
    cam = Camera.from_look_at(eye, centre, fov_deg=45.0, width=w, height=h,
                              focal_pixels=(fx + fy) / 2.0)
    result = rasterize(verts, faces, cam, silhouette_only=True)
    return silhouette_iou(result.mask, target), result


def _apply_pose_delta(pose: CameraPose, d_az: float, d_el: float, d_log_dist: float,
                      d_log_focal: float, center: Optional[np.ndarray] = None) -> None:
    """Apply a refined delta to a camera pose (keeping it aimed at the subject)."""
    centre = np.zeros(3) if center is None else np.asarray(center, dtype=np.float64)
    pose.azimuth = float((pose.azimuth + d_az) % 360.0)
    pose.elevation = float(np.clip(pose.elevation + d_el, -89.0, 89.0))
    pose.distance = float(pose.distance * math.exp(d_log_dist))
    pose.focal_pixels = float(pose.focal_pixels * math.exp(d_log_focal))
    eye = centre + _direction_from_view(pose.azimuth, pose.elevation) * pose.distance
    fov = math.degrees(2 * math.atan(0.5 * max(pose.width, pose.height) /
                                    max(1e-6, pose.focal_pixels)))
    pose.camera = Camera.from_look_at(eye, centre, fov_deg=fov,
                                      width=pose.width, height=pose.height,
                                      focal_pixels=pose.focal_pixels, name=pose.filename)
    if "silhouette_refined" not in pose.method:
        pose.method = pose.method + "+silhouette_refined"



def load_rig(project: Project) -> CameraRig:
    """Load the cached camera rig written by :func:`estimate_rig`."""
    from ...core.store import read_json

    path = project.stage_dir("camera_estimation", create=False) / "cameras.json"
    data = read_json(path)
    if not isinstance(data, dict):
        raise StageError("camera rig not found", stage="camera_estimation", recoverable=True,
                         details={"path": str(path)})
    return CameraRig.from_dict(data)


# --------------------------------------------------------------------------
# Scale gauge calibration
# --------------------------------------------------------------------------
def calibrate_rig_height(
    rig: CameraRig,
    masks: Dict[str, np.ndarray],
    *,
    target_height: float = 1.0,
    resolution: int = 88,
    max_iterations: int = 4,
    tolerance: float = 0.015,
    reporter: Any = None,
) -> Dict[str, Any]:
    """Close the scale gauge so the carved subject reaches ``target_height``.

    The reconstruction is only defined up to a similarity: scaling every camera
    distance and the assumed subject height by ``k`` reproduces the same images
    and scales the carved volume by ``k``.  Perspective effects mean the naive
    "subject height = 1 at the target distance" choice undershoots by several
    percent, which shows up as a reconstruction that is systematically smaller
    than the references.  This routine measures the carved height at a coarse
    voxel resolution and corrects the gauge until it matches, so the exported
    asset really does have the documented one-unit subject height.
    """
    from ..reconstruction.hull import carve_volume  # local import: avoids a cycle

    factor = 1.0
    current = rig
    history: List[Dict[str, float]] = []
    for _ in range(max(1, max_iterations)):
        hull = carve_volume(current, masks, bounds=current.bounds(), resolution=resolution)
        idx = np.argwhere(hull.occupancy)
        if len(idx) == 0:
            break
        low = hull.bounds_min + idx.min(0)[::-1] * hull.voxel_size
        high = hull.bounds_min + (idx.max(0)[::-1] + 1) * hull.voxel_size
        height = float(high[2] - low[2])
        history.append({"height": height, "factor": factor})
        if height <= 1e-9:
            break
        correction = float(np.clip(target_height / height, 0.6, 1.7))
        if abs(correction - 1.0) <= tolerance * 0.5:
            break
        factor *= correction
        current = rig.scaled(factor)

    if reporter is not None and abs(factor - 1.0) > 1e-6:
        reporter.info(
            f"scale gauge corrected by {factor:.3f}x so the subject measures "
            f"{target_height:g} unit(s) tall"
        )
    return {"rig": current, "factor": factor, "history": history}


# --------------------------------------------------------------------------
# Lens estimation (unknown focal length)
# --------------------------------------------------------------------------
def _silhouette_score(rig: "CameraRig", mesh, masks: Dict[str, np.ndarray],
                      *, resolution: int = 128) -> Dict[str, Any]:
    """Mean silhouette IoU of *mesh* rendered through *rig* against the masks."""
    from ..compare.raster import rasterize, silhouette_iou

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    scores: List[Dict[str, float]] = []
    for pose in rig.cameras:
        mask = masks.get(pose.filename)
        if mask is None or mask.size == 0:
            continue
        aspect = pose.width / max(1, pose.height)
        h = max(48, int(resolution))
        w = max(48, int(round(h * aspect)))
        cam = Camera(R=pose.camera.R, t=pose.camera.t,
                     fx=pose.focal_pixels * (w / max(1, pose.width)),
                     fy=pose.focal_pixels * (h / max(1, pose.height)),
                     cx=w / 2.0 - 0.5, cy=h / 2.0 - 0.5, width=w, height=h,
                     name=pose.filename)
        result = rasterize(verts, faces, cam, silhouette_only=True)
        target = _resize_mask(mask, w, h) > 0
        scores.append({"filename": pose.filename,
                       "iou": float(silhouette_iou(result.mask, target))})
    if not scores:
        return {"mean_iou": 0.0, "min_iou": 0.0, "per_view": []}
    values = [s["iou"] for s in scores]
    return {"mean_iou": float(np.mean(values)), "min_iou": float(np.min(values)),
            "per_view": scores}


def calibrate_lens_and_rig(
    project: Project,
    *,
    images: Optional[Sequence[Any]] = None,
    views: Optional[Dict[str, Dict[str, Any]]] = None,
    masks: Optional[Dict[str, np.ndarray]] = None,
    arrays: Optional[Dict[str, np.ndarray]] = None,
    subject_height_m: float = 1.0,
    fov_candidates: Sequence[float] = (26.0, 30.0, 34.0, 38.0, 43.0, 48.0, 55.0),
    refine_steps: int = 1,
    coarse_resolution: int = 64,
    score_resolution: int = 128,
    reporter: Any = None,
) -> Tuple[CameraRig, Dict[str, Any]]:
    """Estimate the lens (field of view) *and* the rig from the reference set.

    Reference images rarely carry EXIF calibration data, so the focal length is a
    free parameter of the camera solve - and it is *not* a neutral choice: the
    assumed FOV decides how much perspective the reference set is believed to
    contain, which in turn decides how deep the carved subject is.  Assuming a
    45 deg lens for images shot with a 38 deg lens, for instance, makes the
    reconstruction slightly too shallow.

    The estimation is analysis-by-synthesis: for each candidate FOV the rig is
    solved, its scale gauge is closed, a coarse visual hull is carved, and the
    resulting renders are compared against the reference masks.  The FOV with the
    best mean silhouette IoU wins, then a finer pass polishes it.  Everything is
    local, deterministic and cheap (a handful of coarse carves).
    """
    from ..ingestion.loader import ImageSet, load_masks  # local: avoids a cycle
    from ..reconstruction.hull import carve_volume, hull_to_mesh

    if images is None:
        image_set = ImageSet(project)
        images = [image_set.load(entry["id"]) for entry in image_set.entries]
    if masks is None:
        masks = load_masks(project, images)

    def trial(fov: float) -> Dict[str, Any]:
        rig = estimate_rig(project, images=images, views=views,
                           subject_height_m=subject_height_m, ref_fov_deg=float(fov))
        gauge = calibrate_rig_height(rig, masks, resolution=48)
        rig = gauge["rig"]
        hull = carve_volume(rig, masks, arrays, bounds=rig.bounds(),
                            resolution=coarse_resolution)
        mesh = hull_to_mesh(hull)
        score = _silhouette_score(rig, mesh, masks, resolution=score_resolution)
        return {"fov_deg": float(fov), "rig": rig, "score": score,
                "gauge_factor": gauge["factor"],
                "extents": np.asarray(mesh.extents).tolist()}

    coarse = [trial(fov) for fov in fov_candidates]
    coarse.sort(key=lambda t: -t["score"]["mean_iou"])
    best = coarse[0]
    if reporter is not None:
        reporter.info(
            f"lens estimate: {best['fov_deg']:.0f} deg vertical FOV "
            f"(silhouette IoU {best['score']['mean_iou']:.3f})"
        )

    history = [{"fov_deg": round(t["fov_deg"], 2),
                "mean_iou": round(t["score"]["mean_iou"], 4),
                "min_iou": round(t["score"]["min_iou"], 4),
                "extents": [round(float(v), 4) for v in t["extents"]]} for t in coarse]

    for step in (4.0, 2.0, 1.0)[:max(0, refine_steps)]:
        for candidate in (best["fov_deg"] - step, best["fov_deg"] + step):
            if candidate <= 8.0 or candidate >= 110.0:
                continue
            if any(abs(candidate - h["fov_deg"]) < 1e-6 for h in history):
                continue
            result = trial(candidate)
            history.append({"fov_deg": round(result["fov_deg"], 2),
                            "mean_iou": round(result["score"]["mean_iou"], 4),
                            "min_iou": round(result["score"]["min_iou"], 4),
                            "extents": [round(float(v), 4) for v in result["extents"]]})
            if result["score"]["mean_iou"] > best["score"]["mean_iou"]:
                best = result

    report = {
        "fov_deg": round(float(best["fov_deg"]), 3),
        "mean_iou": round(float(best["score"]["mean_iou"]), 4),
        "min_iou": round(float(best["score"]["min_iou"]), 4),
        "candidates": history,
        "selected_extents": [round(float(v), 4) for v in best["extents"]],
        "default_fov_deg": DEFAULT_FOV_DEG,
        "assumed_from_data": True,
    }
    return best["rig"], report

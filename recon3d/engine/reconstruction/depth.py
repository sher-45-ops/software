"""Multi-view depth estimation and dense point cloud fusion (spec #7, #49).

Two local, classical mechanisms:

* **Plane-sweep / ray-marching MVS.**  For each reference view we march along
  pixel rays through the already-reconstructed volume, and score each candidate
  depth by the *cross-view photometric agreement*: the pixel colour is compared
  with the colour reprojected into every other view.  The depth with the best
  agreement is the surface.  This recovers concavities the visual hull cannot
  represent and produces a dense, photometrically supported point cloud.
* **Depth fusion.**  Points from all views are merged in world space, filtered
  by confidence and statistical outlier removal, and (optionally) used to carve
  the volume - see :func:`recon3d.engine.reconstruction.hull.fuse_depth_points`.

Everything is CPU, vectorised, tiled along pixel rows so that memory stays
bounded on large images.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...errors import StageError
from ..cameras.rig import CameraRig
from .hull import _sample_colors, _sample_masks


@dataclass
class DepthMap:
    filename: str
    depth: np.ndarray  # HxW float32, NaN where unknown
    confidence: np.ndarray  # HxW float32 0-1
    valid: np.ndarray  # HxW bool
    statistics: Dict[str, Any] = field(default_factory=dict)


def _grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32)
    return (0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]).astype(np.float32)


def estimate_depth_for_view(
    pose,
    rig: CameraRig,
    images: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    *,
    depth_range: Tuple[float, float],
    steps: int = 32,
    pixel_step: int = 4,
    min_views: int = 2,
    reporter: Any = None,
) -> DepthMap:
    """Ray-march one view and pick the depth with the best cross-view agreement."""
    camera = pose.camera
    H, W = camera.height, camera.width
    depth = np.full((H, W), np.nan, dtype=np.float32)
    confidence = np.zeros((H, W), dtype=np.float32)
    valid = np.zeros((H, W), dtype=bool)
    mask = masks.get(pose.filename)
    if mask is None:
        return DepthMap(pose.filename, depth, confidence, valid, {"reason": "no mask"})

    ref_image = images.get(pose.filename)
    if ref_image is None:
        return DepthMap(pose.filename, depth, confidence, valid, {"reason": "no image"})
    ref_gray = _grayscale(ref_image)

    others = [p for p in rig.cameras if p.filename != pose.filename]
    if not others or min_views > len(others):
        return DepthMap(pose.filename, depth, confidence, valid, {"reason": "not enough other views"})

    z_near, z_far = depth_range
    samples = np.linspace(z_near, z_far, max(4, steps))

    ys, xs = np.mgrid[0:H:pixel_step, 0:W:pixel_step]
    ys = ys.reshape(-1)
    xs = xs.reshape(-1)
    inside = _sample_masks(mask, xs.astype(np.float64), ys.astype(np.float64)).astype(bool)
    ys, xs = ys[inside], xs[inside]
    if len(ys) == 0:
        return DepthMap(pose.filename, depth, confidence, valid, {"reason": "empty mask"})

    # Ray directions in world space for the sampled pixels.
    cam_dirs = np.stack([(xs.astype(np.float64) - camera.cx) / camera.fx,
                         (ys.astype(np.float64) - camera.cy) / camera.fy,
                         np.ones(len(xs))], axis=1)
    cam_dirs /= np.linalg.norm(cam_dirs, axis=1, keepdims=True)
    world_dirs = cam_dirs.dot(camera.R)
    origin = camera.position()

    n_rays = len(xs)
    n_steps = len(samples)
    # cost[ray, step]
    cost = np.zeros((n_rays, n_steps), dtype=np.float32)
    votes = np.zeros((n_rays, n_steps), dtype=np.int16)

    ref_colors = ref_image[ys, xs].astype(np.float32)

    for other in others:
        o_cam = other.camera
        o_image = images.get(other.filename)
        o_mask = masks.get(other.filename)
        if o_image is None:
            continue
        # Project every (ray, step) candidate into this view in one batch.
        # Shape: (n_steps, n_rays, 3)
        # (n_steps, n_rays, 3): the camera origin (1,1,3) plus the ray directions
        # (1, n_rays, 3) scaled by each candidate depth (n_steps, 1, 1).
        candidate = origin[None, None, :] + world_dirs[None, :, :] * samples[:, None, None]
        flat = candidate.reshape(-1, 3)
        uv, cam_depth = o_cam.project(flat)
        u, v = uv[:, 0], uv[:, 1]
        occ = cam_depth > 1e-6
        if o_mask is not None:
            occ &= _sample_masks(o_mask, u, v).astype(bool)
        colors, valid_px = _sample_colors(o_image, u, v)
        good = (occ & valid_px).reshape(n_steps, n_rays)
        stacked = np.repeat(ref_colors, n_steps, axis=0)
        diff = np.abs(colors - stacked).sum(axis=1)
        diff = diff.reshape(n_steps, n_rays)
        cost += np.where(good, diff, 0.0).T.astype(np.float32)
        votes += good.T.astype(np.int16)
        # Only the per-view temporaries are released here: ``depth`` is the output
        # buffer of this function and must survive the loop.
        del candidate, flat, u, v, cam_depth, occ, colors, valid_px, good, diff
        if reporter is not None:
            reporter.check_cancelled()

    enough = votes >= max(1, min_views)
    cost_norm = np.where(enough, cost / np.maximum(1, votes), np.inf)
    best = np.argmin(cost_norm, axis=1)
    best_cost = cost_norm[np.arange(n_rays), best]
    good_rays = np.isfinite(best_cost) & np.any(enough, axis=1)

    if reporter is not None:
        reporter.info(f"depth for {pose.filename}: {int(good_rays.sum())}/{n_rays} rays matched")

    if not good_rays.any():
        return DepthMap(pose.filename, depth, confidence, valid,
                        {"rays": int(n_rays), "matched": 0, "reason": "no photometric agreement"})

    # Sub-step parabolic refinement around the best sample.
    best_t = samples[best]
    idx = np.arange(n_rays)
    can_refine = (best > 0) & (best < n_steps - 1)
    c0 = cost_norm[idx, np.clip(best - 1, 0, n_steps - 1)]
    c1 = cost_norm[idx, best]
    c2 = cost_norm[idx, np.clip(best + 1, 0, n_steps - 1)]
    # Unmatched samples carry ``inf`` cost, so this difference warns on the NaNs it
    # produces; those entries are discarded by ``ok`` below anyway.
    with np.errstate(invalid="ignore"):
        denom = (c0 - 2 * c1 + c2)
    delta = np.zeros(n_rays, dtype=np.float32)
    ok = can_refine & np.isfinite(c0) & np.isfinite(c2) & (np.abs(denom) > 1e-6) & good_rays
    delta[ok] = np.clip(0.5 * (c0[ok] - c2[ok]) / denom[ok], -0.5, 0.5)
    dt = float(samples[1] - samples[0]) if n_steps > 1 else 0.0
    depth_values = best_t + delta * dt

    # Confidence: agreement quality plus multi-view support.
    vote_conf = np.clip(votes[idx, best] / max(1, len(others)), 0, 1).astype(np.float32)
    contrast = np.clip(1.0 - best_cost / 120.0, 0.0, 1.0).astype(np.float32)
    conf = (0.6 * contrast + 0.4 * vote_conf)

    sel = good_rays
    fu = xs[sel].astype(np.float32)
    fv = ys[sel].astype(np.float32)
    depth[ys[sel], xs[sel]] = depth_values[sel].astype(np.float32)
    confidence[ys[sel], xs[sel]] = conf[sel]
    valid[ys[sel], xs[sel]] = True

    stats = {
        "rays": int(n_rays),
        "matched": int(good_rays.sum()),
        "match_ratio": round(float(good_rays.mean()), 3),
        "mean_confidence": round(float(conf[sel].mean()), 3) if sel.any() else 0.0,
        "depth_min": round(float(np.nanmin(depth_values[sel])), 4) if sel.any() else 0.0,
        "depth_max": round(float(np.nanmax(depth_values[sel])), 4) if sel.any() else 0.0,
    }
    return DepthMap(pose.filename, depth, confidence, valid, stats)


def unproject_depth(depth_map: DepthMap, camera, *, min_confidence: float = 0.15,
                    stride: int = 2, colors: Optional[np.ndarray] = None
                    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Turn a depth map into world-space points (and optionally their colours)."""
    d = depth_map.depth
    H, W = d.shape
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    z = d[ys, xs]
    conf = depth_map.confidence[ys, xs]
    ok = np.isfinite(z) & (z > 0) & (conf >= min_confidence)
    if not ok.any():
        return np.zeros((0, 3)), (np.zeros((0, 3), np.uint8) if colors is not None else None)
    u = xs[ok].astype(np.float64)
    v = ys[ok].astype(np.float64)
    zz = z[ok].astype(np.float64)
    cam_pts = np.stack([(u - camera.cx) / camera.fx * zz,
                        (v - camera.cy) / camera.fy * zz,
                        zz], axis=1)
    world = camera.camera_to_world(cam_pts)
    if colors is None:
        return world, None
    return world, colors[ys[ok], xs[ok]].astype(np.uint8)


def statistical_outlier_removal(points: np.ndarray, *, k: int = 12, std_ratio: float = 2.0,
                                max_points: int = 300_000) -> np.ndarray:
    """Standard SOR filter (PCL-compatible behaviour) using a KD-tree."""
    if len(points) < k + 1:
        return points
    from scipy.spatial import cKDTree

    sample = points
    if len(points) > max_points:
        idx = np.random.default_rng(0).choice(len(points), max_points, replace=False)
        sample = points[idx]
    tree = cKDTree(sample)
    dist, _ = tree.query(sample, k=min(k, len(sample)), workers=-1)
    mean_dist = dist[:, 1:].mean(axis=1)
    threshold = mean_dist.mean() + std_ratio * mean_dist.std()
    if len(sample) < len(points):
        # Filter the full cloud against the per-sample threshold.
        full_dist, _ = tree.query(points, k=1, workers=-1)
        return points[full_dist <= threshold]
    return sample[mean_dist <= threshold]


def fuse_depth_maps(
    rig: CameraRig,
    images: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    *,
    depth_margin: float = 0.35,
    steps: int = 28,
    pixel_step: int = 4,
    min_confidence: float = 0.2,
    reporter: Any = None,
    progress_range: Tuple[float, float] = (0.0, 100.0),
    max_views: int = 8,
) -> Dict[str, Any]:
    """Compute depth for every view and fuse the results into a point cloud."""
    radius = rig.bounding_radius()
    depth_maps: List[DepthMap] = []
    points_all: List[np.ndarray] = []
    colors_all: List[np.ndarray] = []
    poses = list(rig.cameras)[: max(1, max_views)]
    lo, hi = progress_range

    for i, pose in enumerate(poses):
        center_depth = float(np.linalg.norm(pose.camera.position() - rig.center))
        depth_range = (max(0.05, center_depth - radius - depth_margin),
                       center_depth + radius + depth_margin)
        dm = estimate_depth_for_view(pose, rig, images, masks, depth_range=depth_range,
                                     steps=steps, pixel_step=pixel_step, reporter=None)
        depth_maps.append(dm)
        image = images.get(pose.filename)
        pts, cols = unproject_depth(dm, pose.camera, min_confidence=min_confidence,
                                    stride=max(1, int(pixel_step // 2)),
                                    colors=image)
        if len(pts):
            points_all.append(pts)
            if cols is not None:
                colors_all.append(cols)
        if reporter is not None:
            reporter.update(lo + (hi - lo) * (i + 1) / max(1, len(poses)),
                            f"multi-view depth for {pose.filename} "
                            f"({dm.statistics.get('matched', 0)} matched rays)")

    if not points_all:
        return {"points": np.zeros((0, 3)), "colors": np.zeros((0, 3), np.uint8),
                "depth_maps": [], "statistics": {"views": len(depth_maps), "fused": 0}}

    raw = np.vstack(points_all)
    colors = np.vstack(colors_all) if colors_all else None
    filtered = statistical_outlier_removal(raw, k=10, std_ratio=2.5)
    # Voxel-grid downsample so overlapping views do not bias the density.
    voxel = max(1e-4, rig.bounding_radius() / 180.0)
    keys = np.floor(filtered / voxel).astype(np.int64)
    _uniq, index = np.unique(keys, axis=0, return_index=True)
    fused = filtered[index]
    fused_colors = colors[index] if colors is not None and len(colors) == len(raw) else None

    stats = {
        "views": len(depth_maps),
        "raw_points": int(len(raw)),
        "after_outlier_removal": int(len(filtered)),
        "fused_voxels": int(len(fused)),
        "voxel_size": round(float(voxel), 5),
        "per_view": [dm.statistics for dm in depth_maps],
        "mean_match_ratio": round(float(np.mean([dm.statistics.get("match_ratio", 0)
                                                 for dm in depth_maps])), 3) if depth_maps else 0.0,
    }
    return {"points": fused, "colors": fused_colors, "depth_maps": depth_maps, "statistics": stats}


def depth_statistics_consistency(depth_maps: Sequence[DepthMap]) -> Dict[str, Any]:
    """Cross-view consistency check on the fused depth maps."""
    ratios = [dm.statistics.get("match_ratio", 0.0) for dm in depth_maps]
    confs = [dm.statistics.get("mean_confidence", 0.0) for dm in depth_maps]
    warnings: List[str] = []
    if ratios and float(np.mean(ratios)) < 0.25:
        warnings.append(
            "fewer than 25% of rays found photometric agreement across views: the "
            "references may be inconsistent (different subject, mirrored images, "
            "strong lighting changes) - geometry will rely mainly on silhouettes"
        )
    return {
        "views": len(depth_maps),
        "mean_match_ratio": round(float(np.mean(ratios)), 3) if ratios else 0.0,
        "mean_confidence": round(float(np.mean(confs)), 3) if confs else 0.0,
        "warnings": warnings,
    }


def save_point_cloud(path, points: np.ndarray, colors: Optional[np.ndarray] = None,
                     normals: Optional[np.ndarray] = None) -> str:
    """Write a PLY point cloud."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    header = ["ply", "format ascii 1.0", f"element vertex {len(points)}",
              "property float x", "property float y", "property float z"]
    if normals is not None:
        header += ["property float nx", "property float ny", "property float nz"]
    if colors is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header.append("end_header")
    lines: List[str] = list(header)
    normals = np.asarray(normals).reshape(-1, 3) if normals is not None else None
    colors = np.asarray(colors).reshape(-1, 3) if colors is not None else None
    for i, p in enumerate(points):
        row = f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}"
        if normals is not None and i < len(normals):
            row += f" {normals[i][0]:.4f} {normals[i][1]:.4f} {normals[i][2]:.4f}"
        if colors is not None and i < len(colors):
            c = np.clip(colors[i], 0, 255).astype(int)
            row += f" {c[0]} {c[1]} {c[2]}"
        lines.append(row)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)

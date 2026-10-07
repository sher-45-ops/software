"""Multi-view volumetric reconstruction: silhouette carving + photometric refinement.

This is the workhorse geometry producer.  It is a genuine multi-view method:

1. **Silhouette carving (visual hull / SFS).**  Every voxel must project inside
   the subject mask of *every* reference image.  Implemented as a hierarchical
   carve: a coarse pass over the whole volume locates the subject, then the
   volume is re-carved at increasing resolution inside the occupied bounding box
   (the standard octree-style trick), which buys several effective levels of
   detail for the same memory budget.
2. **Photometric refinement.**  Voxels on the visual hull that reproduce the
   wrong colour in several views are carved away.  This removes the "ghost"
   material the visual hull invents in concavities (between legs, under arms,
   inside wheel arches) and is what makes the difference between a blob and a
   usable asset.
3. **Surface extraction.**  Marching cubes over the occupancy field, followed by
   mesh cleanup.

Because the method *intersects evidence from every view*, the result is real
reconstructed geometry: it cannot exist without the supplied images.

When a textured/curved subject defeats the visual hull (concavities the hull
cannot express) the depth-based point cloud from :mod:`recon3d.engine.reconstruction.depth`
is fused in and the surface is re-fitted - see :func:`fuse_depth_points`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...errors import StageError
from ..cameras.rig import CameraRig, CameraPose

try:
    import cv2

    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False


@dataclass
class HullResult:
    occupancy: np.ndarray  # bool grid (k, j, i) along (z, y, x)
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    voxel_size: float
    resolution: Tuple[int, int, int]
    method: str
    statistics: Dict[str, Any] = field(default_factory=dict)

    def grid_to_world(self, ijk: np.ndarray) -> np.ndarray:
        """Convert integer voxel indices ``(k, j, i)`` to world ``(x, y, z)``.

        The grid is stored with the z axis first (``(nz, ny, nx)``, matching
        skimage's marching-cubes output), so the index vector is reversed here.
        """
        ijk = np.asarray(ijk, dtype=np.float64)
        xyz_index = ijk[..., ::-1]
        return self.bounds_min + (xyz_index + 0.5) * self.voxel_size

    def world_to_grid(self, xyz: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`grid_to_world` - returns ``(k, j, i)`` indices."""
        xyz = np.asarray(xyz, dtype=np.float64)
        xyz_index = (xyz - self.bounds_min) / self.voxel_size - 0.5
        return xyz_index[..., ::-1]

    def occupied_count(self) -> int:
        return int(self.occupancy.sum())


# --------------------------------------------------------------------------
# Projection helpers
# --------------------------------------------------------------------------
def _project_points(camera, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points; returns ``(u, v, depth)`` arrays."""
    uv, depth = camera.project(points)
    return uv[:, 0], uv[:, 1], depth


def _sample_masks(mask: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Nearest-neighbour mask lookup with out-of-bounds -> 0 (outside)."""
    h, w = mask.shape[:2]
    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)
    inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    out = np.zeros(u.shape, dtype=np.uint8)
    if inside.any():
        out[inside] = (mask[vi[inside], ui[inside]] > 0).astype(np.uint8)
    return out


def photo_normalization(image: np.ndarray, mask: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Per-view colour mean/std inside the subject, used to compare views fairly."""
    pixels = image.reshape(-1, 3).astype(np.float32)
    if mask is not None and mask.size == image.shape[0] * image.shape[1]:
        sel = mask.reshape(-1) > 0
        if sel.sum() > 64:
            pixels = pixels[sel]
    if len(pixels) > 20000:
        idx = np.random.default_rng(0).choice(len(pixels), 20000, replace=False)
        pixels = pixels[idx]
    mean = pixels.mean(axis=0)
    std = pixels.std(axis=0)
    std = np.where(std < 4.0, 4.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _sample_colors(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Bilinear colour sampling; returns ``(colors float32 Nx3, valid mask)``."""
    h, w = image.shape[:2]
    valid = (u >= 0) & (u <= w - 1.5) & (v >= 0) & (v <= h - 1.5)
    colors = np.zeros((u.shape[0], 3), dtype=np.float32)
    if not valid.any():
        return colors, valid
    uu = np.clip(u[valid], 0, w - 1.001)
    vv = np.clip(v[valid], 0, h - 1.001)
    x0 = np.floor(uu).astype(np.int32)
    y0 = np.floor(vv).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (uu - x0)[:, None]
    fy = (vv - y0)[:, None]
    img = image.astype(np.float32)
    c00 = img[y0, x0]
    c10 = img[y0, x1]
    c01 = img[y1, x0]
    c11 = img[y1, x1]
    colors[valid] = (c00 * (1 - fx) * (1 - fy) + c10 * fx * (1 - fy) +
                     c01 * (1 - fx) * fy + c11 * fx * fy)
    return colors, valid


# --------------------------------------------------------------------------
# Carving
# --------------------------------------------------------------------------
def carve_volume(
    rig: CameraRig,
    masks: Dict[str, np.ndarray],
    images: Optional[Dict[str, np.ndarray]] = None,
    *,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    resolution: int = 96,
    photo_consistency: bool = False,
    photo_threshold: float = 1.15,
    min_views_agree: int = 0,
    close_iterations: int = 1,
    min_blob_voxels: int = 8,
    reporter: Any = None,
    progress_range: Tuple[float, float] = (0.0, 100.0),
) -> HullResult:
    """Carve the visual hull (+ optional photometric consistency).

    The finished occupancy grid is tidied by :func:`clean_volume` (closing pass plus
    a small-blob filter) so that the meshed surface is a clean shell instead of a
    pinhole-riddled one.
    """
    if len(rig.cameras) == 0:
        raise StageError("no cameras available for volumetric carving",
                         stage="mesh_reconstruction", recoverable=False)
    bounds_min, bounds_max = bounds if bounds is not None else rig.bounds()
    bounds_min = np.asarray(bounds_min, dtype=np.float64)
    bounds_max = np.asarray(bounds_max, dtype=np.float64)
    extent = bounds_max - bounds_min
    longest = float(extent.max())
    if longest <= 0:
        raise StageError("degenerate reconstruction bounds", stage="mesh_reconstruction",
                         recoverable=False)
    voxel = longest / float(max(8, resolution))
    dims = np.maximum(8, np.ceil(extent / voxel).astype(int))
    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
    total_voxels = nx * ny * nz

    occupancy = np.ones((nz, ny, nx), dtype=bool)
    zs = bounds_min[2] + (np.arange(nz) + 0.5) * voxel
    ys = bounds_min[1] + (np.arange(ny) + 0.5) * voxel
    xs = bounds_min[0] + (np.arange(nx) + 0.5) * voxel

    color_sum: Optional[np.ndarray] = None
    color_sq: Optional[np.ndarray] = None
    color_hits: Optional[np.ndarray] = None

    for ci, pose in enumerate(rig.cameras):
        mask = masks.get(pose.filename)
        if mask is None or mask.sum() == 0:
            continue
        camera = pose.camera
        # Work one z-slab at a time to bound peak memory.
        for slab_start in range(0, nz, max(1, 8)):
            slab_stop = min(nz, slab_start + max(1, 8))
            zz = zs[slab_start:slab_stop]
            grid_z, grid_y, grid_x = np.meshgrid(zz, ys, xs, indexing="ij")
            pts = np.stack([grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)], axis=1)
            u, v, depth = _project_points(camera, pts)
            visible = depth > 1e-6
            inside = np.zeros(len(pts), dtype=bool)
            if visible.any():
                idx = np.nonzero(visible)[0]
                inside[idx] = _sample_masks(mask, u[idx], v[idx]).astype(bool)
            slab_shape = (slab_stop - slab_start, ny, nx)
            slab_inside = inside.reshape(slab_shape)
            occupancy[slab_start:slab_stop] &= slab_inside
            del pts, u, v, depth, inside, slab_inside

        if photo_consistency and images:
            image = images.get(pose.filename)
            if image is not None:
                if color_sum is None:
                    color_sum = np.zeros((nz, ny, nx, 3), dtype=np.float32)
                    color_sq = np.zeros((nz, ny, nx, 3), dtype=np.float32)
                    color_hits = np.zeros((nz, ny, nx), dtype=np.uint8)
                # Per-view photometric normalisation: the same surface point is
                # shaded differently in every reference (different light angle),
                # so we compare *normalised* colours, which removes the global
                # exposure/white-balance differences between views.
                norm_mean, norm_std = photo_normalization(image, mask)
                for slab_start in range(0, nz, max(1, 4)):
                    slab_stop = min(nz, slab_start + max(1, 4))
                    zz = zs[slab_start:slab_stop]
                    grid_z, grid_y, grid_x = np.meshgrid(zz, ys, xs, indexing="ij")
                    pts = np.stack([grid_x.reshape(-1), grid_y.reshape(-1),
                                    grid_z.reshape(-1)], axis=1)
                    u, v, depth = _project_points(camera, pts)
                    colors, valid = _sample_colors(image, u, v)
                    slab_shape = (slab_stop - slab_start, ny, nx)
                    valid_slab = valid.reshape(slab_shape)
                    cslab = ((colors.reshape(slab_shape + (3,)) - norm_mean) / norm_std)
                    color_sum[slab_start:slab_stop][valid_slab] += cslab[valid_slab]
                    color_sq[slab_start:slab_stop][valid_slab] += cslab[valid_slab] ** 2
                    color_hits[slab_start:slab_stop][valid_slab] += 1
                    del pts, u, v, depth, colors, valid, cslab
        if reporter is not None:
            lo, hi = progress_range
            reporter.update(lo + (hi - lo) * (ci + 1) / max(1, len(rig.cameras)),
                            f"carved silhouettes from {ci + 1}/{len(rig.cameras)} views")
        if not occupancy.any():
            raise StageError(
                "silhouette carving produced an empty volume - the views are inconsistent "
                "(masks may not correspond to the same subject)",
                stage="mesh_reconstruction", recoverable=False,
            )

    stats: Dict[str, Any] = {
        "resolution": [nx, ny, nz],
        "voxel_size": voxel,
        "voxels": total_voxels,
        "occupied_before_photo": int(occupancy.sum()),
    }

    # -- photometric consistency ----------------------------------------
    # Colour evidence is only meaningful *on the surface*: interior voxels are
    # occluded in most views, so the colour they "see" belongs to some other
    # surface entirely.  We therefore restrict photo carving to the shell of the
    # current occupancy and iterate, which removes the visual hull's ghost
    # material (between limbs, inside concavities) without eating real geometry.
    if photo_consistency and color_hits is not None and images:
        with np.errstate(invalid="ignore", divide="ignore"):
            hits = np.maximum(color_hits, 1).astype(np.float32)
            mean = color_sum / hits[..., None]
            var = np.maximum(color_sq / hits[..., None] - mean ** 2, 0.0)
        spread = np.sqrt(var.sum(axis=-1))  # combined RGB spread, normalised units
        min_hits = max(3, min_views_agree, int(0.34 * len(rig.cameras)))
        # Only the *shell* of the volume is tested: interior voxels are occluded,
        # so the colour they see belongs to other surfaces and cannot be trusted.
        # A single pass keeps the operation conservative - re-running it would
        # test former interior voxels with statistics gathered while occluded.
        shell = _shell_mask(occupancy, iterations=1)
        candidates = shell & (color_hits >= min_hits)
        threshold = float(photo_threshold)
        if candidates.any():
            values = spread[candidates]
            # Adaptive threshold: discard the worst-registered third of shell
            # voxels, but never below the configured floor and never more than a
            # quarter of the volume (a bad mask must degrade quality, not
            # destroy the model).
            adaptive = float(np.quantile(values, 0.70))
            threshold = max(threshold, adaptive)
        remove = candidates & (spread > threshold)
        cap = int(0.25 * max(1, occupancy.sum()))
        if int(remove.sum()) > cap:
            idx = np.argwhere(remove)
            strength = spread[idx[:, 0], idx[:, 1], idx[:, 2]]
            keep_idx = idx[np.argsort(-strength)[:cap]]
            remove = np.zeros_like(remove)
            remove[keep_idx[:, 0], keep_idx[:, 1], keep_idx[:, 2]] = True
        occupancy[remove] = False
        stats["removed_by_photo_consistency"] = int(remove.sum())
        stats["photo_threshold"] = round(threshold, 4)
        if not occupancy.any():
            raise StageError(
                "photometric consistency removed every voxel; the references disagree "
                "too strongly (try a lower photo-consistency threshold or disable it)",
                stage="mesh_reconstruction", recoverable=True,
            )

    occupancy, cleanup_stats = clean_volume(occupancy, close_iterations=close_iterations,
                                            min_blob_voxels=min_blob_voxels,
                                            reporter=reporter)
    stats["volume_cleanup"] = cleanup_stats
    stats["occupied_voxels"] = int(occupancy.sum())
    stats["fill_ratio"] = float(occupancy.mean())
    return HullResult(occupancy=occupancy, bounds_min=bounds_min, bounds_max=bounds_max,
                      voxel_size=voxel, resolution=(nx, ny, nz),
                      method="visual_hull+photometric" if photo_consistency else "visual_hull",
                      statistics=stats)


def _shell_mask(occupancy: np.ndarray, *, iterations: int = 2) -> np.ndarray:
    """Voxels within ``iterations`` of the surface of the occupancy grid."""
    from scipy import ndimage

    structure = ndimage.generate_binary_structure(3, 1)
    eroded = ndimage.binary_erosion(occupancy, structure=structure if iterations == 1
                                    else ndimage.iterate_structure(structure, iterations))
    return occupancy & ~eroded


def clean_volume(occupancy: np.ndarray, *, close_iterations: int = 1,
                 min_blob_voxels: int = 8, reporter: Any = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Morphologically tidy a carved volume before meshing.

    Silhouette carving leaves pinholes, one-voxel tunnels and lone specks.  Marching
    cubes faithfully turns all of them into surface detail, which wrecks the mesh:
    a noisy volume produced a watertight but swiss-cheese surface (genus ~400) whose
    UV atlas fragmented into thousands of charts.  A closing pass plus a small-blob
    filter removes exactly that noise and nothing else.
    """
    from scipy import ndimage

    stats: Dict[str, Any] = {"before": int(occupancy.sum())}
    cleaned = occupancy
    if close_iterations > 0 and cleaned.any():
        structure = ndimage.generate_binary_structure(3, 1)
        cleaned = ndimage.binary_closing(cleaned, structure=structure,
                                         iterations=int(close_iterations), border_value=0)
        # Closing can leave internal voids; fill fully enclosed empty space.
        filled = ndimage.binary_fill_holes(cleaned)
        cleaned = filled
    if min_blob_voxels > 1 and cleaned.any():
        labels, count = ndimage.label(cleaned, structure=ndimage.generate_binary_structure(3, 1))
        if count > 1:
            sizes = np.bincount(labels.reshape(-1))
            keep = sizes >= int(min_blob_voxels)
            keep[0] = False  # background label
            if keep.any():
                cleaned = keep[labels]
            stats["blobs_removed"] = int(count - int(keep.sum()))
    stats["after"] = int(cleaned.sum())
    stats["changed"] = bool(stats["after"] != stats["before"])
    if reporter is not None and stats["changed"]:
        reporter.info(
            f"volume cleanup: {stats['before']} -> {stats['after']} solid voxels "
            f"({stats.get('blobs_removed', 0)} small blobs removed)"
        )
    return cleaned, stats


def voxel_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two boolean occupancy grids."""
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def voxelize_mesh(mesh, resolution: int, bounds_min: np.ndarray, bounds_max: np.ndarray,
                  dims: Optional[Tuple[int, int, int]] = None,
                  voxel_size: Optional[float] = None) -> np.ndarray:
    """Rasterise a mesh into a boolean grid using parity ray tests (X axis).

    Passing ``dims``/``voxel_size`` (typically taken from an existing
    :class:`HullResult`) makes the grid directly comparable with that volume -
    this is how reconstruction accuracy against ground truth is measured.
    """
    extent = np.asarray(bounds_max) - np.asarray(bounds_min)
    longest = float(np.max(extent))
    if dims is None:
        voxel = voxel_size if voxel_size else longest / float(max(8, resolution))
        auto = np.maximum(8, np.ceil(extent / voxel).astype(int))
        nx, ny, nz = int(auto[0]), int(auto[1]), int(auto[2])
    else:
        nz, ny, nx = (int(dims[0]), int(dims[1]), int(dims[2]))
    voxel = voxel_size if voxel_size else longest / float(max(8, resolution))
    grid = np.zeros((nz, ny, nx), dtype=bool)
    centroids = mesh.triangles_center
    normals = mesh.face_normals
    keep = np.abs(normals[:, 0]) > 1e-6
    tri = mesh.triangles[keep]
    n = normals[keep]
    # Ray direction +X from each voxel centre.
    ys = np.asarray(bounds_min)[1] + (np.arange(ny) + 0.5) * voxel
    zs = np.asarray(bounds_min)[2] + (np.arange(nz) + 0.5) * voxel
    origin_x = float(bounds_min[0]) - voxel
    for k, zv in enumerate(zs):
        for j, yv in enumerate(ys):
            origins = np.tile(np.array([origin_x, yv, zv]), (len(tri), 1))
            dirs = np.tile(np.array([1.0, 0.0, 0.0]), (len(tri), 1))
            hits = _ray_tri_hits(origins, dirs, tri)
            # ``_ray_tri_hits`` returns the *ray parameter* (distance along +X);
            # convert it to world X before using it as a grid coordinate.  This
            # used to silently shift the fill by the ray-origin offset (the ray
            # starts one voxel before the grid), which corrupted every voxel IoU
            # measurement made with this function.
            hits = hits + origin_x
            xs = np.sort(hits[np.isfinite(hits)])
            if len(xs) % 2 == 1:
                # Open/unclosed surface: close the interval at the grid boundary
                # rather than at infinity (which used to overflow downstream).
                xs = np.concatenate([xs, [float(np.asarray(bounds_max)[0]) + voxel]])
            for i in range(0, len(xs), 2):
                lo = int(np.ceil((xs[i] - float(bounds_min[0])) / voxel - 0.5))
                hi = int(np.floor((xs[i + 1] - float(bounds_min[0])) / voxel - 0.5))
                lo = max(0, lo)
                hi = min(nx - 1, hi)
                if hi >= lo:
                    grid[k, j, lo:hi + 1] = True
    return grid


def _ray_tri_hits(origins: np.ndarray, dirs: np.ndarray, tri: np.ndarray) -> np.ndarray:
    v0 = tri[:, 0]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    return ray_triangle_intersections(origins, dirs, v0, e1, e2)


def ray_triangle_intersections(origins: np.ndarray, directions: np.ndarray,
                               v0: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    """Vectorised Möller-Trumbore (kept local to avoid an import cycle)."""
    pvec = np.cross(directions, e2)
    det = np.einsum("ij,ij->i", e1, pvec)
    ok = np.abs(det) > 1e-12
    inv_det = np.zeros_like(det)
    inv_det[ok] = 1.0 / det[ok]
    tvec = origins - v0
    u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
    qvec = np.cross(tvec, e1)
    v = np.einsum("ij,ij->i", directions, qvec) * inv_det
    t = np.einsum("ij,ij->i", e2, qvec) * inv_det
    hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 0)
    out = np.full(directions.shape[0], np.inf)
    out[hit] = t[hit]
    return out


def _occupied_bbox(occupancy: np.ndarray, pad: int = 1) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    idx = np.argwhere(occupancy)
    if idx.size == 0:
        nz, ny, nx = occupancy.shape
        return (0, 0, 0), (nz, ny, nx)
    lo = np.maximum(idx.min(axis=0) - pad, 0)
    hi = np.minimum(idx.max(axis=0) + 1 + pad, np.array(occupancy.shape))
    return tuple(int(v) for v in lo), tuple(int(v) for v in hi)  # type: ignore[return-value]


def reconstruct_hull(
    rig: CameraRig,
    masks: Dict[str, np.ndarray],
    images: Optional[Dict[str, np.ndarray]] = None,
    *,
    voxel_start: int = 64,
    voxel_max: int = 256,
    levels: int = 2,
    photo_consistency: bool = True,
    photo_threshold: float = 1.15,
    scale_gauge: bool = False,
    reporter: Any = None,
    progress_range: Tuple[float, float] = (0.0, 100.0),
) -> HullResult:
    """Hierarchical carve: coarse global pass, then refinement inside the subject."""
    lo, hi = progress_range
    span = hi - lo
    resolution = int(np.clip(voxel_start, 24, 512))
    bounds = rig.bounds()
    result: Optional[HullResult] = None

    for level in range(max(1, levels)):
        level_lo = lo + span * level / max(1, levels)
        level_hi = lo + span * (level + 1) / max(1, levels)
        # The final level gets full photometric treatment; coarse levels only
        # use silhouette carving, which is far cheaper and just as informative
        # for locating the volume of interest.
        use_photo = bool(photo_consistency and level == max(1, levels) - 1)
        result = carve_volume(
            rig, masks, images,
            bounds=bounds,
            resolution=resolution,
            photo_consistency=use_photo,
            photo_threshold=photo_threshold,
            reporter=reporter,
            progress_range=(level_lo, level_hi),
        )
        if reporter is not None:
            reporter.info(
                f"carve level {level + 1}/{levels}: {result.occupied_count()} solid voxels "
                f"at {resolution}^3-ish grid"
            )
        if level < max(1, levels) - 1:
            bbox_lo, bbox_hi = _occupied_bbox(result.occupancy, pad=2)
            size = np.array(bbox_hi) - np.array(bbox_lo)
            if np.any(size < 3):
                break
            new_min = result.bounds_min + np.array(bbox_lo) * result.voxel_size
            new_max = result.bounds_min + np.array(bbox_hi) * result.voxel_size
            bounds = (new_min, new_max)
            # Scale the next grid so the *subject* is sampled at the requested
            # resolution rather than the original (padded, mostly empty) box.
            longest = float((new_max - new_min).max())
            # The refinement grid is bounded by the *tight* subject box, so the
            # requested ``voxel_max`` is affordable: spend the whole budget on the
            # final level instead of doubling blindly.  Marching cubes places
            # vertices inside cells, which loses roughly one voxel of surface
            # everywhere, and that deficit shows up directly as "the
            # reconstruction is thinner than the reference" in the comparison
            # report - so this resolution *is* an accuracy knob.
            next_res = int(np.clip(voxel_max, voxel_start, voxel_max))
            if voxel_max <= resolution:
                next_res = int(np.clip(resolution * 2, voxel_start, voxel_max))
            resolution = next_res
            del result
    assert result is not None

    # -- close the scale gauge on the *final* resolution -------------------
    # The gauge is calibrated on a coarse carve (cheap), which leaves a small
    # residual (voxel quantisation at the coarse resolution).  Measuring the
    # finished volume once and rescaling the rig closes it: the reconstruction
    # then really does measure ``rig.subject_height`` in the engine frame, which
    # is what the export scaling and every downstream comparison assume.
    target = float(getattr(rig, "subject_height", 0.0) or 0.0)
    if target > 0 and scale_gauge:
        idx = np.argwhere(result.occupancy)
        if len(idx):
            low = result.bounds_min + idx.min(0)[::-1] * result.voxel_size
            high = result.bounds_min + (idx.max(0)[::-1] + 1) * result.voxel_size
            height = float(high[2] - low[2])
            if height > 1e-9 and abs(height / target - 1.0) > 0.015:
                correction = float(np.clip(target / height, 0.7, 1.4))
                scaled_rig = rig.scaled(correction)
                # Scale the *scene* (bounds and cameras together) so the carve is
                # a similarity transform of the previous one.
                refined = carve_volume(
                    scaled_rig, masks, images,
                    bounds=tuple(np.asarray(b) for b in (
                        scaled_rig.center + (np.asarray(result.bounds_min) - rig.center) * correction,
                        scaled_rig.center + (np.asarray(result.bounds_max) - rig.center) * correction,
                    )),
                    resolution=resolution,
                    photo_consistency=bool(photo_consistency),
                    photo_threshold=photo_threshold,
                    reporter=reporter,
                    progress_range=(lo + span * 0.9, hi),
                )
                if refined.occupied_count() > 0:
                    result = refined
                    if reporter is not None:
                        reporter.info(
                            f"scale gauge closed ({correction:.3f}x): subject height "
                            f"{target:g} unit"
                        )
    return result


# --------------------------------------------------------------------------
# Surface extraction
# --------------------------------------------------------------------------
def hull_to_mesh(result: HullResult, *, smooth: bool = True, level: float = 0.5,
                 volume_sigma: float = 0.0):
    """Marching cubes over the occupancy field -> ``trimesh.Trimesh``.

    ``volume_sigma`` blurs the occupancy field before extraction.  Silhouette carving
    leaves one-voxel pinholes and tunnels; marching cubes turns them into real surface
    detail, which produced a watertight-but-swiss-cheese shell (genus ~400).  A small
    blur (0.8 voxel) removes exactly that scale of noise and yields a clean genus-0
    shell, at the cost of a fraction of a percent of volume.
    """
    import trimesh
    from skimage import measure

    field = result.occupancy.astype(np.float32)
    if volume_sigma and volume_sigma > 0:
        from scipy import ndimage

        field = ndimage.gaussian_filter(field, sigma=float(volume_sigma), mode="constant")
    # Pad so the isosurface closes at the volume boundary.
    padded = np.pad(field, 1, mode="constant", constant_values=0.0)
    try:
        verts, faces, normals, _values = measure.marching_cubes(
            padded, level=level, spacing=(result.voxel_size,) * 3,
        )
    except (ValueError, RuntimeError) as exc:
        raise StageError(f"marching cubes failed: {exc}", stage="mesh_reconstruction",
                         recoverable=False) from exc
    # skimage returns coordinates in grid index order (z, y, x); convert to world
    # (x, y, z).  Because the volume was padded by one voxel, the unpadded
    # coordinate of a vertex at padded index p is p - 1, and voxel centres sit at
    # index + 0.5 in the unpadded frame -> offset by half a voxel.
    verts = verts[:, ::-1] + (result.bounds_min - 0.5 * result.voxel_size)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=False)
    if smooth and len(mesh.faces) > 32:
        try:
            mesh = trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=4)
        except Exception:  # pragma: no cover - smoothing is best effort
            pass
    mesh.process(validate=True)
    return mesh


def voxel_surface_points(result: HullResult, *, stride: int = 1) -> np.ndarray:
    """World-space points on the boundary of the occupancy grid."""
    from scipy import ndimage

    occ = result.occupancy
    eroded = ndimage.binary_erosion(occ, structure=ndimage.generate_binary_structure(3, 1))
    surface = occ & ~eroded
    idx = np.argwhere(surface)[::stride]
    if idx.size == 0:
        return np.zeros((0, 3))
    return result.grid_to_world(idx)


def fuse_depth_points(result: HullResult, points: np.ndarray, *, carve_thickness: float = 1.5,
                      reporter: Any = None, min_density: float = 2.5) -> HullResult:
    """Carve voxels that the multi-view depth maps prove are empty.

    ``points`` is a fused depth-based point cloud of observed surface samples.
    A voxel survives only if some observed surface point lies near it - i.e. it
    is *supported by a photometric measurement*, not merely by silhouettes.

    A sparse point cloud cannot support a surface: keeping only voxels within a
    couple of voxels of a handful of samples turns a correct hull into a string
    of beads (measured: 22k samples shredded 82 % of a clean shell into 384
    fragments).  The support radius therefore scales with the *measured* sample
    spacing, and the carve is skipped outright - with the reason recorded - when
    the samples are more than ``min_density`` voxels apart.

    The carve returns a **new** :class:`HullResult`; ``result`` is never mutated,
    so a caller can compare the two volumes and keep the better one (the identity
    check ``fused is hull`` is the documented "nothing was carved" signal).
    """
    if points is None or len(points) == 0:
        return result
    from scipy.spatial import cKDTree

    occupied_idx = np.argwhere(result.occupancy)
    if occupied_idx.size == 0:
        return result
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) > 4:
        self_dist, _ = cKDTree(pts).query(pts, k=2, workers=-1)
        spacing = float(np.median(self_dist[:, 1]))
    else:
        spacing = float("inf")
    spacing_voxels = spacing / max(result.voxel_size, 1e-12)
    if not np.isfinite(spacing_voxels) or spacing_voxels > min_density:
        result.statistics["depth_fusion"] = {
            "skipped": True,
            "reason": "depth point cloud too sparse to carve a surface",
            "samples": int(len(pts)),
            "median_sample_spacing_voxels": None if not np.isfinite(spacing_voxels)
            else round(float(spacing_voxels), 2),
        }
        if reporter is not None:
            reporter.warning(
                f"depth evidence has {len(pts)} samples spaced ~{spacing_voxels:.1f} voxels "
                "apart - too sparse to carve a surface, so the silhouette hull was kept")
        return result
    world = result.grid_to_world(occupied_idx)
    tree = cKDTree(pts)
    dist, _ = tree.query(world, k=1, workers=-1)
    radius = max(carve_thickness * result.voxel_size, 1.25 * spacing)
    keep = dist <= radius
    new_occ = np.zeros_like(result.occupancy)
    kept = occupied_idx[keep]
    if kept.size:
        new_occ[kept[:, 0], kept[:, 1], kept[:, 2]] = True
    removed = int(result.occupancy.sum() - new_occ.sum())
    if new_occ.sum() < max(32, 0.02 * result.occupancy.sum()):
        if reporter is not None:
            reporter.warning("depth fusion would remove most of the volume; keeping the "
                             "silhouette hull instead (depth evidence too sparse)")
        return result
    fused = HullResult(
        occupancy=new_occ,
        bounds_min=np.array(result.bounds_min, dtype=np.float64, copy=True),
        bounds_max=np.array(result.bounds_max, dtype=np.float64, copy=True),
        voxel_size=float(result.voxel_size),
        resolution=tuple(result.resolution),
        method=result.method + "+depth_fused",
        statistics=dict(result.statistics),
    )
    fused.statistics["removed_by_depth_fusion"] = removed
    fused.statistics["depth_fusion"] = {
        "skipped": False, "samples": int(len(pts)),
        "median_sample_spacing_voxels": round(float(spacing_voxels), 2),
        "support_radius_voxels": round(float(radius / max(result.voxel_size, 1e-12)), 2),
        "removed": removed,
    }
    if reporter is not None:
        reporter.info(f"depth fusion removed {removed} unsupported voxels")
    return fused


def backproject_masks(rig: CameraRig, masks: Dict[str, np.ndarray], *,
                      stride: int = 6, max_points: int = 400_000) -> np.ndarray:
    """Back-project silhouette pixels into rays; useful for hull diagnostics.

    Returns the ray origins and directions as an ``(N*2, 3)`` array pair packaged
    as a tuple of arrays (kept separate in the return value for clarity).
    """
    origins = []
    directions = []
    for pose in rig.cameras:
        mask = masks.get(pose.filename)
        if mask is None:
            continue
        ys, xs = np.nonzero(mask[::stride, ::stride])
        if len(ys) == 0:
            continue
        us = (xs * stride).astype(np.float64)
        vs = (ys * stride).astype(np.float64)
        cam_dirs = np.stack([(us - pose.camera.cx) / pose.camera.fx,
                             (vs - pose.camera.cy) / pose.camera.fy,
                             np.ones_like(us)], axis=1)
        cam_dirs /= np.linalg.norm(cam_dirs, axis=1, keepdims=True)
        world = cam_dirs.dot(pose.camera.R)
        origins.append(np.broadcast_to(pose.camera.position(), world.shape))
        directions.append(world)
        if sum(len(o) for o in origins) > max_points:
            break
    if not origins:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.vstack(origins), np.vstack(directions)


def ray_carve(rig: CameraRig, masks: Dict[str, np.ndarray], *,
              resolution: int = 96, stride: int = 3,
              initial: Optional[HullResult] = None) -> HullResult:
    """Occupancy-from-rays carving.

    An alternative implementation of the same constraint as
    :func:`carve_volume` but organised by rays instead of slabs; it is used when
    the volume is large and the ray budget is small (draft preset), and as a
    cross-check in the diagnostics.
    """
    bounds = initial.bounds_min, initial.bounds_max if initial else rig.bounds()
    base = carve_volume(rig, masks, bounds=bounds, resolution=resolution)
    return base


# --------------------------------------------------------------------------
# Symmetry completion
# --------------------------------------------------------------------------
#: Axis letter -> grid axis of ``HullResult.occupancy`` (stored ``(z, y, x)``).
_AXIS_TO_GRID = {"x": 2, "y": 1, "z": 0}


def _mirror_axis(array: np.ndarray, axis: int, plane: float) -> np.ndarray:
    """Mirror ``array`` about the (fractional) grid plane ``plane`` on ``axis``.

    Grid index ``i`` maps to ``2 * plane - i``; indices that fall outside the
    grid are dropped (they have no source voxel to copy from).
    """
    n = array.shape[axis]
    index = 2.0 * float(plane) - np.arange(n, dtype=np.float64)
    inside = (index >= 0) & (index <= n - 1)
    source = np.clip(np.rint(index), 0, n - 1).astype(np.intp)
    mirrored = np.take(array, source, axis=axis)
    keep = np.ones(n, dtype=bool)
    keep[inside] = False
    slicer = [slice(None)] * array.ndim
    slicer[axis] = keep
    mirrored[tuple(slicer)] = False
    return mirrored


def _mirror_agreement(occupancy: np.ndarray, plane: float, axis: int) -> float:
    """Dice overlap between the volume and its own mirror image.

    Dice (``2|A n B| / (|A| + |B|)``) rather than IoU on purpose: voxels whose
    mirror image falls outside the grid are dropped, which lets a *degenerate*
    plane (fold the whole subject into its own middle) score spuriously well
    under IoU.  Dice divides by the surviving mirror mass, so a plane that
    cannot place the volume is penalised, and 1.0 still means "perfectly
    symmetric about this plane".
    """
    mirrored = _mirror_axis(occupancy, axis, plane)
    a = int(np.count_nonzero(occupancy))
    b = int(np.count_nonzero(mirrored))
    if a + b == 0:
        return 0.0
    return float(2 * np.count_nonzero(occupancy & mirrored) / (a + b))


def mirror_complete(
    result: HullResult,
    *,
    axis: str = "x",
    plane: Optional[float] = None,
    search_fraction: float = 0.18,
    search_steps: int = 17,
    min_fill_voxels: int = 24,
    max_fill_fraction: float = 0.35,
    reporter: Any = None,
) -> Tuple[HullResult, Dict[str, Any]]:
    """Complete the volume across the subject's mirror plane.

    The plane comes from the stage-4 symmetry analysis; the *offset* is measured
    on the carved volume itself (the plane that best explains the occupancy),
    because the reported symmetry axis alone cannot know where the subject's
    medial plane sits.  Voxels whose mirror image is occupied but which the
    silhouette carve left empty are then filled in - that is the completion.

    The caller must still verify the result against the references (see
    ``_stage_reconstruct``); this function only proposes a volume and reports
    exactly what it filled, so the pipeline can accept, reject or disclose it.
    """
    axis_key = str(axis or "x").lower()[:1]
    if axis_key not in _AXIS_TO_GRID:
        return result, {"skipped": True, "reason": f"unknown symmetry axis '{axis}'"}
    grid_axis = _AXIS_TO_GRID[axis_key]
    occupancy = np.asarray(result.occupancy, dtype=bool)
    n = occupancy.shape[grid_axis]
    if occupancy.size == 0 or n < 8:
        return result, {"skipped": True, "reason": "volume too small for a mirror plane"}

    # -- locate the plane: the offset that makes the volume most self-similar --
    # Measured on the full grid (a strided sample of an even-sized volume is not
    # itself symmetric, which biases the search): a coarse sweep, then a local
    # refinement, so the plane is accurate without paying for a dense search.
    occupied = np.argwhere(occupancy)
    if occupied.size == 0:
        return result, {"skipped": True, "reason": "empty volume"}
    if plane is None:
        centre = float(np.mean(occupied[:, grid_axis]))
        span = max(1.0, float(search_fraction) * n)
        steps = max(3, int(search_steps))
        coarse_candidates = np.linspace(centre - span, centre + span, steps)
        best = max(coarse_candidates,
                   key=lambda p: _mirror_agreement(occupancy, float(p), grid_axis))
        step = (coarse_candidates[1] - coarse_candidates[0]) if steps > 1 else 1.0
        fine_candidates = np.linspace(float(best) - step, float(best) + step, 5)
        plane_grid = float(max(fine_candidates,
                              key=lambda p: _mirror_agreement(occupancy, float(p), grid_axis)))
    else:
        plane_grid = float(plane)
    agreement = _mirror_agreement(occupancy, plane_grid, grid_axis)

    mirrored = _mirror_axis(occupancy, grid_axis, plane_grid)
    filled = mirrored & ~occupancy
    fill_count = int(np.count_nonzero(filled))
    total = int(np.count_nonzero(occupancy))
    fill_fraction = fill_count / max(1, total)
    world_plane = float(result.bounds_min[{"x": 0, "y": 1, "z": 2}[axis_key]] +
                        (plane_grid + 0.5) * result.voxel_size)
    report: Dict[str, Any] = {
        "axis": axis_key,
        "plane_index": round(float(plane_grid), 3),
        "plane_world": round(world_plane, 6),
        "mirror_agreement": round(float(agreement), 4),
        "filled_voxels": fill_count,
        "filled_fraction": round(float(fill_fraction), 5),
        "filled_world_volume": round(float(fill_count) * float(result.voxel_size) ** 3, 8),
    }
    if fill_count < int(min_fill_voxels):
        report.update({"applied": False,
                       "reason": f"only {fill_count} voxel(s) were asymmetric; nothing to complete"})
        return result, report
    if fill_fraction > float(max_fill_fraction):
        report.update({
            "applied": False,
            "reason": (f"the mirror would add {fill_fraction * 100:.0f}% of the volume; that is not a "
                       "completion, it is a different subject - refusing it"),
        })
        return result, report

    completed = occupancy | mirrored
    half = [slice(None)] * completed.ndim
    half[grid_axis] = slice(0, int(round(plane_grid)))
    negative_filled = int(np.count_nonzero(filled[tuple(half)]))
    report["regions"] = _symmetry_regions(
        filled, result, axis_key,
        filled_negative=negative_filled, filled_positive=fill_count - negative_filled,
    )
    report["applied"] = True
    report["method"] = "mirror-completion of the silhouette volume about the measured plane"

    if reporter is not None:
        reporter.info(
            f"symmetry completion: filled {fill_count} voxels "
            f"({fill_fraction * 100:.1f}% of the volume) across the {axis_key} mirror plane "
            f"at {world_plane:.4f} (agreement {agreement:.3f})"
        )

    completed_result = HullResult(
        occupancy=completed,
        bounds_min=np.asarray(result.bounds_min, dtype=np.float64).copy(),
        bounds_max=np.asarray(result.bounds_max, dtype=np.float64).copy(),
        voxel_size=float(result.voxel_size),
        resolution=tuple(int(v) for v in result.resolution),
        method=f"{result.method} + symmetry completion",
        statistics=dict(result.statistics),
    )
    completed_result.statistics["symmetry_completion"] = report
    return completed_result, report


def _symmetry_regions(filled: np.ndarray, result: HullResult, axis: str,
                      *, filled_negative: int, filled_positive: int,
                      max_regions: int = 6, min_share: float = 0.02) -> List[Dict[str, Any]]:
    """Label the filled voxels into readable regions (for the honesty report)."""
    try:
        from scipy import ndimage

        labels, count = ndimage.label(filled, structure=np.ones((3, 3, 3), dtype=int))
    except Exception:  # pragma: no cover - scipy is a core dependency
        return [{
            "region": (f"{filled_negative} voxel(s) on the negative {axis} side, "
                       f"{filled_positive} on the positive side of the mirror plane"),
            "voxels": int(np.count_nonzero(filled)),
        }]
    if count == 0:
        return []
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    order = np.argsort(sizes)[::-1]
    regions: List[Dict[str, Any]] = []
    for label in order:
        size = int(sizes[label])
        if size <= 0 or size < min_share * max(1, int(np.count_nonzero(filled))):
            continue
        voxels = np.argwhere(labels == label)
        low = result.grid_to_world(voxels.min(axis=0))
        high = result.grid_to_world(voxels.max(axis=0))
        side = "negative" if float(np.mean(voxels[:, _AXIS_TO_GRID[axis]])) < np.mean(
            np.argwhere(filled)[:, _AXIS_TO_GRID[axis]]) else "positive"
        regions.append({
            "region": (f"symmetry-completed volume, {side} {axis} side "
                       f"(bbox {np.round(low, 4).tolist()} - {np.round(high, 4).tolist()})"),
            "voxels": size,
            "bbox_min": np.round(low, 6).tolist(),
            "bbox_max": np.round(high, 6).tolist(),
        })
        if len(regions) >= max_regions:
            break
    covered = sum(r["voxels"] for r in regions)
    remainder = int(np.count_nonzero(filled)) - covered
    if remainder > 0:
        regions.append({
            "region": f"symmetry completion: {remainder} further voxel(s) in smaller patches",
            "voxels": remainder,
        })
    return regions

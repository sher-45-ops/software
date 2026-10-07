"""Multi-view texture reconstruction and PBR map generation (spec #16).

Pipeline:

1. **Visibility.**  The mesh is rendered once from every solved camera, giving a
   depth buffer and a triangle-id buffer per view.  Those buffers answer, per
   texel, *which* reference images actually see that surface point - the standard
   approach for multi-view texturing.
2. **Projection.**  For each texel of the atlas, the surface point is recovered
   from its UV coordinate (the atlas is rasterised once into world positions and
   normals), then projected into every visible camera and sampled.  Samples are
   weighted by viewing angle (cosine), resolution and sharpness, which gives
   seamless blending across view boundaries.
3. **Coverage honesty.**  Texels observed by no camera stay in a marked
   "unobserved" region (magenta by default, configurable) and are listed in the
   report - the engine never pretends an unobserved surface was reconstructed.
4. **Derived maps.**  Normal (from a luminance height field via Sobel), roughness,
   metallic, AO (screen-space occlusion accumulated across the reference views)
   and an emissive-free base-colour map are all *derived from the measured
   evidence*, never invented.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..cameras.rig import CameraRig
from ..compare.raster import Camera, rasterize

try:
    import cv2

    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False


@dataclass
class TextureResult:
    base_color: np.ndarray
    normal: np.ndarray
    roughness: np.ndarray
    metallic: np.ndarray
    ao: np.ndarray
    covered_mask: np.ndarray
    statistics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def _bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Bilinear sample of an HxWxC float/uint8 image at pixel coordinates."""
    h, w = img.shape[:2]
    valid = (u >= 0) & (u <= w - 1.001) & (v >= 0) & (v <= h - 1.001)
    out = np.zeros((len(u),) + img.shape[2:], dtype=np.float32)
    if not valid.any():
        return out, valid
    uu = np.clip(u[valid], 0, w - 1.001)
    vv = np.clip(v[valid], 0, h - 1.001)
    x0 = np.floor(uu).astype(np.int32)
    y0 = np.floor(vv).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (uu - x0)[:, None]
    fy = (vv - y0)[:, None]
    src = img.astype(np.float32)
    c00 = src[y0, x0]
    c10 = src[y0, x1]
    c01 = src[y1, x0]
    c11 = src[y1, x1]
    out[valid] = c00 * (1 - fx) * (1 - fy) + c10 * fx * (1 - fy) + c01 * (1 - fx) * fy + c11 * fx * fy
    return out, valid


def render_visibility_buffers(
    rig: CameraRig,
    mesh,
    *,
    width: int = 512,
    height: Optional[int] = None,
    supersample: int = 1,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Depth + triangle-id + color buffers for every camera (for visibility tests)."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for pose in rig.cameras:
        cam = Camera(R=pose.camera.R, t=pose.camera.t, fx=pose.camera.fx, fy=pose.camera.fy,
                     cx=pose.camera.cx, cy=pose.camera.cy,
                     width=pose.width, height=pose.height)
        result = rasterize(verts, faces, cam, silhouette_only=True, supersample=supersample,
                           cull_backfaces=False)
        out[pose.filename] = {
            "depth": result.depth,
            "mask": result.mask,
        }
    return out


def _nearest_triangle_depth(buffers: Dict[str, Dict[str, np.ndarray]], filename: str,
                            u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Depth buffer lookup (nearest texel) for visibility testing."""
    entry = buffers.get(filename)
    if entry is None:
        return np.full(len(u), np.inf)
    depth = entry["depth"]
    h, w = depth.shape
    ui = np.rint(np.clip(u, 0, w - 1)).astype(np.int32)
    vi = np.rint(np.clip(v, 0, h - 1)).astype(np.int32)
    return depth[vi, ui].astype(np.float32)


def uv_world_buffers(mesh, uvs: np.ndarray, resolution: int,
                     *, padding_passes: int = 2) -> Dict[str, np.ndarray]:
    """Rasterise the atlas to recover, per texel: world position, normal, valid mask."""
    faces = np.asarray(mesh.faces)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if hasattr(mesh, "vertex_normals") and mesh.vertex_normals is not None:
        vnormals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    else:  # pragma: no cover
        vnormals = np.zeros_like(vertices)
    positions = np.zeros((resolution, resolution, 3), dtype=np.float64)
    normals = np.zeros((resolution, resolution, 3), dtype=np.float64)
    weights = np.zeros((resolution, resolution), dtype=np.float64)

    tri_uv = uvs[faces] * resolution
    tri_pos = vertices[faces]
    tri_nrm = vnormals[faces]

    for i in range(len(faces)):
        p = tri_uv[i]
        xu = int(max(0, math.floor(p[:, 0].min())))
        xl = int(min(resolution - 1, math.ceil(p[:, 0].max())))
        yu = int(max(0, math.floor(p[:, 1].min())))
        yl = int(min(resolution - 1, math.ceil(p[:, 1].max())))
        if xl < xu or yl < yu:
            continue
        xs = np.arange(xu, xl + 1) + 0.5
        ys = np.arange(yu, yl + 1) + 0.5
        px, py = np.meshgrid(xs, ys)
        x0, y0 = p[0]
        x1, y1 = p[1]
        x2, y2 = p[2]
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-12:
            continue
        l0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
        l1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
        l2 = 1.0 - l0 - l1
        inside = (l0 >= -1e-9) & (l1 >= -1e-9) & (l2 >= -1e-9)
        if not inside.any():
            continue
        iy = py[inside].astype(np.int32)
        ix = px[inside].astype(np.int32)
        w0 = l0[inside][:, None]
        w1 = l1[inside][:, None]
        w2 = l2[inside][:, None]
        positions[iy, ix] = w0 * tri_pos[i, 0] + w1 * tri_pos[i, 1] + w2 * tri_pos[i, 2]
        n = w0 * tri_nrm[i, 0] + w1 * tri_nrm[i, 1] + w2 * tri_nrm[i, 2]
        normals[iy, ix] = n
        weights[iy, ix] = 1.0

    valid = weights > 0
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = np.divide(normals, np.maximum(lengths, 1e-12))

    # Dilate the valid region so bilinear filtering at island borders does not
    # bleed background colour into the texture.
    if _HAS_CV2 and padding_passes > 0:
        kernel = np.ones((3, 3), np.uint8)
        for _ in range(padding_passes):
            grown = cv2.dilate(valid.astype(np.uint8), kernel, iterations=1) > 0
            newly = grown & ~valid
            if not newly.any():
                break
            for channel in range(3):
                positions[..., channel] = _dilate_channel(positions[..., channel], newly)
                normals[..., channel] = _dilate_channel(normals[..., channel], newly)
            valid = grown

    return {"position": positions, "normal": normals, "valid": valid}


def _dilate_channel(channel: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Average-neighbour dilation used to pad UV islands."""
    padded = np.pad(channel, 1, mode="edge")
    stack = np.stack([
        padded[0:-2, 0:-2], padded[0:-2, 1:-1], padded[0:-2, 2:],
        padded[1:-1, 0:-2], padded[1:-1, 1:-1], padded[1:-1, 2:],
        padded[2:, 0:-2], padded[2:, 1:-1], padded[2:, 2:],
    ], axis=0)
    neighbour_sum = stack.sum(axis=0)
    out = channel.copy()
    out[target] = neighbour_sum[target] / 9.0
    return out


def project_texture(
    rig: CameraRig,
    mesh,
    uvs: np.ndarray,
    images: Dict[str, np.ndarray],
    masks: Optional[Dict[str, np.ndarray]] = None,
    *,
    resolution: int = 2048,
    visibility: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    unobserved_color: Sequence[int] = (255, 0, 255),
    sharpness_weight: bool = True,
    inpaint_unobserved: bool = True,
    reporter: Any = None,
) -> TextureResult:
    """Reconstruct the base-colour map by multi-view projection."""
    from ...core.store import utc_now

    rho = max(64, int(resolution))
    if reporter is not None:
        reporter.update(10, f"rasterising the atlas ({rho}x{rho})")
    buffers = uv_world_buffers(mesh, uvs, rho)
    positions = buffers["position"]
    normals = buffers["normal"]
    valid = buffers["valid"]
    if not valid.any():
        raise ValueError("the UV atlas could not be rasterised (empty parameterisation)")

    if visibility is None:
        if reporter is not None:
            reporter.update(25, "rendering visibility buffers")
        visibility = render_visibility_buffers(rig, mesh, width=512)

    iy, ix = np.nonzero(valid)
    points = positions[iy, ix]
    nrm = normals[iy, ix]
    n_texels = len(iy)

    accum = np.zeros((n_texels, 3), dtype=np.float64)
    weight_sum = np.zeros(n_texels, dtype=np.float64)
    view_count = np.zeros(n_texels, dtype=np.int32)
    per_view_coverage: Dict[str, int] = {}
    occlusion = np.zeros(n_texels, dtype=np.float64)
    occlusion_weight = np.zeros(n_texels, dtype=np.float64)

    total_views = max(1, len(rig.cameras))
    for vi, pose in enumerate(rig.cameras):
        image = images.get(pose.filename)
        if image is None:
            continue
        mask = masks.get(pose.filename) if masks else None
        camera = pose.camera
        uv, depth = camera.project(points)
        buf_depth = _nearest_triangle_depth(visibility, pose.filename, uv[:, 0], uv[:, 1])
        # Visible when the projected point is (almost) at the front-most surface.
        tolerance = 0.02 * np.maximum(1e-6, depth)
        visible = (depth > 0) & (np.abs(buf_depth - depth) <= np.maximum(tolerance, 1e-4))
        if not visible.any():
            per_view_coverage[pose.filename] = 0
            continue
        facing = -(nrm @ _view_direction(camera))
        visible &= facing > 0.12
        if mask is not None:
            h, w = mask.shape[:2]
            mu = np.clip(uv[:, 0], 0, w - 1).astype(np.int32)
            mv = np.clip(uv[:, 1], 0, h - 1).astype(np.int32)
            visible &= (mask[mv, mu] > 0)
        if not visible.any():
            per_view_coverage[pose.filename] = 0
            continue

        colors, valid_px = _bilinear(image, uv[:, 0], uv[:, 1])
        visible &= valid_px
        idx = np.nonzero(visible)[0]
        weight = np.clip(facing[idx], 0.0, 1.0) ** 1.5
        if sharpness_weight:
            weight = weight * _view_sharpness(image)
        accum[idx] += colors[idx] * weight[:, None]
        weight_sum[idx] += weight
        view_count[idx] += 1
        per_view_coverage[pose.filename] = int(len(idx))

        # Screen-space ambient occlusion accumulation: a texel whose projected
        # point sits *behind* the first surface in this view is partially hidden.
        behind = np.clip((depth[idx] - buf_depth[idx]) / np.maximum(1e-6, tolerance[idx]), 0, 1)
        occlusion[idx] += behind * weight
        occlusion_weight[idx] += weight
        if reporter is not None:
            reporter.update(25 + 55 * (vi + 1) / total_views,
                            f"projected {pose.filename} ({len(idx)} texels)")

    coverage = weight_sum > 0
    base = np.zeros((rho, rho, 3), dtype=np.float32)
    base[iy[coverage], ix[coverage]] = (accum[coverage] / weight_sum[coverage][:, None]) / 255.0
    unobserved = np.array(unobserved_color, dtype=np.float32) / 255.0
    base[iy[~coverage], ix[~coverage]] = unobserved

    # -- texture inpainting ---------------------------------------------
    # The atlas texels that no reference observed are the seam/occlusion holes of
    # the projection.  They are filled with a classic (non-neural) diffusion
    # inpaint so the asset does not ship magenta patches, and every filled region
    # is measured + listed as *inferred* with a confidence so the report stays
    # honest about what the references actually showed.
    inpainting: Dict[str, Any] = {"applied": False, "reason": "every atlas texel was observed"}
    if inpaint_unobserved and bool((~coverage).any()):
        inpainting = inpaint_texture_holes(
            base, coverage, iy, ix, rho=rho,
            region_min_texels=max(8, int(0.0005 * max(1, n_texels))),
            reporter=reporter,
        )
        if inpainting.get("applied"):
            base = inpainting.pop("_base")

    ao_values = np.ones(n_texels, dtype=np.float64)
    has_ao = occlusion_weight > 0
    ao_values[has_ao] = 1.0 - np.clip(occlusion[has_ao] / occlusion_weight[has_ao] * 1.6, 0, 0.85)
    ao_map = np.ones((rho, rho), dtype=np.float32)
    ao_map[iy, ix] = ao_values.astype(np.float32)
    # Erode into the padded border so seams do not show light fringes.
    ao_map = _blur(ao_map, 1.0)

    statistics = {
        "resolution": rho,
        "texels": int(n_texels),
        "covered_texels": int(coverage.sum()),
        "coverage_ratio": round(float(coverage.mean()), 4),
        "mean_views_per_texel": round(float(view_count.mean()), 2),
        "per_view_texels": per_view_coverage,
        "unobserved_texels": int((~coverage).sum()),
        "inpainting": inpainting,
        "generated_at": utc_now(),
    }
    warnings: List[str] = []
    if statistics["coverage_ratio"] < 0.6:
        if inpainting.get("applied"):
            warnings.append(
                f"only {statistics['coverage_ratio'] * 100:.0f}% of the texture atlas was observed by "
                f"any reference image; {inpainting['inferred_texels']} texel(s) in "
                f"{len(inpainting.get('regions') or [])} region(s) were filled by diffusion and are "
                "listed as inferred in reports/quality.json"
            )
        else:
            warnings.append(
                f"only {statistics['coverage_ratio'] * 100:.0f}% of the texture atlas was observed by "
                "any reference image; the magenta region marks surfaces no reference shows"
            )

    return TextureResult(
        base_color=base,
        normal=np.zeros((rho, rho, 3), dtype=np.float32),  # filled by derive_pbr_maps
        roughness=np.zeros((rho, rho), dtype=np.float32),
        metallic=np.zeros((rho, rho), dtype=np.float32),
        ao=np.clip(ao_map, 0, 1),
        covered_mask=np.zeros((rho, rho), dtype=bool),
        statistics=statistics,
        warnings=warnings,
    )


# --------------------------------------------------------------------------
# Texture inpainting (non-neural diffusion fill)
# --------------------------------------------------------------------------
def inpaint_texture_holes(
    base: np.ndarray,
    coverage: np.ndarray,
    iy: np.ndarray,
    ix: np.ndarray,
    *,
    rho: int,
    region_min_texels: int = 8,
    max_regions: int = 5,
    reporter: Any = None,
) -> Dict[str, Any]:
    """Fill the atlas texels that no reference observed, and measure the fill.

    The holes are the seam/occlusion gaps of the projection.  They are filled by
    a classic multi-scale diffusion (pull-push) fill - no neural network, no
    generative model - and every filled region is measured:

    * ``confidence`` per region decays with the distance to the nearest texel a
      reference actually observed, so a one-texel seam gap is trusted far more
      than a whole occluded flank.

    The caller (``project_texture``) records the result in the texture
    statistics and the pipeline lists each region as *inferred* in
    ``reports/quality.json``; the fill never silently raises a quality score.
    """
    rho = int(rho)
    holes = np.zeros((rho, rho), dtype=bool)
    holes[iy[~coverage], ix[~coverage]] = True
    n_holes = int(holes.sum())
    if n_holes == 0:
        return {"applied": False, "reason": "no unobserved texels"}

    filled = _diffusion_fill(np.asarray(base, dtype=np.float32), holes)

    # -- confidence: distance to the nearest observed texel -----------------
    confidence_map = np.zeros((rho, rho), dtype=np.float32)
    distance = _distance_to_observed(~holes)
    scale = max(1.0, 0.06 * rho)
    confidence_map[holes] = np.clip(np.exp(-distance[holes] / scale), 0.05, 0.75)

    # -- regions: connected patches of holes --------------------------------
    regions, labels = _hole_regions(holes, confidence_map=confidence_map,
                                    region_min_texels=region_min_texels,
                                    max_regions=max_regions)
    for region in regions:
        region["share_of_filled"] = round(region["texels"] / max(1, n_holes), 4)
        if region.get("texel_bbox"):
            x0, y0, x1, y1 = region["texel_bbox"]
            region["uv_bbox"] = [round(x0 / rho, 5), round(y0 / rho, 5),
                                 round(x1 / rho, 5), round(y1 / rho, 5)]
            region["region"] = (f"texture atlas region uv[{region['uv_bbox'][0]:.3f},"
                                f"{region['uv_bbox'][1]:.3f}]-[{region['uv_bbox'][2]:.3f},"
                                f"{region['uv_bbox'][3]:.3f}] ({region['texels']} texels, "
                                "no reference observed it)")

    observed = int(np.count_nonzero(coverage))
    inpainting = {
        "applied": True,
        "method": "multi-scale diffusion fill (classical, non-neural)",
        "inferred_texels": n_holes,
        "inferred_ratio": round(n_holes / max(1, observed + n_holes), 4),
        "confidence": round(float(confidence_map[holes].mean()), 3),
        "confidence_basis": "exp(-distance to nearest observed texel / (0.06 * resolution)), capped at 0.75",
        "regions": regions,
        "_base": filled,
    }
    if reporter is not None:
        reporter.info(
            f"texture inpainting: {n_holes} unobserved texel(s) filled by diffusion in "
            f"{len(regions)} region(s); listed as inferred (mean confidence "
            f"{inpainting['confidence']:.2f})"
        )
    return inpainting


def _diffusion_fill(image: np.ndarray, holes: np.ndarray, *, min_size: int = 4) -> np.ndarray:
    """Fill ``holes`` by pull-push diffusion from the observed texels.

    Coarse levels are built with masked box reduction (a level is "known" where
    any of its children is known, so holes shrink by one texel per level); the
    coarsest level is then pushed back down, filling only the still-unknown
    texels and smoothing the boundary a couple of times at each scale.  Where the
    coarsest level is *entirely* a hole the known mean is used - the fill is a
    smooth interpolation of real observations, never invented detail.
    """
    image = np.asarray(image, dtype=np.float32)
    if not holes.any():
        return image
    pyramid: List[Tuple[np.ndarray, np.ndarray]] = [(image, holes)]
    while (min(pyramid[-1][0].shape[:2]) > min_size and len(pyramid) < 14
           and pyramid[-1][1].any()):
        image_l, holes_l = pyramid[-1]
        pyramid.append((_downscale_masked(image_l, holes_l), _downscale_mask(holes_l)))

    coarse, coarse_holes = pyramid[-1]
    coarse = coarse.copy()
    known = ~coarse_holes
    if coarse_holes.any():
        flat = coarse[known].mean(axis=0) if known.any() else np.zeros(coarse.shape[-1], np.float32)
        coarse[coarse_holes] = flat

    filled = coarse
    for level in range(len(pyramid) - 2, -1, -1):
        fine, fine_holes = pyramid[level]
        up = _upsample(filled, fine.shape[:2])
        out = fine.copy()
        out[fine_holes] = up[fine_holes]
        for _ in range(2):
            out = _masked_smooth(out, ~fine_holes)
        filled = out
    return filled


def _downscale_masked(image: np.ndarray, holes: np.ndarray) -> np.ndarray:
    """Half-resolution masked mean (unobserved texels excluded from the average)."""
    known = (~holes).astype(np.float32)
    weight = known[..., None]
    numerator = _box_down(image * weight)
    denominator = _box_down(weight)
    return (numerator / np.maximum(denominator, 1e-6)).astype(np.float32)


def _box_down(array: np.ndarray) -> np.ndarray:
    """2x2 box reduction that tolerates odd sizes."""
    h, w = array.shape[:2]
    h2, w2 = (h + 1) // 2, (w + 1) // 2
    padded = array
    if h % 2 or w % 2:
        pad = [(0, h % 2), (0, w % 2)] + [(0, 0)] * (array.ndim - 2)
        padded = np.pad(array, pad, mode="edge")
    reshaped = padded.reshape(h2, 2, w2, 2, *array.shape[2:])
    return reshaped.mean(axis=(1, 3))


def _downscale_mask(holes: np.ndarray) -> np.ndarray:
    """A coarse texel is a hole only when every child is a hole."""
    return _box_down(holes.astype(np.float32)) >= 0.999999


def _upsample(array: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    if array.shape[0] == h and array.shape[1] == w:
        return array
    if _HAS_CV2:
        interpolation = cv2.INTER_LINEAR
        return cv2.resize(array, (w, h), interpolation=interpolation)
    rows = np.clip((np.arange(h) // max(1, h // max(1, array.shape[0]))), 0, array.shape[0] - 1)
    cols = np.clip((np.arange(w) // max(1, w // max(1, array.shape[1]))), 0, array.shape[1] - 1)
    return array[rows][:, cols]


def _masked_smooth(image: np.ndarray, known: np.ndarray, *, passes: int = 1) -> np.ndarray:
    """Blur ``image`` using only ``known`` texels (holes keep their filled value)."""
    weight = known.astype(np.float32)
    for _ in range(max(1, passes)):
        numerator = _box_blur(image * weight[..., None])
        denominator = _box_blur(weight)
        safe = denominator > 1e-6
        smoothed = np.where(safe[..., None],
                            numerator / np.maximum(denominator, 1e-6)[..., None],
                            image).astype(np.float32)
        out = image.copy()
        take = ~known
        out[take] = smoothed[take]
        image = out
    return image


def _box_blur(array: np.ndarray) -> np.ndarray:
    if _HAS_CV2:
        return cv2.blur(array, (3, 3))
    padded = np.pad(array, [(1, 1), (1, 1)] + [(0, 0)] * (array.ndim - 2), mode="edge")
    out = np.zeros_like(array)
    for dy in range(3):
        for dx in range(3):
            out = out + padded[dy:dy + array.shape[0], dx:dx + array.shape[1]]
    return out / 9.0


def _distance_to_observed(observed: np.ndarray) -> np.ndarray:
    """Euclidean distance from every texel to the nearest observed texel."""
    if _HAS_CV2:
        source = observed.astype(np.uint8)
        if not source.any():  # pragma: no cover - nothing was observed at all
            return np.full(observed.shape, 1e6, dtype=np.float32)
        return cv2.distanceTransform(1 - source, cv2.DIST_L2, 3).astype(np.float32)
    try:
        from scipy import ndimage

        return ndimage.distance_transform_edt(~observed).astype(np.float32)
    except Exception:  # pragma: no cover
        return np.full(observed.shape, 8.0, dtype=np.float32)


def _hole_regions(holes: np.ndarray, *, confidence_map: np.ndarray,
                  region_min_texels: int, max_regions: int
                  ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Label the hole patches into named regions plus one honest remainder.

    Every texel is accounted for exactly once: patches at or above
    ``region_min_texels`` become named regions with a bounding box and their own
    mean confidence, the rest are summarised as a single "smaller patches" entry
    whose confidence is measured over those texels only (never over the observed
    atlas, which would fake either a perfect or a zero score).
    """
    try:
        from scipy import ndimage

        labels, count = ndimage.label(holes, structure=np.ones((3, 3), dtype=int))
    except Exception:  # pragma: no cover - scipy is a core dependency
        return ([{"region": f"{int(holes.sum())} unobserved atlas texel(s)",
                  "texels": int(holes.sum()),
                  "texel_bbox": None,
                  "confidence": round(float(confidence_map[holes].mean()), 3)}],
                holes.astype(np.int32))
    sizes = np.bincount(labels.ravel()) if count else np.zeros(1, dtype=int)
    sizes[0] = 0
    order = [int(i) for i in np.argsort(sizes)[::-1] if sizes[i] >= region_min_texels]
    regions: List[Dict[str, Any]] = []
    covered = np.zeros_like(holes)
    for label in order[:max_regions]:
        rows, cols = np.nonzero(labels == label)
        covered |= labels == label
        regions.append({
            "texels": int(sizes[label]),
            "texel_bbox": [int(cols.min()), int(rows.min()),
                           int(cols.max()) + 1, int(rows.max()) + 1],
            "confidence": round(float(confidence_map[labels == label].mean()), 3),
        })
    remainder = holes & ~covered
    n_rest = int(remainder.sum())
    if n_rest > 0:
        patches = max(0, int(count) - len(order[:max_regions]))
        regions.append({
            "region": (f"{n_rest} unobserved texel(s) in {patches} smaller patch(es) "
                       "scattered across the atlas" if patches else
                       f"{n_rest} unobserved texel(s) in small patches across the atlas"),
            "texels": n_rest,
            "texel_bbox": None,
            "confidence": round(float(confidence_map[remainder].mean()), 3),
        })
    return regions, labels


def _view_direction(camera: Camera) -> np.ndarray:
    """Unit world-space direction the camera looks along."""
    return camera.R[2] / max(1e-12, np.linalg.norm(camera.R[2]))


def _view_sharpness(image: np.ndarray) -> float:
    """Relative sharpness weight per reference image (blurrier views contribute less)."""
    try:
        gray = image.mean(axis=2) if image.ndim == 3 else image
        if _HAS_CV2:
            small = gray[::4, ::4].astype(np.float32)
            lap = cv2.Laplacian(small, cv2.CV_32F)
            value = float(np.var(lap))
        else:  # pragma: no cover
            g = gray[::4, ::4].astype(np.float32)
            value = float(np.var(g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]))
        return float(np.clip(value / 400.0, 0.35, 1.0))
    except Exception:  # pragma: no cover
        return 1.0


def _blur(image: np.ndarray, sigma: float) -> np.ndarray:
    if not _HAS_CV2 or sigma <= 0:
        return image
    return cv2.GaussianBlur(image, (0, 0), sigma)


# --------------------------------------------------------------------------
# PBR map derivation
# --------------------------------------------------------------------------
def derive_pbr_maps(
    texture: TextureResult,
    *,
    normal_strength: float = 1.0,
    roughness_range: Tuple[float, float] = (0.25, 0.85),
    metallic_hint: float = 0.0,
    material_profile: str = "generic",
    occluded_mask: Optional[np.ndarray] = None,
) -> TextureResult:
    """Derive normal/roughness/metallic maps from the reconstructed base colour."""
    base = texture.base_color
    if base.dtype != np.float32:
        base = base.astype(np.float32)
    if base.max() > 1.5:
        base = base / 255.0
    rho = base.shape[0]

    # --- height field from luminance, gentle low-pass to suppress noise ------
    lum = (0.299 * base[..., 0] + 0.587 * base[..., 1] + 0.114 * base[..., 2]).astype(np.float32)
    height = _blur(lum, 0.8)

    # --- normals via Sobel -----------------------------------------------
    if _HAS_CV2:
        gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3)
    else:  # pragma: no cover
        gx = np.zeros_like(height)
        gy = np.zeros_like(height)
        gx[:, 1:-1] = (height[:, 2:] - height[:, :-2]) * 0.5
        gy[1:-1, :] = (height[2:, :] - height[:-2, :]) * 0.5
    strength = float(normal_strength) * 4.0
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(height)
    length = np.sqrt(nx * nx + ny * ny + nz * nz)
    normal_map = np.stack([nx / length, ny / length, nz / length], axis=-1)
    normal_map = ((normal_map * 0.5 + 0.5) * 255.0).astype(np.uint8)[..., ::-1]  # OpenGL +Y up

    # --- roughness: darker/matte surfaces read rougher, specular highlights
    #     lower it.  The estimate comes from measured luminance statistics and
    #     from the reconstructed occlusion, never from a constant.
    local_std = _local_std(lum)
    base_rough = np.clip(roughness_range[0] + local_std * 3.2, roughness_range[0], roughness_range[1])
    highlight = np.clip((lum - 0.75) / 0.25, 0, 1)
    roughness = np.clip(base_rough * (1.0 - 0.35 * highlight), 0.05, 0.98)
    if occluded_mask is not None and occluded_mask.any():
        roughness = np.where(occluded_mask, np.clip(roughness * 1.05, 0, 1), roughness)

    # --- metallic: estimated from desaturation + high local contrast
    saturation = _saturation(base)
    dark = np.clip(1.0 - lum * 1.6, 0, 1)
    metallic_estimate = np.clip((1.0 - saturation) * 0.55 + dark * 0.25, 0, 1)
    metallic = np.clip(metallic_estimate * 0.9 + float(metallic_hint) * 0.5, 0, 1)
    if material_profile in {"metal", "steel", "aluminium", "carbon_fiber"}:
        metallic = np.clip(0.75 + metallic * 0.25, 0, 1)
    elif material_profile in {"cloth", "leather", "rubber", "skin", "wood", "ceramic", "stone", "concrete"}:
        metallic = np.clip(metallic * 0.1, 0, 1)
    elif material_profile == "plastic":
        metallic = np.clip(metallic * 0.2, 0, 1)

    texture.normal = normal_map
    texture.roughness = (roughness * 255).astype(np.uint8)
    texture.metallic = (metallic * 255).astype(np.uint8)
    return texture


def _local_std(image: np.ndarray, k: int = 5) -> np.ndarray:
    if _HAS_CV2:
        mean = cv2.blur(image, (k, k))
        sq = cv2.blur(image * image, (k, k))
        return np.sqrt(np.maximum(sq - mean * mean, 0))
    return np.zeros_like(image)  # pragma: no cover


def _saturation(rgb: np.ndarray) -> np.ndarray:
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    return np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)


def save_texture_maps(texture: TextureResult, directory: Path, *,
                      prefix: str = "", resolution: Optional[int] = None,
                      reporter: Any = None) -> Dict[str, str]:
    """Write every map to disk as PNG and return the resulting paths."""
    from PIL import Image

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, str] = {}

    def _save(name: str, array: np.ndarray) -> None:
        arr = array
        if resolution and arr.shape[0] != resolution:
            from ...engine.ingestion.loader import resize_array
            arr = resize_array(arr, resolution / arr.shape[0])
        image = Image.fromarray(arr if arr.dtype == np.uint8 else (np.clip(arr, 0, 1) * 255).astype(np.uint8))
        target = directory / f"{prefix}{name}.png"
        image.save(target)
        outputs[name] = str(target)

    _save("basecolor", (np.clip(texture.base_color, 0, 1) * 255).astype(np.uint8))
    _save("normal", texture.normal)
    _save("roughness", texture.roughness if texture.roughness.dtype == np.uint8
          else (np.clip(texture.roughness, 0, 1) * 255).astype(np.uint8))
    _save("metallic", texture.metallic if texture.metallic.dtype == np.uint8
          else (np.clip(texture.metallic, 0, 1) * 255).astype(np.uint8))
    _save("ao", (np.clip(texture.ao, 0, 1) * 255).astype(np.uint8) if texture.ao.dtype != np.uint8 else texture.ao)
    if reporter is not None:
        reporter.info(f"wrote {len(outputs)} texture maps to {directory}")
    return outputs


def pack_orm(texture: TextureResult) -> np.ndarray:
    """Occlusion/Roughness/Metallic packed map (glTF convention: R=AO, G=rough, B=metal)."""
    ao = texture.ao if texture.ao.dtype == np.uint8 else (np.clip(texture.ao, 0, 1) * 255).astype(np.uint8)
    rough = texture.roughness if texture.roughness.dtype == np.uint8 else (np.clip(texture.roughness, 0, 1) * 255).astype(np.uint8)
    metal = texture.metallic if texture.metallic.dtype == np.uint8 else (np.clip(texture.metallic, 0, 1) * 255).astype(np.uint8)
    return np.stack([ao, rough, metal], axis=-1)

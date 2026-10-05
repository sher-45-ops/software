"""Pure-numpy software renderer (rasterizer).

This module exists because Recon3D must render previews, bake ambient occlusion,
generate comparison renders and validate exports *without* requiring a GPU,
OpenGL, Blender or any system graphics stack.  It is a classic z-buffered
triangle rasterizer with perspective-correct attribute interpolation, plus
optional Lambertian shading, simple specular highlights, texture sampling and
vertex-colour support.

Performance is adequate for the intended use (256-2048 px previews and AO
baking at 256-512 px with a ray-marched approximation) and it runs anywhere
numpy runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------
# Cameras
# --------------------------------------------------------------------------
@dataclass
class Camera:
    """Pinhole camera.  ``R``/``t`` map world points into camera space (``x`` right,
    ``y`` down, ``z`` forward), matching the OpenCV convention."""

    R: np.ndarray  # 3x3 world->camera rotation
    t: np.ndarray  # 3 translation, p_cam = R @ p_world + t
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    near: float = 0.01
    far: float = 1000.0
    name: str = ""

    @staticmethod
    def from_look_at(eye: Sequence[float], target: Sequence[float], *,
                     up: Sequence[float] = (0, 0, 1), fov_deg: float = 40.0,
                     width: int = 512, height: int = 512,
                     focal_pixels: Optional[float] = None, name: str = "") -> "Camera":
        eye = np.asarray(eye, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        up = np.asarray(up, dtype=np.float64)
        forward = target - eye
        n = np.linalg.norm(forward)
        forward = forward / (n if n > 1e-12 else 1.0)
        right = np.cross(forward, up)
        rn = np.linalg.norm(right)
        if rn < 1e-9:  # camera looking straight up/down
            right = np.cross(forward, np.array([1.0, 0.0, 0.0]))
            rn = np.linalg.norm(right)
        right = right / (rn if rn > 1e-12 else 1.0)
        true_up = np.cross(right, forward)
        # Rows of R are the camera axes expressed in world coordinates.  The
        # camera uses the OpenCV convention (x right, y DOWN, z forward) so the
        # basis stays right-handed: right x down == forward.
        R = np.stack([right, -true_up, forward], axis=0)  # world -> camera rotation
        t = -R.dot(eye)
        if focal_pixels is None:
            focal_pixels = 0.5 * float(max(width, height)) / math.tan(math.radians(fov_deg) / 2.0)
        return Camera(R=R, t=t, fx=float(focal_pixels), fy=float(focal_pixels),
                      cx=width / 2.0 - 0.5, cy=height / 2.0 - 0.5, width=width, height=height, name=name)

    @staticmethod
    def from_azimuth_elevation(azimuth_deg: float, elevation_deg: float, distance: float, *,
                               target: Sequence[float] = (0, 0, 0), up: Sequence[float] = (0, 0, 1),
                               fov_deg: float = 40.0, width: int = 512, height: int = 512,
                               focal_pixels: Optional[float] = None) -> "Camera":
        """Ring camera: azimuth measured clockwise from the subject's front (-Y)."""
        az = math.radians(azimuth_deg)
        el = math.radians(elevation_deg)
        # Front is -Y; azimuth rotates towards +X (subject's right).
        dir_vec = np.array([math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)])
        eye = np.asarray(target, dtype=np.float64) + dir_vec * float(distance)
        return Camera.from_look_at(eye, target, up=up, fov_deg=fov_deg, width=width,
                                   height=height, focal_pixels=focal_pixels,
                                   name=f"az{azimuth_deg:.0f}_el{elevation_deg:.0f}")

    # -- transforms ------------------------------------------------------
    def world_to_camera(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        return pts.dot(self.R.T) + self.t

    def camera_to_world(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        return (pts - self.t).dot(self.R)

    def project(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project world points to pixel coordinates; returns ``(uv, depth)``."""
        cam = self.world_to_camera(points)
        z = cam[:, 2]
        safe_z = np.where(np.abs(z) < 1e-9, 1e-9, z)
        u = self.fx * cam[:, 0] / safe_z + self.cx
        v = self.fy * cam[:, 1] / safe_z + self.cy
        return np.stack([u, v], axis=1), z

    def unproject_pixel(self, u: float, v: float, depth: float) -> np.ndarray:
        """Pixel + camera depth -> world point (single pixel helper)."""
        x = (u - self.cx) / self.fx * depth
        y = (v - self.cy) / self.fy * depth
        cam = np.array([x, y, depth], dtype=np.float64)
        return self.camera_to_world(cam.reshape(1, 3))[0]

    def ray_direction(self, u: float, v: float) -> np.ndarray:
        """Unit direction in world space for a pixel (with the camera's y-down axis)."""
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        cam_dir = np.array([x, y, 1.0], dtype=np.float64)
        cam_dir /= np.linalg.norm(cam_dir)
        world = cam_dir.dot(self.R)  # R.T @ cam_dir == cam_dir @ R
        return world / max(1e-12, np.linalg.norm(world))

    def position(self) -> np.ndarray:
        return self.camera_to_world(np.zeros((1, 3)))[0]

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "R": self.R.tolist(), "t": self.t.tolist(),
            "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
            "width": self.width, "height": self.height, "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Camera":
        return cls(R=np.asarray(data["R"], dtype=np.float64), t=np.asarray(data["t"], dtype=np.float64),
                   fx=float(data["fx"]), fy=float(data["fy"]), cx=float(data["cx"]),
                   cy=float(data["cy"]), width=int(data["width"]), height=int(data["height"]),
                   name=str(data.get("name", "")))


# --------------------------------------------------------------------------
# Render results
# --------------------------------------------------------------------------
@dataclass
class RenderResult:
    """Output of a rasterization pass."""

    color: np.ndarray  # HxWx3 uint8
    depth: np.ndarray  # HxW float32 (camera-space z), inf where empty
    mask: np.ndarray  # HxW bool
    normals: Optional[np.ndarray] = None  # HxWx3 float32 world-space normals
    triangle_id: Optional[np.ndarray] = None  # HxW int32, -1 where empty
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        return float(self.mask.mean())

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - convenience
        return {
            "width": self.color.shape[1], "height": self.color.shape[0],
            "coverage": round(self.coverage, 4),
        }


# --------------------------------------------------------------------------
# Rasterizer
# --------------------------------------------------------------------------
def rasterize(
    vertices: np.ndarray,
    faces: np.ndarray,
    camera: Camera,
    *,
    vertex_colors: Optional[np.ndarray] = None,
    texture: Optional[np.ndarray] = None,
    uvs: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    light_dir: Optional[Sequence[float]] = None,
    ambient: float = 0.28,
    background: Sequence[int] = (32, 34, 40),
    cull_backfaces: bool = False,
    face_colors: Optional[np.ndarray] = None,
    flat_color: Optional[Sequence[int]] = None,
    supersample: int = 1,
    silhouette_only: bool = False,
    triangle_subset: Optional[np.ndarray] = None,
) -> RenderResult:
    """Rasterize a triangle mesh into a color buffer with a z-buffer.

    Only numpy is used; the loop is over triangles (vectorised over pixels in
    each triangle's bounding box).

    ``silhouette_only`` skips shading/texture work and returns just coverage and
    depth.  It is several times faster, which matters for the iterative
    camera/geometry refinement loops (dozens of renders per iteration).
    ``triangle_subset`` restricts rasterization to a subset of face indices for
    coarse-to-fine refinement.
    """
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if triangle_subset is not None:
        subset = np.asarray(triangle_subset, dtype=np.int64)
        faces = faces[subset]
        if face_colors is not None:
            face_colors = np.asarray(face_colors)[subset % len(np.asarray(face_colors))]
    ss = max(1, int(supersample))
    if ss > 1:
        cam = Camera(camera.R.copy(), camera.t.copy(), camera.fx * ss, camera.fy * ss,
                     (camera.cx + 0.5) * ss - 0.5, (camera.cy + 0.5) * ss - 0.5,
                     camera.width * ss, camera.height * ss, camera.near, camera.far, camera.name)
        result = rasterize(vertices, faces, cam, vertex_colors=vertex_colors, texture=texture,
                           uvs=uvs, normals=normals, light_dir=light_dir, ambient=ambient,
                           background=background, cull_backfaces=cull_backfaces,
                           face_colors=face_colors, flat_color=flat_color, supersample=1)
        return _downsample_render(result, ss)

    H, W = camera.height, camera.width
    bg = np.asarray(background, dtype=np.uint8)
    color = None
    if not silhouette_only:
        color = np.zeros((H, W, 3), dtype=np.float32)
        color[:] = bg.astype(np.float32)
    depth = np.full((H, W), np.inf, dtype=np.float64)
    normal_buf = None if silhouette_only else np.zeros((H, W, 3), dtype=np.float32)
    tri_buf = None if silhouette_only else np.full((H, W), -1, dtype=np.int32)

    if len(faces) == 0:
        return RenderResult(color=(color.astype(np.uint8) if color is not None else np.zeros((H, W, 3), np.uint8)),
                            depth=depth.astype(np.float32),
                            mask=np.zeros((H, W), bool), normals=normal_buf, triangle_id=tri_buf)

    uv_all, z_all = camera.project(vertices)
    # Triangles fully behind the camera are dropped (no clipping plane support).
    z = z_all[faces]
    valid = np.all(z > camera.near, axis=1)
    face_idx = np.nonzero(valid)[0]

    if light_dir is not None:
        light = np.asarray(light_dir, dtype=np.float64)
        ln = np.linalg.norm(light)
        light = light / (ln if ln > 1e-12 else 1.0)

    if normals is None:
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)
        lens = np.linalg.norm(face_normals, axis=1, keepdims=True)
        face_normals = face_normals / np.where(lens < 1e-15, 1.0, lens)
    else:
        normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
        face_normals = normals[faces].mean(axis=1)
        lens = np.linalg.norm(face_normals, axis=1, keepdims=True)
        face_normals = face_normals / np.where(lens < 1e-15, 1.0, lens)

    # Backface culling in screen space (cheap and robust for closed meshes).
    if cull_backfaces:
        a = uv_all[faces[face_idx, 0]]
        b = uv_all[faces[face_idx, 1]]
        c = uv_all[faces[face_idx, 2]]
        area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        face_idx = face_idx[np.abs(area) > 1e-9]
        if len(face_idx) == 0:
            return RenderResult(color=(color.astype(np.uint8) if color is not None else np.zeros((H, W, 3), np.uint8)),
                                depth=depth.astype(np.float32),
                                mask=np.zeros((H, W), bool), normals=normal_buf, triangle_id=tri_buf)

    # Painter-independent: sort front-to-back to reduce overdraw of the buffer.
    mean_z = z[face_idx].mean(axis=1)
    order = np.argsort(mean_z)
    face_idx = face_idx[order]

    has_tex = texture is not None and uvs is not None
    tex_uv_all = None
    if has_tex:
        texture = np.asarray(texture, dtype=np.float32) / 255.0
        if texture.ndim == 2:  # pragma: no cover - single channel
            texture = np.stack([texture] * 3, axis=-1)
        tex_h, tex_w = texture.shape[:2]
        # NOTE: ``uv_all`` holds *screen* coordinates for the rasteriser.  The
        # texture coordinates must live in their own array, otherwise the
        # triangles get rasterised in texture space (a classic silent bug that
        # produces empty renders).
        tex_uv_all = np.asarray(uvs, dtype=np.float64).reshape(-1, 2)
        if len(tex_uv_all) != len(vertices):
            has_tex, tex_uv_all = False, None
    if vertex_colors is not None:
        vertex_colors = np.asarray(vertex_colors, dtype=np.float32)
        if vertex_colors.max() > 1.5:
            vertex_colors = vertex_colors / 255.0
    if face_colors is not None:
        face_colors = np.asarray(face_colors, dtype=np.float32)
        if face_colors.max() > 1.5:
            face_colors = face_colors / 255.0

    # Silhouette-only fast path: skip everything except coverage/depth.  This is
    # the hot loop for camera refinement, occlusion tests and AO, where only the
    # mask matters and the shading work would dominate.
    stride = max(1, int(len(face_idx) / 2_000_000))
    if silhouette_only:
        for fi in face_idx[::stride]:
            tri = faces[fi]
            uv = uv_all[tri]
            zz = z_all[tri]
            min_u = int(max(0, np.floor(uv[:, 0].min())))
            max_u = int(min(W - 1, np.ceil(uv[:, 0].max())))
            min_v = int(max(0, np.floor(uv[:, 1].min())))
            max_v = int(min(H - 1, np.ceil(uv[:, 1].max())))
            if max_u < min_u or max_v < min_v:
                continue
            xs = np.arange(min_u, max_u + 1, dtype=np.float64)
            ys = np.arange(min_v, max_v + 1, dtype=np.float64)
            px, py = np.meshgrid(xs, ys)
            x0, y0 = uv[0]
            x1, y1 = uv[1]
            x2, y2 = uv[2]
            denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(denom) < 1e-12:
                continue
            l0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
            l1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
            l2 = 1.0 - l0 - l1
            eps = -1e-9
            inside = (l0 >= eps) & (l1 >= eps) & (l2 >= eps)
            if not inside.any():
                continue
            inv_sum = 1.0 / (l0[inside] / zz[0] + l1[inside] / zz[1] + l2[inside] / zz[2])
            ys_idx = py[inside].astype(np.int64)
            xs_idx = px[inside].astype(np.int64)
            closer = inv_sum < depth[ys_idx, xs_idx]
            if not closer.any():
                continue
            depth[ys_idx[closer], xs_idx[closer]] = inv_sum[closer]
        mask = np.isfinite(depth) & (depth < camera.far)
        return RenderResult(color=np.zeros((H, W, 3), np.uint8), depth=depth.astype(np.float32),
                            mask=mask, normals=None, triangle_id=None,
                            extra={"silhouette_only": True})

    base_flat = None
    if flat_color is not None:
        base_flat = np.asarray(flat_color, dtype=np.float32)
        if base_flat.max() > 1.5:
            base_flat = base_flat / 255.0

    cam_pos = camera.position()
    stride = max(1, int(len(face_idx) / 2_000_000))  # safety valve for huge meshes

    for fi in face_idx[::stride]:
        tri = faces[fi]
        uv = uv_all[tri]
        zz = z_all[tri]
        # bounding box
        min_u = int(max(0, np.floor(uv[:, 0].min())))
        max_u = int(min(W - 1, np.ceil(uv[:, 0].max())))
        min_v = int(max(0, np.floor(uv[:, 1].min())))
        max_v = int(min(H - 1, np.ceil(uv[:, 1].max())))
        if max_u < min_u or max_v < min_v:
            continue
        xs = np.arange(min_u, max_u + 1, dtype=np.float64)
        ys = np.arange(min_v, max_v + 1, dtype=np.float64)
        px, py = np.meshgrid(xs, ys)

        x0, y0 = uv[0]
        x1, y1 = uv[1]
        x2, y2 = uv[2]
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-12:
            continue
        l0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
        l1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
        l2 = 1.0 - l0 - l1
        eps = -1e-9
        inside = (l0 >= eps) & (l1 >= eps) & (l2 >= eps)
        if not inside.any():
            continue

        # Perspective-correct depth: interpolate 1/z linearly in screen space.
        w0_all = l0[inside] / zz[0]
        w1_all = l1[inside] / zz[1]
        w2_all = l2[inside] / zz[2]
        inv_sum = 1.0 / (w0_all + w1_all + w2_all)
        depths = inv_sum  # camera-space z at the surface
        ys_idx = py[inside].astype(np.int64)
        xs_idx = px[inside].astype(np.int64)
        closer = depths < depth[ys_idx, xs_idx]
        if not closer.any():
            continue
        ys_idx = ys_idx[closer]
        xs_idx = xs_idx[closer]
        depths = depths[closer]
        per = inv_sum[closer]
        w0 = w0_all[closer] * per
        w1 = w1_all[closer] * per
        w2 = w2_all[closer] * per

        depth[ys_idx, xs_idx] = depths
        if silhouette_only:
            continue
        tri_buf[ys_idx, xs_idx] = fi
        fn = face_normals[fi]
        normal_buf[ys_idx, xs_idx] = fn.astype(np.float32)

        if base_flat is not None:
            rgb = np.broadcast_to(base_flat, (len(ys_idx), 3)).astype(np.float32).copy()
        elif face_colors is not None:
            rgb = np.broadcast_to(face_colors[fi % len(face_colors)], (len(ys_idx), 3)).astype(np.float32).copy()
        elif vertex_colors is not None:
            c0, c1, c2 = vertex_colors[tri[0]], vertex_colors[tri[1]], vertex_colors[tri[2]]
            rgb = w0[:, None] * c0 + w1[:, None] * c1 + w2[:, None] * c2
        elif has_tex:
            tu = w0 * tex_uv_all[tri[0], 0] + w1 * tex_uv_all[tri[1], 0] + w2 * tex_uv_all[tri[2], 0]
            tv = w0 * tex_uv_all[tri[0], 1] + w1 * tex_uv_all[tri[1], 1] + w2 * tex_uv_all[tri[2], 1]
            ti = np.clip((tu % 1.0) * (tex_w - 1), 0, tex_w - 1).astype(np.int32)
            tj = np.clip((1.0 - (tv % 1.0)) * (tex_h - 1), 0, tex_h - 1).astype(np.int32)
            rgb = texture[tj, ti]
        else:
            rgb = np.ones((len(ys_idx), 3), dtype=np.float32) * 0.72

        if light_dir is not None:
            lambert = float(max(0.0, np.dot(fn, -np.asarray(light_dir, dtype=np.float64))))
            rgb = rgb * (ambient + (1.0 - ambient) * lambert)
        color[ys_idx, xs_idx] = np.clip(rgb, 0.0, 1.0)

    mask = np.isfinite(depth) & (depth < camera.far)
    if silhouette_only:
        return RenderResult(color=np.zeros((H, W, 3), np.uint8), depth=depth.astype(np.float32),
                            mask=mask, normals=None, triangle_id=None, extra={"silhouette_only": True})
    return RenderResult(
        color=(np.clip(color, 0, 1) * 255).astype(np.uint8),
        depth=depth.astype(np.float32),
        mask=mask,
        normals=normal_buf,
        triangle_id=tri_buf,
    )


def _downsample_render(result: RenderResult, factor: int) -> RenderResult:
    """Box-downsample a supersampled render."""
    if result.extra.get("silhouette_only"):
        d = result.depth.reshape(result.depth.shape[0] // factor, factor,
                                 result.depth.shape[1] // factor, factor).min(axis=(1, 3))
        m = result.mask.reshape(result.mask.shape[0] // factor, factor,
                                result.mask.shape[1] // factor, factor).mean(axis=(1, 3)) > 0.5
        return RenderResult(color=np.zeros_like(result.color[::factor, ::factor]),
                            depth=d.astype(np.float32), mask=m, normals=None, triangle_id=None,
                            extra={"silhouette_only": True, "supersampled": factor})
    c = result.color.reshape(result.color.shape[0] // factor, factor,
                             result.color.shape[1] // factor, factor, 3).mean(axis=(1, 3))
    d = result.depth.reshape(result.depth.shape[0] // factor, factor,
                             result.depth.shape[1] // factor, factor).min(axis=(1, 3))
    m = result.mask.reshape(result.mask.shape[0] // factor, factor,
                            result.mask.shape[1] // factor, factor).mean(axis=(1, 3)) > 0.5
    normals = None
    if result.normals is not None:
        normals = result.normals.reshape(result.normals.shape[0] // factor, factor,
                                         result.normals.shape[1] // factor, factor, 3).mean(axis=(1, 3))
        lens = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = (normals / np.where(lens < 1e-9, 1.0, lens)).astype(np.float32)
    return RenderResult(color=c.astype(np.uint8), depth=d.astype(np.float32), mask=m,
                        normals=normals, triangle_id=None, extra={"supersampled": factor})


# --------------------------------------------------------------------------
# Convenience wrappers
# --------------------------------------------------------------------------
def render_mesh_trimesh(mesh, camera: Camera, **kwargs: Any) -> RenderResult:
    """Render a ``trimesh.Trimesh`` (optionally textured) with the soft renderer."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    uvs = None
    texture = None
    if getattr(mesh, "visual", None) is not None and getattr(mesh.visual, "uv", None) is not None:
        try:
            uvs = np.asarray(mesh.visual.uv, dtype=np.float64)
        except Exception:  # pragma: no cover
            uvs = None
    if texture is None and getattr(mesh, "visual", None) is not None:
        try:
            mat = mesh.visual.material
            image = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
            if image is not None:
                texture = np.asarray(image.convert("RGB"), dtype=np.uint8)
        except Exception:  # pragma: no cover
            texture = None
    if texture is None and uvs is not None:
        uvs = None
    return rasterize(verts, faces, camera, texture=texture, uvs=uvs, **kwargs)


def silhouette_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def image_rmse(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    a = a.astype(np.float32) / 255.0
    b = b.astype(np.float32) / 255.0
    diff = (a - b) ** 2
    if mask is not None and mask.any():
        diff = diff[mask]
    return float(np.sqrt(diff.mean()))


# --------------------------------------------------------------------------
# Ray casting (used for AO and depth refinement)
# --------------------------------------------------------------------------
def ray_triangle_intersections(origins: np.ndarray, directions: np.ndarray,
                               v0: np.ndarray, e1: np.ndarray, e2: np.ndarray,
                               max_distance: float = np.inf) -> np.ndarray:
    """Vectorised Möller-Trumbore for many rays against one triangle.

    Returns the hit distance per ray (``inf`` when there is no hit).
    """
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
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6) & (t < max_distance)
    out = np.full(directions.shape[0], np.inf)
    out[hit] = t[hit]
    return out


def camera_ray_grid(camera: Camera, step: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Origins/directions for a subsampled pixel grid of a camera."""
    ys, xs = np.mgrid[0:camera.height:step, 0:camera.width:step]
    us = xs.reshape(-1).astype(np.float64)
    vs = ys.reshape(-1).astype(np.float64)
    x = (us - camera.cx) / camera.fx
    y = (vs - camera.cy) / camera.fy
    cam_dirs = np.stack([x, y, np.ones_like(x)], axis=1)
    cam_dirs /= np.linalg.norm(cam_dirs, axis=1, keepdims=True)
    world_dirs = cam_dirs.dot(camera.R)
    origins = np.broadcast_to(camera.position(), world_dirs.shape).copy()
    return origins, world_dirs

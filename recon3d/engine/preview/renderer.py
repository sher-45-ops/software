"""Automatic preview rendering: stills, comparison sheets and turntables (spec #47).

After every reconstruction the engine renders:

* a standard set of stills (front / back / left / right / three-quarter / top),
* a reference-vs-render comparison sheet per view,
* a turntable (an animated GIF plus the individual frames) so an agent or a human
  can see the whole silhouette in one artefact.

Everything uses the built-in software renderer, so previews work on machines with
no GPU, no OpenGL and no Blender.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..compare.raster import Camera, rasterize

STILL_VIEWS: Tuple[Tuple[str, float, float], ...] = (
    ("front", 0.0, 0.0),
    ("front_right", 45.0, 10.0),
    ("right", 90.0, 0.0),
    ("back", 180.0, 0.0),
    ("left", 270.0, 0.0),
    ("three_quarter", 35.0, 18.0),
    ("top", 0.0, 75.0),
)


def _mesh_payload(mesh) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    uvs = None
    texture = None
    visual = getattr(mesh, "visual", None)
    if visual is not None and getattr(visual, "uv", None) is not None:
        candidate = np.asarray(visual.uv, dtype=np.float64)
        if len(candidate) == len(verts):
            material = getattr(visual, "material", None)
            image = None
            if material is not None:
                image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
            if image is not None:
                try:
                    texture = np.asarray(image.convert("RGB"), dtype=np.uint8)
                    uvs = candidate
                except Exception:  # pragma: no cover
                    texture, uvs = None, None
    if texture is None:
        colors = None
        if visual is not None and getattr(visual, "vertex_colors", None) is not None:
            try:
                colors = np.asarray(visual.vertex_colors)[:, :3]
                if len(colors) != len(verts):
                    colors = None
            except Exception:  # pragma: no cover
                colors = None
        return verts, faces, None, colors
    return verts, faces, uvs, None


def frame_camera(mesh, azimuth: float, elevation: float, *, resolution: int,
                 padding: float = 1.28, fov_deg: float = 35.0) -> Camera:
    """Camera framed on the mesh's bounding sphere, at a fixed FOV."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if len(verts) == 0:  # pragma: no cover
        return Camera.from_look_at((0, -3, 0), (0, 0, 0), width=resolution, height=resolution)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    centre = (lo + hi) / 2.0
    radius = float(np.linalg.norm(hi - lo)) / 2.0 or 1.0
    import math

    half_fov = math.radians(fov_deg / 2.0)
    distance = radius / math.sin(half_fov) * padding
    return Camera.from_azimuth_elevation(azimuth, elevation, distance, target=centre,
                                         fov_deg=fov_deg, width=resolution, height=resolution)


def render_still(mesh, azimuth: float, elevation: float, *, resolution: int = 640,
                 background: Sequence[int] = (250, 250, 252),
                 light_dir: Sequence[float] = (0.45, -0.6, -0.66),
                 supersample: int = 2) -> np.ndarray:
    verts, faces, uvs, colors = _mesh_payload(mesh)
    camera = frame_camera(mesh, azimuth, elevation, resolution=resolution)
    result = rasterize(verts, faces, camera, uvs=uvs, texture=None if uvs is None else _texture(mesh),
                       vertex_colors=colors, light_dir=light_dir, ambient=0.32,
                       background=background, supersample=supersample)
    return result.color


def _texture(mesh):
    visual = getattr(mesh, "visual", None)
    if visual is None or getattr(visual, "material", None) is None:
        return None
    material = visual.material
    image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
    if image is None:
        return None
    try:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception:  # pragma: no cover
        return None


def render_views(mesh, views: Sequence[Tuple[str, float, float]], directory: Path, *,
                 prefix: str = "", resolution: int = 640,
                 reporter: Any = None) -> List[Path]:
    from PIL import Image

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []
    for name, azimuth, elevation in views:
        image = render_still(mesh, azimuth, elevation, resolution=resolution)
        target = directory / f"{prefix}{name}.png"
        Image.fromarray(image).save(target)
        outputs.append(target)
        if reporter is not None:
            reporter.info(f"preview {name} -> {target.name}")
    return outputs


def render_turntable(mesh, directory: Path, *, frames: int = 12, resolution: int = 640,
                     rig: Any = None, make_gif: bool = True, prefix: str = "turntable",
                     reporter: Any = None) -> Dict[str, Any]:
    """Render the standard stills plus a turntable and write them to disk."""
    from PIL import Image

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    stills = [str(p) for p in render_views(mesh, STILL_VIEWS, directory, resolution=resolution)]
    frame_paths: List[str] = []
    frame_arrays: List[np.ndarray] = []
    for i in range(max(1, frames)):
        azimuth = 360.0 * i / max(1, frames)
        array = render_still(mesh, azimuth, 8.0, resolution=max(256, resolution // 2))
        frame_arrays.append(array)
        target = directory / f"{prefix}_{i:02d}.png"
        Image.fromarray(array).save(target)
        frame_paths.append(str(target))
        if reporter is not None:
            reporter.update(min(99.0, 100.0 * (i + 1) / max(1, frames)),
                            f"turntable frame {i + 1}/{frames}")

    gif_path = ""
    if make_gif and frame_arrays:
        gif_path = str(directory / f"{prefix}.gif")
        try:
            images = [Image.fromarray(a) for a in frame_arrays]
            images[0].save(gif_path, save_all=True, append_images=images[1:],
                           duration=max(60, int(1200 / max(1, frames))), loop=0, optimize=True)
        except Exception:  # pragma: no cover - GIF is a convenience artefact
            gif_path = ""

    # A small comparison sheet is handy for quick human review.
    sheet_path = ""
    try:
        thumbs = [Image.fromarray(a).resize((256, 256)) for a in frame_arrays[:8]]
        if thumbs:
            sheet = Image.new("RGB", (256 * len(thumbs), 256), (255, 255, 255))
            for i, thumb in enumerate(thumbs):
                sheet.paste(thumb, (i * 256, 0))
            sheet_path = str(directory / f"{prefix}_sheet.png")
            sheet.save(sheet_path)
    except Exception:  # pragma: no cover
        sheet_path = ""

    return {
        "stills": stills,
        "frames": frame_paths,
        "gif": gif_path,
        "sheet": sheet_path,
        "resolution": resolution,
    }


def render_depth_preview(mesh, *, azimuth: float = 35.0, elevation: float = 15.0,
                         resolution: int = 512) -> np.ndarray:
    """Normalised depth render - useful to inspect geometry without textures."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    camera = frame_camera(mesh, azimuth, elevation, resolution=resolution)
    result = rasterize(verts, faces, camera, silhouette_only=True)
    depth = result.depth.copy()
    finite = np.isfinite(depth)
    if finite.any():
        lo, hi = float(depth[finite].min()), float(depth[finite].max())
        span = max(1e-9, hi - lo)
        normalised = (depth - lo) / span
        normalised[~finite] = 1.0
    else:  # pragma: no cover
        normalised = np.ones_like(depth)
    grey = ((1.0 - np.clip(normalised, 0, 1)) * 255).astype(np.uint8)
    return np.stack([grey] * 3, axis=-1)


def render_normal_preview(mesh, *, azimuth: float = 35.0, elevation: float = 15.0,
                          resolution: int = 512) -> np.ndarray:
    """World-space normal visualisation (spec #23 normals view)."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    camera = frame_camera(mesh, azimuth, elevation, resolution=resolution)
    result = rasterize(verts, faces, camera, silhouette_only=False, flat_color=None,
                       background=(0, 0, 0))
    normals = result.normals
    if normals is None:  # pragma: no cover
        return np.zeros((resolution, resolution, 3), dtype=np.uint8)
    rgb = ((np.clip(normals, -1, 1) * 0.5 + 0.5) * 255).astype(np.uint8)
    rgb[~result.mask] = 0
    return rgb


def render_wireframe(mesh, *, azimuth: float = 35.0, elevation: float = 15.0,
                     resolution: int = 512, max_edges: int = 24000) -> np.ndarray:
    """Wireframe visualisation drawn with Bresenham lines (viewer requirement #23)."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    camera = frame_camera(mesh, azimuth, elevation, resolution=resolution)
    uv, depth = camera.project(verts)
    canvas = np.full((resolution, resolution, 3), 255, dtype=np.uint8)

    edges = set()
    step = max(1, len(faces) // max(1, max_edges // 3))
    for face in faces[::step]:
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = (min(int(a), int(b)), max(int(a), int(b)))
            edges.add(key)
    for a, b in edges:
        if depth[a] <= 0 or depth[b] <= 0:
            continue
        x0, y0 = int(round(uv[a][0])), int(round(uv[a][1]))
        x1, y1 = int(round(uv[b][0])), int(round(uv[b][1]))
        _bresenham(canvas, x0, y0, x1, y1, (40, 44, 52))
    return canvas


def _bresenham(canvas: np.ndarray, x0: int, y0: int, x1: int, y1: int,
               color: Tuple[int, int, int]) -> None:
    h, w = canvas.shape[:2]
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if 0 <= x0 < w and 0 <= y0 < h:
            canvas[y0, x0] = color
        if x0 == x1 and y0 == y1:
            break
        err2 = 2 * err
        if err2 >= dy:
            err += dy
            x0 += sx
        if err2 <= dx:
            err += dx
            y0 += sy


def render_uv_preview(mesh, *, resolution: int = 512, checker: int = 16) -> np.ndarray:
    """Checkerboard UV layout preview (spec #23 UV visualisation)."""
    from ..textures.uv import uv_overlap_ratio

    visual = getattr(mesh, "visual", None)
    uv = getattr(visual, "uv", None) if visual is not None else None
    if uv is None:
        return np.full((resolution, resolution, 3), 240, dtype=np.uint8)
    uv = np.asarray(uv, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    yy, xx = np.mgrid[0:resolution, 0:resolution]
    pattern = (((xx // checker) + (yy // checker)) % 2).astype(np.uint8)
    canvas = np.where(pattern[..., None] == 0, 205, 245).astype(np.uint8)
    canvas = np.repeat(canvas, 3, axis=2) if canvas.shape[2] == 1 else canvas
    # Draw island edges.
    tri_uv = uv[faces] * (resolution - 1)
    for tri in tri_uv[:: max(1, len(faces) // 4000)]:
        for i in range(3):
            x0, y0 = int(tri[i][0]), int(tri[i][1])
            x1, y1 = int(tri[(i + 1) % 3][0]), int(tri[(i + 1) % 3][1])
            _bresenham(canvas, x0, y0, x1, y1, (200, 60, 60))
    return canvas

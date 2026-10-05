"""Synthetic reference-set generator (spec #51: small test datasets).

Renders a known subject from a full camera ring with the built-in software
renderer and writes:

* ``*.png`` reference images,
* optional masks (ground-truth silhouettes, for pipeline tests that must not
  depend on segmentation quality),
* ``ground_truth.json`` with the exact camera parameters, subject dimensions and
  the world transform - so the test suite can *measure* reconstruction accuracy
  instead of merely checking that a file exists.

This is test/demo tooling.  The reconstruction pipeline never reads these
artefacts; they exist so that "the model is derived from the images" is a
verifiable statement rather than a claim.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..compare.raster import Camera, rasterize
from .shapes import make_test_subject, vertex_color_from_position


@dataclass
class DatasetSpec:
    kind: str = "robot"
    views: Sequence[Tuple[str, float, float]] = (
        ("front", 0.0, 0.0), ("front_right", 45.0, 0.0), ("right", 90.0, 0.0),
        ("back_right", 135.0, 0.0), ("back", 180.0, 0.0), ("back_left", 225.0, 0.0),
        ("left", 270.0, 0.0), ("front_left", 315.0, 0.0), ("top", 0.0, 80.0),
    )
    resolution: int = 512
    distance: float = 3.2
    fov_deg: float = 38.0
    target: Tuple[float, float, float] = (0.0, 0.0, 0.85)
    light_dir: Tuple[float, float, float] = (0.4, -0.7, -0.6)
    background: Tuple[int, int, int] = (240, 240, 240)
    noise: float = 0.0
    jpeg_quality: Optional[int] = None
    textured: bool = True
    write_masks: bool = True


def generate_reference_set(output_dir: Path, spec: Optional[DatasetSpec] = None,
                           *, subject=None) -> Dict[str, Any]:
    """Render a reference set and return its metadata."""
    spec = spec or DatasetSpec()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "masks"
    if spec.write_masks:
        mask_dir.mkdir(exist_ok=True)

    mesh = subject if subject is not None else make_test_subject(spec.kind)
    if spec.textured:
        mesh = vertex_color_from_position(mesh)
    bounds = np.asarray(mesh.bounds)
    centre_true = bounds.mean(axis=0)
    height_true = float(bounds[1][2] - bounds[0][2])

    from PIL import Image

    cameras: List[Dict[str, Any]] = []
    for name, az, el in spec.views:
        camera = Camera.from_azimuth_elevation(az, el, spec.distance, target=spec.target,
                                              fov_deg=spec.fov_deg, width=spec.resolution,
                                              height=spec.resolution)
        result = rasterize(np.asarray(mesh.vertices), np.asarray(mesh.faces), camera,
                           vertex_colors=(np.asarray(mesh.visual.vertex_colors)[:, :3]
                                          if spec.textured else None),
                           light_dir=spec.light_dir, ambient=0.4,
                           background=spec.background)
        image = result.color
        if spec.noise > 0:
            rng = np.random.default_rng(abs(hash(name)) % 2**32)
            image = np.clip(image.astype(np.float32) +
                            rng.normal(0, spec.noise * 255.0, image.shape), 0, 255).astype(np.uint8)
        filename = f"{name}.png"
        if spec.jpeg_quality:
            Image.fromarray(image).save(output_dir / filename.replace(".png", ".jpg"),
                                        quality=spec.jpeg_quality)
        else:
            Image.fromarray(image).save(output_dir / filename)
        if spec.write_masks:
            Image.fromarray((result.mask * 255).astype(np.uint8)).save(mask_dir / filename)
        cameras.append({
            "name": name,
            "filename": filename if not spec.jpeg_quality else filename.replace(".png", ".jpg"),
            "azimuth": az,
            "elevation": el,
            "distance": spec.distance,
            "fov_deg": spec.fov_deg,
            "fx": camera.fx, "fy": camera.fy, "cx": camera.cx, "cy": camera.cy,
            "R": camera.R.tolist(), "t": camera.t.tolist(),
            "silhouette_area": float(result.mask.mean()),
        })

    metadata = {
        "spec": {
            "kind": spec.kind,
            "resolution": spec.resolution,
            "distance": spec.distance,
            "fov_deg": spec.fov_deg,
            "target": list(spec.target),
            "views": [list(v) for v in spec.views],
        },
        "subject": {
            "kind": spec.kind,
            "bounds": bounds.tolist(),
            "centre": centre_true.tolist(),
            "height": height_true,
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
        },
        # Mapping from true world coordinates to the engine's normalised frame:
        # the engine normalises the subject height to 1.0 and centres it on the
        # solved subject centre, so the true subject centre maps to that point.
        "normalisation": {
            "true_centre": list(spec.target),
            "true_height": height_true,
            "scale": 1.0 / height_true,
        },
        "cameras": cameras,
    }
    (output_dir / "ground_truth.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_ground_truth(reference_dir: Path) -> Optional[Dict[str, Any]]:
    path = Path(reference_dir) / "ground_truth.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:  # pragma: no cover
        return None


def transform_to_engine_frame(mesh, ground_truth: Dict[str, Any], rig_center: Sequence[float]):
    """Map a ground-truth mesh into the engine's solved coordinate frame.

    The engine normalises the subject height to 1.0 and places the subject
    centre at the solved ``rig.center``; both operations are rigid+uniform, so
    the same transform applied to the ground-truth mesh makes the two directly
    comparable (that comparison is how reconstruction accuracy is measured).
    """
    norm = ground_truth.get("normalisation", {})
    scale = float(norm.get("scale", 1.0))
    true_centre = np.asarray(norm.get("true_centre", [0, 0, 0]), dtype=np.float64)
    out = mesh.copy()
    out.apply_translation(-true_centre)
    out.apply_scale(scale)
    out.apply_translation(np.asarray(rig_center, dtype=np.float64))
    return out


def silhouette_iou_report(rig, mesh, masks: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Render the mesh from every solved camera and compare with the masks."""
    from ..compare.raster import rasterize, silhouette_iou

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    per_view = []
    for pose in rig.cameras:
        mask = masks.get(pose.filename)
        if mask is None:
            continue
        result = rasterize(verts, faces, pose.camera, flat_color=(255, 255, 255),
                           background=(0, 0, 0))
        per_view.append({
            "filename": pose.filename,
            "view": pose.view,
            "iou": round(silhouette_iou(result.mask, mask > 0), 4),
            "rendered_coverage": round(result.coverage, 4),
            "reference_coverage": round(float((mask > 0).mean()), 4),
        })
    ious = [p["iou"] for p in per_view]
    outliers = [p for p in per_view if p["iou"] < 0.5]
    return {
        "mean_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
        "min_iou": round(float(np.min(ious)), 4) if ious else 0.0,
        "per_view": per_view,
        "weak_views": [p["filename"] for p in outliers],
    }

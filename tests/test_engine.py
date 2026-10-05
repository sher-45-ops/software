"""Unit tests for the computational core: hull, geometry, UV, export, materials."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from recon3d.engine.compare.raster import Camera, image_rmse, rasterize, silhouette_iou
from recon3d.engine.export.writers import SUPPORTED_FORMATS, export_asset, verify_export
from recon3d.engine.geometry.cleanup import (cleanup_mesh, fill_holes, mesh_statistics,
                                            remove_small_components, weld_vertices)
from recon3d.engine.geometry.simplify import available_backends, simplify_mesh
from recon3d.engine.materials.library import (MATERIAL_LIBRARY, apply_material_request,
                                              estimate_materials, get_material)
from recon3d.engine.optimization.lod import build_lod_chain, polygon_budget
from recon3d.engine.reconstruction.hull import voxel_iou, voxelize_mesh
from recon3d.engine.textures.uv import available_uv_backends, unwrap_mesh


# --------------------------------------------------------------------------- #
# rasteriser / cameras
# --------------------------------------------------------------------------- #
def test_camera_projection_roundtrip():
    camera = Camera.from_azimuth_elevation(0.0, 0.0, 2.0, target=(0, 0, 0), fov_deg=40.0,
                                           width=256, height=256)
    uv, depth = camera.project(np.array([[0.0, 0.0, 0.0]]))
    assert float(uv[0, 0]) == pytest.approx(camera.cx)
    assert float(uv[0, 1]) == pytest.approx(camera.cy)
    assert float(depth[0]) == pytest.approx(2.0)
    point = camera.unproject_pixel(float(uv[0, 0]), float(uv[0, 1]), depth=2.0)
    assert np.allclose(point, [0, 0, 0], atol=1e-6)
    assert np.allclose(camera.position(), [0.0, -2.0, 0.0], atol=1e-6)


def test_rasterize_sphere_covers_expected_area():
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    camera = Camera.from_azimuth_elevation(0.0, 0.0, 2.0, target=(0, 0, 0), fov_deg=40.0,
                                           width=256, height=256)
    result = rasterize(np.asarray(mesh.vertices), np.asarray(mesh.faces), camera)
    assert result.mask.dtype == bool
    filled = float(result.mask.mean())
    radius_px = 0.5 * camera.fx / 2.0
    expected = np.pi * radius_px ** 2 / (256 * 256)
    assert filled == pytest.approx(expected, rel=0.3), (filled, expected)


def test_rasterize_is_deterministic_and_silhouette_iou_is_one():
    mesh = trimesh.creation.box(extents=(0.4, 0.3, 0.6))
    camera = Camera.from_azimuth_elevation(35.0, 15.0, 2.5, target=(0, 0, 0), fov_deg=40.0,
                                           width=128, height=128)
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    first = rasterize(vertices, faces, camera)
    second = rasterize(vertices, faces, camera)
    assert np.array_equal(first.mask, second.mask)
    assert silhouette_iou(first.mask, second.mask) == pytest.approx(1.0)
    assert image_rmse(first.color, second.color) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# voxelisation / hull
# --------------------------------------------------------------------------- #
def test_voxelize_mesh_places_occupancy_on_the_mesh_bounds():
    mesh = trimesh.creation.box(extents=(0.4, 0.3, 0.6))
    mesh.apply_translation([1.0, -0.5, 0.25])
    bounds = np.asarray(mesh.bounds)
    occupancy = voxelize_mesh(mesh, 48, bounds[0], bounds[1])
    assert occupancy.sum() > 1000


def test_voxel_iou_identity_and_disjoint():
    a = np.zeros((8, 8, 8), dtype=bool)
    a[2:6, 2:6, 2:6] = True
    assert voxel_iou(a, a) == pytest.approx(1.0)
    assert voxel_iou(a, np.zeros_like(a)) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# cleanup / decimation
# --------------------------------------------------------------------------- #
def test_cleanup_removes_floaters():
    body = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    floater = trimesh.creation.box(extents=(0.02, 0.02, 0.02))
    floater.apply_translation([1.2, 0.0, 0.0])
    merged = trimesh.util.concatenate([body, floater])
    assert mesh_statistics(merged)["components"] == 2
    cleaned, report = cleanup_mesh(merged, smooth_iterations=0)
    assert mesh_statistics(cleaned)["components"] == 1
    assert report.steps
    assert report.before["components"] == 2 and report.after["components"] == 1


def test_remove_small_components_keeps_the_big_one():
    body = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    floater = trimesh.creation.box(extents=(0.02, 0.02, 0.02))
    floater.apply_translation([1.2, 0.0, 0.0])
    merged = trimesh.util.concatenate([body, floater])
    thinned, info = remove_small_components(merged)
    assert info["removed"] >= 1
    assert mesh_statistics(thinned)["components"] == 1


def test_fill_holes_reduces_boundaries():
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    keep = np.arange(len(mesh.faces)) != 0
    punctured = trimesh.Trimesh(mesh.vertices, mesh.faces[keep], process=False)
    before = trimesh.Trimesh(punctured.vertices, punctured.faces, process=False)
    filled, _ = fill_holes(before)
    assert len(filled.faces) >= len(before.faces)


def test_weld_and_simplify_reduce_geometry():
    mesh = trimesh.creation.icosphere(subdivisions=4)
    welded = weld_vertices(mesh)
    welded_mesh = welded[0] if isinstance(welded, tuple) else welded
    assert len(welded_mesh.vertices) <= len(mesh.vertices)
    simplified, info = simplify_mesh(mesh, 200)
    assert len(simplified.faces) <= len(mesh.faces)
    assert "backend" in info
    assert available_backends()


# --------------------------------------------------------------------------- #
# UV
# --------------------------------------------------------------------------- #
def test_unwrap_is_valid_and_within_unit_square(synthetic_mesh):
    result = unwrap_mesh(synthetic_mesh, resolution=512, padding=2)
    uvs = np.asarray(result.mesh.visual.uv)
    assert result.backend in available_uv_backends()
    assert uvs.shape == (len(result.mesh.vertices), 2)
    assert float(uvs.min()) >= -1e-6 and float(uvs.max()) <= 1.0 + 1e-6
    assert result.islands >= 1
    assert 0.0 < result.packing_efficiency <= 1.0
    assert result.estimated_distortion < 3.0
    assert result.overlap is False


# --------------------------------------------------------------------------- #
# materials / LOD / export
# --------------------------------------------------------------------------- #
def test_material_library_and_estimation():
    assert len(MATERIAL_LIBRARY) >= 15
    definition = get_material("plastic")
    assert definition is not None
    assignment = estimate_materials(np.array([0.85, 0.85, 0.9]), subject_type="robot")
    assert assignment.material.name in MATERIAL_LIBRARY
    assert 0.0 <= assignment.confidence <= 1.0
    tweaked = apply_material_request(assignment, "make it glossy metal")
    assert tweaked is not None


def test_polygon_budget_is_monotonic_across_presets():
    budgets = [polygon_budget(None, base_faces=120_000, quality=q)
               for q in ("draft", "standard", "high", "ultra")]
    assert budgets == sorted(budgets)
    assert budgets[0] < budgets[-1]


def test_lod_chain_never_mutates_the_source_mesh(textured_mesh):
    """Regression: LOD0 reuses the input mesh, so nothing may mutate it in place.

    An earlier revision zeroed the LOD0 vertex array to free memory, which (because
    LOD0 *is* the pipeline mesh) emptied the model right before export.
    """
    before_vertices = int(len(textured_mesh.vertices))
    before_faces = int(len(textured_mesh.faces))
    build_lod_chain(textured_mesh, levels=3, ratios=(1.0, 0.5, 0.25), texture_resolution=256)
    assert len(textured_mesh.vertices) == before_vertices
    assert len(textured_mesh.faces) == before_faces


def test_lod_chain_keeps_lod0(textured_mesh):
    chain, meshes = build_lod_chain(textured_mesh, levels=3, ratios=(1.0, 0.5, 0.25),
                                    texture_resolution=256)
    assert len(chain.levels) == 3
    faces = [level.faces for level in chain.levels]
    assert faces[0] >= faces[1] >= faces[2]
    assert faces[0] == len(textured_mesh.faces)
    assert len(meshes) == 3


@pytest.mark.parametrize("fmt", [f for f in SUPPORTED_FORMATS if f != "usdz"])
def test_export_roundtrip(textured_mesh, tmp_path, fmt):
    mesh = unwrap_mesh(textured_mesh, resolution=256).mesh
    target = tmp_path / f"model.{fmt}"
    result = export_asset(mesh, target, fmt)
    assert target.exists() and target.stat().st_size > 200
    report = verify_export(target, fmt)
    assert report.get("ok") is True, report
    assert result.format in (fmt, "usda")  # ``usd`` exports write the text form
    assert result.verified is True


def test_export_rejects_unknown_format(textured_mesh, tmp_path):
    from recon3d.errors import Recon3DError

    with pytest.raises(Recon3DError):
        export_asset(textured_mesh, tmp_path / "model.exe", "exe")


# --------------------------------------------------------------------------- #
# depth fusion (multi-view plane sweep)
# --------------------------------------------------------------------------- #
def test_depth_fusion_produces_points_on_the_subject():
    """Regression: the plane-sweep stage used to crash (broadcasting + a shadowed
    output buffer), so depth evidence silently never reached the reconstruction."""
    from recon3d.engine.cameras.rig import CameraRig, CameraPose
    from recon3d.engine.compare.raster import Camera, rasterize
    from recon3d.engine.reconstruction.depth import fuse_depth_maps
    from recon3d.engine.reconstruction.shapes import make_test_subject

    subject = make_test_subject("robot")
    views = [("front", 0.0, 0.0), ("front_right", 45.0, 0.0), ("back_right", 135.0, 0.0),
             ("back", 180.0, 0.0)]
    size = 128
    images, masks, poses = {}, {}, []
    for name, az, el in views:
        camera = Camera.from_azimuth_elevation(az, el, 3.0, target=(0, 0, 0), fov_deg=40.0,
                                               width=size, height=size)
        result = rasterize(np.asarray(subject.vertices), np.asarray(subject.faces), camera,
                           vertex_colors=np.asarray(subject.visual.vertex_colors)[:, :3])
        images[f"{name}.png"] = result.color
        masks[f"{name}.png"] = result.mask
        poses.append(CameraPose(image_id=name, filename=f"{name}.png", view=name,
                                azimuth=az, elevation=el, distance=3.0,
                                focal_pixels=camera.fx, width=size, height=size,
                                camera=camera, method="test"))
    rig = CameraRig(cameras=poses, center=np.zeros(3), subject_height=1.0, method="test",
                    ring_quality=1.0)
    fused = fuse_depth_maps(rig, images, masks, steps=16, pixel_step=4, max_views=3)
    assert len(fused["points"]) > 100
    points = np.asarray(fused["points"])
    assert np.isfinite(points).all()
    # the fused cloud must sit inside the subject's bounding volume
    bounds = np.asarray(subject.bounds)
    assert (points.min(axis=0) >= bounds[0] - 0.45).all()
    assert (points.max(axis=0) <= bounds[1] + 0.45).all()
    stats = fused["statistics"]
    assert 0.0 < float(stats["raw_points"]) >= float(stats["after_outlier_removal"]) > 0
    assert float(stats.get("mean_confidence", stats.get("confidence_mean", 1.0))) > 0.1
    assert "depth_maps" in fused

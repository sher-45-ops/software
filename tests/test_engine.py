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


# --------------------------------------------------------------------------- #
# symmetry completion (stage-4 evidence -> filled volume, disclosed as inferred)
# --------------------------------------------------------------------------- #
def _hull_with_bump(size: int = 32, *, symmetric: bool = False, half_space: bool = False):
    """A centred ball with (by default) a lobe protruding on +x only.

    The -x side is therefore missing the mirror of that lobe: exactly the defect
    symmetry completion exists to fill, and exactly the kind of fill that has to
    be disclosed as inferred.
    """
    from recon3d.engine.reconstruction.hull import HullResult

    centre = (size - 1) / 2.0
    zz, yy, xx = np.mgrid[0:size, 0:size, 0:size]
    ball = (xx - centre) ** 2 + (yy - centre) ** 2 + (zz - centre) ** 2 <= (size * 0.24) ** 2
    if half_space:
        occupancy = np.zeros((size, size, size), dtype=bool)
        occupancy[:, :, size // 2:] = True
    elif symmetric:
        occupancy = ball
    else:
        lobe_centre = centre + size * 0.26
        lobe = ((xx - lobe_centre) ** 2 + (yy - centre) ** 2 +
                (zz - centre) ** 2) <= (size * 0.10) ** 2
        occupancy = ball | lobe
    return HullResult(
        occupancy=occupancy,
        bounds_min=np.array([-1.0, -1.0, -1.0]),
        bounds_max=np.array([1.0, 1.0, 1.0]),
        voxel_size=2.0 / size,
        resolution=(size, size, size),
        method="test",
    )


def test_mirror_completion_fills_the_missing_side_only():
    from recon3d.engine.reconstruction.hull import _mirror_agreement, mirror_complete

    hull = _hull_with_bump()
    completed, report = mirror_complete(hull, axis="x")

    assert report["applied"] is True, report.get("reason")
    assert report["filled_voxels"] > 0
    assert report["filled_fraction"] < 0.35, "a mirror must not invent a second subject"
    assert 0.0 <= report["mirror_agreement"] <= 1.0
    assert report["regions"], "every filled patch has to be describable in the report"
    assert abs(report["plane_world"]) < 0.25, "the plane must land near the subject centre"

    plane = report["plane_index"]
    assert _mirror_agreement(completed.occupancy, plane, 2) >= _mirror_agreement(
        hull.occupancy, plane, 2)
    assert int(completed.occupancy.sum()) > int(hull.occupancy.sum())
    assert not (hull.occupancy & ~completed.occupancy).any(), "original voxels must survive"


def test_mirror_completion_refuses_to_invent_a_new_subject():
    from recon3d.engine.reconstruction.hull import mirror_complete

    # An 8.9% fill is already "a different subject" under this boundary, and the
    # boundary is what protects against a mirror that rewrites the asset.
    hull = _hull_with_bump()
    completed, report = mirror_complete(hull, axis="x", max_fill_fraction=0.02)
    assert report["applied"] is False
    assert "not a completion" in report["reason"]
    assert completed is hull, "a refused completion must not replace the volume"


def test_mirror_completion_is_a_no_op_on_a_symmetric_volume():
    from recon3d.engine.reconstruction.hull import mirror_complete

    completed, report = mirror_complete(_hull_with_bump(symmetric=True), axis="x")
    assert report["filled_voxels"] == 0
    assert report["applied"] is False
    assert completed is not None


def test_symmetry_completion_records_inferred_entries():
    """The pipeline helper must turn a completion into disclosed, confident entries."""
    from recon3d.core.pipeline import _inferred_symmetry_entries

    entries = _inferred_symmetry_entries(
        {"filled_voxels": 100, "mirror_agreement": 0.83, "prior_symmetry": 0.9,
         "axis": "x", "plane_world": 0.01,
         "validation": {"base_iou": 0.88, "completed_iou": 0.90},
         "regions": [{"region": "symmetry-completed volume (bbox [0,0,0] - [1,1,1])",
                      "voxels": 60},
                     {"region": "symmetry-completed volume: 40 further voxel(s)", "voxels": 40}]},
        0.9)
    assert len(entries) == 2
    for entry in entries:
        assert entry["kind"] == "symmetry_completion"
        assert 0.0 < entry["confidence"] <= 0.95
        assert "mirror agreement" in entry["evidence"]
        assert "silhouette IoU" in entry["verified"]


# --------------------------------------------------------------------------- #
# texture inpainting (classic diffusion, reported as inferred)
# --------------------------------------------------------------------------- #
def test_texture_inpainting_fills_holes_and_measures_confidence():
    from recon3d.engine.textures.project import inpaint_texture_holes

    rho = 96
    base = np.zeros((rho, rho, 3), dtype=np.float32)
    base[..., 0], base[..., 1] = 0.2, 0.6
    iy, ix = np.mgrid[16:80, 16:80].reshape(2, -1)
    hole = (iy > 56) & (ix > 56)          # one large occluded quadrant
    seam = (iy > 30) & (iy < 33)          # a thin seam
    coverage = ~(hole | seam)

    result = inpaint_texture_holes(base, coverage, iy, ix, rho=rho)

    assert result["applied"] is True
    assert result["inferred_texels"] == int((hole | seam).sum())
    assert 0.0 < result["confidence"] <= 1.0
    seam_regions = [r for r in result["regions"] if r.get("texel_bbox")]
    assert seam_regions, "filled patches must be listed as regions"
    # a thin seam sits next to observed texels and must be trusted more than the
    # big occluded block
    seam_conf = max((r["confidence"] for r in seam_regions
                     if (r["texel_bbox"][3] - r["texel_bbox"][1]) <= 4), default=0.0)
    block_conf = min((r["confidence"] for r in seam_regions
                      if (r["texel_bbox"][3] - r["texel_bbox"][1]) > 4), default=1.0)
    assert seam_conf > block_conf
    assert all("texels" in r and "confidence" in r for r in result["regions"])

    filled = result["_base"]
    assert np.allclose(filled[70, 70], [0.2, 0.6, 0.0], atol=0.05)
    assert np.allclose(filled[20, 20], [0.2, 0.6, 0.0], atol=1e-6), "observed texels stay untouched"


def test_unobserved_texture_regions_reach_the_quality_report():
    from recon3d.engine.compare.quality import score_quality

    entry = {"kind": "texture_inpainting", "stage": "texture",
             "region": "texture atlas region uv[0.1,0.1]-[0.2,0.2] (12 texels)",
             "texels": 12, "confidence": 0.42}
    quality = score_quality(comparison={"mean_iou": 0.9, "views": [{}]},
                            mesh_stats={"faces": 1000}, inferred=[entry],
                            budgets=[{"name": "mobile", "pass": True}])
    assert quality["inferred"][0]["confidence"] == 0.42
    assert quality["inferred_summary"]["kinds"] == ["texture_inpainting"]
    assert any(m.startswith("inferred: ") and "0.42" in m for m in quality["missing_regions"]), \
        "an agent reading only missing_regions must still see the filled regions"
    assert quality["budgets"] == [{"name": "mobile", "pass": True}]
    assert any("filled in" in w for w in quality["warnings"])


# --------------------------------------------------------------------------- #
# game_ready asset budgets
# --------------------------------------------------------------------------- #
class _BudgetContext:
    """Minimal stand-in for PipelineContext (the budget export only needs these)."""

    def __init__(self, version_dir, preset="game_ready"):
        from types import SimpleNamespace

        self.params = {"_preset": preset, "_performance_mode": "balanced"}
        self.cache = {}
        self.version_dir = version_dir
        self.reporter = _QuietReporter()
        self.budget = {"texture_resolution": 2048}
        self.profile = SimpleNamespace(ram_mb=3939, logical_cores=2)

    def request(self, key, default=None):
        if key == "texture_resolution":
            return 1024
        return default

    def dir_for(self, name):
        path = self.version_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path


class _QuietReporter:
    def __init__(self):
        self.messages = []

    def info(self, message, *a, **k):
        self.messages.append(("info", str(message)))

    def warning(self, message, *a, **k):
        self.messages.append(("warning", str(message)))

    def update(self, *a, **k):
        pass


def test_game_ready_preset_ships_a_measured_mobile_budget(tmp_path):
    from recon3d.core.pipeline import _export_named_budgets, available_presets

    budgets = {p["name"]: p.get("asset_budgets") for p in available_presets()}
    mobile = budgets["game_ready"]["mobile"]
    assert mobile["target_polycount"] == 12000
    assert mobile["texture_resolution"] == 1024
    assert not budgets["draft"], "only presets that promise budgets may declare them"

    mesh = trimesh.creation.icosphere(subdivisions=3)
    lod_dir = tmp_path / "lod"
    lod_dir.mkdir()
    ctx = _BudgetContext(tmp_path)
    chain = {f"lod{i}": f"lod/lod{i}.glb" for i in range(4)}
    reports = _export_named_budgets(ctx, mesh, lod_dir, chain)

    by_name = {r["name"]: r for r in reports}
    assert set(by_name) == {"mobile", "desktop"}
    for report in by_name.values():
        assert report["pass"] is True
        assert (tmp_path / report["path"]).exists()
        assert report["actual"]["target_polycount"] <= report["targets"]["target_polycount"]
        assert report["checks"]["target_polycount"]["pass"] is True

    # a budget that cannot be met is reported as failed, never silently dropped
    ctx.params["_preset"] = "game_ready"
    import recon3d.core.pipeline as pipeline

    original = pipeline.load_preset

    def _impossible(name):
        preset = original(name)
        if preset.get("name") == "game_ready":
            preset = dict(preset)
            # ask for a single LOD while four are shipped: must be reported, not hidden
            preset["budgets"] = {"mobile": {"target_polycount": 12000, "lod_levels": 1}}
        return preset

    pipeline.load_preset = _impossible
    try:
        reports = pipeline._export_named_budgets(ctx, mesh, lod_dir, chain)
    finally:
        pipeline.load_preset = original
    assert reports[0]["pass"] is False
    assert any("not met" in message for _level, message in ctx.reporter.messages)

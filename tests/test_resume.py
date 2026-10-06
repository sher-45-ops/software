"""Robustness tests: checkpoint re-use, invalidation and cancellation.

The engine advertises resumable stages, per-stage checkpoints tied to their
inputs, and cancellation that leaves no half-built version behind. These tests
exercise that contract on real (small, draft) pipeline runs.
"""

from __future__ import annotations

import json
import struct
import time
from pathlib import Path

import pytest

from recon3d.config import load_config
from recon3d.core.jobs import (JobManager, checkpoint_valid, read_checkpoint,
                               write_checkpoint)
from recon3d.core.pipeline import run_pipeline, select_stages
from recon3d.core.project import ProjectManager


@pytest.fixture()
def small_project(tmp_path: Path, reference_dir: Path):
    cfg = load_config(data_root=str(tmp_path / "resume-data"))
    cfg.ensure_dirs()
    manager = ProjectManager(cfg.projects_path)
    project = manager.create("resume-check", subject_type="robot")
    project.add_images(sorted(reference_dir.glob("*.png")))
    jobs = JobManager(log_dir=cfg.logs_path)
    return cfg, project, jobs


def _stamp(project, stage: str):
    """Identity of a stage's checkpoint: content timestamp + file mtime (ns).

    ``finished_at`` alone has one-second granularity, so two consecutive runs can
    share it; the file's nanosecond mtime makes "was this stage re-run?" exact.
    """
    checkpoint = read_checkpoint(project, stage)
    assert checkpoint is not None, f"no checkpoint for '{stage}'"
    path = project.stage_dir(stage, create=False) / "_checkpoint.json"
    return checkpoint["finished_at"], path.stat().st_mtime_ns


def _run(project, jobs, *, stages=None, label="run", resume=True, params=None):
    job = jobs.create(project.id, kind="reconstruct", params={}, project_obj=project)
    resolved = {"preset": "draft", "texture_resolution": 512}
    resolved.update(params or {})
    return run_pipeline(project, job, params=resolved, stages=stages, label=label,
                        resume=resume)


def test_stage_subset_then_resume_reuses_checkpoints(small_project):
    """Run the first half, then the rest: the finished half must not be redone."""
    cfg, project, jobs = small_project
    first_stages = ["ingestion", "validation", "segmentation", "analysis", "features"]

    first = _run(project, jobs, stages=first_stages, label="first-half")
    assert first["status"] in {"completed", "completed_with_warnings"}, first

    stamps = {}
    for stage in first_stages:
        checkpoint = read_checkpoint(project, stage)
        assert checkpoint is not None, f"stage '{stage}' wrote no checkpoint"
        assert checkpoint["status"] == "completed"
        assert checkpoint["inputs_hash"], "the checkpoint is not tied to its inputs"
        stamps[stage] = _stamp(project, stage)

    second = _run(project, jobs, label="second-half")
    assert second["status"] in {"completed", "completed_with_warnings"}, second
    assert not second.get("stages_failed"), second.get("stages_failed")

    for stage, stamp in stamps.items():
        # A re-used checkpoint is not rewritten at all: the stage really was skipped.
        assert _stamp(project, stage) == stamp, f"stage '{stage}' was recomputed"

    version_dir = Path(second["outputs"]["version_dir"])
    assert version_dir.exists()
    assert list((version_dir / "final").glob("*.glb"))
    assert second["statistics"]["triangles"] > 200


def _write_reference(path: Path, *, seed: int, size: int = 96) -> Path:
    """A small, deterministic PNG that is unique per ``seed``."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    canvas[size // 4: 3 * size // 4, size // 4: 3 * size // 4] = rng.integers(
        0, 200, size=(size // 2, size // 2, 3), dtype=np.uint8)
    Image.fromarray(canvas).save(path)
    return path


def test_duplicate_references_are_detected_and_skipped(small_project):
    """Byte-identical references are recorded as duplicates, not added twice."""
    cfg, project, jobs = small_project
    original = sorted((project.root / "input" / "original").glob("*.png"))[0]
    clone = project.root / "input" / "original" / "clone_front.png"
    clone.write_bytes(original.read_bytes())

    assert project.add_images([clone]) == []
    duplicates = [d for d in project.data.get("duplicates", []) if d.get("duplicate_of")]
    assert duplicates, "the duplicate was not recorded"


def test_changed_inputs_invalidate_the_checkpoint(small_project):
    """A checkpoint is tied to its inputs, not just to the stage name."""
    cfg, project, jobs = small_project
    _run(project, jobs, stages=["ingestion", "validation"], label="first")
    before_hash = read_checkpoint(project, "ingestion")["inputs_hash"]
    before = _stamp(project, "ingestion")

    extra = _write_reference(project.root / "input" / "original" / "extra_top.png", seed=7)
    reopened = ProjectManager(cfg.projects_path).open("resume-check")
    assert reopened.add_images([extra]), "a genuinely new image must be accepted"

    _run(reopened, jobs, stages=["ingestion", "validation"], label="second")
    assert read_checkpoint(reopened, "ingestion")["inputs_hash"] != before_hash, \
        "the input hash did not change"
    assert _stamp(reopened, "ingestion") != before, \
        "the stage was re-used even though its inputs changed"


def test_checkpoint_validity_rules(small_project):
    """`checkpoint_valid` encodes the resume contract; pin it down."""
    cfg, project, jobs = small_project
    _run(project, jobs, stages=["ingestion"], label="checkpoint-rules")
    good = read_checkpoint(project, "ingestion")
    assert good is not None

    # The pipeline validates outputs whenever the checkpoint recorded any
    # (``require_outputs=bool(checkpoint["outputs"])``); ingestion writes no files
    # of its own, so the strict form is False for it by design.
    assert checkpoint_valid(project, "ingestion", good["inputs_hash"], require_outputs=False)
    assert not checkpoint_valid(project, "ingestion", good["inputs_hash"])
    assert not checkpoint_valid(project, "ingestion", "a-different-hash", require_outputs=False)
    assert not checkpoint_valid(project, "no-such-stage", good["inputs_hash"], require_outputs=False)

    # Corrupted-stage detection: an artefact that disappears invalidates the stage.
    stage_dir = project.stage_dir("features")
    artefact = stage_dir / "probe.json"
    artefact.write_text("{}", encoding="utf-8")
    relative = str(artefact.relative_to(project.root))
    write_checkpoint(project, "features", inputs_hash="probe-hash",
                     outputs={"probe": relative})
    assert checkpoint_valid(project, "features", "probe-hash")
    artefact.unlink()
    assert not checkpoint_valid(project, "features", "probe-hash")

    # A failed checkpoint is never resumed.
    write_checkpoint(project, "validation", inputs_hash="x", status="failed",
                     statistics={"error": "simulated crash"})
    assert not checkpoint_valid(project, "validation", "x")


def test_cancellation_stops_the_job_and_leaves_no_half_version(small_project):
    """Cancelling a running job marks it cancelled and rolls back its version."""
    cfg, project, jobs = small_project

    versions_before = {v.id for v in project.versions()}
    state = {"cancelled": False}

    def cancel_when_busy(job_id: str, event) -> None:
        if state["cancelled"]:
            return
        if event.stage in {"camera_estimation", "mesh_reconstruction", "silhouette_refine"}:
            state["cancelled"] = True
            jobs.cancel(job_id, reason="test cancellation")

    job = jobs.create(project.id, kind="reconstruct", params={"preset": "draft"},
                      project_obj=project)
    unsubscribe = jobs.subscribe(cancel_when_busy)
    try:
        jobs.submit(job, lambda j: run_pipeline(
            project, j, params={"preset": "draft", "texture_resolution": 512},
            label="cancel-check", resume=False))
        job = jobs.wait(job.id, timeout=900)
    finally:
        unsubscribe()

    assert state["cancelled"], "the job never reached a cancellable stage"
    assert job.state.value == "cancelled", f"job ended as {job.state.value}"
    assert job.error and "cancel" in str(job.error).lower()

    # The cancelled run must not leave a half-written version in the project.
    reopened = ProjectManager(cfg.projects_path).open("resume-check")
    assert {v.id for v in reopened.versions()} == versions_before
    assert job.resumable is False or job.resumable is True  # flag is present either way

def test_crash_recovery_resumes_from_the_persisted_geometry(small_project):
    """The expensive carve must not be repeated after an interrupted run.

    The mesh_reconstruction stage persists its surface; a later run that finds the
    checkpoint re-uses the geometry *and* the cameras instead of carving again.
    """
    cfg, project, jobs = small_project
    carve_stages = ["ingestion", "validation", "segmentation", "analysis", "features",
                    "camera_estimation", "mesh_reconstruction"]

    first = _run(project, jobs, stages=carve_stages, label="crashed-after-carve")
    assert first["status"] in {"completed", "completed_with_warnings"}, first

    mesh_checkpoint = read_checkpoint(project, "mesh_reconstruction")
    assert mesh_checkpoint is not None
    assert mesh_checkpoint["outputs"], "the geometry stage recorded no artefacts"
    carve_stamp = _stamp(project, "mesh_reconstruction")
    persisted = project.root / "intermediate" / "mesh_reconstruction" / "mesh.ply"
    assert persisted.exists(), "the mesh was not persisted for resumption"

    # Second run: the same stages plus everything downstream.
    second = _run(project, jobs, label="resumed")
    assert second["status"] in {"completed", "completed_with_warnings"}, second
    assert not second.get("stages_failed"), second.get("stages_failed")

    assert _stamp(project, "mesh_reconstruction") == carve_stamp, \
        "the carve was repeated even though its checkpoint and mesh were present"

    version_dir = Path(second["outputs"]["version_dir"])
    assert list((version_dir / "final").glob("*.glb"))
    assert (version_dir / "reports" / "quality.json").exists()
    # LODs/statistics come from the resumed geometry, so they must be real numbers.
    assert second["statistics"]["triangles"] > 200
    assert second["quality"]["metrics"]["mean_silhouette_iou"] > 0.0


def test_deleted_stage_artefacts_force_a_rebuild(small_project):
    """Corrupted-stage detection: a checkpoint whose mesh vanished is not trusted."""
    cfg, project, jobs = small_project
    stages = ["ingestion", "validation", "segmentation", "analysis", "features",
              "camera_estimation", "mesh_reconstruction"]
    _run(project, jobs, stages=stages, label="first")
    before = _stamp(project, "mesh_reconstruction")

    (project.root / "intermediate" / "mesh_reconstruction" / "mesh.ply").unlink()

    _run(project, jobs, stages=["mesh_reconstruction"], label="second")
    assert _stamp(project, "mesh_reconstruction") != before, \
        "a stage whose geometry disappeared was re-used anyway"
    assert (project.root / "intermediate" / "mesh_reconstruction" / "mesh.ply").exists()


def test_full_rerun_reuses_every_checkpoint(small_project):
    """The strongest resume claim: a repeat run recomputes *nothing*.

    Run the whole pipeline, then run it again with identical parameters. Every stage
    must report its checkpoint as re-used (rig, textures and materials included - they
    live in the version directory, so they are the ones that used to be recomputed),
    the measured quality must be identical, and the second run must be much faster.
    """
    cfg, project, jobs = small_project
    params = {"quality": "draft", "texture_resolution": 512, "generate_lods": True,
              "generate_rig": True, "generate_previews": False}

    first = _run(project, jobs, label="first", params=params)
    assert first["status"] in {"completed", "completed_with_warnings"}, first
    stamps = {spec.name: _stamp(project, spec.name) for spec in select_stages(None)}
    first_stats = first["statistics"]
    first_quality = first["quality"]

    started = time.time()
    second = _run(project, jobs, label="second", params=params)
    elapsed = time.time() - started
    assert second["status"] in {"completed", "completed_with_warnings"}, second
    assert not second.get("stages_failed"), second.get("stages_failed")

    # The export stage is deliberately regenerated for every version: its files are
    # named after the version (hero_v002.glb, and the .obj/.mtl pair must keep
    # matching names inside the file), and writing them costs seconds, not minutes.
    recomputed = [name for name, stamp in stamps.items()
                  if name != "export" and _stamp(project, name) != stamp]
    assert not recomputed, f"these stages were recomputed on an identical re-run: {recomputed}"

    assert second["statistics"]["triangles"] == first_stats["triangles"]
    assert second["quality"]["overall"] == first_quality["overall"]
    assert second["quality"]["metrics"] == first_quality["metrics"]
    assert elapsed < 60, f"a fully cached re-run took {elapsed:.1f}s"

    # The rig, textures and material assignment really are back in the version.
    version_dir = Path(second["outputs"]["version_dir"])
    assert (version_dir / "rig" / "skin_weights.npz").exists()
    assert (version_dir / "textures" / "basecolor.png").exists()
    assert (version_dir / "materials" / "materials.json").exists()

    # ...and the re-used mesh still carries its UVs into the exported asset.
    glb = next(version_dir.glob("final/*.glb"))
    assert second["version"] in glb.name, f"the export is not named after this version: {glb}"
    payload = glb.read_bytes()
    assert payload[:4] == b"glTF"
    import struct

    chunk_length, chunk_type = struct.unpack("<II", payload[12:20])
    assert chunk_type == 0x4E4F534A, "first GLB chunk is not JSON"
    document = json.loads(payload[20:20 + chunk_length].decode("utf-8"))
    attributes = [primitive.get("attributes", {})
                  for mesh_entry in document.get("meshes", [])
                  for primitive in mesh_entry.get("primitives", [])]
    assert attributes and all("TEXCOORD_0" in entry for entry in attributes), \
        f"the resumed export lost its texture coordinates: {attributes}"


def test_changing_texture_resolution_reuses_the_geometry(small_project):
    """A parameter change invalidates what depends on it - and nothing else.

    Raising the texture resolution must rebuild the texture-dependent stages while the
    cameras, the carve and the cleanup stay cached: that is the whole point of hashing
    each stage's *own* inputs and its dependencies' hashes.
    """
    cfg, project, jobs = small_project
    geometry_stages = ["ingestion", "validation", "segmentation", "analysis", "features",
                       "camera_estimation", "mesh_reconstruction", "silhouette_refine",
                       "mesh_cleanup", "optimization", "uv"]

    first = _run(project, jobs, label="512", params={"texture_resolution": 512})
    assert first["status"] in {"completed", "completed_with_warnings"}, first
    geometry = {stage: _stamp(project, stage) for stage in geometry_stages}
    triangles = first["statistics"]["triangles"]

    second = _run(project, jobs, label="1024", params={"texture_resolution": 1024})
    assert second["status"] in {"completed", "completed_with_warnings"}, second

    for stage, stamp in geometry.items():
        assert _stamp(project, stage) == stamp, \
            f"'{stage}' was recomputed although only the texture resolution changed"

    # The texture really was rebuilt at the new resolution...
    assert second["statistics"]["texture_resolution"] == 1024
    assert second["statistics"]["triangles"] == triangles, "the geometry changed"
    version_dir = Path(second["outputs"]["version_dir"])
    from PIL import Image

    with Image.open(version_dir / "textures" / "basecolor.png") as image:
        assert image.size == (1024, 1024)

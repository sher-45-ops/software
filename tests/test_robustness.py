"""Robustness controls: cancellation, bounded retry, resume-after-failure, CLI.

These cover the guarantees the spec asks for beyond happy-path reconstruction:

* a running job can be stopped cooperatively, from this process *or* from another
  one, and every finished stage survives the stop;
* a stage that fails for a transient reason is retried (bounded) instead of
  killing the run, while an input error is *not* retried;
* a cancelled or failed run is picked up by ``recon3d retry`` / ``recon3d jobs``;
* checkpoints written before the stop are re-used, so nothing is recomputed.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from recon3d.core.jobs import JobManager, read_checkpoint
from recon3d.core.pipeline import run_pipeline
from recon3d.core.progress import CancellationToken
from recon3d.errors import CancelledError, InsufficientDataError


def _visible_params(**overrides):
    """Parameters shared by every run in this file: a fast, honest draft pipeline."""
    params = {"quality": "draft", "texture_resolution": 512, "generate_lods": False,
              "generate_previews": False}
    params.update(overrides)
    return params


def _run(project, jobs, *, params=None, stages=None, label=""):
    """Run the pipeline with the *same* parameters the job was created for."""
    resolved = _visible_params(**(params or {}))
    job = jobs.create(project.id, kind="reconstruct", params=resolved, project_obj=project)
    jobs.run_sync(job, lambda j: run_pipeline(project, j, params=resolved, stages=stages,
                                              label=label))
    return job


def _patch_stage(monkeypatch, name: str, wrap):
    """Replace one stage function in the pipeline table (keeping its metadata)."""
    import dataclasses

    from recon3d.core import pipeline as pipeline_module

    table = []
    for spec in pipeline_module.STAGE_TABLE:
        table.append(dataclasses.replace(spec, fn=wrap(spec.fn))
                     if spec.name == name else spec)
    monkeypatch.setattr(pipeline_module, "STAGE_TABLE", table)


def _reference_set(directory: Path, *, views: int = 4) -> list[Path]:
    from recon3d.engine.reconstruction.dataset import DatasetSpec, generate_reference_set

    directory.mkdir(parents=True, exist_ok=True)
    views_spec = (("front", 0.0, 0.0), ("front_right", 45.0, 0.0), ("right", 90.0, 0.0),
                  ("back", 180.0, 0.0))[:views]
    generate_reference_set(directory, DatasetSpec(kind="robot", views=views_spec,
                                                  resolution=160, noise=0.004,
                                                  write_masks=False), subject=None)
    return sorted(directory.glob("*.png"))


@pytest.fixture()
def small_project(config, tmp_path: Path):
    from recon3d.core.project import ProjectManager

    refs = _reference_set(tmp_path / "refs")
    project = ProjectManager(config.projects_path).create("controls", subject_type="robot",
                                                          description="robustness fixture")
    project.add_images(refs)
    jobs = JobManager(log_dir=config.logs_path, max_concurrent=1)
    return config, project, jobs


# --------------------------------------------------------------------------
# cancellation
# --------------------------------------------------------------------------
def test_cancel_token_is_honoured_across_processes(tmp_path: Path):
    token = CancellationToken(watch=tmp_path / "job.cancel")
    assert not token.cancelled
    token.raise_if_cancelled()  # no request -> no exception

    (tmp_path / "job.cancel").write_text("stop", encoding="utf-8")
    with pytest.raises(CancelledError):
        token.raise_if_cancelled()


def test_cancelling_a_run_keeps_finished_stages(small_project, monkeypatch):
    """Stop the job mid-pipeline: finished stages must survive for a later resume."""
    cfg, project, jobs = small_project

    def cancel_after_segmentation(original):
        def wrapper(ctx):
            report = original(ctx)
            ctx.reporter.token.cancel("test asked to stop")
            return report

        return wrapper

    _patch_stage(monkeypatch, "segmentation", cancel_after_segmentation)

    job = _run(project, jobs, label="cancelled-run")

    assert job.state.value == "cancelled"
    assert isinstance(job.error, dict) and job.error.get("message")
    # Everything finished before the stop is checkpointed, nothing after it is.
    assert read_checkpoint(project, "ingestion") is not None
    assert read_checkpoint(project, "segmentation") is not None
    assert read_checkpoint(project, "mesh_cleanup") is None
    assert not job.result

    # Retrying the same project resumes instead of repeating the finished half.
    resumed = jobs.create(project.id, params=_visible_params(), project_obj=project,
                          resumed_from=job.id)
    jobs.run_sync(resumed, lambda j: run_pipeline(project, j, params=_visible_params()))
    assert resumed.state.value in {"completed", "partial"}, resumed.error
    assert read_checkpoint(project, "mesh_cleanup") is not None
    assert resumed.result["statistics"]["triangles"] > 0


def test_cancel_request_file_stops_a_running_job(small_project):
    """A second process (the CLI) can stop a job it does not own in memory."""
    cfg, project, jobs = small_project

    params = _visible_params()
    job = jobs.create(project.id, params=params, project_obj=project)
    watcher = job.token.request_path()
    assert watcher is not None and str(watcher).endswith(f"{job.id}.cancel")

    def touch_request() -> None:
        # Give the worker a moment to start, then act like `recon3d cancel`.
        time.sleep(1.0)
        watcher.parent.mkdir(parents=True, exist_ok=True)
        watcher.write_text(json.dumps({"reason": "cli"}), encoding="utf-8")

    thread = threading.Thread(target=touch_request, daemon=True)
    thread.start()
    jobs.run_sync(job, lambda j: run_pipeline(project, j, params=params))
    thread.join(timeout=10)

    assert job.state.value == "cancelled"
    # The request file is cleaned up so a fresh run is not cancelled instantly.
    assert not watcher.exists()

    # A second job of the same project watches its own request file, and the
    # helper the CLI calls writes exactly there.
    other = jobs.create(project.id, params={"quality": "draft"}, project_obj=project)
    request = jobs.request_cancel(other.id, reason="cli")
    assert request == other.token.request_path()
    assert request.exists()


# --------------------------------------------------------------------------
# retry
# --------------------------------------------------------------------------
def test_transient_stage_failure_is_retried(small_project, monkeypatch):
    cfg, project, jobs = small_project
    calls = {"count": 0}

    def flaky(original):
        def wrapper(ctx):
            calls["count"] += 1
            if calls["count"] == 1:
                raise OSError("simulated transient disk failure")
            return original(ctx)

        return wrapper

    _patch_stage(monkeypatch, "validation", flaky)

    job = _run(project, jobs, params={"stage_retries": 1}, label="flaky")

    assert calls["count"] == 2, "the stage should have been retried exactly once"
    assert job.state.value in {"completed", "partial"}, job.error
    assert job.stages["validation"].get("attempts") == 2
    assert read_checkpoint(project, "validation")["status"] == "completed"
    assert any("retrying" in warning for warning in job.warnings), job.warnings


def test_input_errors_are_not_retried(small_project, monkeypatch):
    """A bad request must fail fast - retrying it would only waste the operator's time."""
    cfg, project, jobs = small_project
    calls = {"count": 0}

    def refuse(original):
        def wrapper(ctx):
            calls["count"] += 1
            raise InsufficientDataError("simulated: references cannot support the request")

        return wrapper

    # 'segmentation' is a mandatory stage, so its failure fails the job.
    _patch_stage(monkeypatch, "segmentation", refuse)

    job = _run(project, jobs, params={"stage_retries": 3}, label="refused")

    assert calls["count"] == 1, "an input error must not be retried"
    assert job.state.value == "failed"
    failed_checkpoint = read_checkpoint(project, "segmentation")
    assert failed_checkpoint is not None and failed_checkpoint["status"] == "failed"
    assert job.resumable is False


# --------------------------------------------------------------------------
# CLI control surface
# --------------------------------------------------------------------------
def test_cli_jobs_cancel_and_retry(small_project, monkeypatch, capsys):
    """`recon3d jobs|cancel|retry` drive the persisted job records, JSON included."""
    cfg, project, jobs = small_project
    from recon3d.cli import main as cli

    # A finished run appears in `recon3d jobs`, with a persisted record.
    job = _run(project, jobs, stages=["ingestion", "validation"], label="cli")
    assert job.state.value in {"completed", "partial"}, job.error

    monkeypatch.setenv("RECON3D_HOME", str(cfg.data_root))
    assert cli.main(["jobs", "--project", project.id, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] >= 1
    assert any(record["id"] == job.id for record in payload["jobs"])

    # Cancelling a finished job is refused (nothing to cancel).
    assert cli.main(["cancel", job.id, "--json"]) == cli.EXIT_ERROR
    json.loads(capsys.readouterr().out)

    # `retry` re-runs the persisted job with the same parameters; the finished
    # stages are cached, so it is quick and the report proves the resume.
    assert cli.main(["retry", job.id, "--json"]) == 0
    retried = json.loads(capsys.readouterr().out)
    assert retried["ok"] is True
    assert retried["job"]["resumed_from"] == job.id
    assert retried["job"]["state"] in {"completed", "partial"}
    assert read_checkpoint(project, "ingestion") is not None


def test_cli_info_and_doctor_are_json_clean(capsys, monkeypatch, tmp_path: Path):
    """`--json` output must stay machine-readable (stdout carries JSON only)."""
    monkeypatch.setenv("RECON3D_HOME", str(tmp_path / "cli-home"))
    from recon3d.cli import main as cli

    assert cli.main(["info", "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert "standard" in info["presets"] and "glb" in info["formats"]
    assert any(stage["name"] == "mesh_reconstruction" for stage in info["stages"])
    assert info["no_external_ai"] is True

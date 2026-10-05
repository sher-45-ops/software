"""CLI tests: real subprocess calls against a throwaway data root.

These cover the contract an agent relies on: every command exits with the documented code
and ``--json`` always emits exactly one JSON object on stdout.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def run_cli(*args: str, data_root: Path, timeout: int = 300):
    env = {**os.environ, "RECON3D_HOME": str(data_root),
           "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    return subprocess.run([sys.executable, "-m", "recon3d.cli.main", *args],
                          capture_output=True, text=True, env=env, timeout=timeout)


@pytest.fixture(scope="module")
def cli_root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("cli-data")


def test_version_and_doctor(cli_root):
    version = run_cli("--version", data_root=cli_root)
    assert version.returncode == 0 and "recon3d" in version.stdout

    doctor = run_cli("doctor", "--json", data_root=cli_root)
    assert doctor.returncode == 0
    payload = json.loads(doctor.stdout)
    assert payload["ok"] is True
    assert payload["version"]
    assert "dependencies" in payload or "hardware" in payload


def test_info_is_a_full_capability_report(cli_root):
    info = run_cli("info", "--json", data_root=cli_root)
    assert info.returncode == 0
    payload = json.loads(info.stdout)
    assert len(payload["stages"]) >= 15
    assert payload["formats"]
    assert payload["parameters"]["defaults"]
    assert payload["interfaces"]["cli"]


def test_project_lifecycle(cli_root):
    created = run_cli("create", "cli-test", "--subject", "robot", "--json", data_root=cli_root)
    assert created.returncode == 0, created.stderr
    assert json.loads(created.stdout)["id"] == "cli-test"

    listed = run_cli("projects", "--json", data_root=cli_root)
    assert "cli-test" in [p["id"] for p in json.loads(listed.stdout)["projects"]]

    shown = run_cli("project", "cli-test", "--json", data_root=cli_root)
    assert json.loads(shown.stdout)["id"] == "cli-test"

    versions = run_cli("versions", "cli-test", "--json", data_root=cli_root)
    assert json.loads(versions.stdout)["versions"] == []

    # destructive operations require --yes
    refused = run_cli("projects", "--delete", "cli-test", data_root=cli_root)
    assert refused.returncode != 0
    deleted = run_cli("projects", "--delete", "cli-test", "--yes", "--json", data_root=cli_root)
    assert deleted.returncode == 0


def test_add_images_rejects_non_images(cli_root, tmp_path):
    run_cli("create", "cli-images", "--json", data_root=cli_root)
    junk = tmp_path / "not-an-image.txt"
    junk.write_text("hello")
    result = run_cli("add-images", "cli-images", str(junk), "--json", data_root=cli_root)
    assert result.returncode == 1
    assert "image" in (result.stdout + result.stderr).lower()


def test_unknown_project_exits_nonzero(cli_root):
    result = run_cli("project", "does-not-exist", "--json", data_root=cli_root)
    assert result.returncode == 1


def test_usage_error_exit_code(cli_root):
    result = run_cli("reconstruct", data_root=cli_root)
    assert result.returncode == 2  # argparse usage error


@pytest.mark.slow
def test_full_reconstruction_through_the_cli(cli_root, reference_dir):
    """A complete draft run: the CLI must emit parseable JSON and real outputs."""
    assert run_cli("create", "cli-e2e", "--subject", "robot", "--json",
                   data_root=cli_root).returncode == 0
    added = run_cli("add-images", "cli-e2e", str(reference_dir), "--json", data_root=cli_root)
    assert added.returncode == 0, added.stderr
    assert json.loads(added.stdout)["total"] == 7

    run = run_cli("reconstruct", "cli-e2e", "--preset", "draft",
                  "--texture-resolution", "512", "--formats", "glb",
                  "--no-lods", "--json", data_root=cli_root, timeout=1800)
    payload = json.loads(run.stdout)
    assert payload["status"] in {"completed", "ok"}
    assert payload["statistics"]["triangles"] > 200
    version_dir = Path(payload["outputs"]["version_dir"])
    assert version_dir.exists()
    assert (version_dir / "final").exists()
    assert list((version_dir / "final").glob("*.glb"))
    assert (version_dir / "reports" / "quality.json").exists()


def test_rest_health_and_project_lifecycle(cli_root):
    """The REST API must answer /health and return serialisable JSON payloads.

    Skipped when the optional ``api`` extra is not installed; the CLI path it
    exercises works without FastAPI.
    """
    import time
    import urllib.error
    import urllib.request

    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")

    port = 8791
    env = {**os.environ, "RECON3D_HOME": str(cli_root),
           "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "--factory", "recon3d.api.server:create_app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "error"],
        env=env, cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 30
        health = None
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                    health = json.loads(r.read())
                break
            except Exception:
                time.sleep(0.5)
        assert health is not None, "the API never became reachable"
        assert health["ok"] is True and health["version"]
        assert isinstance(health["active_jobs"], int)

        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/projects", method="POST",
            data=json.dumps({"name": "rest-check", "subject_type": "robot"}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=15) as r:
            created = json.loads(r.read())
        project_id = (created.get("project") or created).get("id")
        assert project_id

        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/projects/missing", timeout=10)
            raise AssertionError("a missing project must not return 200")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
            body = json.loads(exc.read())
            assert body["ok"] is False and body["code"] == "not_found"
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            server.kill()

"""End-to-end tests for the REST + WebSocket interface.

These start a real ``uvicorn`` server in a subprocess, drive it over HTTP exactly
like an agent would, and watch the job's progress over the WebSocket.  They are
the only tests that exercise the async progress channel, so they are marked
``slow`` (a draft reconstruction runs inside them).

Skipped cleanly when the optional ``api`` extra is not installed.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _Server:
    """Context manager around a ``recon3d serve`` subprocess."""

    def __init__(self, data_root: Path, port: int) -> None:
        self.data_root = data_root
        self.port = port
        self.process: subprocess.Popen | None = None

    def __enter__(self) -> "_Server":
        env = {**os.environ, "RECON3D_HOME": str(self.data_root),
               "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
        self.process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "--factory", "recon3d.api.server:create_app",
             "--host", "127.0.0.1", "--port", str(self.port), "--log-level", "error"],
            env=env, cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 40
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(self.get("/health"), timeout=5):
                    return self
            except Exception:
                if self.process.poll() is not None:
                    raise AssertionError("the API server exited during startup")
                time.sleep(0.4)
        raise AssertionError("the API server never became reachable")

    def __exit__(self, *exc) -> None:
        assert self.process is not None
        self.process.terminate()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:  # pragma: no cover
            self.process.kill()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def ws_url(self, path: str) -> str:
        return f"ws://127.0.0.1:{self.port}{path}"

    def get(self, path: str) -> str:
        return self.url(path)

    def request(self, path: str, *, method: str = "GET", payload=None, timeout: int = 30):
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.url(path), method=method, data=data,
            headers={"Content-Type": "application/json"} if data else {})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read() or b"{}")


@pytest.fixture()
def api_server(tmp_path: Path):
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    data_root = tmp_path / "api-data"
    with _Server(data_root, _free_port()) as server:
        yield server


def test_health_and_error_envelope(api_server):
    health = api_server.request("/health")
    assert health["ok"] is True and health["version"]
    assert isinstance(health["active_jobs"], int)

    capabilities = api_server.request("/v1/capabilities")
    assert len(capabilities["stages"]) >= 15
    assert "glb" in capabilities["formats"]

    with pytest.raises(urllib.error.HTTPError) as excinfo:
        api_server.request("/v1/projects/does-not-exist")
    assert excinfo.value.code == 404
    body = json.loads(excinfo.value.read())
    assert body["ok"] is False and body["code"] == "not_found"


@pytest.mark.slow
def test_websocket_streams_job_progress(api_server, reference_dir: Path, tmp_path: Path):
    """A real job, watched live over ``WS /v1/ws/jobs/{id}``."""
    from websockets.sync.client import connect

    from recon3d.config import load_config
    from recon3d.core.project import ProjectManager

    cfg = load_config(data_root=str(api_server.data_root))
    manager = ProjectManager(cfg.projects_path)
    project = manager.create("ws-check", subject_type="robot")
    project.add_images(sorted(reference_dir.glob("*.png")))
    assert len(project.images) > 0

    started = api_server.request(
        f"/v1/projects/{project.id}/reconstruct", method="POST",
        payload={"preset": "draft", "texture_resolution": 512, "export_formats": ["glb"],
                 "generate_lods": False})
    job_id = started["job"]["id"]
    assert job_id

    events, stages, done = [], set(), None
    with connect(api_server.ws_url(f"/v1/ws/jobs/{job_id}"), open_timeout=20,
                 close_timeout=10, max_size=None) as socket_conn:
        deadline = time.time() + 900
        while time.time() < deadline:
            message = json.loads(socket_conn.recv(timeout=900))
            if message.get("kind") == "done":
                done = message
                break
            if message.get("kind") == "error":  # pragma: no cover - would fail below
                pytest.fail(f"job socket reported an error: {message}")
            events.append(message)
            if message.get("stage"):
                stages.add(message["stage"])

    assert events, "the socket produced no progress events"
    assert done is not None, "the socket never reported completion"
    assert done["state"] == "completed", done
    # Progress must mention pipeline stages, not just a heartbeat.
    assert len(stages) >= 3, f"only saw stages {sorted(stages)}"
    assert any("mesh" in name or "reconstruct" in name for name in stages), sorted(stages)
    # Every event carries the documented fields.
    for event in events[:5]:
        assert set(event) >= {"stage", "progress", "message", "job", "level", "timestamp"}
        assert event["job"] == job_id
    result = done.get("result") or {}
    assert result.get("status") in {"completed", "ok"}
    assert (result.get("outputs") or {}).get("version_dir")

    # The same job is still queryable over plain HTTP after the socket closes.
    status = api_server.request(f"/v1/jobs/{job_id}")
    assert status["state"] == "completed"
    artifacts = api_server.request(f"/v1/jobs/{job_id}/artifacts")
    assert artifacts.get("files") or artifacts.get("artifacts") or artifacts.get("ok")


def test_job_artifacts_reject_path_traversal(api_server, tmp_path: Path):
    """`artifacts/{path}` must never escape the project directory."""
    from recon3d.config import load_config
    from recon3d.core.project import ProjectManager

    cfg = load_config(data_root=str(api_server.data_root))
    manager = ProjectManager(cfg.projects_path)
    project = manager.create("traversal-check", subject_type="prop")
    job = api_server.request(f"/v1/projects/{project.id}/reconstruct", method="POST",
                             payload={"stages": ["ingestion"]})
    job_id = job["job"]["id"]

    for attempt in ("../../../../etc/passwd", "..%2f..%2fetc%2fpasswd", "/etc/passwd"):
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            api_server.request(f"/v1/jobs/{job_id}/artifacts/{attempt}")
        assert excinfo.value.code in {400, 403, 404}

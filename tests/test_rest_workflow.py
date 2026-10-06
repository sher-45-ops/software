"""REST-only workflow tests: multipart upload, artefact download, model manager.

The engine's promise is that *everything* works headlessly through the API. These
tests drive the whole journey the way an external agent would - create a project
over HTTP, upload reference images as multipart files, run a reconstruction, list
the artefacts and download one - and check that the model manager is safe.
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
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _Server:
    def __init__(self, data_root: Path, port: int) -> None:
        self.data_root = data_root
        self.port = port

    def __enter__(self):
        env = {**os.environ, "RECON3D_HOME": str(self.data_root),
               "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
        self.process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "--factory", "recon3d.api.server:create_app",
             "--host", "127.0.0.1", "--port", str(self.port), "--log-level", "error"],
            env=env, cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 40
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(self.url("/health"), timeout=5):
                    return self
            except Exception:
                if self.process.poll() is not None:
                    raise AssertionError("the API server exited during startup")
                time.sleep(0.4)
        raise AssertionError("the API server never became reachable")

    def __exit__(self, *exc) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:  # pragma: no cover
            self.process.kill()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def request(self, path: str, *, method: str = "GET", payload=None, timeout: int = 30):
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.url(path), method=method, data=data,
            headers={"Content-Type": "application/json"} if data else {})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
        return json.loads(body or b"{}")

    def upload(self, path: str, files: list[Path], timeout: int = 60):
        """Multipart/form-data upload, built by hand to keep the test dependency-free."""
        boundary = f"----recon3d{uuid.uuid4().hex}"
        parts = []
        for file in files:
            parts.append(
                f"--{boundary}\r\n".encode()
                + f'Content-Disposition: form-data; name="files"; filename="{file.name}"\r\n'
                .encode()
                + b"Content-Type: image/png\r\n\r\n"
                + file.read_bytes()
                + b"\r\n")
        body = b"".join(parts) + f"--{boundary}--\r\n".encode()
        request = urllib.request.Request(
            self.url(path), method="POST", data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read() or b"{}")

    def download(self, path: str, timeout: int = 60) -> bytes:
        with urllib.request.urlopen(self.url(path), timeout=timeout) as response:
            return response.read()


@pytest.fixture()
def api(tmp_path: Path):
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    with _Server(tmp_path / "api-data", _free_port()) as server:
        yield server


@pytest.mark.slow
def test_full_workflow_over_http_only(api, reference_dir: Path):
    """Create -> upload -> reconstruct -> list artefacts -> download, over HTTP."""
    created = api.request("/v1/projects", method="POST",
                          payload={"name": "http-only", "subject_type": "robot"})
    project_id = (created.get("project") or created)["id"]
    assert api.request(f"/v1/projects/{project_id}")["id"] == project_id

    images = sorted(reference_dir.glob("*.png"))[:4]
    uploaded = api.upload(f"/v1/projects/{project_id}/images", images)
    assert len(uploaded["added"]) == len(images), uploaded
    listed = api.request(f"/v1/projects/{project_id}/images")
    assert len(listed["images"]) == len(images)

    started = api.request(f"/v1/projects/{project_id}/reconstruct", method="POST",
                          payload={"preset": "draft", "texture_resolution": 512,
                                   "export_formats": ["glb"], "generate_lods": False})
    job_id = started["job"]["id"]

    deadline = time.time() + 900
    state = "queued"
    while time.time() < deadline:
        status = api.request(f"/v1/jobs/{job_id}")
        state = status["state"]
        if state in {"completed", "failed", "cancelled", "partial"}:
            break
        time.sleep(2)
    assert state in {"completed", "partial"}, f"job ended as {state}: {status.get('error')}"

    artifacts = api.request(f"/v1/jobs/{job_id}/artifacts")
    names = [entry["path"] for entry in artifacts["files"]]
    assert any(name.endswith(".glb") for name in names), names
    assert any(name.endswith("quality.json") for name in names), names

    glb = next(name for name in names if name.endswith(".glb"))
    payload = api.download(f"/v1/jobs/{job_id}/artifacts/{glb}")
    assert payload[:4] == b"glTF", "the downloaded GLB is not a GLB (bad magic header)"
    assert len(payload) > 10_000

    # Deleting a project requires an explicit confirmation.
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        api.request(f"/v1/projects/{project_id}", method="DELETE")
    assert excinfo.value.code in {400, 403, 409, 422}
    assert api.request(f"/v1/projects/{project_id}?confirm=true",
                       method="DELETE")["deleted"] == project_id


def test_model_registry_is_honest_and_offline_safe(api):
    """/v1/models lists the optional registry; nothing downloads implicitly."""
    models = api.request("/v1/models")
    entries = models.get("models") or models.get("available") or []
    assert entries, models
    for entry in entries:
        assert entry.get("id")
        assert "license" in entry or "licence" in entry, entry
        assert "installed" in entry, entry
    # Nothing is installed in a fresh data root, and no download was attempted.
    assert not any(entry.get("installed") for entry in entries)

    # A traversal attempt in the model id must never touch the filesystem.
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        api.request("/v1/models/..%2f..%2fetc%2fpasswd/download", method="POST")
    assert excinfo.value.code in {400, 403, 404}


def test_model_manager_rejects_traversal_and_needs_confirmation(tmp_path: Path):
    """Direct API check: unsafe ids and unconfirmed deletes are refused."""
    from recon3d.config import load_config
    from recon3d.errors import SecurityError
    from recon3d.modelzoo.manager import ModelManager

    cfg = load_config(data_root=str(tmp_path / "models-data"))
    cfg.ensure_dirs()
    manager = ModelManager(cfg)

    listed = manager.list()
    assert listed
    with pytest.raises(Exception):
        manager.download("../../../tmp/evil")
    with pytest.raises(SecurityError):
        manager.delete("u2netp_salient")  # no confirmation

    offline = load_config(data_root=str(tmp_path / "models-data"),
                          overrides={"offline": True})
    assert offline.offline is True

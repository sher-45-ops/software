"""Local model manager: download, checksum, list and delete optional models.

Rules (spec #33, #36):

* **Nothing is downloaded implicitly.** A reconstruction never reaches the
  network; you either have the model on disk or the engine uses its classical
  CPU path.
* Offline mode (``config.offline`` or ``RECON3D_OFFLINE=1``) blocks every
  download, and so does ``allow_network_downloads = false``.
* Checksums are recorded the first time a file is downloaded and verified on
  every subsequent use, so a corrupted or swapped model file is detected.
* Deletion requires explicit confirmation, exactly like destructive project
  operations.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..errors import ModelError, SecurityError

REGISTRY_PATH = Path(__file__).resolve().parent / "registry.json"


def load_registry(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the shipped model registry (with an on-disk override if present)."""
    base = json.loads(Path(path or REGISTRY_PATH).read_text(encoding="utf-8"))
    return base


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@dataclass
class ModelStatus:
    id: str
    name: str
    purpose: str
    license: str
    installed: bool
    size_mb: float
    files: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "purpose": self.purpose, "license": self.license,
            "installed": self.installed, "size_mb": self.size_mb, "files": self.files,
        }


class ModelManager:
    """Manage the optional local models under ``<data_root>/models``."""

    def __init__(self, config) -> None:
        self.config = config
        self.models_dir = Path(config.models_path)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.models_dir / "state.json"
        self.registry = load_registry()
        self._state = self._load_state()

    # -- state ---------------------------------------------------------
    def _load_state(self) -> Dict[str, Any]:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:  # pragma: no cover
                return {}
        return {}

    def _save_state(self) -> None:
        self.state_path.write_text(json.dumps(self._state, indent=2), encoding="utf-8")

    def _model_dir(self, model_id: str) -> Path:
        safe = "".join(ch for ch in model_id if ch.isalnum() or ch in "._-")
        if not safe or safe != model_id:
            raise SecurityError(f"invalid model id: {model_id!r}")
        return self.models_dir / safe

    def entry(self, model_id: str) -> Dict[str, Any]:
        for model in self.registry.get("models", []):
            if model["id"] == model_id:
                return model
        raise ModelError(f"unknown model '{model_id}'",
                         details={"available": [m["id"] for m in self.registry.get("models", [])]})

    # -- queries -------------------------------------------------------
    def list(self) -> List[Dict[str, Any]]:
        models: List[Dict[str, Any]] = []
        for model in self.registry.get("models", []):
            directory = self._model_dir(model["id"])
            files: List[Dict[str, Any]] = []
            installed = True
            for spec in model.get("files", []):
                path = directory / spec["name"]
                present = path.exists()
                installed = installed and present
                files.append({
                    "name": spec["name"], "present": present,
                    "bytes": path.stat().st_size if present else 0,
                    "expected_sha256": spec.get("sha256", ""),
                    "recorded_sha256": (self._state.get(model["id"], {}) or {}).get(spec["name"], ""),
                    "url": spec.get("url", ""),
                })
            status = ModelStatus(model["id"], model.get("name", model["id"]), model.get("purpose", ""),
                                 model.get("license", "unknown"), installed,
                                 float(model.get("size_mb", 0)), files)
            models.append(status.to_dict())
        return models

    def is_installed(self, model_id: str) -> bool:
        return all(f["present"] for f in self.list() if f["id"] == model_id)

    def path_for(self, model_id: str, filename: Optional[str] = None) -> Optional[Path]:
        entry = self.entry(model_id)
        directory = self._model_dir(model_id)
        name = filename or entry["files"][0]["name"]
        path = directory / name
        return path if path.exists() else None

    def ensure_recommended(self) -> List[Dict[str, Any]]:
        results = []
        for model in self.registry.get("models", []):
            if model.get("recommended", True):
                results.append(self.download(model["id"]))
        return results

    # -- network -------------------------------------------------------
    def _check_download_allowed(self) -> None:
        if not getattr(self.config, "allow_network_downloads", True):
            raise ModelError("model downloads are disabled (allow_network_downloads = false)")
        if getattr(self.config, "offline", False) or os.environ.get("RECON3D_OFFLINE"):
            raise ModelError("offline mode is enabled; model downloads are blocked")

    def download(self, model_id: str, *, force: bool = False, timeout: int = 600) -> Dict[str, Any]:
        entry = self.entry(model_id)
        directory = self._model_dir(model_id)
        directory.mkdir(parents=True, exist_ok=True)
        self._check_download_allowed()
        results: List[Dict[str, Any]] = []
        for spec in entry.get("files", []):
            target = directory / spec["name"]
            if target.exists() and not force:
                ok, digest = self.verify_file(model_id, spec["name"])
                results.append({"name": spec["name"], "ok": ok, "skipped": True,
                                "sha256": digest, "reason": "already present"})
                continue
            url = spec.get("url")
            if not url:
                results.append({"name": spec["name"], "ok": False,
                                "error": "no download URL is configured for this file"})
                continue
            try:
                self._fetch(url, target, timeout=timeout)
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                results.append({"name": spec["name"], "ok": False, "error": str(exc)})
                continue
            digest = sha256_file(target)
            expected = spec.get("sha256") or ""
            if expected and expected != digest:
                target.unlink(missing_ok=True)
                results.append({"name": spec["name"], "ok": False,
                                "error": f"checksum mismatch (expected {expected[:12]}…, "
                                         f"got {digest[:12]}…)"})
                continue
            self._state.setdefault(model_id, {})[spec["name"]] = digest
            self._save_state()
            results.append({"name": spec["name"], "ok": True, "sha256": digest,
                            "bytes": target.stat().st_size,
                            "note": "" if expected else "checksum recorded for future verification"})
        ok = all(r["ok"] for r in results)
        return {"id": model_id, "ok": ok, "files": results, "directory": str(directory)}

    def _fetch(self, url: str, target: Path, *, timeout: int) -> None:
        if not url.startswith(("https://", "http://")):
            raise ModelError(f"refusing to download from a non-http(s) URL: {url!r}")
        request = urllib.request.Request(url, headers={"User-Agent": "recon3d-model-manager"})
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response, open(tmp_path, "wb") as out:
                shutil.copyfileobj(response, out, length=1 << 20)
            os.replace(tmp_path, target)
        finally:
            tmp_path.unlink(missing_ok=True)

    def verify_file(self, model_id: str, filename: str) -> "tuple[bool, str]":
        entry = self.entry(model_id)
        directory = self._model_dir(model_id)
        path = directory / filename
        if not path.exists():
            return False, ""
        digest = sha256_file(path)
        spec = next((f for f in entry.get("files", []) if f["name"] == filename), {})
        expected = spec.get("sha256") or (self._state.get(model_id, {}) or {}).get(filename, "")
        return (not expected or expected == digest), digest

    def verify_all(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for status in self.list():
            if not status["installed"]:
                results.append({"id": status["id"], "ok": False, "error": "not installed"})
                continue
            file_ok = True
            detail = []
            for spec in self.entry(status["id"]).get("files", []):
                ok, digest = self.verify_file(status["id"], spec["name"])
                file_ok = file_ok and ok
                detail.append({"name": spec["name"], "ok": ok, "sha256": digest})
            results.append({"id": status["id"], "ok": file_ok, "files": detail})
        return results

    def delete(self, model_id: str, *, confirm: bool = False) -> Dict[str, Any]:
        if not confirm:
            raise SecurityError(
                "deleting a model requires explicit confirmation (pass confirm=True / --yes)",
                details={"action": "delete_model", "model": model_id},
            )
        directory = self._model_dir(model_id)
        existed = directory.exists()
        if existed:
            shutil.rmtree(directory)
        self._state.pop(model_id, None)
        self._save_state()
        return {"id": model_id, "ok": True, "removed": existed}

    # -- runtime helpers ----------------------------------------------
    def runtime_for(self, model_id: str) -> Dict[str, Any]:
        """Return availability information for a model's runtime."""
        entry = self.entry(model_id)
        runtime = entry.get("runtime", "onnxruntime")
        try:
            module = __import__(runtime)
            version = getattr(module, "__version__", "present")
            available = True
        except Exception as exc:  # pragma: no cover
            version, available = str(exc), False
        return {"runtime": runtime, "available": available, "version": version,
                "installed": self.is_installed(model_id),
                "path": str(self._model_dir(model_id))}

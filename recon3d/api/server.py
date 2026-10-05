"""REST + WebSocket API.

The API is a thin, explicit layer over the same core the CLI uses, so an agent can
drive the engine without touching a shell (spec #35: no arbitrary shell
execution, whitelisted operations only).

Endpoints (all JSON, all under ``/v1``)::

    GET    /health
    GET    /v1/capabilities
    GET    /v1/doctor
    GET    /v1/presets
    POST   /v1/projects                      {name, subject_type, style, notes}
    GET    /v1/projects
    GET    /v1/projects/{project}
    DELETE /v1/projects/{project}?confirm=true
    POST   /v1/projects/{project}/images     multipart files (or {"paths": [...]})
    GET    /v1/projects/{project}/images
    POST   /v1/projects/{project}/reconstruct {params...}
    GET    /v1/projects/{project}/versions
    GET    /v1/jobs/{job}
    POST   /v1/jobs/{job}/cancel
    GET    /v1/jobs/{job}/artifacts
    GET    /v1/jobs/{job}/artifacts/{path...}
    WS     /v1/ws/jobs/{job}                 progress events (live)
    GET    /v1/models
    POST   /v1/models/{model}/download
    POST   /v1/export                        re-export a finished version

The studio UI is served at ``/`` from ``recon3d/studio/static``.
"""

from __future__ import annotations

import json
import mimetypes
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import VERSION
from ..config import load_config
from ..core.project import ProjectManager
from ..errors import Recon3DError

STATIC_DIR = Path(__file__).resolve().parent.parent / "studio" / "static"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except Exception:  # pragma: no cover
            return str(value)
    return value


def create_app(config=None):
    """Build the FastAPI application (imported lazily so the CLI stays light)."""
    from fastapi import Body, FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    cfg = config or load_config()
    cfg.ensure_dirs()
    manager = ProjectManager(cfg.projects_path)

    from ..core.jobs import JobManager

    jobs = JobManager(log_dir=cfg.logs_path)

    app = FastAPI(title="Recon3D Engine", version=VERSION,
                  description="Local-first multi-view image to 3D reconstruction engine")

    def error_response(exc: Recon3DError) -> JSONResponse:
        return JSONResponse(status_code=getattr(exc, "http_status", 400),
                            content={"ok": False, "code": getattr(exc, "code", "error"),
                                     "message": str(exc),
                                     "details": getattr(exc, "details", {})})

    @app.exception_handler(Recon3DError)
    async def _handle(request, exc: Recon3DError):  # pragma: no cover - framework hook
        return error_response(exc)

    # -- meta ----------------------------------------------------------
    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"ok": True, "version": VERSION, "data_root": str(cfg.data_root),
                "offline": bool(cfg.offline), "active_jobs": jobs.active_count}

    @app.get("/v1/capabilities")
    def capabilities() -> Dict[str, Any]:
        from ..agent.manifest import capability_report

        return capability_report()

    @app.get("/v1/doctor")
    def doctor() -> Dict[str, Any]:
        from ..core.resources import hardware_report
        from ..engine.backends.detect import detect_backends

        return {"ok": True, "version": VERSION, "hardware": hardware_report(),
                "backends": detect_backends(), "data_root": str(cfg.data_root)}

    @app.get("/v1/presets")
    def presets() -> Dict[str, Any]:
        from ..core.pipeline import available_presets

        return {"presets": available_presets()}

    # -- projects ------------------------------------------------------
    @app.post("/v1/projects")
    def create_project(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        project = manager.create(str(payload.get("name") or "untitled"),
                                 subject_type=str(payload.get("subject_type", "auto")),
                                 style=str(payload.get("style", "realistic")),
                                 description=str(payload.get("notes")
                                                 or payload.get("description", "")))
        return _json_safe(project.to_dict(include_images=False))

    @app.get("/v1/projects")
    def list_projects() -> Dict[str, Any]:
        return {"projects": _json_safe(manager.list())}

    @app.get("/v1/projects/{project_id}")
    def get_project(project_id: str) -> Dict[str, Any]:
        project = manager.resolve(project_id)
        return _json_safe(project.to_dict())

    @app.delete("/v1/projects/{project_id}")
    def delete_project(project_id: str, confirm: bool = False) -> Dict[str, Any]:
        manager.delete(project_id, confirm=confirm)
        return {"ok": True, "deleted": project_id}

    @app.post("/v1/projects/{project_id}/images")
    async def add_images(project_id: str, files: List[UploadFile] = File(default=None),
                         payload: Optional[Dict[str, Any]] = Body(default=None)) -> Dict[str, Any]:
        project = manager.resolve(project_id)
        stored: List[Path] = []
        if files:
            staging = project.root / "input" / "uploads"
            staging.mkdir(parents=True, exist_ok=True)
            for upload in files:
                name = Path(upload.filename or "upload.png").name
                target = staging / f"{int(time.time() * 1000)}_{name}"
                target.write_bytes(await upload.read())
                stored.append(target)
        elif payload and payload.get("paths"):
            stored = [Path(p) for p in payload["paths"]]
        else:
            raise HTTPException(status_code=400, detail="provide files or {'paths': [...]}")
        added = project.add_images(stored)
        return {"ok": True, "added": _json_safe([img.to_dict() for img in added]),
                "total": len(project.images)}

    @app.get("/v1/projects/{project_id}/images")
    def project_images(project_id: str) -> Dict[str, Any]:
        project = manager.resolve(project_id)
        return {"images": _json_safe([img.to_dict() for img in project.images])}

    @app.get("/v1/projects/{project_id}/versions")
    def project_versions(project_id: str) -> Dict[str, Any]:
        project = manager.resolve(project_id)
        return {"versions": _json_safe([v.to_dict() for v in project.versions()]),
                "current": project.current_version}

    # -- jobs ----------------------------------------------------------
    @app.post("/v1/projects/{project_id}/reconstruct")
    def reconstruct(project_id: str, payload: Dict[str, Any] = Body(default=None)) -> Dict[str, Any]:
        from ..core.pipeline import run_pipeline

        project = manager.resolve(project_id)
        params = dict(payload or {})
        stages = params.pop("stages", None)
        job = jobs.create(project.id, kind="reconstruct", params=params, project_obj=project)
        jobs.submit(job, lambda j: run_pipeline(project, j, params=params, stages=stages))
        return {"ok": True, "job": _json_safe(job.to_dict())}

    @app.get("/v1/jobs")
    def list_jobs(project: Optional[str] = None) -> Dict[str, Any]:
        return {"jobs": _json_safe([j.to_dict() for j in jobs.list(project)])}

    @app.get("/v1/jobs/{job_id}")
    def job_status(job_id: str, events: int = 50) -> Dict[str, Any]:
        job = jobs.get(job_id)
        return _json_safe(job.to_dict(include_events=True, event_limit=events))

    @app.post("/v1/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> Dict[str, Any]:
        job = jobs.cancel(job_id)
        return {"ok": True, "job": _json_safe(job.to_dict())}

    @app.get("/v1/jobs/{job_id}/artifacts")
    def job_artifacts(job_id: str) -> Dict[str, Any]:
        job = jobs.get(job_id)
        result = job.result or {}
        outputs = result.get("outputs") or {}
        version_dir = outputs.get("version_dir")
        files: List[Dict[str, Any]] = []
        if version_dir and Path(version_dir).exists():
            root = Path(version_dir)
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    files.append({"path": str(path.relative_to(root)),
                                  "bytes": path.stat().st_size,
                                  "url": f"/v1/jobs/{job_id}/artifacts/{path.relative_to(root)}"})
        return {"job": job_id, "version_dir": version_dir, "files": files,
                "outputs": _json_safe(outputs)}

    @app.get("/v1/jobs/{job_id}/artifacts/{artifact_path:path}")
    def download_artifact(job_id: str, artifact_path: str):
        job = jobs.get(job_id)
        outputs = (job.result or {}).get("outputs") or {}
        version_dir = outputs.get("version_dir")
        if not version_dir:
            raise HTTPException(status_code=404, detail="job has no outputs yet")
        root = Path(version_dir).resolve()
        target = (root / artifact_path).resolve()
        if root not in target.parents and target != root:
            raise HTTPException(status_code=403, detail="path escapes the version directory")
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="artifact not found")
        media, _ = mimetypes.guess_type(target.name)
        return FileResponse(target, media_type=media or "application/octet-stream",
                            filename=target.name)

    @app.post("/v1/export")
    def re_export(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        from ..engine.export.writers import export_asset, verify_export
        import trimesh

        project = manager.resolve(str(payload.get("project")))
        version = project.get_version(payload.get("version"))
        base = project.version_path(version.id)
        if not version.assets.mesh:
            raise HTTPException(status_code=409, detail="version has no mesh")
        source = base / next(iter(version.assets.mesh.values()))
        mesh = trimesh.load(str(source), force="mesh", process=False)
        out_dir = Path(payload.get("output_dir") or (base / "exports")).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for fmt in payload.get("formats", ["glb"]):
            target = out_dir / f"{project.id}_{version.id}.{fmt}"
            try:
                export_asset(mesh, target, fmt)
                results.append({"format": fmt, "path": str(target),
                                "verification": verify_export(target, fmt)})
            except Exception as exc:
                results.append({"format": fmt, "error": str(exc)})
        return {"ok": all("error" not in r for r in results), "exports": results}

    # -- models --------------------------------------------------------
    @app.get("/v1/models")
    def models() -> Dict[str, Any]:
        from ..modelzoo.manager import ModelManager

        return {"models": ModelManager(cfg).list(), "offline": bool(cfg.offline)}

    @app.post("/v1/models/{model_id}/download")
    def download_model(model_id: str) -> Dict[str, Any]:
        from ..modelzoo.manager import ModelManager

        return ModelManager(cfg).download(model_id)

    # -- websocket progress -------------------------------------------
    @app.websocket("/v1/ws/jobs/{job_id}")
    async def job_socket(websocket: WebSocket, job_id: str) -> None:
        await websocket.accept()
        try:
            job = jobs.get(job_id)
        except Recon3DError as exc:
            await websocket.send_json({"kind": "error", "message": str(exc)})
            await websocket.close()
            return
        queue: List[Dict[str, Any]] = []
        lock = threading.Lock()

        def on_event(job_event_id: str, event) -> None:
            if job_event_id != job_id:
                return
            with lock:
                queue.append(event.to_dict())

        unsubscribe = jobs.subscribe(on_event)
        try:
            sent = 0
            while True:
                with lock:
                    pending = queue[sent:]
                    sent = len(queue)
                for payload in pending:
                    await websocket.send_json(_json_safe(payload))
                if job.terminal and not pending:
                    await websocket.send_json({"kind": "done", "state": job.state.value,
                                               "result": _json_safe(job.result)})
                    break
                await _async_sleep(0.25)
        except (WebSocketDisconnect, RuntimeError):  # pragma: no cover - client went away
            pass
        finally:
            unsubscribe()
            try:
                await websocket.close()
            except RuntimeError:  # pragma: no cover
                pass

    # -- studio --------------------------------------------------------
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/", response_class=HTMLResponse)
        def studio() -> HTMLResponse:
            return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))

    return app


async def _async_sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)


def run_server(config=None, *, host: str = "127.0.0.1", port: int = 8760,
               reload: bool = False, open_browser: bool = False) -> None:  # pragma: no cover
    """Start the API server (blocking)."""
    try:
        import uvicorn
    except ImportError as exc:
        raise Recon3DError(
            "the API server needs FastAPI + uvicorn: `pip install recon3d[api]`"
        ) from exc

    cfg = config or load_config()
    app = create_app(cfg)
    url = f"http://{host}:{port}"
    print(f"Recon3D API listening on {url}  (studio UI: {url}/)")
    print("Press Ctrl+C to stop.")
    if open_browser:
        import threading as _threading
        import webbrowser

        _threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="info", reload=False)

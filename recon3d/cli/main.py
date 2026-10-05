"""The ``recon3d`` command line interface.

Every operation the engine can perform is reachable from here, headless, on
Windows / Linux / macOS - this is the primary operator interface (the API and MCP
server wrap the same functions).  Add ``--json`` to any command for
machine-readable output.

Examples
--------
::

    recon3d doctor
    recon3d create hero --subject character
    recon3d add-images hero ./refs/*
    recon3d reconstruct hero --preset high --target-polycount 60000
    recon3d status hero --watch
    recon3d export hero --formats glb,fbx,usdz
    recon3d models list
    recon3d serve --port 8760
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .. import VERSION
from ..config import load_config, save_config
from ..errors import Recon3DError

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _emit(payload: Any, args: argparse.Namespace, human: Optional[str] = None) -> None:
    """Print either machine-readable JSON or a human summary."""
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, default=str))
    elif human is not None:
        print(human)
    else:
        print(json.dumps(payload, indent=2, default=str))


def _project_manager(args: argparse.Namespace):
    from ..core.project import ProjectManager

    cfg = load_config()
    cfg.ensure_dirs()
    return cfg, ProjectManager(cfg.projects_path)


def _resolve_project(args: argparse.Namespace):
    cfg, manager = _project_manager(args)
    return cfg, manager, manager.resolve(args.project)


def _job_manager(cfg):
    from ..core.jobs import JobManager

    return JobManager(log_dir=cfg.logs_path)


def _human_bytes(value: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} TB"


def _progress_printer(args: argparse.Namespace):
    state = {"last": 0.0}

    def show(event) -> None:
        if getattr(args, "json", False):
            # Machine-readable stdout must stay a *single* JSON document, so the
            # event stream goes to stderr (still one JSON object per line).
            print(json.dumps(event.to_dict(), default=str), file=sys.stderr, flush=True)
            return
        now = time.time()
        if event.kind in {"progress", "stage_start"} and now - state["last"] < 0.4:
            return
        state["last"] = now
        stage = event.stage or "-"
        bar_len = 24
        filled = int(bar_len * max(0.0, min(100.0, event.progress)) / 100.0)
        bar = "#" * filled + "." * (bar_len - filled)
        print(f"[{bar}] {event.progress:5.1f}%  {stage:20s} {event.message}", flush=True)

    return show


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def cmd_doctor(args: argparse.Namespace) -> int:
    """Diagnose the installation: hardware, dependencies, paths, writability."""
    from ..core.resources import hardware_report

    cfg = load_config()
    checks: List[Dict[str, Any]] = []
    ok = True

    try:
        import numpy

        checks.append({"check": "numpy", "ok": True, "detail": numpy.__version__})
    except Exception as exc:  # pragma: no cover
        ok = False
        checks.append({"check": "numpy", "ok": False, "detail": str(exc)})
    for module, purpose in (("cv2", "segmentation / image IO"), ("scipy", "numerics"),
                            ("skimage", "marching cubes"), ("trimesh", "mesh IO"),
                            ("PIL", "image IO"), ("fast_simplification", "fast decimation")):
        try:
            mod = __import__(module)
            checks.append({"check": module, "ok": True,
                           "detail": getattr(mod, "__version__", "present"),
                           "purpose": purpose})
        except Exception as exc:
            optional = module in {"fast_simplification"}
            ok = ok and optional
            checks.append({"check": module, "ok": False, "detail": str(exc),
                           "purpose": purpose, "optional": optional})

    from ..engine.backends.detect import detect_backends

    backends = detect_backends()
    hardware = hardware_report()
    problems: List[str] = []
    for directory in (cfg.data_root, cfg.projects_path, cfg.cache_path, cfg.models_path, cfg.logs_path):
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            probe = Path(directory) / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            writable = True
        except OSError as exc:
            writable = False
            ok = False
            problems.append(f"{directory}: {exc}")
        checks.append({"check": f"writable:{directory}", "ok": writable})

    payload = {
        "ok": ok,
        "version": VERSION,
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "data_root": str(cfg.data_root),
        "offline": cfg.offline,
        "hardware": hardware,
        "backends": backends,
        "checks": checks,
        "problems": problems,
        "advice": _doctor_advice(backends, hardware),
    }
    lines = [f"recon3d {VERSION} on Python {sys.version.split()[0]} ({sys.platform})",
             f"data root: {cfg.data_root}",
             f"cpu: {hardware['cpu_count']} cores, RAM {_human_bytes(hardware['ram_total_mb'] * 1024 * 1024)}, "
             f"device: {hardware['device']}"
             + (f", GPU: {hardware['gpu']['name']}" if hardware.get("gpu") else ""),
             "", "dependencies:"]
    for check in checks:
        if check["check"].startswith("writable:"):
            continue
        mark = "ok  " if check["ok"] else "MISS"
        lines.append(f"  [{mark}] {check['check']}: {check.get('detail', '')}")
    lines.append("")
    lines.append("optional local backends:")
    for backend in backends["backends"]:
        mark = "yes" if backend["available"] else "no "
        lines.append(f"  [{mark}] {backend['name']:20s} {backend['purpose']}")
    if problems:
        lines.append("")
        lines.append("problems:")
        lines.extend(f"  - {p}" for p in problems)
    lines.append("")
    lines.append("advice:")
    lines.extend(f"  - {tip}" for tip in payload["advice"])
    _emit(payload, args, "\n".join(lines))
    return EXIT_OK if ok else EXIT_ERROR


def _doctor_advice(backends: Dict[str, Any], hardware: Dict[str, Any]) -> List[str]:
    tips: List[str] = []
    names = {b["name"]: b for b in backends["backends"]}
    if not names.get("fast_simplification", {}).get("available"):
        tips.append("install `fast-simplification` (pip install recon3d[quality]) for fast decimation")
    if not names.get("xatlas", {}).get("available"):
        tips.append("install `xatlas` (pip install recon3d[quality]) for production-quality UVs")
    if not names.get("onnxruntime", {}).get("available"):
        tips.append("optional: `pip install recon3d[neural]` + `recon3d models download` for "
                    "local neural depth/segmentation")
    if hardware["ram_total_mb"] < 6000:
        tips.append(f"only {_human_bytes(hardware['ram_total_mb'] * 1024 * 1024)} RAM detected - "
                    "use --preset draft or --performance-mode balanced for large image sets")
    if not tips:
        tips.append("everything needed for a full-quality local reconstruction is present")
    return tips


def cmd_setup(args: argparse.Namespace) -> int:
    """Create the data root + config, optionally download the optional models."""
    cfg = load_config()
    if args.data_root:
        cfg.data_root = str(Path(args.data_root).expanduser().resolve())
    cfg.ensure_dirs()
    if args.performance_mode:
        cfg.performance_mode = args.performance_mode
    if args.offline:
        cfg.offline = True
    path = save_config(cfg)
    payload = {"ok": True, "config": str(path), "data_root": str(cfg.data_root),
               "projects": str(cfg.projects_path), "models": str(cfg.models_path)}
    if args.with_models:
        from ..modelzoo.manager import ModelManager

        manager = ModelManager(cfg)
        payload["models"] = manager.ensure_recommended()
    _emit(payload, args, f"setup complete: {cfg.data_root}")
    return EXIT_OK


def cmd_create(args: argparse.Namespace) -> int:
    cfg, manager = _project_manager(args)
    project = manager.create(args.name, subject_type=args.subject, style=args.style,
                             description=args.notes or "")
    payload = project.to_dict(include_images=False)
    payload["path"] = str(project.root)
    _emit(payload, args, f"created project '{project.id}' at {project.root}")
    return EXIT_OK


def cmd_add_images(args: argparse.Namespace) -> int:
    cfg, manager, project = _resolve_project(args)
    paths = [Path(p) for p in args.images]
    expanded: List[Path] = []
    for path in paths:
        if path.is_dir():
            for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.tif", "*.tiff"):
                expanded.extend(sorted(path.glob(pattern)))
        else:
            expanded.append(path)
    if not expanded:
        print("no images found", file=sys.stderr)
        return EXIT_USAGE
    added = project.add_images(expanded, views=args.views)
    payload = {"added": [img.to_dict() for img in added], "count": len(added),
               "project": project.id, "total": len(project.images)}
    _emit(payload, args, f"added {len(added)} image(s); project now has {len(project.images)}")
    return EXIT_OK


def cmd_projects(args: argparse.Namespace) -> int:
    cfg, manager = _project_manager(args)
    if args.delete:
        manager.delete(args.delete, confirm=args.yes)
        _emit({"deleted": args.delete, "ok": True}, args, f"deleted project '{args.delete}'")
        return EXIT_OK
    projects = manager.list()
    lines = [f"{p['id']:24s} {p.get('subject_type', ''):18s} {p.get('images', 0):3d} images  "
             f"{p.get('name', '')}" for p in projects] or ["no projects yet"]
    _emit({"projects": projects}, args, "\n".join(lines))
    return EXIT_OK


def cmd_reconstruct(args: argparse.Namespace) -> int:
    cfg, manager, project = _resolve_project(args)
    from ..core.pipeline import run_pipeline, resolve_params

    params: Dict[str, Any] = {"preset": args.preset, "quality": args.quality or args.preset}
    if args.target_polycount is not None:
        params["target_polycount"] = args.target_polycount
    if args.texture_resolution:
        params["texture_resolution"] = args.texture_resolution
    if args.formats:
        params["export_formats"] = [f.strip() for f in args.formats.split(",") if f.strip()]
    if args.units:
        params["units"] = args.units
    if args.subject_height:
        params["subject_height_m"] = args.subject_height
    for flag, key in ((args.rig, "generate_rig"), (args.lods, "generate_lods"),
                      (args.no_uvs, "generate_uvs"), (args.preview, "generate_previews")):
        if flag is not None:
            params[key] = bool(flag)
    if args.stages:
        params["stages"] = [s.strip() for s in args.stages.split(",") if s.strip()]
    if args.param:
        for item in args.param:
            if "=" not in item:
                print(f"--param expects key=value, got '{item}'", file=sys.stderr)
                return EXIT_USAGE
            key, _, raw = item.partition("=")
            try:
                params[key.strip()] = json.loads(raw)
            except json.JSONDecodeError:
                params[key.strip()] = raw

    jobs = _job_manager(cfg)
    job = jobs.create(project.id, kind="reconstruct", params=params, project_obj=project)
    unsubscribe = jobs.subscribe(lambda job_id, event: _progress_printer(args)(event)
                                 if job_id == job.id else None)
    try:
        if args.watch or not args.json:
            jobs.run_sync(job, lambda j: run_pipeline(project, j, params=params,
                                                      stages=params.get("stages")))
        else:
            jobs.submit(job, lambda j: run_pipeline(project, j, params=params,
                                                    stages=params.get("stages")))
            job = jobs.wait(job.id)
    finally:
        unsubscribe()
    # Flat, agent-friendly envelope: the pipeline result keys (status, quality,
    # statistics, outputs, stages_failed, validation) sit at the top level *and*
    # under "result" for callers that prefer the nested shape.
    result = job.result or {}
    payload = {**result,
               "job": job.to_dict(include_events=True, event_limit=200),
               "result": result,
               "state": job.state.value if hasattr(job.state, "value") else str(job.state)}
    if job.state.value == "failed":
        _emit(payload, args, f"reconstruction failed: {job.error}")
        return EXIT_ERROR
    human = _format_result(job)
    _emit(payload, args, human)
    return EXIT_OK


def _num(value: Any, default: float = 0.0) -> float:
    """Coerce a possibly-missing metric to a float (measurements are optional)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number == number else default  # NaN -> default


def _format_result(job) -> str:
    result = job.result or {}
    quality = result.get("quality") or {}
    stats = result.get("statistics") or {}
    outputs = result.get("outputs") or {}
    lines = [f"job {job.id}: {job.state.value} in {_num(job.duration_s):.1f}s",
             f"version: {result.get('version', '-')}",
             f"quality: {quality.get('grade', '-')} ({_num(quality.get('overall')):.1f}/100)"]
    metrics = (quality.get("metrics") or {})
    if metrics:
        lines.append(f"  silhouette IoU {_num(metrics.get('mean_silhouette_iou')):.3f} | "
                     f"coverage delta {_num(metrics.get('mean_coverage_delta')):+.3f} | "
                     f"colour RMSE {_num(metrics.get('mean_color_rmse')):.3f}")
    if stats:
        lines.append(f"mesh: {stats.get('triangles') or 0} tris, {stats.get('vertices') or 0} verts, "
                     f"LODs {stats.get('lod_levels') or 0}, "
                     f"texture {stats.get('texture_resolution') or 0}px")
    if outputs.get("version_dir"):
        lines.append(f"output: {outputs['version_dir']}")
    for warning in (quality.get("warnings") or [])[:4]:
        lines.append(f"  ! {warning}")
    return "\n".join(lines)


def cmd_status(args: argparse.Namespace) -> int:
    cfg, manager, project = _resolve_project(args)
    jobs = _job_manager(cfg)
    if args.job:
        job = jobs.get(args.job)
    else:
        project_jobs = jobs.list(project.id)
        job = project_jobs[0] if project_jobs else None
    if args.watch and job is not None:
        printer = _progress_printer(args)
        seen = 0
        while not job.terminal:
            events = job.events
            while seen < len(events):
                printer(events[seen])
                seen += 1
            time.sleep(0.5)
        for event in job.events[seen:]:
            printer(event)
    payload = {
        "project": project.to_dict(include_images=False),
        "current_version": project.current_version,
        "disk_usage": project.disk_usage(),
        "job": job.to_dict(include_events=True, event_limit=100) if job else None,
    }
    lines = [f"project {project.id} '{project.name}' ({len(project.images)} images, "
             f"subject={project.subject_type})",
             f"current version: {project.current_version or '-'}"]
    if job:
        lines.append(f"latest job: {job.id} {job.state.value} ({job.progress:.0f}%) {job.message}")
    for version in project.versions():
        lines.append(f"  {version.id}: {version.status} quality={(version.quality or {}).get('grade', '-')} "
                     f"created {version.created_at}")
    _emit(payload, args, "\n".join(lines))
    return EXIT_OK


def cmd_export(args: argparse.Namespace) -> int:
    cfg, manager, project = _resolve_project(args)
    from ..engine.export.writers import export_asset, verify_export

    version = project.get_version(args.version)
    base = project.version_path(version.id)
    if not version.assets.mesh:
        print("this version has no mesh to export", file=sys.stderr)
        return EXIT_ERROR
    fmt_source = next(iter(version.assets.mesh))
    mesh_path = base / version.assets.mesh[fmt_source]
    import trimesh

    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    formats = [f.strip().lower() for f in (args.formats or fmt_source).split(",") if f.strip()]
    out_dir = Path(args.out).expanduser().resolve() if args.out else base / "exports"
    results = []
    for fmt in formats:
        target = out_dir / f"{project.id}_{version.id}.{ 'gltf' if fmt == 'gltf' else fmt }"
        try:
            written = export_asset(mesh, target, fmt)
            verification = verify_export(target, fmt)
            results.append({"format": fmt, "path": str(target), "ok": verification.get("ok", False),
                            "bytes": target.stat().st_size if target.exists() else 0,
                            "verification": verification})
        except Exception as exc:
            results.append({"format": fmt, "error": str(exc), "ok": False})
    ok = all(r.get("ok") for r in results)
    payload = {"version": version.id, "exports": results, "output_dir": str(out_dir)}
    lines = [f"exported {len(results)} format(s) to {out_dir}"]
    lines += [f"  [{'ok' if r.get('ok') else 'FAIL'}] {r['format']}: {r.get('path', r.get('error'))}"
              for r in results]
    _emit(payload, args, "\n".join(lines))
    return EXIT_OK if ok else EXIT_ERROR


def cmd_versions(args: argparse.Namespace) -> int:
    cfg, manager, project = _resolve_project(args)
    versions = project.versions()
    payload = {"project": project.id,
               "versions": [v.to_dict() for v in versions],
               "current": project.data.get("current_version")}
    lines = [f"{v.id:10s} {v.status:10s} {(v.quality or {}).get('grade', '-'):8s} {v.label}" for v in versions]
    _emit(payload, args, "\n".join(lines) or "no versions yet")
    return EXIT_OK


def cmd_serve(args: argparse.Namespace) -> int:
    from ..api.server import run_server

    cfg = load_config()
    run_server(cfg, host=args.host, port=args.port, reload=args.reload,
               open_browser=args.open_browser)
    return EXIT_OK


def cmd_models(args: argparse.Namespace) -> int:
    from ..modelzoo.manager import ModelManager

    cfg = load_config()
    manager = ModelManager(cfg)
    if args.action == "list":
        payload = {"models": manager.list(), "models_dir": str(cfg.models_path),
                   "offline": cfg.offline}
        lines = []
        for entry in payload["models"]:
            mark = "installed" if entry["installed"] else "missing"
            lines.append(f"  [{mark:9s}] {entry['id']:24s} {entry['license']:16s} {entry['purpose']}")
        _emit(payload, args, "\n".join(lines) or "no models registered")
        return EXIT_OK
    if args.action == "download":
        targets = args.ids or ["depth_anything_v2_small"]
        results = [manager.download(model_id, force=args.force) for model_id in targets]
        _emit({"downloaded": results}, args,
              "\n".join(f"  {r['id']}: {'ok' if r['ok'] else r.get('error')}" for r in results))
        return EXIT_OK if all(r["ok"] for r in results) else EXIT_ERROR
    if args.action == "verify":
        results = manager.verify_all()
        _emit({"verified": results}, args,
              "\n".join(f"  {r['id']}: {'ok' if r['ok'] else r.get('error')}" for r in results))
        return EXIT_OK if all(r["ok"] for r in results) else EXIT_ERROR
    if args.action == "delete":
        if not args.ids:
            print("delete needs at least one model id", file=sys.stderr)
            return EXIT_USAGE
        results = [manager.delete(model_id, confirm=args.yes) for model_id in args.ids]
        _emit({"deleted": results}, args, "\n".join(f"  {r['id']}: {r['ok']}" for r in results))
        return EXIT_OK
    print(f"unknown action {args.action}", file=sys.stderr)
    return EXIT_USAGE


def cmd_studio(args: argparse.Namespace) -> int:
    args.host = getattr(args, "host", "127.0.0.1")
    args.port = getattr(args, "port", 8760)
    args.reload = False
    args.open_browser = True
    return cmd_serve(args)


def cmd_info(args: argparse.Namespace) -> int:
    """Print the full capability report (used by agents and the manifest)."""
    from ..agent.manifest import capability_report

    payload = capability_report()
    _emit(payload, args)
    return EXIT_OK


def cmd_project_get(args: argparse.Namespace) -> int:
    cfg, manager, project = _resolve_project(args)
    data = project.to_dict()
    payload = {"id": data.get("id", project.id), **data,
               "versions": [v.to_dict() for v in project.versions()],
               "path": str(project.root)}
    _emit(payload, args, json.dumps(payload, indent=2, default=str))
    return EXIT_OK


# --------------------------------------------------------------------------
# argument parser
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recon3d",
        description="Local-first multi-view image -> 3D reconstruction engine (CPU friendly, "
                    "no external generative AI services).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run `recon3d doctor` first, then `recon3d create` / `add-images` / `reconstruct`.",
    )
    parser.add_argument("--version", action="version", version=f"recon3d {VERSION}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_json(p):
        p.add_argument("--json", action="store_true", help="machine-readable JSON output")

    p = sub.add_parser("doctor", help="diagnose hardware, dependencies, paths and backends")
    add_json(p)
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("setup", help="create the data root and configuration")
    p.add_argument("--data-root")
    p.add_argument("--performance-mode", choices=["auto", "draft", "balanced", "quality", "maximum"])
    p.add_argument("--offline", action="store_true")
    p.add_argument("--with-models", action="store_true", help="download the optional local models")
    add_json(p)
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("create", help="create a project")
    p.add_argument("name")
    p.add_argument("--subject", default="auto",
                   help="auto|character|creature|vehicle|prop|object|environment|robot")
    p.add_argument("--style", default="realistic")
    p.add_argument("--notes")
    add_json(p)
    p.set_defaults(func=cmd_create)

    p = sub.add_parser("add-images", help="copy reference images into a project")
    p.add_argument("project")
    p.add_argument("images", nargs="+")
    p.add_argument("--views", nargs="*", default=None,
                   help="optional explicit view labels (front back left right ...), same order")
    add_json(p)
    p.set_defaults(func=cmd_add_images)

    p = sub.add_parser("projects", help="list or delete projects")
    p.add_argument("--delete")
    p.add_argument("--yes", action="store_true", help="confirm destructive operations")
    add_json(p)
    p.set_defaults(func=cmd_projects)

    p = sub.add_parser("project", help="show one project")
    p.add_argument("project")
    add_json(p)
    p.set_defaults(func=cmd_project_get)

    p = sub.add_parser("reconstruct", help="run the reconstruction pipeline")
    p.add_argument("project")
    p.add_argument("--preset", default="standard",
                   choices=["draft", "standard", "high", "ultra", "game_ready", "cinematic"])
    p.add_argument("--quality", choices=["draft", "standard", "high", "ultra", "game_ready", "cinematic"])
    p.add_argument("--target-polycount", type=int)
    p.add_argument("--texture-resolution", type=int)
    p.add_argument("--formats", help="comma separated export formats (glb,obj,fbx,stl,ply,usdz)")
    p.add_argument("--units", choices=["normalized", "meters", "centimeters", "millimeters"])
    p.add_argument("--subject-height", type=float, help="real-world height in metres (for metric export)")
    p.add_argument("--stages", help="comma separated subset of stages to run")
    p.add_argument("--param", action="append", help="extra parameter, key=value (JSON value allowed)")
    p.add_argument("--rig", dest="rig", action="store_true", default=None)
    p.add_argument("--no-rig", dest="rig", action="store_false")
    p.add_argument("--lods", dest="lods", action="store_true", default=None)
    p.add_argument("--no-lods", dest="lods", action="store_false")
    p.add_argument("--no-uvs", dest="no_uvs", action="store_true", default=None)
    p.add_argument("--preview", dest="preview", action="store_true", default=None)
    p.add_argument("--no-preview", dest="preview", action="store_false")
    p.add_argument("--watch", action="store_true", help="stream progress (default)")
    add_json(p)
    p.set_defaults(func=cmd_reconstruct)

    p = sub.add_parser("status", help="project/job status")
    p.add_argument("project")
    p.add_argument("--job")
    p.add_argument("--watch", action="store_true")
    add_json(p)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("versions", help="list project versions")
    p.add_argument("project")
    add_json(p)
    p.set_defaults(func=cmd_versions)

    p = sub.add_parser("export", help="re-export a finished version in other formats")
    p.add_argument("project")
    p.add_argument("--version")
    p.add_argument("--formats", default="")
    p.add_argument("--out")
    add_json(p)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("serve", help="run the REST API + WebSocket server (+ studio UI)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8760)
    p.add_argument("--reload", action="store_true")
    p.add_argument("--open-browser", action="store_true")
    add_json(p)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("studio", help="start the API server and open the studio UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8760)
    add_json(p)
    p.set_defaults(func=cmd_studio)

    p = sub.add_parser("models", help="manage optional local models")
    p.add_argument("action", choices=["list", "download", "verify", "delete"])
    p.add_argument("ids", nargs="*")
    p.add_argument("--force", action="store_true")
    p.add_argument("--yes", action="store_true")
    add_json(p)
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("info", help="capability report (stages, presets, formats, limits)")
    add_json(p)
    p.set_defaults(func=cmd_info)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "json", False):
        # Keep stdout parseable: diagnostics go to stderr for --json runs.
        from ..core.progress import setup_logging

        setup_logging(level="warning", stream=sys.stderr)
    try:
        return int(args.func(args))
    except Recon3DError as exc:
        payload = {"ok": False, "error": exc.code if hasattr(exc, "code") else "error",
                   "message": str(exc)}
        if getattr(args, "json", False):
            print(json.dumps(payload, indent=2, default=str))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

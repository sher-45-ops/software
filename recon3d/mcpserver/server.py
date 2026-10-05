"""Model Context Protocol server.

Agents (Claude Code, Codex, Arena, any MCP client) drive the engine through five
whitelisted tools - no shell access is ever required (spec #35):

``recon3d_doctor``            hardware/dependency/backends report
``recon3d_create_project``    create a project and optionally add images
``recon3d_reconstruct``       run the pipeline (blocking or async)
``recon3d_job_status``        progress, stage table, quality, artefact list
``recon3d_list_outputs``      machine-readable index of everything produced

Run it with::

    python -m recon3d.mcpserver            # stdio transport (default)
    recon3d-mcp                            # same thing, installed console script

Example ``mcpServers`` configuration::

    { "mcpServers": { "recon3d": { "command": "python", "args": ["-m", "recon3d.mcpserver"] } } }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import VERSION
from ..config import load_config

#: Tool names are namespaced so they never collide with other MCP servers.
TOOL_NAMES = (
    "recon3d_doctor",
    "recon3d_create_project",
    "recon3d_reconstruct",
    "recon3d_job_status",
    "recon3d_list_outputs",
)


def _serialise(payload: Any) -> str:
    return json.dumps(payload, indent=2, default=str)


def _mcp_server_class():
    """Return the MCP server class across SDK generations.

    The official Python SDK renamed ``FastMCP`` to ``MCPServer`` in 2.x (and moved
    it to ``mcp.server.mcpserver``).  Supporting both keeps the engine usable no
    matter which SDK version an agent platform ships.
    """
    try:
        from mcp.server.mcpserver import MCPServer  # type: ignore

        return MCPServer
    except Exception:
        pass
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore

        return FastMCP
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "the MCP server needs the `mcp` package: `pip install recon3d[mcp]`"
        ) from exc


def build_server(config=None) -> "Any":
    """Create the MCP server object (requires the optional ``mcp`` package)."""
    server_class = _mcp_server_class()
    cfg = config or load_config()
    cfg.ensure_dirs()
    try:
        server = server_class("recon3d", version=VERSION)
    except TypeError:  # older/newer signatures may not accept ``version``
        server = server_class("recon3d")

    from ..core.jobs import JobManager
    from ..core.project import ProjectManager

    projects = ProjectManager(cfg.projects_path)
    jobs = JobManager(log_dir=cfg.logs_path)

    @server.tool(description="Diagnose the local Recon3D install: hardware, dependencies, "
                             "optional backends, data paths. Run this first.")
    def recon3d_doctor() -> str:
        from ..core.resources import hardware_report
        from ..engine.backends.detect import detect_backends

        return _serialise({"version": VERSION, "ok": True, "data_root": str(cfg.data_root),
                           "offline": bool(cfg.offline), "hardware": hardware_report(),
                           "backends": detect_backends()})

    @server.tool(description="Create a project and (optionally) register reference images by "
                             "absolute path. Returns the project id to pass to recon3d_reconstruct.")
    def recon3d_create_project(name: str, image_paths: Optional[List[str]] = None,
                               subject_type: str = "auto", style: str = "realistic") -> str:
        project = projects.create(name, subject_type=subject_type, style=style)
        added = []
        if image_paths:
            added = project.add_images([Path(p) for p in image_paths])
        return _serialise({"project": project.id, "path": str(project.root),
                           "images_added": len(added), "total_images": len(project.images),
                           "subject_type": project.subject_type})

    @server.tool(description="Reconstruct a 3D asset from a project's reference images. "
                             "blocks until the pipeline finishes. Returns quality, statistics "
                             "and the artefact paths (glb/obj/fbx/textures/lods/rig).")
    def recon3d_reconstruct(project: str, preset: str = "standard",
                            target_polycount: Optional[int] = None,
                            texture_resolution: Optional[int] = None,
                            export_formats: Optional[List[str]] = None,
                            generate_rig: bool = False,
                            generate_lods: bool = True,
                            units: str = "normalized",
                            subject_height_m: Optional[float] = None,
                            stages: Optional[List[str]] = None,
                            timeout_s: int = 3600) -> str:
        from ..core.pipeline import run_pipeline

        proj = projects.resolve(project)
        params: Dict[str, Any] = {"preset": preset, "quality": preset,
                                  "generate_rig": generate_rig, "generate_lods": generate_lods,
                                  "units": units}
        if target_polycount:
            params["target_polycount"] = int(target_polycount)
        if texture_resolution:
            params["texture_resolution"] = int(texture_resolution)
        if export_formats:
            params["export_formats"] = list(export_formats)
        if subject_height_m:
            params["subject_height_m"] = float(subject_height_m)
        job = jobs.create(proj.id, kind="reconstruct", params=params, project_obj=proj)
        jobs.submit(job, lambda j: run_pipeline(proj, j, params=params, stages=stages))
        finished = jobs.wait(job.id, timeout=float(timeout_s))
        return _serialise(_job_summary(finished, include_artifacts=True))

    @server.tool(description="Poll a reconstruction job: state, progress, per-stage time and "
                             "status, measured quality and warnings.")
    def recon3d_job_status(job_id: str, include_stages: bool = True) -> str:
        job = jobs.get(job_id)
        return _serialise(_job_summary(job, include_artifacts=False,
                                       include_stages=include_stages))

    @server.tool(description="List every produced file for a finished job (or project version) "
                             "with sizes and absolute paths.")
    def recon3d_list_outputs(job_id: str) -> str:
        job = jobs.get(job_id)
        result = job.result or {}
        outputs = result.get("outputs") or {}
        version_dir = outputs.get("version_dir")
        files: List[Dict[str, Any]] = []
        if version_dir and Path(version_dir).exists():
            root = Path(version_dir)
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    files.append({"path": str(path),
                                  "relative": str(path.relative_to(root)),
                                  "bytes": path.stat().st_size})
        return _serialise({"job": job_id, "version_dir": version_dir,
                           "outputs": outputs, "files": files,
                           "quality": result.get("quality")})

    return server


def _job_summary(job, *, include_artifacts: bool, include_stages: bool = True) -> Dict[str, Any]:
    result = job.result or {}
    payload: Dict[str, Any] = {
        "job": job.id, "project": job.project, "state": job.state.value,
        "progress": round(float(job.progress), 2), "message": job.message,
        "duration_s": round(float(job.duration_s or 0.0), 2),
        "error": job.error,
    }
    if include_stages:
        payload["stages"] = getattr(job, "stages", {})
        payload["events"] = [e.to_dict() for e in job.events[-40:]]
    if result:
        payload["version"] = result.get("version")
        payload["quality"] = result.get("quality")
        payload["statistics"] = result.get("statistics")
        payload["warnings"] = result.get("warnings", [])
        if include_artifacts:
            payload["outputs"] = result.get("outputs")
    return payload


def main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover - transport loop
    """Entry point used by ``python -m recon3d.mcpserver``."""
    import argparse

    parser = argparse.ArgumentParser(prog="recon3d-mcp",
                                     description="Recon3D MCP server (stdio transport)")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse", "streamable-http"])
    parser.add_argument("--json", action="store_true", help="print the tool list and exit")
    args = parser.parse_args(argv)

    if args.json:
        print(_serialise({"server": "recon3d", "version": VERSION, "tools": list(TOOL_NAMES)}))
        return 0

    server = build_server()
    run = getattr(server, "run", None)
    if run is None:  # pragma: no cover - SDK mismatch
        raise SystemExit("this mcp SDK version is not supported; use mcp>=1.2")
    try:
        run(transport=args.transport)
    except TypeError:  # pragma: no cover - transport keyword renamed
        run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

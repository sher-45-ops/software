#!/usr/bin/env python3
"""Drive Recon3D from Python: create a project, reconstruct, read the report.

    python examples/python_api.py ./demo/refs

Every call here is the same code path the CLI, REST API and MCP server use.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from recon3d.config import load_config
from recon3d.core.jobs import JobManager
from recon3d.core.pipeline import run_pipeline
from recon3d.core.project import ProjectManager


def main() -> int:
    refs = Path(sys.argv[1] if len(sys.argv) > 1 else "./demo/refs")
    images = sorted(p for p in refs.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                    and p.parent.name != "masks")
    if not images:
        print(f"no images in {refs}; generate a set with scripts/make_demo_dataset.py")
        return 2

    cfg = load_config()                       # honours RECON3D_HOME / config.json
    cfg.ensure_dirs()
    projects = ProjectManager(cfg.projects_path)
    project = (projects.open("python-example") if projects.exists("python-example")
               else projects.create("python-example", subject_type="auto"))
    if not project.images:
        project.add_images(images)
    print(f"project {project.id}: {len(project.images)} image(s) -> {project.root}")

    jobs = JobManager(log_dir=cfg.logs_path)
    job = jobs.create(project.id, kind="reconstruct", params={"preset": "draft"},
                      project_obj=project)

    # Progress is also available as structured events: jobs.subscribe(callback)
    def on_event(event) -> None:
        print(f"  [{event.stage or '-':<20}] {event.progress:5.1f}% {event.message}")

    unsubscribe = jobs.subscribe(lambda job_id, event: on_event(event) if job_id == job.id else None)
    try:
        jobs.run_sync(job, lambda j: run_pipeline(project, j,
                                                  params={"preset": "draft",
                                                          "texture_resolution": 1024,
                                                          "export_formats": ["glb", "obj"]}))
    finally:
        unsubscribe()

    result = job.result or {}
    print(json.dumps({"state": job.state.value,
                      "quality": result.get("quality", {}).get("overall"),
                      "statistics": result.get("statistics"),
                      "mesh": (result.get("outputs") or {}).get("mesh")}, indent=2))
    return 0 if job.state.value == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

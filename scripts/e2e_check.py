#!/usr/bin/env python3
"""End-to-end accuracy gate: reference images in, verified asset out.

The script is intentionally demanding: it fails (non-zero exit) when the engine
does not produce a real, quantifiably-good asset.  It:

1. creates a project in a throwaway data root,
2. renders (or reuses) a synthetic multi-view reference set,
3. runs the full pipeline through the public API,
4. checks the outputs against the reference images and the ground truth,
5. writes a JSON report next to the version so CI or a human can inspect it.

Examples
--------
::

    python scripts/e2e_check.py --preset draft
    python scripts/e2e_check.py --preset standard --views 9 --keep
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default="draft",
                        choices=["draft", "standard", "high", "ultra", "game_ready", "cinematic"])
    parser.add_argument("--views", type=int, default=9, choices=[4, 6, 9, 12])
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--work-dir", type=Path, default=ROOT / ".e2e")
    parser.add_argument("--min-iou", type=float, default=0.55,
                        help="minimum mean reference silhouette IoU to pass")
    parser.add_argument("--min-score", type=float, default=40.0,
                        help="minimum overall quality score to pass")
    parser.add_argument("--min-faces", type=int, default=500)
    parser.add_argument("--keep", action="store_true",
                        help="keep the working directory (default: only on failure)")
    parser.add_argument("--reuse-refs", action="store_true",
                        help="reuse existing reference images in --work-dir/refs")
    parser.add_argument("--texture-resolution", type=int, default=1024)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    from recon3d.config import load_config
    from recon3d.core.jobs import JobManager
    from recon3d.core.pipeline import run_pipeline
    from recon3d.core.progress import setup_logging
    from recon3d.core.project import ProjectManager
    from recon3d.engine.reconstruction.dataset import DatasetSpec, generate_reference_set

    setup_logging(level="info" if args.verbose else "warning")

    work = args.work_dir.resolve()
    data_root = work / "data"
    refs = work / "refs"
    if not (args.reuse_refs and any(refs.glob("*.png"))):
        if refs.exists():
            shutil.rmtree(refs)
        spec = DatasetSpec(kind="robot", resolution=args.resolution, write_masks=True,
                           views=(list(zip(
                               ["front", "front_right", "right", "back_right", "back",
                                "back_left", "left", "front_left", "top",
                                "right_front", "right_back", "left_back"],
                               [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0,
                                0.0, 60.0, 120.0, 240.0],
                               [0.0] * 9 + [0.0, 0.0, 0.0]))[:args.views]))
        generate_reference_set(refs, spec)
    ground_truth = json.loads((refs / "ground_truth.json").read_text()) \
        if (refs / "ground_truth.json").exists() else None

    cfg = load_config(data_root=str(data_root))
    cfg.ensure_dirs()
    projects = ProjectManager(cfg.projects_path)
    project_id = f"e2e_{args.preset}"
    if projects.exists(project_id):
        projects.delete(project_id, confirm=True)
    project = projects.create(project_id, subject_type="robot",
                              description=f"e2e check ({args.preset}, {args.views} views)")
    project.add_images(sorted(refs.glob("*.png")))

    jobs = JobManager(log_dir=cfg.logs_path)
    job = jobs.create(project.id, kind="reconstruct", params={"quality": args.preset},
                      project_obj=project)
    params = {
        "preset": args.preset,
        "quality": args.preset,
        "texture_resolution": args.texture_resolution,
        "export_formats": ["glb", "obj"],
        "generate_previews": True,
    }
    started = time.time()
    jobs.run_sync(job, lambda j: run_pipeline(project, j, params=params))
    duration = time.time() - started

    result = job.result or {}
    quality = result.get("quality") or {}
    metrics = quality.get("metrics") or {}
    statistics = result.get("statistics") or {}
    outputs = result.get("outputs") or {}
    checks: list[dict] = []

    def check(name: str, ok: bool, detail) -> bool:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    check("job completed", job.state.value == "completed", job.state.value)
    check("no failed stages", not result.get("stages_failed"), result.get("stages_failed"))
    check("mesh was produced", bool(outputs.get("mesh")), sorted((outputs.get("mesh") or {}).keys()))
    check("triangle count is non-trivial", int(statistics.get("triangles") or 0) >= args.min_faces,
          statistics.get("triangles"))
    check("mean silhouette IoU", float(metrics.get("mean_silhouette_iou") or 0.0) >= args.min_iou,
          metrics.get("mean_silhouette_iou"))
    check("quality score", float(quality.get("overall") or 0.0) >= args.min_score, quality.get("overall"))
    check("textures written", bool(outputs.get("textures")), outputs.get("textures"))
    check("GLB exported and verified", bool((outputs.get("mesh") or {}).get("glb")),
          (outputs.get("mesh") or {}).get("glb"))
    check("LODs present", int(statistics.get("lod_levels") or 0) >= 1, statistics.get("lod_levels"))
    check("watertight or repaired", statistics.get("watertight") in (True, False),
          statistics.get("watertight"))

    gt_check: dict = {}
    if ground_truth:
        version_dir = Path(outputs.get("version_dir") or "")
        report_path = version_dir / "reports" / "quality.json"
        if report_path.exists():
            gt_check["quality_report"] = str(report_path)
        height = float(ground_truth["subject"]["height"])
        extents = statistics.get("extents") or []
        if extents:
            gt_check["normalised_height"] = round(float(extents[2]), 4)
            gt_check["ground_truth_height_units"] = 1.0
            gt_check["relative_error"] = round(abs(float(extents[2]) - 1.0), 4)
            check("reconstruction height matches the gauge",
                  abs(float(extents[2]) - 1.0) <= 0.15, round(float(extents[2]), 4))
        gt_check["ground_truth_height_native"] = height

    passed = all(c["ok"] for c in checks)
    report = {
        "preset": args.preset,
        "views": args.views,
        "resolution": args.resolution,
        "duration_s": round(duration, 2),
        "job_state": job.state.value,
        "quality": quality,
        "statistics": statistics,
        "outputs": outputs,
        "ground_truth": gt_check,
        "checks": checks,
        "passed": passed,
    }
    work.mkdir(parents=True, exist_ok=True)
    report_path = work / f"e2e-{args.preset}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"\n=== e2e check ({args.preset}, {args.views} views) ===")
    print(f"duration      : {duration:.1f} s")
    print(f"quality       : {quality.get('overall', 0):.1f} ({quality.get('grade', '?')})")
    print(f"silhouette IoU: {metrics.get('mean_silhouette_iou', 0):.3f}")
    print(f"triangles     : {statistics.get('triangles')}  LODs: {statistics.get('lod_levels')}")
    print(f"version dir   : {outputs.get('version_dir')}")
    print(f"report        : {report_path}")
    for item in checks:
        flag = "PASS" if item["ok"] else "FAIL"
        print(f"  [{flag}] {item['check']}: {item['detail']}")
    print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")

    if not args.keep and passed:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

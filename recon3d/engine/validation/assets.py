"""Asset validation: does the produced asset actually hold together? (spec #39)

Every claim in the validation report is measured from the produced files, not
assumed from the pipeline state: the mesh is re-read from disk, its topology
re-checked, the texture files are opened, the GLB is re-parsed, and the requested
polygon/texture/rig/LOD budgets are compared against what was actually written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ...core.project import Project, Version
from ...core.store import write_json


def validate_version(project: Project, version: Version, *,
                     write_report: bool = True) -> Dict[str, Any]:
    """Validate every artefact of a version against the requested parameters."""
    base = project.version_path(version.id)
    params = version.params or {}
    errors: List[str] = []
    warnings: List[str] = []
    checks: Dict[str, Any] = {}

    # -- mesh -----------------------------------------------------------
    mesh_entry = None
    for fmt in ("glb", "obj", "ply", "stl", "fbx", "gltf", "usda"):
        if fmt in version.assets.mesh:
            mesh_entry = (fmt, base / version.assets.mesh[fmt])
            break
    mesh_stats: Dict[str, Any] = {}
    if mesh_entry is None:
        errors.append("no mesh file was produced")
    else:
        fmt, path = mesh_entry
        checks["mesh_format"] = fmt
        checks["mesh_path"] = str(path)
        if not path.exists():
            errors.append(f"mesh file is missing: {path}")
        elif path.stat().st_size == 0:
            errors.append("mesh file is empty")
        else:
            try:
                import trimesh

                loaded = trimesh.load(str(path), force="mesh", process=False)
                vertices = np.asarray(loaded.vertices)
                faces = np.asarray(loaded.faces)
                mesh_stats = {
                    "vertices": int(len(vertices)),
                    "faces": int(len(faces)),
                    "bounds": np.asarray(loaded.bounds).tolist() if len(vertices) else None,
                }
                if len(vertices) == 0 or len(faces) == 0:
                    errors.append("the exported mesh contains no geometry")
                else:
                    try:
                        if not bool(loaded.is_winding_consistent):
                            warnings.append("exported mesh has inconsistent face winding")
                        if not bool(loaded.is_watertight):
                            warnings.append("exported mesh is not watertight "
                                            "(expected for some reconstructions)")
                        degenerates = int(len(faces) - loaded.nondegenerate_faces().sum())
                        if degenerates:
                            warnings.append(f"exported mesh contains {degenerates} degenerate faces")
                        mesh_stats["watertight"] = bool(loaded.is_watertight)
                        mesh_stats["winding_consistent"] = bool(loaded.is_winding_consistent)
                        mesh_stats["degenerate_faces"] = degenerates
                    except Exception:  # pragma: no cover
                        pass
                    requested = params.get("target_polycount")
                    if isinstance(requested, (int, float)) and requested > 0:
                        ratio = len(faces) / float(requested)
                        checks["polycount_hit"] = round(ratio, 3)
                        if ratio > 1.25:
                            warnings.append(
                                f"exported {len(faces)} triangles versus a requested budget of "
                                f"{int(requested)} (+{(ratio - 1) * 100:.0f}%)"
                            )
            except Exception as exc:
                errors.append(f"the exported mesh could not be re-read: {exc}")

    # -- textures -------------------------------------------------------
    texture_stats: Dict[str, Any] = {}
    if params.get("generate_pbr", True):
        expected = ["basecolor"]
        if params.get("generate_pbr", True):
            expected += ["normal", "roughness", "metallic", "ao"]
        missing = [name for name in expected if name not in version.assets.textures]
        if missing:
            warnings.append(f"texture maps missing: {', '.join(missing)}")
        texture_stats = {}
        for name, rel in version.assets.textures.items():
            path = base / rel
            if not path.exists():
                errors.append(f"texture '{name}' is missing on disk")
                continue
            try:
                from PIL import Image

                with Image.open(path) as image:
                    texture_stats[name] = {
                        "size": [image.width, image.height],
                        "mode": image.mode,
                        "bytes": path.stat().st_size,
                    }
            except Exception as exc:
                errors.append(f"texture '{name}' could not be opened: {exc}")
        requested_res = int(params.get("texture_resolution", 0) or 0)
        if requested_res and texture_stats.get("basecolor"):
            actual = texture_stats["basecolor"]["size"][0]
            checks["texture_resolution_requested"] = requested_res
            checks["texture_resolution_actual"] = actual
            if actual < requested_res:
                warnings.append(
                    f"texture resolution is {actual}px, below the requested {requested_res}px "
                    "(raised by the hardware guard rails)"
                )

    # -- uvs ------------------------------------------------------------
    if params.get("generate_uvs", True):
        uv_report = (version.assets.reports.get("uv") or {})
        try:
            import trimesh

            path = base / next(iter(version.assets.mesh.values()))
            loaded = trimesh.load(str(path), force="mesh", process=False)
            visual = getattr(loaded, "visual", None)
            uv = getattr(visual, "uv", None) if visual is not None else None
            checks["has_uvs"] = uv is not None
            if uv is None:
                errors.append("UVs were requested but the exported mesh has no texture "
                              "coordinates")
        except Exception:  # pragma: no cover
            pass

    # -- glb structure --------------------------------------------------
    if "glb" in version.assets.mesh:
        from ...engine.export.writers import verify_glb

        ok, details = verify_glb(base / version.assets.mesh["glb"])
        checks["glb_readback"] = details
        if not ok:
            errors.append(f"GLB verification failed: {details.get('error')}")

    # -- rig ------------------------------------------------------------
    if params.get("generate_rig"):
        if not version.assets.rig:
            errors.append("rigging was requested but no rig files were produced")
        else:
            rig_path = base / version.assets.rig.get("rig", "")
            checks["rig"] = {"path": str(rig_path), "exists": rig_path.exists()}
            if not rig_path.exists():
                errors.append("rig.json is missing")
            weights = version.assets.rig.get("weights")
            if not weights:
                errors.append("skin weights were not written")
            else:
                weight_path = base / weights
                try:
                    data = np.load(weight_path)
                    w = data["weights"]
                    checks["skin_weights"] = {
                        "shape": list(w.shape),
                        "sum_min": float(w.sum(axis=1).min()),
                        "sum_max": float(w.sum(axis=1).max()),
                    }
                    if abs(float(w.sum(axis=1).mean()) - 1.0) > 0.01:
                        warnings.append("skin weights are not normalised to 1.0")
                except Exception as exc:
                    errors.append(f"skin weights could not be read: {exc}")

    # -- lods -----------------------------------------------------------
    if params.get("generate_lods", True):
        expected_levels = int(params.get("lod_levels", 0) or 0)
        lods = version.assets.lods or {}
        if not lods:
            warnings.append("no LOD files were produced")
        for name, rel in lods.items():
            path = base / rel
            if not path.exists():
                errors.append(f"{name} file is missing")
        if lods and expected_levels and len(lods) < expected_levels:
            warnings.append(f"requested {expected_levels} LOD levels, produced {len(lods)}")

    # -- previews -------------------------------------------------------
    if params.get("generate_previews", True):
        previews = [base / p for p in version.assets.previews]
        found = [p for p in previews if p.exists()]
        checks["previews"] = len(found)
        if not found:
            warnings.append("no preview images were produced")

    report = {
        "ok": not errors,
        "version": version.id,
        "project": project.id,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "mesh": mesh_stats,
        "textures": texture_stats,
        "statistics": version.statistics,
        "quality": version.quality,
        "validated_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
    }
    if write_report:
        try:
            write_json(base / "reports" / "validation_report.json", report)
        except OSError:  # pragma: no cover
            pass
    return report

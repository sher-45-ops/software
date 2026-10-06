"""Machine-readable capability manifest for agents.

An agent that has never seen this project should be able to answer three
questions from one call: *what can this engine do, what do I have to pass, and
what will I get back*.  :func:`capability_report` returns that description, and it
is what ``recon3d info`` prints and what ``agent-manifest.json`` in the
repository mirrors.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .. import VERSION

STAGE_DESCRIPTIONS: Dict[str, str] = {
    "ingestion": "load reference images, EXIF/orientation normalisation, hashing",
    "validation": "per-image quality checks (blur, exposure, duplicate, resolution)",
    "segmentation": "subject masks (colour/edge based, optional local salient model)",
    "analysis": "subject-type classification, symmetry detection, material hints",
    "features": "feature extraction + cross-view matching evidence",
    "camera_estimation": "ring/lens/scale solve for every reference image",
    "mesh_reconstruction": "visual hull carving, photometric shell carve, depth fusion",
    "silhouette_refine": "re-render vs references, adjust cameras if it objectively improves",
    "mesh_cleanup": "floaters, holes, non-manifold, normals, weld, degenerate faces",
    "optimization": "topology + polycount control (game-ready budgets)",
    "uv": "UV unwrap (xatlas, built-in fallback) + quality report",
    "texture": "multi-view texture projection, PBR map derivation, ORM packing",
    "materials": "evidence-based PBR material estimation + agent overrides",
    "rigging": "humanoid/creature skeleton + skin weights + deformation stress test",
    "lod": "LOD0-3 chain with error metrics",
    "preview": "stills, turntable GIF, comparison sheets, wireframe/normal/UV views",
    "comparison": "render vs reference comparison, quality score, similarity refinement",
    "export": "GLB/GLTF/OBJ/FBX/STL/PLY/USD(Z) writers + independent verification",
}

PRESET_DESCRIPTIONS: Dict[str, str] = {
    "draft": "fastest, lowest detail (preview / iteration)",
    "standard": "balanced default for everyday assets",
    "high": "high quality, larger textures, camera refinement",
    "ultra": "maximum quality, every stage at full effort",
    "game_ready": "strict polycount + full LOD chain for real-time engines",
    "cinematic": "maximum detail for offline rendering, no polycount limit",
}

OUTPUT_TREE: Dict[str, str] = {
    "final/": "the model itself (glb/obj/mtl/fbx/stl/ply/usd/usdz as requested)",
    "textures/": "basecolor, normal, roughness, metallic, ao, packed orm",
    "lod/": "lod0..lod3 meshes",
    "materials/": "materials.json (PBR parameters, evidence, overrides)",
    "rig/": "rig.json, skin_weights.npz, skin.json",
    "previews/": "stills, turntable.gif, wireframe/normal/uv/coverage renders",
    "renders/": "full-resolution reference-angle renders used for comparison",
    "reports/": "quality.json, statistics.json, stages.json, reference_comparison.json, "
                "image_quality.json, validation_report.json",
    "intermediate/": "point cloud, carved volume info, per-stage reports",
    "version.json": "everything this version was built from and with",
}


#: One-line description per pipeline parameter.  ``test_manifest.py`` fails when a
#: parameter exists in ``DEFAULT_PARAMS`` without an entry here, so agents never
#: discover features they cannot name (or names they cannot use).
PARAMETER_DESCRIPTIONS: Dict[str, str] = {
    "quality": "draft|standard|high|ultra|game_ready|cinematic",
    "preset": "explicit preset name; wins over 'quality'",
    "style": "keep the supplied style; 'stylized' avoids photorealistic post-processing",
    "geometry": "low|medium|high|auto - carve fidelity before the preset budget",
    "texture_resolution": "512..8192 (clamped by hardware, reported)",
    "target_polycount": "int or 'auto' - triangle budget for the exported mesh",
    "preserve_sharp_edges": "bool - keep mechanical/hard edges instead of smoothing them away",
    "symmetry": "auto|none|on|x|y|z - enforce a mirror plane when the subject has one",
    "generate_uvs": "bool",
    "generate_pbr": "bool - derive normal/roughness/metallic/AO maps",
    "generate_rig": "bool - skeleton + skin weights (humanoid/creature)",
    "generate_lods": "bool",
    "lod_levels": "int (0 -> from the preset)",
    "generate_previews": "bool - turntable/reference previews into previews/",
    "generate_pointcloud": "bool - write the multi-view point cloud as an intermediate",
    "photo_consistency": "bool - reject silhouette voxels whose colour disagrees across views",
    "material_request": "free text, e.g. 'brushed steel with red paint'",
    "material": "material name from the built-in library (see 'materials')",
    "subject_type": "auto|character_humanoid|character_creature|robot_mech|animal|unknown",
    "refine_passes": "int; reference-comparison refinement passes (0 disables, -1 -> preset)",
    "compare_renders": "int; reference views rendered into renders/ for inspection (0 -> preset)",
    "export_formats": "list of glb|gltf|obj|fbx|stl|ply|usd|usda|usdz",
    "units": "normalized|meters|centimeters|millimeters",
    "subject_height_m": "real height in metres (for metric export)",
    "seed": "int - seeds the pipeline's sampling for reproducible runs",
    "backend": "auto|numpy|<optional backend> - decimation/UV backend preference",
    "stage_retries": "int, extra attempts for a transient stage failure (default 1)",
    "force_stages": "list of stage names to rebuild even if cached",
    "skip_stages": "list of stage names to skip",
    "stages": "optional subset of stage names to run",
    "label": "text - label stored on the produced version",
    "notes": "text - operator notes stored on the produced version",
    "material_overrides": "{name: {...}} forced materials/colours",
    "perform_refinement": "bool (reference comparison + similarity refinement)",
    "defaults": "the full default parameter set",
}


def capability_report() -> Dict[str, Any]:
    """Full, serialisable description of the engine's capabilities."""
    from ..core.pipeline import DEFAULT_PARAMS, STAGE_PROGRESS_NAMES
    from ..engine.backends.detect import detect_backends
    from ..engine.export.writers import SUPPORTED_FORMATS
    from ..engine.materials.library import MATERIAL_LIBRARY
    from ..engine.optimization.lod import DEFAULT_LOD_RATIOS

    return {
        "name": "recon3d",
        "version": VERSION,
        "summary": ("Local-first multi-view image to 3D reconstruction engine. Turns "
                    "front/back/left/right/three-quarter/top reference images of a subject "
                    "into a real textured mesh (with UVs, PBR maps, LODs and an optional rig). "
                    "Runs entirely on the local CPU; no external generative AI service is used."),
        "stages": [{"name": name, "description": STAGE_DESCRIPTIONS.get(name, "")}
                   for name in STAGE_PROGRESS_NAMES],
        "presets": PRESET_DESCRIPTIONS,
        "parameters": dict(PARAMETER_DESCRIPTIONS, defaults=DEFAULT_PARAMS),
        "outputs": OUTPUT_TREE,
        "formats": sorted(SUPPORTED_FORMATS),
        "materials": sorted(MATERIAL_LIBRARY.keys()),
        "lod_ratios": DEFAULT_LOD_RATIOS,
        "interfaces": {
            "cli": ("recon3d <command> [--json]: doctor setup create add-images projects project "
                    "reconstruct status jobs cancel retry versions export serve studio models "
                    "info  (see docs/CLI.md)"),
            "rest": ("POST /v1/projects/{id}/reconstruct, GET /v1/jobs/{id}, "
                     "POST /v1/jobs/{id}/cancel, GET /v1/jobs/{id}/artifacts/{path}, "
                     "WS /v1/ws/jobs/{id}"),
            "mcp": "python -m recon3d.mcpserver (tools: recon3d_doctor, recon3d_create_project, "
                   "recon3d_reconstruct, recon3d_job_status, recon3d_list_outputs)",
        },
        "no_external_ai": True,
        "offline_capable": True,
        "gpu_required": False,
        "backends": detect_backends(),
        "recovery": ("Runs are checkpointed per stage and resumable: re-run the same command, "
                     "or use `recon3d jobs` to find a failed/cancelled run and `recon3d retry "
                     "<job-id>` to continue it. `recon3d cancel <job-id>` stops a job running "
                     "in any process; finished stages are kept."),
        "honesty_policy": ("Every quality number in reports/ is measured from the produced files. "
                           "Unobserved texture regions are reported as missing rather than "
                           "invented, and a failed stage degrades instead of silently emitting a "
                           "worse model."),
    }

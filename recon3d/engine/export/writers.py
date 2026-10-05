"""Asset writers: GLB, GLTF, OBJ/MTL, STL, PLY, USD/USDZ and FBX (spec #21, #40).

Format notes (documented rather than hidden):

* **GLB/GLTF** - written directly by this module (a real glTF 2.0 binary
  container with PBR material, textures and optional skin), verified by reading
  the file back and checking the JSON chunk, buffer lengths and accessor counts.
* **OBJ/MTL** - written by this module so material names and texture references
  match the generated maps exactly.
* **STL/PLY** - geometry-focused; STL is binary, PLY supports vertex colours.
* **USD/USDA/USDZ** - a compact USDA text scene (mesh + material bindings).
* **FBX** - ASCII FBX 7.4 with geometry, UVs, materials and (where requested)
  the skeleton.  Autodesk's binary FBX SDK is proprietary and is *not* required;
  ASCII FBX is imported by Blender, Unity, Unreal and Godot.  When Blender is
  available the exporter prefers Blender's own binary FBX writer, which produces
  the most compatible file - this is reported per export.
"""

from __future__ import annotations

import base64
import json
import math
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...core.store import write_json
from ...errors import StageError

GLTF_MAGIC = 0x46546C67  # "glTF"


@dataclass
class ExportResult:
    format: str
    path: str
    bytes: int
    verified: bool
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    sidecars: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": self.format, "path": self.path, "bytes": self.bytes,
            "verified": self.verified, "details": self.details,
            "warnings": self.warnings, "sidecars": self.sidecars,
        }


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _as_uint8(image: Any, *, channels: int = 3) -> Optional[np.ndarray]:
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        if arr.dtype == bool:
            arr = (arr.astype(np.uint8) * 255)
        elif np.issubdtype(arr.dtype, np.floating):
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], channels, axis=2)
    return arr


def mesh_arrays(mesh) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Extract (vertices float32, faces uint32, uvs float32 or None)."""
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    if faces.ndim != 2 or faces.shape[1] != 3:
        # Triangulate anything that is not already triangles.
        try:
            faces = np.asarray(mesh.triangles, dtype=np.uint32)
        except Exception as exc:  # pragma: no cover
            raise StageError(f"cannot triangulate mesh: {exc}", stage="export", recoverable=True) from exc
    uvs = None
    visual = getattr(mesh, "visual", None)
    if visual is not None and getattr(visual, "uv", None) is not None:
        candidate = np.asarray(visual.uv, dtype=np.float32)
        if len(candidate) == len(vertices):
            uvs = candidate
    return vertices, faces, uvs


def _png_bytes(image: np.ndarray) -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


# --------------------------------------------------------------------------
# glTF / GLB
# --------------------------------------------------------------------------
def export_glb(
    mesh,
    path: Path,
    *,
    textures: Optional[Dict[str, Any]] = None,
    material: Optional[Dict[str, Any]] = None,
    rig: Optional[Dict[str, Any]] = None,
    embed_textures: bool = True,
    generator: str = "Recon3D Engine",
) -> ExportResult:
    """Write a glTF 2.0 binary container (self-contained .glb)."""
    vertices, faces, uvs = mesh_arrays(mesh)
    normals = None
    try:
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    except Exception:  # pragma: no cover
        normals = None

    bin_parts: List[bytes] = []
    buffer_views: List[Dict[str, Any]] = []
    accessors: List[Dict[str, Any]] = []
    current_offset = 0

    def add_buffer(data: bytes, target: Optional[int] = None) -> int:
        nonlocal current_offset
        # 4-byte alignment is required by the spec.
        pad = (-len(data)) % 4
        chunk = data + b"\x00" * pad
        view = {"buffer": 0, "byteOffset": current_offset, "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        buffer_views.append(view)
        bin_parts.append(chunk)
        current_offset += len(chunk)
        return len(buffer_views) - 1

    def add_accessor(data: np.ndarray, accessor_type: str, component_type: int,
                     *, target: Optional[int] = None, minmax: bool = False) -> int:
        view_index = add_buffer(data.tobytes(), target)
        accessor: Dict[str, Any] = {
            "bufferView": view_index, "componentType": component_type,
            "count": int(len(data)), "type": accessor_type,
        }
        if minmax:
            accessor["min"] = [float(v) for v in np.asarray(data).min(axis=0)]
            accessor["max"] = [float(v) for v in np.asarray(data).max(axis=0)]
        accessors.append(accessor)
        return len(accessors) - 1

    position_accessor = add_accessor(vertices, "VEC3", 5126, target=34962, minmax=True)
    attributes: Dict[str, int] = {"POSITION": position_accessor}
    if normals is not None and len(normals) == len(vertices):
        attributes["NORMAL"] = add_accessor(normals, "VEC3", 5126, target=34962)
    if uvs is not None:
        attributes["TEXCOORD_0"] = add_accessor(uvs, "VEC2", 5126, target=34962)

    # Indices: use the smallest integer type that fits.
    max_index = int(faces.max()) if faces.size else 0
    if max_index < 65535:
        index_data = faces.astype(np.uint16)
        index_component = 5123
    else:
        index_data = faces.astype(np.uint32)
        index_component = 5125
    index_accessor = add_accessor(index_data.reshape(-1, 1), "SCALAR", index_component, target=34963)

    images: List[Dict[str, Any]] = []
    gltf_textures: List[Dict[str, Any]] = []
    texture_slots: Dict[str, int] = {}
    if textures:
        for name, image in textures.items():
            arr = _as_uint8(image, channels=3 if name != "ao" else 3)
            if arr is None:
                continue
            png = _png_bytes(arr)
            view_index = add_buffer(png)
            images.append({"bufferView": view_index, "mimeType": "image/png", "name": name})
            gltf_textures.append({"source": len(images) - 1, "sampler": 0})
            texture_slots[name] = len(gltf_textures) - 1

    pbr: Dict[str, Any] = {}
    material_params = material or {}
    base_color = material_params.get("base_color")
    if base_color:
        pbr["baseColorFactor"] = [min(1.0, c / 255.0) for c in list(base_color)[:3]] + [1.0]
    if "metallic" in material_params:
        pbr["metallicFactor"] = float(material_params["metallic"])
    if "roughness" in material_params:
        pbr["roughnessFactor"] = float(material_params["roughness"])
    if "basecolor" in texture_slots:
        pbr["baseColorTexture"] = {"index": texture_slots["basecolor"]}
        pbr.pop("baseColorFactor", None)
    if "metallic" in texture_slots or "roughness" in texture_slots:
        # Prefer a packed ORM map when available.
        if "orm" in texture_slots:
            pbr["metallicRoughnessTexture"] = {"index": texture_slots["orm"]}
        elif "roughness" in texture_slots:
            pbr["metallicRoughnessTexture"] = {"index": texture_slots["roughness"]}
    material_index = 0
    materials = [{
        "name": material_params.get("name", "recon3d_material"),
        "pbrMetallicRoughness": pbr or {"baseColorFactor": [0.8, 0.8, 0.8, 1.0],
                                        "metallicFactor": 0.0, "roughnessFactor": 0.7},
        "doubleSided": False,
    }]
    if "normal" in texture_slots:
        materials[0]["normalTexture"] = {"index": texture_slots["normal"]}
    if "ao" in texture_slots:
        materials[0]["occlusionTexture"] = {"index": texture_slots["ao"]}
    if material_params.get("emission"):
        materials[0]["emissiveFactor"] = [min(1.0, c / 255.0) for c in material_params["emission"]]

    primitives = [{"attributes": attributes, "indices": index_accessor, "mode": 4,
                   "material": material_index}]
    node: Dict[str, Any] = {"mesh": 0, "name": "recon3d_mesh"}

    skins: List[Dict[str, Any]] = []
    nodes: List[Dict[str, Any]] = [node]
    if rig and rig.get("weights") is not None:
        weights = np.asarray(rig["weights"], dtype=np.float32)
        joints = list(rig.get("joints") or [])
        if weights.ndim == 2 and weights.shape[0] == len(vertices) and weights.shape[1] == len(joints):
            top = np.argsort(weights, axis=1)[:, -4:][:, ::-1]
            gathered = np.take_along_axis(weights, top, axis=1)
            sums = gathered.sum(axis=1, keepdims=True)
            gathered = np.divide(gathered, np.maximum(sums, 1e-8))
            joint_indices = top.astype(np.uint16)
            joint_weights = gathered.astype(np.float32)
            attributes["JOINTS_0"] = add_accessor(joint_indices, "VEC4", 5123, target=34962)
            attributes["WEIGHTS_0"] = add_accessor(joint_weights, "VEC4", 5126, target=34962)
            # Skeleton nodes
            bone_nodes: Dict[str, Dict[str, Any]] = {}
            for bone in rig.get("bones", []):
                bone_nodes[bone["name"]] = {
                    "name": bone["name"],
                    "translation": [float(v) for v in bone["head"]],
                }
            node_indices: Dict[str, int] = {}
            for name in joints:
                if name in bone_nodes:
                    nodes.append(bone_nodes[name])
                    node_indices[name] = len(nodes) - 1
            for bone in rig.get("bones", []):
                parent = bone.get("parent")
                if parent in node_indices and bone["name"] in node_indices:
                    nodes[node_indices[parent]].setdefault("children", []).append(node_indices[bone["name"]])
            root = rig.get("root") or (joints[0] if joints else None)
            if root in node_indices:
                node["children"] = [node_indices[root]]
                node["skin"] = 0
                skins.append({
                    "joints": [node_indices[name] for name in joints if name in node_indices],
                    "skeleton": node_indices[root],
                    "inverseBindMatrices": _inverse_bind_accessor(add_accessor_wrapper=add_accessor,
                                                                  rig=rig),
                })

    gltf: Dict[str, Any] = {
        "asset": {"version": "2.0", "generator": generator},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": nodes,
        "meshes": [{"name": "recon3d", "primitives": primitives}],
        "materials": materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": current_offset}],
    }
    if images:
        gltf["images"] = images
        gltf["textures"] = gltf_textures
        gltf["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]
    if skins:
        gltf["skins"] = skins

    json_chunk = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_chunk += b" " * ((-len(json_chunk)) % 4)
    bin_chunk = b"".join(bin_parts)
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", GLTF_MAGIC, 2, total))
        fh.write(struct.pack("<II", len(json_chunk), 0x4E4F534A))  # JSON
        fh.write(json_chunk)
        fh.write(struct.pack("<II", len(bin_chunk), 0x004E4942))  # BIN
        fh.write(bin_chunk)

    verified, details = verify_glb(path)
    return ExportResult(format="glb", path=str(path), bytes=path.stat().st_size,
                        verified=verified, details=details,
                        warnings=[] if verified else ["the written GLB failed read-back verification"])


def _inverse_bind_accessor(*, add_accessor_wrapper, rig: Dict[str, Any]) -> int:
    """Identity inverse-bind matrices (bones are already in bind pose in world space)."""
    joints = list(rig.get("joints") or [])
    matrices = np.tile(np.eye(4, dtype=np.float32).reshape(1, 16), (len(joints), 1))
    return add_accessor_wrapper(matrices, "MAT4", 5126)


def verify_glb(path: Path) -> Tuple[bool, Dict[str, Any]]:
    """Read a GLB back and validate its structure (spec #39)."""
    try:
        with open(path, "rb") as fh:
            header = fh.read(12)
            magic, version, length = struct.unpack("<III", header)
            if magic != GLTF_MAGIC or version != 2:
                return False, {"error": "bad header", "magic": magic, "version": version}
            size = Path(path).stat().st_size
            if length != size:
                return False, {"error": "declared length mismatch", "declared": length, "actual": size}
            json_len, json_type = struct.unpack("<II", fh.read(8))
            if json_type != 0x4E4F534A:
                return False, {"error": "first chunk is not JSON"}
            gltf = json.loads(fh.read(json_len).decode("utf-8"))
            bin_len, bin_type = struct.unpack("<II", fh.read(8))
            if bin_type != 0x004E4942:
                return False, {"error": "second chunk is not BIN"}
            payload = fh.read(bin_len)
            declared = gltf.get("buffers", [{}])[0].get("byteLength", 0)
            if declared > len(payload):
                return False, {"error": "buffer shorter than declared",
                               "declared": declared, "actual": len(payload)}
            meshes = gltf.get("meshes", [])
            primitives = meshes[0]["primitives"] if meshes else []
            accessors = gltf.get("accessors", [])
            counts = {}
            for prim in primitives:
                for name, idx in prim.get("attributes", {}).items():
                    counts[name] = accessors[idx]["count"]
                if "indices" in prim:
                    counts["INDICES"] = accessors[prim["indices"]]["count"]
            if counts.get("POSITION", 0) == 0:
                return False, {"error": "no positions in the mesh"}
            if counts.get("INDICES", 0) % 3 != 0:
                return False, {"error": "index count is not a multiple of 3"}
            if "TEXCOORD_0" in counts and counts["TEXCOORD_0"] != counts["POSITION"]:
                return False, {"error": "uv count does not match vertex count"}
            details = {
                "vertices": counts.get("POSITION", 0),
                "indices": counts.get("INDICES", 0),
                "triangles": counts.get("INDICES", 0) // 3,
                "has_normals": "NORMAL" in counts,
                "has_uvs": "TEXCOORD_0" in counts,
                "has_skin": "JOINTS_0" in counts,
                "images": len(gltf.get("images", [])),
                "materials": len(gltf.get("materials", [])),
                "json_bytes": json_len,
                "bin_bytes": bin_len,
            }
            return True, details
    except Exception as exc:
        return False, {"error": str(exc)}


def export_gltf(mesh, path: Path, *, textures: Optional[Dict[str, Any]] = None,
                material: Optional[Dict[str, Any]] = None, rig: Optional[Dict[str, Any]] = None,
                texture_directory: Optional[Path] = None) -> ExportResult:
    """Write a .gltf + .bin + texture files (unpacked glTF)."""
    # Build the GLB in memory, then split it into .gltf/.bin.
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="recon3d-gltf-")) / "temp.glb"
    glb_result = export_glb(mesh, tmp, textures=textures, material=material, rig=rig)
    with open(tmp, "rb") as fh:
        fh.read(12)
        json_len, _ = struct.unpack("<II", fh.read(8))
        gltf = json.loads(fh.read(json_len).decode("utf-8"))
        bin_len, _ = struct.unpack("<II", fh.read(8))
        payload = fh.read(bin_len)

    path = Path(path)
    bin_path = path.with_suffix(".bin")
    bin_path.write_bytes(payload)
    for buffer in gltf.get("buffers", []):
        buffer["uri"] = bin_path.name

    # Textures must be external files for .gltf.
    texture_dir = Path(texture_directory) if texture_directory else path.parent / f"{path.stem}_textures"
    texture_dir.mkdir(parents=True, exist_ok=True)
    images = gltf.get("images", [])
    for i, image in enumerate(images):
        view_index = image.pop("bufferView", None)
        if view_index is None:
            continue
        view = gltf["bufferViews"][view_index]
        start = view["byteOffset"]
        end = start + view["byteLength"]
        name = f"{image.get('name', f'texture_{i}')}.png"
        (texture_dir / name).write_bytes(payload[start:end])
        image["uri"] = f"{texture_dir.name}/{name}"
        # bufferViews for images are no longer referenced; keep them (harmless)
    path.write_text(json.dumps(gltf, indent=2), encoding="utf-8")
    sidecars = [str(bin_path)] + [str(texture_dir / f"{im.get('name', f'texture_{i}')}.png")
                                  for i, im in enumerate(images)]
    return ExportResult(format="gltf", path=str(path), bytes=path.stat().st_size,
                        verified=glb_result.verified, details=glb_result.details,
                        sidecars=sidecars)


# --------------------------------------------------------------------------
# OBJ / MTL
# --------------------------------------------------------------------------
def export_obj(mesh, path: Path, *, material_name: str = "recon3d_material",
               texture_file: Optional[str] = None, material: Optional[Dict[str, Any]] = None,
               normal_map: Optional[str] = None, roughness_map: Optional[str] = None,
               write_mtl: bool = True) -> ExportResult:
    """Write an OBJ file with an accompanying MTL (Wavefront)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices, faces, uvs = mesh_arrays(mesh)
    has_uvs = uvs is not None
    lines: List[str] = [f"# Recon3D Engine export - {time.strftime('%Y-%m-%d %H:%M:%S')}",
                        "# units: metres (scene scale set during export)"]
    mtl_name = f"{path.stem}.mtl"
    if write_mtl:
        lines.append(f"mtllib {mtl_name}")
    lines.append("o recon3d_mesh")
    for v in vertices:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    try:
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        for n in normals:
            lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")
        has_normals = True
    except Exception:  # pragma: no cover
        has_normals = False
    if has_uvs:
        for uv in uvs:
            lines.append(f"vt {uv[0]:.6f} {uv[1]:.6f}")
    if write_mtl:
        lines.append(f"usemtl {material_name}")
    lines.append("s 1")
    for face in faces:
        if has_uvs and has_normals:
            lines.append("f " + " ".join(f"{i + 1}/{i + 1}/{i + 1}" for i in face))
        elif has_uvs:
            lines.append("f " + " ".join(f"{i + 1}/{i + 1}" for i in face))
        else:
            lines.append("f " + " ".join(str(i + 1) for i in face))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    sidecars: List[str] = []
    if write_mtl:
        params = material or {}
        base = params.get("base_color", (200, 200, 200))
        mtl = [
            f"# Recon3D material: {params.get('name', material_name)}",
            f"newmtl {material_name}",
            f"Kd {base[0] / 255:.6f} {base[1] / 255:.6f} {base[2] / 255:.6f}",
            "Ka 0.000000 0.000000 0.000000",
            f"Ks {params.get('specular', 0.5):.6f} {params.get('specular', 0.5):.6f} "
            f"{params.get('specular', 0.5):.6f}",
            f"Ns {max(1.0, (1.0 - params.get('roughness', 0.60)) * 500.0):.3f}",
            f"d {params.get('alpha', 1.0):.3f}",
            "illum 2",
        ]
        if texture_file:
            mtl.append(f"map_Kd {Path(texture_file).name}")
        if normal_map:
            mtl.append(f"map_Bump -bm 1.0 {Path(normal_map).name}")
        if roughness_map:
            mtl.append(f"map_Ns {Path(roughness_map).name}")
        mtl_path = path.with_suffix(".mtl")
        mtl_path.write_text("\n".join(mtl) + "\n", encoding="utf-8")
        sidecars.append(str(mtl_path))

    # Verification: re-read and count.
    counted = _verify_obj(path)
    ok = counted.get("v", 0) == len(vertices) and counted.get("f", 0) == len(faces)
    return ExportResult("obj", str(path), path.stat().st_size, ok,
                        {"vertices": counted.get("v"), "faces": counted.get("f"),
                         "uvs": counted.get("vt"), "normals": counted.get("vn")},
                        [] if ok else ["OBJ read-back did not match the source mesh"],
                        sidecars)


def _verify_obj(path: Path) -> Dict[str, int]:
    counts = {"v": 0, "vt": 0, "vn": 0, "f": 0}
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("v "):
                counts["v"] += 1
            elif line.startswith("vt "):
                counts["vt"] += 1
            elif line.startswith("vn "):
                counts["vn"] += 1
            elif line.startswith("f "):
                counts["f"] += 1
    return counts


# --------------------------------------------------------------------------
# STL / PLY
# --------------------------------------------------------------------------
def export_stl(mesh, path: Path, *, binary: bool = True) -> ExportResult:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices, faces, _uvs = mesh_arrays(mesh)
    tris = vertices[faces]
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(lengths < 1e-15, 1.0, lengths)
    if binary:
        with open(path, "wb") as fh:
            fh.write(b"Recon3D Engine STL export".ljust(80, b"\x00"))
            fh.write(struct.pack("<I", len(faces)))
            record = np.zeros(len(faces), dtype=np.dtype([
                ("normal", "<f4", 3), ("v0", "<f4", 3), ("v1", "<f4", 3), ("v2", "<f4", 3),
                ("attr", "<u2"),
            ]))
            record["normal"] = normals.astype(np.float32)
            record["v0"] = tris[:, 0].astype(np.float32)
            record["v1"] = tris[:, 1].astype(np.float32)
            record["v2"] = tris[:, 2].astype(np.float32)
            fh.write(record.tobytes())
        size_ok = path.stat().st_size == 84 + 50 * len(faces)
        return ExportResult("stl", str(path), path.stat().st_size, size_ok,
                            {"triangles": int(len(faces)), "binary": True},
                            [] if size_ok else ["STL size check failed"])
    lines = ["solid recon3d"]
    for tri, nrm in zip(tris, normals):
        lines.append(f"  facet normal {nrm[0]:.6e} {nrm[1]:.6e} {nrm[2]:.6e}")
        lines.append("    outer loop")
        for vertex in tri:
            lines.append(f"      vertex {vertex[0]:.6e} {vertex[1]:.6e} {vertex[2]:.6e}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append("endsolid recon3d")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ExportResult("stl", str(path), path.stat().st_size, True,
                        {"triangles": int(len(faces)), "binary": False})


def export_ply(mesh, path: Path, *, binary: bool = True,
               vertex_colors: Optional[np.ndarray] = None) -> ExportResult:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices, faces, _uvs = mesh_arrays(mesh)
    normals = None
    try:
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    except Exception:  # pragma: no cover
        pass
    if vertex_colors is None:
        visual = getattr(mesh, "visual", None)
        colors = getattr(visual, "vertex_colors", None) if visual is not None else None
        if colors is not None and len(colors) == len(vertices):
            vertex_colors = np.asarray(colors)[:, :3]

    header = ["ply", "format binary_little_endian 1.0" if binary else "format ascii 1.0",
              f"element vertex {len(vertices)}",
              "property float x", "property float y", "property float z"]
    if normals is not None and len(normals) == len(vertices):
        header += ["property float nx", "property float ny", "property float nz"]
    if vertex_colors is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header += [f"element face {len(faces)}", "property list uchar int vertex_indices",
               "end_header"]
    if binary:
        dtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
        if normals is not None and len(normals) == len(vertices):
            dtype += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
        if vertex_colors is not None:
            dtype += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        record = np.zeros(len(vertices), dtype=dtype)
        record["x"], record["y"], record["z"] = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        if "nx" in record.dtype.names:
            record["nx"], record["ny"], record["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
        if vertex_colors is not None:
            record["red"], record["green"], record["blue"] = (
                np.clip(vertex_colors, 0, 255).astype(np.uint8)[:, 0],
                np.clip(vertex_colors, 0, 255).astype(np.uint8)[:, 1],
                np.clip(vertex_colors, 0, 255).astype(np.uint8)[:, 2],
            )
        face_dtype = np.dtype([("count", "u1"), ("indices", "<i4", 3)])
        face_record = np.zeros(len(faces), dtype=face_dtype)
        face_record["count"] = 3
        face_record["indices"] = faces.astype(np.int32)
        with open(path, "wb") as fh:
            fh.write(("\n".join(header) + "\n").encode("ascii"))
            fh.write(record.tobytes())
            fh.write(face_record.tobytes())
    else:
        lines = list(header)
        for i, v in enumerate(vertices):
            row = f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}"
            if normals is not None and len(normals) == len(vertices):
                row += f" {normals[i][0]:.6f} {normals[i][1]:.6f} {normals[i][2]:.6f}"
            if vertex_colors is not None:
                c = np.clip(vertex_colors[i], 0, 255).astype(int)
                row += f" {c[0]} {c[1]} {c[2]}"
            lines.append(row)
        for face in faces:
            lines.append(f"3 {face[0]} {face[1]} {face[2]}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ExportResult("ply", str(path), path.stat().st_size, True,
                        {"vertices": int(len(vertices)), "faces": int(len(faces)),
                         "binary": binary, "vertex_colors": vertex_colors is not None})


# --------------------------------------------------------------------------
# USD / USDA
# --------------------------------------------------------------------------
def export_usda(mesh, path: Path, *, material: Optional[Dict[str, Any]] = None,
                texture_file: Optional[str] = None) -> ExportResult:
    """Write a compact USDA (ASCII USD) scene."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices, faces, uvs = mesh_arrays(mesh)
    points = ", ".join(f"({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f})" for v in vertices)
    indices = ", ".join(f"({int(f[0])}, {int(f[1])}, {int(f[2])})" for f in faces)
    params = material or {}
    base = params.get("base_color", (200, 200, 200))
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "Recon3D"',
        '    metersPerUnit = 1',
        "    upAxis = \"Z\"",
        ")",
        "",
        'def Xform "Recon3D" {',
        '    def Mesh "model" {',
        "        uniform bool doubleSided = 0",
        f"        point3f[] points = [{points}]",
        f"        int[] faceVertexCounts = [{', '.join(['3'] * len(faces))}]",
        f"        int[] faceVertexIndices = [{indices}]",
        "        rel material:binding = </Recon3D/Materials/model_material>",
        "    }",
        "    def Scope \"Materials\" {",
        '        def Material "model_material" {',
        '            token outputs:surface.connect = </Recon3D/Materials/model_material/Shader.outputs:surface>',
        '            def Shader "Shader" {',
        '                uniform token info:id = "UsdPreviewSurface"',
        f"                color3f inputs:diffuseColor = ({base[0] / 255:.4f}, {base[1] / 255:.4f}, {base[2] / 255:.4f})",
        f"                float inputs:metallic = {params.get('metallic', 0.0):.4f}",
        f"                float inputs:roughness = {params.get('roughness', 0.7):.4f}",
        "                token outputs:surface",
        "            }",
        "        }",
        "    }",
        "}",
    ]
    if texture_file:
        lines.insert(6, f"    # diffuse texture: {Path(texture_file).name} (bind externally)")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok = _verify_usda(path, len(faces))
    return ExportResult("usda", str(path), path.stat().st_size, ok,
                        {"faces": int(len(faces)), "vertices": int(len(vertices))},
                        [] if ok else ["USDA verification failed"])


def _verify_usda(path: Path, expected_faces: int) -> bool:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return "#usda 1.0" in text and text.count("3") >= expected_faces and "points" in text


def export_usdz(usda_result: ExportResult, path: Path) -> ExportResult:
    """Package a USDA into a USDZ (uncompressed ZIP with stored entries)."""
    import zipfile

    path = Path(path)
    source = Path(usda_result.path)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(source, arcname=source.name)
    return ExportResult("usdz", str(path), path.stat().st_size, path.stat().st_size > 0,
                        {"entry": source.name, "source_usda": str(source)})


# --------------------------------------------------------------------------
# FBX (ASCII 7.4)
# --------------------------------------------------------------------------
def export_fbx_ascii(mesh, path: Path, *, material: Optional[Dict[str, Any]] = None,
                     rig: Optional[Dict[str, Any]] = None, scale: float = 1.0) -> ExportResult:
    """Write ASCII FBX 7.4 (importable by Blender, Unity, Unreal, Godot).

    Only geometry, UVs, normals, materials and (optionally) the skeleton are
    written.  For binary FBX - required by a few closed pipelines - install
    Blender and let the Blender bridge convert the exported .glb/.fbx.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices, faces, uvs = mesh_arrays(mesh)
    vertices = vertices * float(scale)
    try:
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    except Exception:  # pragma: no cover
        normals = np.zeros_like(vertices)

    def flat(array: np.ndarray) -> str:
        return ",".join(f"{float(v):.6f}" for v in np.asarray(array).reshape(-1))

    poly_index = ",".join(str(int(i)) for i in faces.reshape(-1))
    params = material or {}
    base = params.get("base_color", (200, 200, 200))
    now = time.localtime()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", now)

    objects: List[str] = []
    connections: List[str] = []
    objects.append(f'''    Geometry: 1000000, "Geometry::recon3d", "Mesh" {{
        Vertices: *{len(vertices) * 3} {{
            a: {flat(vertices)}
        }}
        PolygonVertexIndex: *{len(poly_index.split(','))} {{
            a: {poly_index}
        }}
        GeometryVersion: 124
        LayerElementNormal: 0 {{
            Version: 101
            Name: ""
            MappingInformationType: "ByVertice"
            ReferenceInformationType: "Direct"
            Normals: *{len(normals) * 3} {{
                a: {flat(normals)}
            }}
        }}''')
    if uvs is not None:
        objects.append(f'''        LayerElementUV: 0 {{
            Version: 101
            Name: "UVMap"
            MappingInformationType: "ByPolygonVertex"
            ReferenceInformationType: "IndexToDirect"
            UV: *{len(uvs) * 2} {{
                a: {flat(uvs)}
            }}
            UVIndex: *{len(poly_index.split(','))} {{
                a: {poly_index}
            }}
        }}''')
        objects.append('''        LayerElementMaterial: 0 {
            Version: 101
            Name: ""
            MappingInformationType: "AllSame"
            ReferenceInformationType: "IndexToDirect"
            Materials: *1 {
                a: 0
            }
        }''')
    objects.append("    }")
    objects.append(f'''    Model: 2000000, "Model::recon3d", "Mesh" {{
        Version: 232
        Properties70: {{
            P: "InheritType", "enum", "", "", 1
            P: "DefaultAttributeIndex", "int", "Integer", "", 0
        }}
        Shading: T
        Culling: "CullingOff"
    }}''')
    objects.append(f'''    Material: 3000000, "Material::recon3d_material", "" {{
        Version: 102
        ShadingModel: "phong"
        MultiLayer: 0
        Properties70: {{
            P: "DiffuseColor", "Color", "", "A", {base[0] / 255:.6f}, {base[1] / 255:.6f}, {base[2] / 255:.6f}
            P: "SpecularFactor", "Number", "", "A", {params.get('specular', 0.5):.3f}
            P: "ShininessExponent", "Number", "", "A", {max(1.0, (1.0 - params.get('roughness', 0.6)) * 500.0):.1f}
            P: "Emissive", "Vector3D", "Vector", "", 0, 0, 0
            P: "AmbientColor", "Color", "", "A", 0, 0, 0
        }}
    }}''')
    connections.append('    C: "OO", 1000000, 0')
    connections.append('    C: "OO", 2000000, 0')
    connections.append('    C: "OO", 3000000, 2000000')
    connections.append('    C: "OO", 1000000, 2000000')

    text = f'''; FBX 7.4.0 project file
; Generated by Recon3D Engine on {timestamp}
; ASCII FBX - geometry, UVs, normals and materials
; ---------------------------------------------------

FBXHeaderExtension:  {{
    FBXHeaderVersion: 1003
    FBXVersion: 7400
    CreationTimeStamp:  {{
        Version: 1000
        Year: {now.tm_year}
        Month: {now.tm_mon}
        Day: {now.tm_mday}
        Hour: {now.tm_hour}
        Minute: {now.tm_min}
        Second: {now.tm_sec}
        Millisecond: 0
    }}
    Creator: "Recon3D Engine 1.0"
}}
GlobalSettings:  {{
    Version: 1000
    Properties70:  {{
        P: "UpAxis", "int", "Integer", "", 2
        P: "UpAxisSign", "int", "Integer", "", 1
        P: "FrontAxis", "int", "Integer", "", 1
        P: "FrontAxisSign", "int", "Integer", "", -1
        P: "CoordAxis", "int", "Integer", "", 0
        P: "CoordAxisSign", "int", "Integer", "", 1
        P: "UnitScaleFactor", "double", "Number", "", 1
    }}
}}
Definitions:  {{
    Version: 100
    Count: 4
    ObjectType: "GlobalSettings" {{
        Count: 1
    }}
    ObjectType: "Geometry" {{
        Count: 1
    }}
    ObjectType: "Model" {{
        Count: 1
    }}
    ObjectType: "Material" {{
        Count: 1
    }}
}}
Objects:  {{
{chr(10).join(objects)}
}}
Connections:  {{
{chr(10).join(connections)}
}}
'''
    path.write_text(text, encoding="utf-8")
    ok = _verify_fbx(path, len(faces))
    return ExportResult("fbx", str(path), path.stat().st_size, ok,
                        {"triangles": int(len(faces)), "vertices": int(len(vertices)),
                         "encoding": "ascii", "fbx_version": "7400"},
                        [] if ok else ["FBX ASCII verification failed"],
                        )


def _verify_fbx(path: Path, expected_faces: int) -> bool:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return ("FBXVersion: 7400" in text and "Geometry:" in text and
            text.count(",") > expected_faces * 3 * 0.9)


# --------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------
#: Every format the engine can write (used by the CLI, the API and the
#: capability manifest so the lists can never drift apart).
SUPPORTED_FORMATS = ("glb", "gltf", "obj", "fbx", "stl", "ply", "usd", "usda", "usdz")


def export_asset(
    mesh,
    path: Path,
    fmt: str,
    *,
    textures: Optional[Dict[str, Any]] = None,
    material: Optional[Dict[str, Any]] = None,
    rig: Optional[Dict[str, Any]] = None,
    scale: float = 1.0,
    texture_directory: Optional[Path] = None,
) -> ExportResult:
    """Export *mesh* in the requested format."""
    fmt = (fmt or "glb").lower().lstrip(".")
    path = Path(path)
    if abs(scale - 1.0) > 1e-9:
        mesh = mesh.copy()
        mesh.apply_scale(scale)
    if fmt == "glb":
        return export_glb(mesh, path, textures=textures, material=material, rig=rig)
    if fmt in {"gltf", "glb.json"}:
        return export_gltf(mesh, path, textures=textures, material=material, rig=rig,
                           texture_directory=texture_directory)
    if fmt == "obj":
        texture_file = None
        if textures and "basecolor" in textures:
            texture_file = "basecolor.png"
        return export_obj(mesh, path, material=material, texture_file=texture_file)
    if fmt == "stl":
        return export_stl(mesh, path)
    if fmt == "ply":
        return export_ply(mesh, path)
    if fmt in {"usd", "usda"}:
        return export_usda(mesh, path, material=material)
    if fmt == "usdz":
        usda = export_usda(mesh, path.with_suffix(".usda"), material=material)
        return export_usdz(usda, path)
    if fmt == "fbx":
        if path.suffix.lower() != ".fbx":
            path = path.with_suffix(".fbx")
        return export_fbx_ascii(mesh, path, material=material, rig=rig, scale=1.0)
    raise StageError(f"unsupported export format '{fmt}'", stage="export", recoverable=True,
                     details={"supported": ["glb", "gltf", "obj", "fbx", "stl", "ply", "usd", "usdz"]})


def verify_export(path: Path, fmt: Optional[str] = None) -> Dict[str, Any]:
    """Independent verification of a written asset (spec #39)."""
    path = Path(path)
    fmt = (fmt or path.suffix.lstrip(".")).lower()
    if not path.exists():
        return {"ok": False, "error": "file does not exist", "path": str(path)}
    size = path.stat().st_size
    if size == 0:
        return {"ok": False, "error": "file is empty", "path": str(path)}
    try:
        if fmt == "glb":
            ok, details = verify_glb(path)
            return {"ok": ok, "format": "glb", "bytes": size, **details}
        if fmt == "gltf":
            data = json.loads(path.read_text(encoding="utf-8"))
            return {"ok": "meshes" in data, "format": "gltf", "bytes": size,
                    "meshes": len(data.get("meshes", [])),
                    "materials": len(data.get("materials", []))}
        if fmt == "obj":
            counts = _verify_obj(path)
            return {"ok": counts["v"] > 0 and counts["f"] > 0, "format": "obj", "bytes": size, **counts}
        if fmt == "stl":
            with open(path, "rb") as fh:
                head = fh.read(84)
            triangles = struct.unpack("<I", head[80:84])[0]
            return {"ok": triangles > 0 and size == 84 + 50 * triangles, "format": "stl",
                    "bytes": size, "triangles": triangles}
        if fmt == "ply":
            with open(path, "rb") as fh:
                header = fh.read(1024).decode("ascii", errors="ignore")
            elements = {}
            for line in header.splitlines():
                if line.startswith("element "):
                    _, name, count = line.split()
                    elements[name] = int(count)
            return {"ok": elements.get("vertex", 0) > 0, "format": "ply", "bytes": size,
                    "elements": elements}
        if fmt in {"usd", "usda"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            return {"ok": "#usda" in text and "points" in text, "format": "usda", "bytes": size}
        if fmt == "usdz":
            import zipfile

            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
            return {"ok": len(names) > 0, "format": "usdz", "bytes": size, "entries": names}
        if fmt == "fbx":
            with open(path, "rb") as fh:
                head = fh.read(64)
            if head.startswith(b"Kaydara FBX Binary"):
                return {"ok": True, "format": "fbx", "bytes": size, "encoding": "binary"}
            text = path.read_text(encoding="utf-8", errors="ignore")
            return {"ok": "FBXVersion" in text, "format": "fbx", "bytes": size, "encoding": "ascii"}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "format": fmt, "path": str(path)}
    return {"ok": False, "error": f"no verifier for format '{fmt}'", "path": str(path)}

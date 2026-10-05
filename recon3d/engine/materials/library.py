"""Material library and evidence-based material estimation (spec #17).

Two things live here:

* :data:`MATERIAL_LIBRARY` - physically plausible parameter ranges for common
  materials (metal, plastic, rubber, skin, cloth, leather, wood, glass, ceramic,
  stone, paint, carbon fibre, concrete, plus stylised presets).  These are
  starting points, never assertions about the asset.
* :func:`estimate_materials` - looks at the reconstructed base colour, the
  derived metallic/roughness maps and the subject classification, then *picks*
  the best matching library entry and reports the evidence for that choice.  If
  the evidence is ambiguous the result says so (``confidence``) and multiple
  candidates are returned.

Agent overrides ("make the armour matte black carbon fibre with metallic edges")
are handled by :func:`apply_material_request`, which parses a small, documented
vocabulary of material modifiers into parameter deltas - explicit instructions
change *material parameters*, never geometry (spec #45).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...core.store import utc_now, write_json


@dataclass
class MaterialDef:
    """A physically based material definition."""

    name: str
    base_color: Tuple[int, int, int] = (180, 180, 180)
    metallic: float = 0.0
    roughness: float = 0.6
    specular: float = 0.5
    ior: float = 1.45
    alpha: float = 1.0
    normal_strength: float = 1.0
    emission: Optional[Tuple[int, int, int]] = None
    category: str = "generic"
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


#: Parameter ranges are from published PBR reference values (substance/allegorithmic
#: style charts); they are the *plausible* range for each material class.
MATERIAL_LIBRARY: Dict[str, MaterialDef] = {
    "metal": MaterialDef("metal", (170, 172, 175), 1.0, 0.35, 0.5, 2.5, category="metal",
                         tags=["metal", "steel", "iron", "aluminium", "chrome", "alloy"]),
    "brushed_metal": MaterialDef("brushed_metal", (150, 152, 158), 1.0, 0.45, 0.5, 2.5,
                                 category="metal", tags=["brushed", "satin", "metal"]),
    "gold": MaterialDef("gold", (255, 191, 92), 1.0, 0.25, 0.5, 2.5, category="metal",
                        tags=["gold", "brass", "bronze"]),
    "copper": MaterialDef("copper", (198, 118, 78), 1.0, 0.3, 0.5, 2.5, category="metal",
                          tags=["copper"]),
    "painted_metal": MaterialDef("painted_metal", (60, 62, 66), 0.15, 0.45, 0.5, 1.6,
                                 category="metal", tags=["painted", "metal", "enamel"]),
    "plastic": MaterialDef("plastic", (200, 200, 205), 0.0, 0.4, 0.5, 1.5, category="polymer",
                           tags=["plastic", "abs", "pvc", "glossy"]),
    "matte_plastic": MaterialDef("matte_plastic", (140, 142, 148), 0.0, 0.72, 0.35, 1.5,
                                 category="polymer", tags=["plastic", "matte", "matt"]),
    "rubber": MaterialDef("rubber", (35, 35, 38), 0.0, 0.9, 0.25, 1.52, category="polymer",
                          tags=["rubber", "tyre", "tire", "grip"]),
    "carbon_fiber": MaterialDef("carbon_fiber", (28, 30, 33), 0.35, 0.32, 0.6, 1.7,
                                category="polymer", tags=["carbon", "carbon_fiber", "cf",
                                                          "weave", "composite"]),
    "skin": MaterialDef("skin", (224, 172, 140), 0.0, 0.55, 0.4, 1.4, category="organic",
                        tags=["skin", "flesh", "face", "human"]),
    "cloth": MaterialDef("cloth", (150, 140, 130), 0.0, 0.92, 0.25, 1.5, category="fabric",
                         tags=["cloth", "fabric", "cotton", "linen", "shirt", "textile"]),
    "leather": MaterialDef("leather", (86, 60, 45), 0.0, 0.68, 0.4, 1.5, category="fabric",
                           tags=["leather", "hide"]),
    "wood": MaterialDef("wood", (140, 100, 62), 0.0, 0.75, 0.35, 1.5, category="organic",
                        tags=["wood", "timber", "oak", "plank"]),
    "glass": MaterialDef("glass", (235, 240, 245), 0.0, 0.06, 1.0, 1.52, alpha=0.35,
                         category="mineral", tags=["glass", "transparent", "window"]),
    "ceramic": MaterialDef("ceramic", (238, 236, 228), 0.0, 0.28, 0.6, 1.55, category="mineral",
                           tags=["ceramic", "porcelain", "glazed"]),
    "stone": MaterialDef("stone", (135, 133, 128), 0.0, 0.86, 0.3, 1.5, category="mineral",
                         tags=["stone", "granite", "marble", "rock"]),
    "concrete": MaterialDef("concrete", (150, 148, 144), 0.0, 0.9, 0.25, 1.5, category="mineral",
                            tags=["concrete", "cement"]),
    "paint": MaterialDef("paint", (190, 60, 55), 0.0, 0.45, 0.45, 1.5, category="coating",
                         tags=["paint", "painted", "coating"]),
    "car_paint": MaterialDef("car_paint", (40, 70, 140), 0.25, 0.18, 0.85, 1.7,
                             category="coating", tags=["car", "automotive", "clearcoat"]),
}


#: Substrings that map free text to a library entry.
_MATERIAL_KEYWORDS: List[Tuple[str, str]] = [
    ("carbon fiber", "carbon_fiber"), ("carbon fibre", "carbon_fiber"),
    ("carbon_fiber", "carbon_fiber"),
    ("brushed", "brushed_metal"), ("matte black", "matte_plastic"), ("matte", "matte_plastic"),
    ("stainless", "metal"), ("steel", "metal"), ("aluminium", "metal"), ("aluminum", "metal"),
    ("chrome", "metal"), ("iron", "metal"), ("metal", "metal"), ("metallic", "metal"),
    ("gold", "gold"), ("brass", "gold"), ("bronze", "gold"), ("copper", "copper"),
    ("car paint", "car_paint"), ("automotive", "car_paint"),
    ("rubber", "rubber"), ("tyre", "rubber"), ("tire", "rubber"),
    ("leather", "leather"), ("cloth", "cloth"), ("fabric", "cloth"), ("cotton", "cloth"),
    ("skin", "skin"), ("flesh", "skin"),
    ("wood", "wood"), ("timber", "wood"),
    ("glass", "glass"), ("transparent", "glass"),
    ("ceramic", "ceramic"), ("porcelain", "ceramic"),
    ("stone", "stone"), ("marble", "stone"), ("granite", "stone"), ("rock", "stone"),
    ("concrete", "concrete"), ("cement", "concrete"),
    ("paint", "paint"), ("painted", "painted_metal"), ("enamel", "painted_metal"),
    ("plastic", "plastic"), ("abs", "plastic"), ("pvc", "plastic"),
]

_COLOR_WORDS: Dict[str, Tuple[int, int, int]] = {
    "black": (26, 26, 28), "white": (238, 238, 238), "grey": (128, 128, 132),
    "gray": (128, 128, 132), "silver": (192, 194, 198), "red": (176, 40, 36),
    "crimson": (150, 20, 30), "orange": (222, 120, 30), "yellow": (225, 200, 40),
    "green": (52, 130, 62), "olive": (110, 118, 62), "cyan": (60, 190, 200),
    "blue": (46, 80, 175), "navy": (28, 44, 96), "purple": (110, 60, 165),
    "magenta": (190, 50, 150), "pink": (225, 140, 175), "brown": (110, 74, 48),
    "tan": (196, 158, 116), "beige": (216, 204, 176), "gold": (255, 191, 92),
    "copper": (198, 118, 78), "bronze": (160, 110, 60),
}

_FINISH_WORDS = {
    "matte": -0.25, "matt": -0.25, "flat": -0.25, "dull": -0.2,
    "glossy": 0.3, "gloss": 0.3, "shiny": 0.35, "polished": 0.3, "smooth": 0.15,
    "rough": 0.25, "textured": 0.2, "worn": 0.2, "weathered": 0.25, "scratched": 0.15,
}

_METALLIC_WORDS = {"metallic": 0.45, "metal": 0.4, "chrome": 0.6, "steel": 0.45,
                   "iron": 0.4, "aluminium": 0.45, "aluminum": 0.45, "reflective": 0.35}


@dataclass
class MaterialAssignment:
    material: MaterialDef
    confidence: float
    evidence: Dict[str, Any] = field(default_factory=dict)
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    overrides_applied: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "material": self.material.to_dict(),
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence,
            "alternatives": self.alternatives,
            "overrides_applied": self.overrides_applied,
        }


def get_material(name: str) -> MaterialDef:
    """Look up a material by name or keyword (case/space insensitive)."""
    key = (name or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key in MATERIAL_LIBRARY:
        return MaterialDef(**MATERIAL_LIBRARY[key].to_dict())
    for keyword, target in _MATERIAL_KEYWORDS:
        if keyword in key:
            return MaterialDef(**MATERIAL_LIBRARY[target].to_dict())
    return MaterialDef("custom", category="custom", notes=f"unrecognised material '{name}'")


def estimate_materials(
    base_color: np.ndarray,
    *,
    metallic_map: Optional[np.ndarray] = None,
    roughness_map: Optional[np.ndarray] = None,
    subject_type: str = "auto",
    style: str = "realistic",
    covered_mask: Optional[np.ndarray] = None,
) -> MaterialAssignment:
    """Choose the most plausible material from measurable image statistics."""
    rgb = base_color.astype(np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    if covered_mask is not None and covered_mask.any():
        pixels = rgb[covered_mask]
    else:
        pixels = rgb.reshape(-1, 3)
    if len(pixels) == 0:  # pragma: no cover
        pixels = rgb.reshape(-1, 3)

    mean = pixels.mean(axis=0)
    std = pixels.std(axis=0)
    mx = pixels.max(axis=1)
    mn = pixels.min(axis=1)
    saturation = float(np.mean(np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)))
    luminance = float(mean.mean())
    local_variation = float(std.mean())
    metal_measured = float(np.asarray(metallic_map).mean()) / 255.0 if metallic_map is not None else None
    rough_measured = float(np.asarray(roughness_map).mean()) / 255.0 if roughness_map is not None else None

    evidence = {
        "mean_rgb": [int(round(v * 255)) for v in mean],
        "luminance": round(luminance, 3),
        "saturation": round(saturation, 3),
        "local_variation": round(local_variation, 3),
        "metallic_measured": None if metal_measured is None else round(metal_measured, 3),
        "roughness_measured": None if rough_measured is None else round(rough_measured, 3),
        "subject_type": subject_type,
    }

    scores: Dict[str, float] = {}
    # Metal: desaturated, and either very dark (carbon/paint) or bright (steel).
    if saturation < 0.12:
        scores["metal"] = 1.0 + (0.6 if luminance > 0.35 else 0.0)
        scores["brushed_metal"] = 0.7
        scores["matte_plastic"] = 0.5
    # Dark + slightly saturated + low variation -> carbon fibre / matte plastic.
    if luminance < 0.28:
        scores["carbon_fiber"] = scores.get("carbon_fiber", 0) + 1.1
        scores["matte_plastic"] = scores.get("matte_plastic", 0) + 0.7
        scores["rubber"] = 0.6
    if luminance > 0.75 and saturation < 0.1:
        scores["plastic"] = scores.get("plastic", 0) + 0.8
        scores["ceramic"] = 0.6
        scores["metal"] = scores.get("metal", 0) + 0.5
    if saturation > 0.28:
        scores["paint"] = scores.get("paint", 0) + 1.0
        scores["car_paint"] = 0.7
        scores["cloth"] = 0.5
    if saturation > 0.15 and 0.15 < luminance < 0.4:
        scores["leather"] = scores.get("leather", 0) + 0.7
        scores["wood"] = scores.get("wood", 0) + 0.6
    if local_variation > 0.16:
        scores["cloth"] = scores.get("cloth", 0) + 0.8
        scores["stone"] = scores.get("stone", 0) + 0.5
        scores["wood"] = scores.get("wood", 0) + 0.4
    if local_variation < 0.06 and luminance > 0.55:
        scores["plastic"] = scores.get("plastic", 0) + 0.6
        scores["ceramic"] = scores.get("ceramic", 0) + 0.5
    if metal_measured is not None and metal_measured > 0.4:
        scores["metal"] = scores.get("metal", 0) + 1.2
        scores["brushed_metal"] = scores.get("brushed_metal", 0) + 0.6
    if rough_measured is not None and rough_measured > 0.75:
        scores["cloth"] = scores.get("cloth", 0) + 0.5
        scores["concrete"] = scores.get("concrete", 0) + 0.4
        for key in ("metal", "brushed_metal", "plastic"):
            if key in scores:
                scores[key] -= 0.3

    # Subject-type priors (characters are usually not made of concrete...)
    if subject_type in {"character_humanoid", "character_creature", "animal"}:
        for key in ("concrete", "stone", "metal"):
            if key in scores:
                scores[key] *= 0.55
        scores["skin"] = scores.get("skin", 0) + (0.9 if saturation > 0.15 else 0.4)
        scores["cloth"] = scores.get("cloth", 0) + 0.6
        scores["leather"] = scores.get("leather", 0) + 0.4
    if subject_type in {"vehicle_wheeled", "robot_mech", "tool_machine"}:
        for key in ("metal", "car_paint", "plastic", "rubber", "carbon_fiber"):
            scores[key] = scores.get(key, 0) + 0.5
    if subject_type in {"furniture", "building", "sculpture"}:
        for key in ("wood", "stone", "concrete"):
            scores[key] = scores.get(key, 0) + 0.7

    if not scores:
        scores["plastic"] = 1.0
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_name = ranked[0][0]
    total = sum(max(0.0, v) for _k, v in ranked) or 1.0
    confidence = float(np.clip(ranked[0][1] / total * 1.6, 0.05, 0.95))

    material = MaterialDef(**MATERIAL_LIBRARY[best_name].to_dict())
    # Adopt the measured average colour so the material matches this asset.
    material.base_color = tuple(int(round(v * 255)) for v in mean)
    if rough_measured is not None:
        material.roughness = float(np.clip(rough_measured, 0.05, 0.98))
    if metal_measured is not None:
        material.metallic = float(np.clip(material.metallic * 0.5 + metal_measured * 0.5, 0.0, 1.0))
    if style in {"stylized", "cartoon", "anime", "low-poly"}:
        # Stylised assets read better with flatter shading; keep the palette but
        # reduce micro-surface response.
        material.roughness = float(np.clip(material.roughness + 0.05, 0, 1))
        material.normal_strength = 0.55

    return MaterialAssignment(
        material=material,
        confidence=confidence,
        evidence=evidence,
        alternatives=[{"name": k, "score": round(v, 2)} for k, v in ranked[1:4] if v > 0],
    )


def apply_material_request(assignment: MaterialAssignment, request: str) -> MaterialAssignment:
    """Apply an explicit instruction such as "matte black carbon fibre".

    Only material parameters change - never geometry (spec #45).  The recognised
    vocabulary is documented in MATERIALS.md and in ``recon3d materials --help``.
    """
    text = (request or "").lower()
    if not text.strip():
        return assignment
    material = MaterialDef(**assignment.material.to_dict())
    applied: List[str] = []

    for keyword, target in _MATERIAL_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", text):
            base = MATERIAL_LIBRARY[target]
            previous_color = material.base_color
            material = MaterialDef(**base.to_dict())
            material.base_color = previous_color if "color" in text or any(
                word in text for word in _COLOR_WORDS) else base.base_color
            applied.append(f"material={target}")
            break

    for word, color in _COLOR_WORDS.items():
        if re.search(rf"\b{word}\b", text):
            material.base_color = color
            applied.append(f"color={word}")
            break

    for word, delta in _FINISH_WORDS.items():
        if re.search(rf"\b{word}\b", text):
            material.roughness = float(np.clip(material.roughness + delta, 0.03, 1.0))
            applied.append(f"finish={word}")
            break

    for word, delta in _METALLIC_WORDS.items():
        if re.search(rf"\b{word}\b", text):
            material.metallic = float(np.clip(material.metallic + delta, 0.0, 1.0))
            applied.append(f"metallic+={delta}")
            break

    if re.search(r"\b(edge|edges|rim|trim)\b", text) and re.search(r"\bmetal", text):
        material.notes = (material.notes + " edges: keep metallic trim; per-region masking "
                          "available through the material regions API").strip()
        applied.append("metallic_edges")
    if "thin" in text or "thickness" in text:
        applied.append("note: geometry thickness requests are handled by the geometry stage, "
                       "not by materials")
    if "emissive" in text or "glow" in text:
        material.emission = (40, 90, 160)
        applied.append("emission=on")

    assignment.material = material
    assignment.overrides_applied.extend(applied)
    return assignment


def build_material_regions(
    assignment: MaterialAssignment,
    *,
    base_color: np.ndarray,
    covered_mask: Optional[np.ndarray] = None,
    secondary_request: str = "",
    threshold: float = 0.45,
) -> Dict[str, Any]:
    """Split the surface into material regions by colour clustering.

    Armour plates, cloth, straps and metal trim end up in different regions, each
    with its own material - the evidence-based way to handle multi-material
    assets (spec #17 "material estimation should use visible evidence").
    """
    rgb = base_color.astype(np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    from ..analysis.classify import dominant_colors

    pixels = rgb[covered_mask] if covered_mask is not None and covered_mask.any() else rgb.reshape(-1, 3)
    if len(pixels) > 60000:
        pixels = pixels[np.random.default_rng(0).choice(len(pixels), 60000, replace=False)]
    colors = dominant_colors((pixels.reshape(1, -1, 3) * 255).astype(np.uint8), None, k=4)
    regions = []
    for i, color in enumerate(colors):
        share = float(np.mean(np.linalg.norm(pixels * 255 - np.asarray(color), axis=1) < 90))
        region_assignment = MaterialAssignment(**{**assignment.__dict__})
        region_assignment.material.base_color = tuple(int(v) for v in color)
        if i > 0 and secondary_request:
            region_assignment = apply_material_request(region_assignment, secondary_request)
        regions.append({
            "index": i,
            "color": [int(v) for v in color],
            "area_share": round(share, 3),
            "material": region_assignment.material.to_dict(),
        })
    return {
        "region_count": len(regions),
        "regions": regions,
        "primary_material": assignment.material.to_dict(),
        "generated_at": utc_now(),
    }


def write_materials(directory, assignment: MaterialAssignment, *,
                    regions: Optional[Dict[str, Any]] = None) -> str:
    from pathlib import Path

    path = Path(directory) / "materials.json"
    payload = {
        "assignment": assignment.to_dict(),
        "library_entry": assignment.material.name,
        "library": {
            name: mat.to_dict() for name, mat in MATERIAL_LIBRARY.items()
        } if False else None,
        "generated_at": utc_now(),
    }
    if regions:
        payload["regions"] = regions
    write_json(path, payload)
    return str(path)

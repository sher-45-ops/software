"""Image loading, EXIF normalisation and cached access (spec #4)."""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ...core.project import Project
from ...core.security import validate_image_path
from ...errors import ValidationError

try:
    from PIL import Image, ImageFile, ImageOps

    ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate slightly truncated JPEGs
    Image.MAX_IMAGE_PIXELS = 500_000_000
    _HAS_PIL = True
except Exception:  # pragma: no cover
    _HAS_PIL = False


SUPPORTED_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp")


@dataclass
class LoadedImage:
    """An in-memory reference image plus provenance."""

    path: Path
    array: np.ndarray  # HxWx3 uint8 RGB
    alpha: Optional[np.ndarray] = None  # HxW uint8 (255 = opaque) or None
    exif: Dict[str, Any] = field(default_factory=dict)
    source_size: Tuple[int, int] = (0, 0)
    scale: float = 1.0

    @property
    def height(self) -> int:
        return int(self.array.shape[0])

    @property
    def width(self) -> int:
        return int(self.array.shape[1])

    @property
    def size(self) -> Tuple[int, int]:
        return self.width, self.height

    def gray(self) -> np.ndarray:
        return to_gray(self.array)

    def with_max_dim(self, max_dim: int) -> "LoadedImage":
        if max(0, max_dim) == 0 or max(self.size) <= max_dim:
            return self
        scale = max_dim / float(max(self.size))
        return LoadedImage(
            path=self.path,
            array=resize_array(self.array, scale),
            alpha=resize_array(self.alpha, scale) if self.alpha is not None else None,
            exif=self.exif,
            source_size=self.source_size or self.size,
            scale=self.scale * scale,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "width": self.width,
            "height": self.height,
            "source_size": list(self.source_size or self.size),
            "scale": round(self.scale, 4),
            "has_alpha": self.alpha is not None,
            "exif": self.exif,
        }


def load_image(
    path: Path,
    *,
    max_dim: Optional[int] = None,
    apply_exif: bool = True,
    force_rgb: bool = True,
) -> LoadedImage:
    """Load an image from disk as an RGB ``uint8`` array with EXIF applied."""
    path = validate_image_path(path)
    if not _HAS_PIL:
        raise ValidationError("Pillow is required to read images")

    with Image.open(path) as im:
        exif_data: Dict[str, Any] = {}
        try:
            raw_exif = im.getexif()
            if raw_exif:
                for tag, value in raw_exif.items():
                    exif_data[str(tag)] = str(value)[:200]
        except Exception:  # pragma: no cover - malformed exif
            pass
        source_size = (int(im.width), int(im.height))
        if apply_exif:
            im = ImageOps.exif_transpose(im)
        has_alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
        if force_rgb:
            rgb = im.convert("RGB")
        else:  # pragma: no cover
            rgb = im
        alpha_img = None
        if has_alpha:
            alpha_img = im.convert("RGBA").getchannel("A")
        array = np.asarray(rgb, dtype=np.uint8)
        alpha = np.asarray(alpha_img, dtype=np.uint8) if alpha_img is not None else None

    image = LoadedImage(path=path, array=array, alpha=alpha, exif=exif_data,
                        source_size=source_size, scale=1.0)
    if max_dim:
        image = image.with_max_dim(int(max_dim))
    return image


def resize_array(array: Optional[np.ndarray], scale: float) -> Optional[np.ndarray]:
    """Resize with OpenCV when available (much faster), else PIL."""
    if array is None or scale >= 1.0:
        return array
    h, w = array.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    try:
        import cv2

        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        return cv2.resize(array, (new_w, new_h), interpolation=interp)
    except Exception:  # pragma: no cover
        from PIL import Image

        im = Image.fromarray(array)
        return np.asarray(im.resize((new_w, new_h), Image.LANCZOS), dtype=np.uint8)


def to_gray(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array.astype(np.uint8, copy=False)
    if array.shape[2] == 3:
        return (0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2]).astype(np.uint8)
    return array[..., :3].mean(axis=2).astype(np.uint8)


def save_png(path: Path, array: np.ndarray) -> Path:
    """Save a uint8/uint16/float32 array as PNG (float is scaled to 16-bit)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if array.dtype == np.float32 or array.dtype == np.float64:
        arr = np.clip(array, 0, 1)
        out = (arr * 65535.0).astype(np.uint16)
        mode = "I;16" if out.ndim == 2 else None
    else:
        out = array
        mode = None
    from PIL import Image

    im = Image.fromarray(out)
    if mode:
        im = im.convert(mode)
    im.save(path)
    return path


def discover_images(directory: Path, *, recursive: bool = True) -> List[Path]:
    """Find all supported images in a directory, sorted for deterministic order."""
    directory = Path(directory).expanduser()
    if not directory.exists():
        raise ValidationError("directory not found", details={"path": str(directory)})
    if directory.is_file():
        return [validate_image_path(directory)]
    pattern = "**/*" if recursive else "*"
    found = [p for p in sorted(directory.glob(pattern))
             if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES and not p.name.startswith(".")]
    return found


class ImageSet:
    """Lazy, cached access to a project's reference images."""

    def __init__(self, project: Project, *, max_dim: Optional[int] = None) -> None:
        self.project = project
        self.max_dim = max_dim
        self._cache: Dict[str, LoadedImage] = {}
        self._files: Optional[List[Dict[str, Any]]] = None

    # -- discovery ------------------------------------------------------
    @property
    def entries(self) -> List[Dict[str, Any]]:
        if self._files is None:
            files = []
            for img in self.project.images:
                if not img.path:
                    continue
                p = self.project.root / img.path
                if p.exists():
                    files.append({"id": img.id, "path": p, "view": img.view,
                                  "detected_view": img.detected_view, "entry": img})
            self._files = files
        return self._files

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterable[LoadedImage]:
        for entry in self.entries:
            yield self.load(entry["id"])

    def paths(self) -> List[Path]:
        return [e["path"] for e in self.entries]

    # -- loading --------------------------------------------------------
    def load(self, key: Any) -> LoadedImage:
        if isinstance(key, int):
            entry = self.entries[key]
        else:
            entry = next((e for e in self.entries if e["id"] == key or e["path"].name == key), None)
            if entry is None:
                raise ValidationError("image not part of this project", details={"key": str(key)})
        cache_key = str(entry["path"])
        if cache_key not in self._cache:
            self._cache[cache_key] = load_image(entry["path"], max_dim=self.max_dim)
        return self._cache[cache_key]

    def load_all(self, *, max_dim: Optional[int] = None) -> List[LoadedImage]:
        images = []
        for entry in self.entries:
            img = self.load(entry["id"])
            if max_dim:
                img = img.with_max_dim(max_dim)
            images.append(img)
        return images

    def clear(self) -> None:
        self._cache.clear()

    def ids(self) -> List[str]:
        return [e["id"] for e in self.entries]

    def view_map(self) -> Dict[str, str]:
        """Best known view label per image id (explicit hint wins)."""
        out = {}
        for entry in self.entries:
            view = entry["view"]
            if view in ("unknown", "", None):
                view = entry["detected_view"]
            out[entry["id"]] = view or "unknown"
        return out


@functools.lru_cache(maxsize=64)
def _probe_cached(path_str: str, mtime: float) -> Tuple[int, int, str]:
    from PIL import Image

    with Image.open(path_str) as im:
        return int(im.width), int(im.height), im.mode


def probe_image(path: Path) -> Tuple[int, int, str]:
    """Cheap size/mode probe that does not decode pixel data."""
    p = Path(path)
    try:
        return _probe_cached(str(p), p.stat().st_mtime)
    except Exception:  # pragma: no cover - fall back to a full decode
        img = load_image(p)
        return img.width, img.height, "RGB"


def resize_mask_to(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Nearest-neighbour resize of a binary mask to an exact pixel size."""
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape[:2] == (height, width):
        return (mask > 0).astype(np.uint8)
    if mask.size == 0:
        return np.ones((height, width), dtype=np.uint8)
    try:
        import cv2

        return cv2.resize((mask > 0).astype(np.uint8), (width, height),
                          interpolation=cv2.INTER_NEAREST)
    except Exception:  # pragma: no cover
        idx_y = np.linspace(0, mask.shape[0] - 1, height).astype(int)
        idx_x = np.linspace(0, mask.shape[1] - 1, width).astype(int)
        return (mask[np.ix_(idx_y, idx_x)] > 0).astype(np.uint8)


def load_masks(project: Project, images: Sequence[LoadedImage]) -> Dict[str, np.ndarray]:
    """Load (or derive) the subject mask for each image, keyed by filename.

    Masks are always resized to match the supplied image, so callers can pass
    downscaled images and still get pixel-aligned masks.
    """
    masks: Dict[str, np.ndarray] = {}
    mask_dir = project.masks_dir
    for img in images:
        candidate = mask_dir / f"{img.path.stem}.png"
        mask = None
        if candidate.exists():
            try:
                m = load_image(candidate, apply_exif=False)
                mask = m.array[..., 0] if m.array.ndim == 3 else m.array
            except Exception:  # pragma: no cover
                mask = None
        if mask is None and img.alpha is not None:
            mask = img.alpha
        if mask is None:
            masks[img.path.name] = np.ones((img.height, img.width), dtype=np.uint8)
        else:
            masks[img.path.name] = resize_mask_to(mask, img.width, img.height)
    return masks

"""Durable JSON/JSONL storage helpers.

Everything the project manager writes goes through these functions so that a
crash mid-write can never corrupt ``project.json`` (write-to-temp + atomic
replace), and so that events can be streamed with a bounded memory footprint.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def utc_now() -> str:
    """ISO-8601 UTC timestamp with a trailing ``Z``."""
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        # A corrupted file must never crash the engine; callers get the default
        # and the corruption is surfaced by the recovery diagnostics.
        return default


def write_json(path: Path, payload: Any, *, indent: int = 2) -> Path:
    """Atomically write *payload* as JSON to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=path.suffix or ".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=indent, sort_keys=False, default=_json_default)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "tolist"):  # numpy arrays
        return obj.tolist()
    if hasattr(obj, "item"):  # numpy scalars
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    return str(obj)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, default=_json_default) + "\n")


def read_jsonl(path: Path, *, limit: Optional[int] = None, tail: bool = False) -> List[Dict[str, Any]]:
    """Read a JSONL file, tolerating truncated final lines (crash recovery)."""
    path = Path(path)
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return records
    if limit is not None:
        return records[-limit:] if tail else records[:limit]
    return records


def read_text_safe(path: Path, default: str = "") -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return default


def dir_size_bytes(path: Path) -> int:
    total = 0
    path = Path(path)
    if not path.exists():
        return 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total


def human_bytes(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}PB"


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive dict merge used for parameter resolution."""
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_dicts(out[key], value)
        else:
            out[key] = value
    return out


def iter_files(root: Path, suffixes: Iterable[str]) -> List[Path]:
    wanted = {s.lower() for s in suffixes}
    root = Path(root)
    if not root.exists():
        return []
    found = [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in wanted]
    return found

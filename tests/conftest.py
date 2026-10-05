"""Shared fixtures.

Tests never touch the user's real data root: every fixture builds a throwaway
directory under pytest's ``tmp_path`` and points the config at it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recon3d.config import load_config  # noqa: E402  (after sys.path tweak)


@pytest.fixture()
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "recon3d-data"
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture()
def config(data_root: Path):
    cfg = load_config(data_root=str(data_root))
    cfg.ensure_dirs()
    return cfg


@pytest.fixture(scope="session")
def reference_dir(tmp_path_factory) -> Path:
    """A small synthetic seven-view reference set (no ground-truth leakage)."""
    from recon3d.engine.reconstruction.dataset import DatasetSpec, generate_reference_set

    out = tmp_path_factory.mktemp("refs")
    spec = DatasetSpec(
        kind="robot",
        views=(("front", 0.0, 0.0), ("front_right", 45.0, 0.0), ("right", 90.0, 0.0),
               ("back", 180.0, 0.0), ("left", 270.0, 0.0), ("front_left", 315.0, 0.0),
               ("top", 0.0, 80.0)),
        resolution=192,
        noise=0.004,
        write_masks=False,
    )
    generate_reference_set(out, spec, subject=None)
    return out


@pytest.fixture()
def project(config, reference_dir: Path):
    from recon3d.core.project import ProjectManager

    manager = ProjectManager(config.projects_path)
    proj = manager.create("unit-test-mini", subject_type="robot")
    proj.add_images(sorted(reference_dir.glob("*.png")))
    return proj


@pytest.fixture(scope="session")
def synthetic_mesh():
    import trimesh

    return trimesh.creation.icosphere(subdivisions=2)


@pytest.fixture(scope="session")
def textured_mesh():
    """A small procedural subject: enough structure for carving/UV/texturing."""
    from recon3d.engine.reconstruction.shapes import make_test_subject

    return make_test_subject("robot")

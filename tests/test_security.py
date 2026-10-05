"""Security tests: sandboxing, path traversal, destructive-operation gates."""

from __future__ import annotations

import pytest

from recon3d.core.project import ProjectManager
from recon3d.core.security import (PathSandbox, require_permission, validate_export_path,
                                   validate_image_path)
from recon3d.errors import SecurityError, ValidationError


def test_sandbox_blocks_traversal(tmp_path):
    sandbox = PathSandbox([tmp_path])
    with pytest.raises(SecurityError):
        sandbox.check("../outside.txt")
    with pytest.raises(SecurityError):
        sandbox.check(tmp_path.parent / "also_outside.txt")
    inside = sandbox.check("nested/inside.txt")
    assert str(inside).startswith(str(tmp_path.resolve()))


def test_sandbox_keeps_originals_read_only(tmp_path):
    sandbox = PathSandbox([tmp_path])
    original = tmp_path / "input" / "original" / "ref.png"
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_bytes(b"x")
    with pytest.raises(SecurityError):
        sandbox.check(original, writable=True)


def test_export_path_blocks_escape(tmp_path):
    sandbox = PathSandbox([tmp_path])
    with pytest.raises(SecurityError):
        validate_export_path("/etc/passwd", "obj", sandbox=sandbox)
    with pytest.raises(SecurityError):
        validate_export_path("../../escape.glb", "glb", sandbox=sandbox)
    with pytest.raises(ValidationError):
        validate_export_path("model.exe", "exe", sandbox=sandbox)
    ok = validate_export_path("final/model.glb", "glb", sandbox=sandbox)
    assert str(ok).endswith("model.glb")


def test_image_path_validation(tmp_path):
    image = tmp_path / "ref.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    assert validate_image_path(image) == image
    with pytest.raises(ValidationError):
        validate_image_path(tmp_path / "missing.png")
    script = tmp_path / "evil.sh"
    script.write_text("#!/bin/sh\nrm -rf /\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        validate_image_path(script)
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    with pytest.raises(ValidationError):
        validate_image_path(empty)


def test_destructive_operations_need_confirmation(tmp_path):
    manager = ProjectManager(tmp_path / "projects")
    project = manager.create("doomed")
    with pytest.raises(SecurityError):
        manager.delete(project.id)
    manager.delete(project.id, confirm=True)
    assert not manager.exists(project.id)


def test_require_permission_helper():
    require_permission("delete_project", True)
    with pytest.raises(SecurityError):
        require_permission("delete_project", False)


def test_project_add_images_rejects_non_images(tmp_path):
    manager = ProjectManager(tmp_path / "projects")
    project = manager.create("images")
    script = tmp_path / "payload.sh"
    script.write_text("#!/bin/sh\nrm -rf /\n", encoding="utf-8")
    with pytest.raises((SecurityError, ValidationError)):
        project.add_images([script])


def test_model_manager_blocks_network_when_offline(tmp_path, monkeypatch):
    from recon3d.config import load_config
    from recon3d.errors import ModelError
    from recon3d.modelzoo.manager import ModelManager

    cfg = load_config(data_root=str(tmp_path / "data"))
    cfg.ensure_dirs()
    cfg.offline = True
    manager = ModelManager(cfg)
    with pytest.raises(ModelError):
        manager.download("depth_anything_v2_small")
    cfg.offline = False
    cfg.allow_network_downloads = False
    with pytest.raises(ModelError):
        manager.download("depth_anything_v2_small")


def test_model_manager_rejects_unknown_and_unsafe_ids(tmp_path):
    from recon3d.config import load_config
    from recon3d.errors import ModelError, SecurityError
    from recon3d.modelzoo.manager import ModelManager

    cfg = load_config(data_root=str(tmp_path / "data"))
    cfg.ensure_dirs()
    manager = ModelManager(cfg)
    with pytest.raises(ModelError):
        manager.entry("does_not_exist")
    with pytest.raises(SecurityError):
        manager._model_dir("../../etc/passwd")

"""Documentation/manifest drift guards.

`agent-manifest.json`, `recon3d info` and the docs are what an external agent reads
*before* it has any other information about the engine. If they drift away from the
code, an agent will call commands that do not exist or pass parameters the pipeline
ignores - so the manifest is checked against the real CLI parser, the real HTTP
routes and the real default parameters here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "agent-manifest.json"


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_is_valid_and_describes_this_version(manifest):
    import recon3d

    assert manifest["version"] == recon3d.VERSION
    assert manifest["name"] == "recon3d"
    assert manifest["license"], "the manifest must state the licence"
    repository = manifest["repository"]
    url = repository["url"] if isinstance(repository, dict) else repository
    assert url.startswith("https://github.com/")
    for key in ("summary", "install", "quickstart", "interfaces", "policy", "capabilities"):
        assert manifest.get(key), f"the manifest is missing '{key}'"


def test_manifest_cli_commands_match_the_real_parser(manifest):
    from recon3d.cli.main import build_parser

    parser = build_parser()
    # The sub-parser is the first positional group; argparse exposes it this way.
    actions = [action for action in parser._actions if action.dest == "command"]
    assert actions, "the CLI parser exposes no sub-commands"
    real = set(actions[0].choices or {})
    documented = set(manifest["interfaces"]["cli"]["commands"])
    assert real == documented, (
        f"agent-manifest.json documents {sorted(documented - real)} which the CLI does "
        f"not provide, and misses {sorted(real - documented)}")


def test_manifest_rest_endpoints_match_the_real_app(manifest):
    fastapi = pytest.importorskip("fastapi")
    from recon3d.api.server import create_app

    app = create_app()
    assert isinstance(app, fastapi.FastAPI)
    schema = app.openapi()
    real = {f"{method.upper()} {path}" for path, methods in schema["paths"].items()
            for method in methods}
    documented = {entry.strip() for entry in manifest["interfaces"]["rest"]["endpoints"]}
    # A handful of endpoints are documented without the {id}/{job} placeholder names
    # used by FastAPI, so normalise the parameters before comparing.
    def normalise(endpoint: str) -> str:
        method, _, path = endpoint.partition(" ")
        path = "/".join(("{id}" if part.startswith("{") else part) for part in path.split("/"))
        return f"{method} {path}"

    missing = {normalise(entry) for entry in documented} - {normalise(entry) for entry in real}
    assert not missing, f"agent-manifest.json documents non-existent endpoints: {sorted(missing)}"

    for endpoint in ("GET /health", "POST /v1/projects/{id}/reconstruct",
                     "POST /v1/jobs/{id}/cancel", "GET /v1/jobs/{id}/artifacts/{path}"):
        assert normalise(endpoint) in {normalise(entry) for entry in real}, endpoint


def test_capability_report_documents_every_pipeline_parameter(manifest):
    from recon3d.agent.manifest import capability_report
    from recon3d.core.pipeline import DEFAULT_PARAMS

    report = capability_report()
    documented = set(report["parameters"])
    documented.update(manifest["capabilities"]["parameters"])
    undocumented = {key for key in DEFAULT_PARAMS
                    if key not in documented and not key.startswith("_")}
    assert not undocumented, (
        f"these pipeline parameters are accepted but never documented for agents: "
        f"{sorted(undocumented)}")


def test_manifest_recovery_and_policy_claims_are_real(manifest):
    """The promises the manifest makes must exist in the CLI and in the code."""
    from recon3d.cli.main import build_parser

    actions = [action for action in build_parser()._actions if action.dest == "command"]
    commands = set(actions[0].choices or {})
    for command in ("jobs", "cancel", "retry"):
        assert command in commands, f"'{command}' is advertised for recovery but missing"

    recovery = manifest["capabilities"]["recovery"]
    assert {"checkpoints", "resume", "cancel", "retry"} <= set(recovery)

    from recon3d.core.pipeline import DEFAULT_PARAMS

    assert "stage_retries" in DEFAULT_PARAMS, "the manifest documents a retry budget"

    policy = manifest["policy"]
    assert policy["no_external_generative_ai"] is True
    assert policy["destructive_operations_require_confirmation"] is True
    # The engine really does refuse a delete without confirmation.
    import tempfile

    from recon3d.config import load_config
    from recon3d.core.project import ProjectManager
    from recon3d.errors import SecurityError

    root = Path(tempfile.mkdtemp())
    cfg = load_config(data_root=str(root / "data"))
    cfg.ensure_dirs()
    manager = ProjectManager(cfg.projects_path)
    manager.create("guarded", subject_type="robot")
    with pytest.raises(SecurityError):
        manager.delete("guarded")


def test_presets_are_documented_and_selectable():
    """Every shipped preset must be documented, selectable and resolvable.

    `fast` was added for iteration speed; if a preset exists as a file but not in
    `PRESET_DESCRIPTIONS`/the CLI choices, an agent reading the manifest cannot use it -
    exactly the drift this module exists to catch.
    """
    from recon3d.agent.manifest import PRESET_DESCRIPTIONS
    from recon3d.cli.main import build_parser
    from recon3d.core.pipeline import available_presets, resolve_params

    names = [preset["name"] for preset in available_presets()]
    assert set(names) == set(PRESET_DESCRIPTIONS), (
        f"presets without docs: {sorted(set(names) - set(PRESET_DESCRIPTIONS))}, "
        f"docs without presets: {sorted(set(PRESET_DESCRIPTIONS) - set(names))}")

    parser = build_parser()
    reconstruct = next(action for action in parser._actions
                       if action.dest == "command").choices["reconstruct"]
    for dest in ("preset", "quality"):
        choices = next(action for action in reconstruct._actions if action.dest == dest).choices
        assert set(names) <= set(choices), f"--{dest} cannot select every preset"

    for name in names:
        params = resolve_params({"quality": name})
        assert params["_preset"] == name, f"'{name}' resolves to {params['_preset']}"


def test_neural_backends_stay_optional():
    """The engine must never import onnxruntime/torch for its classical path."""
    import subprocess
    import sys

    code = (
        "import sys, recon3d.core.pipeline\n"
        "bad = [m for m in ('onnxruntime', 'torch') if m in sys.modules]\n"
        "print(','.join(bad))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(ROOT))
    assert out.returncode == 0, out.stderr[-800:]
    assert out.stdout.strip() == "", f"the pipeline imported neural backends: {out.stdout!r}"


def test_every_package_in_the_tree_is_packaged():
    """A directory that works from the checkout but is absent from the wheel is a trap.

    `recon3d/agent/` had no `__init__.py` and was not listed in pyproject, so
    `pip install recon3d-*.whl` produced an engine whose `recon3d info` crashed with
    ModuleNotFoundError while the source checkout worked. This guard walks the tree and
    fails when any package directory is missing from the packaging list.
    """
    import tomllib

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    listed = set(project["tool"]["setuptools"]["packages"])
    missing = []
    for path in sorted((ROOT / "recon3d").rglob("*.py")):
        package = ".".join(path.parent.relative_to(ROOT).parts)
        if package not in listed:
            missing.append(package)
        if not (path.parent / "__init__.py").exists():
            missing.append(f"{package} (no __init__.py)")
    assert not missing, f"these packages would be missing from the wheel: {sorted(set(missing))}"

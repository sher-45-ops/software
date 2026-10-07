"""Agent-facing surface: the machine-readable capability manifest.

``recon3d.agent.manifest`` is what ``recon3d info --json`` prints and what
``agent-manifest.json`` mirrors, so an agent that has never seen the project can
discover the stages, parameters, presets, interfaces and policies in one call.

This module lives in the wheel: it used to be an implicit namespace package that
worked from a source checkout but was missing from installed distributions, which
broke ``recon3d info`` for anyone who installed the wheel.  The explicit
``__init__.py`` plus the ``recon3d.agent`` entry in ``pyproject.toml`` fixes that,
and ``tests/test_manifest.py`` guards the package list against the tree.
"""

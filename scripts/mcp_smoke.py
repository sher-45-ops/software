#!/usr/bin/env python3
"""Exercise the MCP server in-process: list tools, call the read-only ones.

Running this proves the MCP surface works without needing a client, which is useful in CI
and when an agent platform has not been configured yet.

::

    python scripts/mcp_smoke.py
    python scripts/mcp_smoke.py --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _text(result) -> str:
    """Normalise FastMCP/MCPServer tool results to text."""
    if isinstance(result, tuple):
        result = result[0]
    content = getattr(result, "content", None) or result
    first = content[0]
    return getattr(first, "text", str(first))


async def _run(include_project: bool) -> dict:
    from recon3d.mcpserver.server import TOOL_NAMES, build_server

    server = build_server()
    tools = await server.list_tools()
    report: dict = {"tools": [tool.name for tool in tools],
                    "expected": list(TOOL_NAMES), "checks": []}

    doctor = json.loads(_text(await server.call_tool("recon3d_doctor", {})))
    report["checks"].append({"name": "recon3d_doctor", "ok": bool(doctor.get("ok")),
                             "device": doctor.get("hardware", {}).get("device"),
                             "version": doctor.get("version")})

    if include_project:
        created = json.loads(_text(await server.call_tool(
            "recon3d_create_project", {"name": "mcp-smoke", "subject_type": "object"})))
        report["checks"].append({"name": "recon3d_create_project",
                                 "ok": bool(created.get("project")),
                                 "project": created.get("project")})
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--project", action="store_true",
                        help="also create a throwaway project (writes to the data root)")
    args = parser.parse_args(argv)

    try:
        report = asyncio.run(_run(args.project))
    except RuntimeError as exc:
        print(f"MCP unavailable: {exc}")
        print('Install the extra with: pip install "recon3d[mcp]"')
        return 2

    report["ok"] = (report["tools"] == report["expected"]
                    and all(check["ok"] for check in report["checks"]))
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"tools ({len(report['tools'])}): {', '.join(report['tools'])}")
        for check in report["checks"]:
            print(f"  [{'ok' if check['ok'] else 'FAIL'}] {check['name']}: "
                  f"{ {k: v for k, v in check.items() if k not in ('name', 'ok')} }")
        print(f"RESULT: {'PASS' if report['ok'] else 'FAIL'}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

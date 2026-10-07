# MCP server

Recon3D ships a [Model Context Protocol](https://modelcontextprotocol.io) server so an
agent (Claude Code, Codex, Arena, or any MCP client) can drive reconstructions as native
tool calls — **no shell access required**.

```bash
python -m recon3d.mcpserver            # stdio transport (what MCP clients launch)
python -m recon3d.mcpserver --json     # print the tool list and exit (for CI)
```

Requires the `mcp` extra: `pip install "recon3d[mcp]"`. Both SDK generations are
supported (`mcp.server.fastmcp.FastMCP` and the 2.x `mcp.server.mcpserver.MCPServer`).

## Client configuration

Claude Code / Claude Desktop (`claude_desktop_config.json` or `.mcp.json`):

```json
{
  "mcpServers": {
    "recon3d": {
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "recon3d.mcpserver"],
      "env": { "RECON3D_HOME": "/path/to/recon3d-data" }
    }
  }
}
```

Windows: `"command": "C:\\path\\to\\.venv\\Scripts\\python.exe"`.

Codex / generic stdio clients: command `python`, args `["-m", "recon3d.mcpserver"]`.
HTTP transports (`--transport sse`, `--transport streamable-http`) are available for
clients that prefer them.

Point `RECON3D_HOME` at the data root you want the agent to work in. Everything the server
writes stays inside it.

## Tools

### `recon3d_doctor()`

Hardware, dependency and backend diagnosis plus the resolved data root. Call this first:
it tells you whether optional accelerators are present and what performance mode the
machine can support.

```json
{"version":"1.1.0","ok":true,"data_root":"/home/user/.recon3d","offline":false,
 "hardware":{"cpu":"…","cpu_count":2,"ram_total_mb":3939,"device":"cpu","ram_available_mb":3600},
 "backends":{…}}
```

### `recon3d_create_project(name, image_paths=None, subject_type="auto", style="realistic")`

Creates a project and registers reference images by absolute path. Returns the project id
used by the other tools. Images are copied into the project (your originals are never
modified), validated, and rejected if they are not readable images of a supported type.

```json
{"project":"goblin","path":"/home/user/.recon3d/projects/goblin",
 "images_added":8,"total_images":8,"subject_type":"character"}
```

### `recon3d_reconstruct(project, preset="standard", target_polycount=None, texture_resolution=None, export_formats=None, generate_rig=False, generate_lods=True, units="normalized", subject_height_m=None, stages=None, timeout_s=3600)`

Runs the full pipeline and blocks until it finishes, returning quality, statistics and
artefact paths. Long runs are bounded by `timeout_s`; on timeout the job keeps running and
`recon3d_job_status` reports it.

```json
{"job":"j-…","state":"completed","progress":100.0,"duration_s":186.4,
 "quality":{"overall":72.4,"grade":"fair","metrics":{"mean_silhouette_iou":0.71,…},
            "missing_regions":["back-top"],"warnings":["…"]},
 "statistics":{"triangles":24000,"watertight":true,"lod_levels":2,"texture_resolution":2048},
 "outputs":{"version_dir":"…/versions/v001","mesh":{"glb":"…","obj":"…"},"textures":{…}}}
```

Guidance for agents: start with `preset="draft"` to validate the reference set (it is fast
and its quality numbers already tell you whether the images are usable), then re-run with
`preset="standard"` or `"high"` for the asset you keep.

### `recon3d_job_status(job_id, include_stages=True)`

State, progress, per-stage status, recent events and — when finished — the measured quality
and every warning. Use it to follow a job started elsewhere (CLI or REST) or to inspect a
finished one.

### `recon3d_list_outputs(job_id)`

The complete artefact index: absolute paths, relative paths and sizes for the mesh files,
textures, LODs, rig, previews and JSON reports, plus the quality block. Feed the relevant
paths straight into whatever consumes the asset next.

## Behaviour and safety guarantees

- **No shell execution.** The server never runs commands; all work is in-process API calls.
- **Whitelisted tools.** Five tools, each with a typed schema; there is no "run arbitrary
  code" escape hatch.
- **Sandboxed writes.** Projects live under the data root; export paths are validated and
  cannot escape it.
- **Explicit consent for destructive work.** Deleting a project is not exposed as a tool at
  all — it must be done from the CLI/API with `--yes` / `confirm=true`.
- **No network by default.** The tools only touch the network if you explicitly download an
  optional model; offline mode blocks even that.
- **Honest results.** A degraded or failed stage is reported in the result (`warnings`,
  `quality.missing_regions`, `stages_failed`), never hidden.

## Verified locally

```bash
$ python -m recon3d.mcpserver --json
{"server": "recon3d", "version": "1.1.0",
 "tools": ["recon3d_doctor", "recon3d_create_project", "recon3d_reconstruct",
           "recon3d_job_status", "recon3d_list_outputs"]}
```

`scripts/mcp_smoke.py` exercises listing and calling the tools in-process (no client
required), which is also what the test suite covers.

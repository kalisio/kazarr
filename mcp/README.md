# Kazarr MCP Server

An [MCP (Model Context Protocol)](https://modelcontextprotocol.io) server that exposes the **Kazarr** Zarr dataset service to Claude and other MCP-compatible clients.

Kazarr is a lightweight FastAPI service for accessing multi-dimensional Zarr datasets stored in S3. This MCP server wraps every Kazarr endpoint as a tool so you can explore datasets, extract spatial/temporal subsets, probe point values, compute isolines and retrieve mesh representations — all from a natural-language conversation.

---

## Prerequisites

- Node.js >= 18
- A running Kazarr instance (see the [Kazarr documentation](../README.md))
- [Claude Desktop](https://claude.ai/download) or another MCP-compatible client

---

## Installation

```bash
cd mcp
npm install
```

---

## Configuration

| Variable       | Default                  | Description                                                                                          |
|----------------|--------------------------|------------------------------------------------------------------------------------------------------|
| `KAZARR_URL`   | `http://localhost:8000`  | Base URL of the Kazarr service                                                                       |
| `KAZARR_JWT`   | *(unset)*                | JWT sent as `Authorization: Bearer <token>` on every request. Required only when Kazarr is deployed behind an authenticating proxy. |
| `MCP_PORT`     | *(unset)*                | When set, starts an HTTP server on this port (Streamable HTTP transport) instead of stdio. Use for remote deployments. |

---

## Transport modes

### Stdio (default — local use)

The server runs as a subprocess and communicates over standard I/O. This is the standard mode for Claude Desktop and for `claude mcp add` without a URL.

### HTTP / Streamable HTTP (remote deployments)

When `MCP_PORT` is set, the server starts an HTTP server and exposes the MCP endpoint at `POST /mcp` using the [MCP Streamable HTTP transport](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports#streamable-http). This allows Claude Code to connect to a Kazarr MCP server deployed on a remote host using a URL.

```bash
MCP_PORT=3002 KAZARR_URL=https://my-kazarr-server.example.com KAZARR_JWT=your.jwt.token node server.js
```

Then add it to Claude Code:

```bash
claude mcp add kazarr --transport http http://localhost:3002/mcp
# or for a remote deployment:
claude mcp add kazarr --transport http https://my-mcp-host.example.com/mcp
```

> **Note:** When deploying behind a reverse proxy (nginx, Traefik…), ensure the `/mcp` path is forwarded and that long-lived POST connections are not cut off by proxy timeouts.

---

## Usage with Claude Desktop

Add the server to your `claude_desktop_config.json`
(usually `~/.config/Claude/claude_desktop_config.json` on Linux,
`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "kazarr": {
      "command": "node",
      "args": ["/absolute/path/to/kazarr/mcp/server.js"],
      "env": {
        "KAZARR_URL": "http://localhost:8000",
        "KAZARR_JWT": "your.jwt.token"
      }
    }
  }
}
```

Replace `/absolute/path/to/kazarr` with the actual path to this repository.
Restart Claude Desktop after saving the file.

### Pointing at a remote (or authenticated) instance

```json
"env": {
  "KAZARR_URL": "https://my-kazarr-server.example.com",
  "KAZARR_JWT": "your.jwt.token"
}
```

`KAZARR_JWT` is optional — omit it for unauthenticated deployments.

---

## Usage with Claude Code (CLI)

### Local subprocess (stdio)

```bash
claude mcp add kazarr -- node /absolute/path/to/kazarr/mcp/server.js
```

With a custom URL and JWT:

```bash
claude mcp add kazarr \
  -e KAZARR_URL=https://my-kazarr-server.example.com \
  -e KAZARR_JWT=your.jwt.token \
  -- node /absolute/path/to/kazarr/mcp/server.js
```

### Remote server (HTTP transport)

If the MCP server is already running remotely (started with `MCP_PORT`):

```bash
claude mcp add kazarr --transport http https://my-mcp-host.example.com/mcp
```

---

## Available Tools

### `healthcheck`
Check that the Kazarr service is reachable. Returns `{"status": "ok"}` when healthy.

---

### `get_info`
Return basic information about the Kazarr API: name, version, description and the list of available endpoints.

---

### `list_datasets`
List all Zarr datasets available in the service. Returns an array of dataset identifiers (paths).

| Parameter     | Type   | Required | Description                                                   |
|---------------|--------|----------|---------------------------------------------------------------|
| `search_path` | string | no       | Path prefix to narrow the search, e.g. `"gfs"` or `"era5/wind"` |

---

### `get_dataset_metadata`
Return metadata for a specific Zarr dataset: available variables, coordinate dimensions, spatial bounding box, vertical axis, time bounds and raw Zarr attributes.

| Parameter | Type   | Required | Description                                                              |
|-----------|--------|----------|--------------------------------------------------------------------------|
| `dataset` | string | yes      | Dataset identifier, e.g. `"gfs/wind"`. Forward slashes are allowed for nested paths. |

---

### `extract_data`
Extract a spatial/temporal subset of a Zarr dataset for a given variable. Supports bounding box, vertical range, time selection, resolution limiting, interpolation and multiple output formats.

| Parameter              | Type                              | Required | Description |
|------------------------|-----------------------------------|----------|-------------|
| `dataset`              | string                            | yes      | Dataset identifier |
| `variable`             | string                            | yes      | Variable name, e.g. `"u"` or `"temperature"` |
| `lon_min`              | number                            | no       | Western edge of bounding box (degrees) |
| `lat_min`              | number                            | no       | Southern edge of bounding box (degrees) |
| `lon_max`              | number                            | no       | Eastern edge of bounding box (degrees) |
| `lat_max`              | number                            | no       | Northern edge of bounding box (degrees) |
| `level_min`            | number                            | no       | Minimum vertical coordinate |
| `level_max`            | number                            | no       | Maximum vertical coordinate |
| `level`                | number                            | no       | Single vertical level to extract |
| `time`                 | string                            | no       | ISO 8601 time instant or interval |
| `interp_time`          | boolean                           | no       | Interpolate along the time axis |
| `resolution_limit`     | number                            | no       | Maximum spatial resolution factor (>1 downsamples) |
| `format`               | `raw`\|`geojson`\|`mesh`          | no       | Output format (default: `raw`) |
| `mesh_tile_size`       | integer                           | no       | [format=mesh] Mesh tile size in grid cells |
| `mesh_data_mapping`    | `vertices`\|`cells`               | no       | [format=mesh] Data value association |
| `is_3d`                | boolean                           | no       | Full 3D volume extraction |
| `interp_vars`          | string                            | no       | Comma-separated coordinate variables to interpolate |
| `interp_vars_method`   | string                            | no       | Interpolation method for coordinate variables |
| `interp_vars_params`   | string                            | no       | Extra params for variable interpolation |
| `interp_spatial_method`| string                            | no       | Spatial interpolation method |
| `interp_spatial_params`| string                            | no       | Extra params for spatial interpolation |
| `as_dims`              | string                            | no       | Comma-separated dimension variable names |

---

### `probe_point`
Query the value of one or more variables at a single geographic coordinate.

| Parameter              | Type              | Required | Description |
|------------------------|-------------------|----------|-------------|
| `dataset`              | string            | yes      | Dataset identifier |
| `variables`            | string            | yes      | Comma-separated variable names, e.g. `"u,v"` |
| `lon`                  | number            | yes      | Longitude of the probe point (degrees) |
| `lat`                  | number            | yes      | Latitude of the probe point (degrees) |
| `level`                | number            | no       | Vertical level coordinate for 3D datasets |
| `time`                 | string            | no       | ISO 8601 time instant or interval |
| `interp_time`          | boolean           | no       | Interpolate along the time axis |
| `format`               | `raw`\|`geojson`  | no       | Output format |
| `interp_vars`          | string            | no       | Comma-separated coordinate variables to interpolate |
| `interp_vars_method`   | string            | no       | Variable interpolation method |
| `interp_vars_params`   | string            | no       | Variable interpolation extra params |
| `interp_spatial_method`| string            | no       | Spatial interpolation method |
| `interp_spatial_params`| string            | no       | Spatial interpolation extra params |
| `as_dims`              | string            | no       | Comma-separated dimension variable names |

---

### `probe_points`
Query one or more variables at multiple geographic coordinates in a single request (batch probe). More efficient than calling `probe_point` repeatedly.

| Parameter              | Type              | Required | Description |
|------------------------|-------------------|----------|-------------|
| `dataset`              | string            | yes      | Dataset identifier |
| `variables`            | string            | yes      | Comma-separated variable names |
| `points`               | string            | yes      | JSON array of `{lon, lat, level?}` objects, e.g. `[{"lon": 2.35, "lat": 48.85}]` |
| `time`                 | string            | no       | ISO 8601 time instant or interval |
| `interp_time`          | boolean           | no       | Interpolate along the time axis |
| `format`               | `raw`\|`geojson`  | no       | Output format |
| `interp_vars`          | string            | no       | Comma-separated coordinate variables to interpolate |
| `interp_vars_method`   | string            | no       | Variable interpolation method |
| `interp_vars_params`   | string            | no       | Variable interpolation extra params |
| `interp_spatial_method`| string            | no       | Spatial interpolation method |
| `interp_spatial_params`| string            | no       | Spatial interpolation extra params |
| `as_dims`              | string            | no       | Comma-separated dimension variable names |

> **Note:** `probe_points` sends the points as a POST body `{"points": [...]}` to the Kazarr `/probes` endpoint.

---

### `get_isoline`
Compute contour lines (isolines) for a variable at specified threshold values. Returns a GeoJSON FeatureCollection of LineString features, each with a `"level"` property.

| Parameter           | Type              | Required | Description |
|---------------------|-------------------|----------|-------------|
| `dataset`           | string            | yes      | Dataset identifier |
| `variable`          | string            | yes      | Variable to compute isolines for, e.g. `"temperature"` |
| `levels`            | string            | yes      | Comma-separated threshold values, e.g. `"0,5,10,15,20"` |
| `time`              | string            | no       | ISO 8601 time instant or interval |
| `interp_time`       | boolean           | no       | Interpolate along the time axis |
| `format`            | `raw`\|`geojson`  | no       | Output format (default: `geojson`) |
| `interp_vars`       | string            | no       | Comma-separated coordinate variables to interpolate |
| `interp_vars_method`| string            | no       | Variable interpolation method |
| `interp_vars_params`| string            | no       | Variable interpolation extra params |
| `as_dims`           | string            | no       | Comma-separated dimension variable names |

---

### `select_data`
Free-form dimension selection on a dataset variable. Returns the raw multi-dimensional array without spatial subsetting. Useful for exploring the full range of a variable.

| Parameter           | Type   | Required | Description |
|---------------------|--------|----------|-------------|
| `dataset`           | string | yes      | Dataset identifier |
| `variable`          | string | yes      | Variable to select |
| `time`              | string | no       | ISO 8601 time instant or interval |
| `interp_time`       | boolean| no       | Interpolate along the time axis |
| `interp_vars`       | string | no       | Comma-separated coordinate variables to interpolate |
| `interp_vars_method`| string | no       | Variable interpolation method |
| `interp_vars_params`| string | no       | Variable interpolation extra params |
| `as_dims`           | string | no       | Comma-separated dimension variable names |

---

### `get_mesh`
Retrieve the support mesh (geometry) of a dataset. Returns the triangulated mesh structure that underlies the dataset grid, optionally including per-vertex data values.

| Parameter           | Type                  | Required | Description |
|---------------------|-----------------------|----------|-------------|
| `dataset`           | string                | yes      | Dataset identifier |
| `format`            | `mesh`\|`geojson`     | no       | Output format (default: `mesh`) |
| `mesh_data_mapping` | `vertices`\|`cells`   | no       | Data value association |
| `is_3d`             | boolean               | no       | Generate a 3D volumetric mesh |
| `variable`          | string                | no       | Variable to use as mesh geometry basis |
| `level_variable`    | string                | no       | Variable to use as the level (vertical) coordinate |

---

## Example conversation

Once the MCP server is connected, you can ask Claude things like:

> *"List all available datasets in Kazarr."*

> *"Show me the metadata for the `gfs/wind` dataset — what variables and time range does it cover?"*

> *"Extract the `u` wind component for the region between -5°W, 41°N and 9.5°E, 51°N at 2024-01-15T12:00:00Z."*

> *"Probe the wind speed at Paris (lon=2.35, lat=48.85) for variables u and v."*

> *"Compute isolines for temperature at 0, 5, 10, 15 and 20°C from the `era5/temperature` dataset."*

> *"Query the wind values at multiple stations: Paris (2.35, 48.85), Lyon (4.83, 45.75) and Marseille (5.37, 43.30)."*

---

## Running the server manually (for testing)

```bash
# Stdio mode (unauthenticated)
KAZARR_URL=http://localhost:8000 node server.js

# Stdio mode (with JWT)
KAZARR_URL=https://my-kazarr-server.example.com KAZARR_JWT=your.jwt.token node server.js

# HTTP mode on port 3002
MCP_PORT=3002 KAZARR_URL=http://localhost:8000 node server.js
```

In stdio mode the server communicates over stdin/stdout (standard MCP transport) and opens no HTTP port.
In HTTP mode the server listens on `0.0.0.0:<MCP_PORT>` and exposes `POST /mcp`.

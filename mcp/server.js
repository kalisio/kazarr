#!/usr/bin/env node
/**
 * MCP Server for Kazarr
 *
 * Exposes Kazarr's Zarr dataset service endpoints as MCP tools so that
 * Claude (or any MCP-compatible client) can explore datasets, extract
 * spatial/temporal subsets, probe point values, compute isolines, and
 * retrieve mesh representations.
 *
 * Configuration (environment variables):
 *   KAZARR_URL  – Base URL of the Kazarr service (default: http://localhost:8000)
 *   KAZARR_JWT  – JWT sent as Authorization: Bearer <token> on every request (optional)
 *   MCP_PORT    – When set, starts an HTTP server on this port instead of stdio.
 *                 Exposes the MCP endpoint at POST /mcp (Streamable HTTP transport).
 *                 Use this for remote deployments reachable via URL.
 */

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js'
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js'
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js'
import { createServer } from 'node:http'
import { fileURLToPath } from 'node:url'
import { z } from 'zod'

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const MCP_PORT = process.env.MCP_PORT ? parseInt(process.env.MCP_PORT, 10) : null

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

/**
 * Split a comma-separated string into a trimmed, non-empty string array.
 * Returns an empty array for falsy input.
 */
function splitList (value) {
  if (!value) return []
  return value.split(',').map(v => v.trim()).filter(Boolean)
}

function makeApiFetch (baseUrl, jwt) {
  return async function apiFetch (path, params = {}, body = null) {
    const url = new URL(`${baseUrl}${path}`)

    Object.entries(params).forEach(([k, v]) => {
      if (v === undefined || v === null || v === '') return
      // Arrays map to repeated query parameters: ?k=a&k=b
      if (Array.isArray(v)) {
        v.forEach(item => url.searchParams.append(k, String(item)))
      } else {
        url.searchParams.set(k, String(v))
      }
    })

    const headers = { Accept: 'application/json' }
    if (jwt) headers.Authorization = `Bearer ${jwt}`
    if (body !== null) headers['Content-Type'] = 'application/json'

    const response = await fetch(url.toString(), {
      method: body !== null ? 'POST' : 'GET',
      headers,
      body: body !== null ? JSON.stringify(body) : undefined
    })

    const text = await response.text()
    if (!response.ok) {
      throw new Error(`HTTP ${response.status} ${response.statusText} — ${text.slice(0, 300)}`)
    }

    try {
      return JSON.parse(text)
    } catch {
      return text
    }
  }
}

function ok (data) {
  return {
    content: [{
      type: 'text',
      text: typeof data === 'string' ? data : JSON.stringify(data, null, 2)
    }]
  }
}

function err (error) {
  return {
    isError: true,
    content: [{ type: 'text', text: String(error) }]
  }
}

// ---------------------------------------------------------------------------
// Server factory
// ---------------------------------------------------------------------------

export function buildServer (options = {}) {
  const baseUrl = (options.url || process.env.KAZARR_URL || 'http://localhost:8000').replace(/\/$/, '')
  const jwt = options.jwt !== undefined ? options.jwt : (process.env.KAZARR_JWT || null)
  const apiFetch = makeApiFetch(baseUrl, jwt)

  const server = new McpServer({
    name: 'kazarr',
    version: '1.0.0'
  })

  // -------------------------------------------------------------------------
  // Tool: healthcheck
  // -------------------------------------------------------------------------

  server.tool(
    'healthcheck',
    'Check whether the Kazarr service is up. Returns {"status": "ok"} when healthy.',
    {},
    async () => {
      try {
        return ok(await apiFetch('/health'))
      } catch (e) {
        return err(e)
      }
    }
  )

  // -------------------------------------------------------------------------
  // Tool: get_info
  // -------------------------------------------------------------------------

  server.tool(
    'get_info',
    'Return basic information about the Kazarr API: name, version, description and the list of available endpoints.',
    {},
    async () => {
      try {
        return ok(await apiFetch('/'))
      } catch (e) {
        return err(e)
      }
    }
  )

  // -------------------------------------------------------------------------
  // Tool: list_datasets
  // -------------------------------------------------------------------------

  server.tool(
    'list_datasets',
    'List all Zarr datasets available in the service. ' +
    'Returns an array of dataset identifiers (paths) that can be used with the other tools.',
    {
      search_path: z.string().optional().describe(
        'Optional path prefix to narrow the search, e.g. "gfs" or "era5/wind". ' +
        'Returned ids are always relative to the service root.'
      )
    },
    async ({ search_path }) => {
      try {
        return ok(await apiFetch('/datasets', { search_path }))
      } catch (e) {
        return err(e)
      }
    }
  )

  // -------------------------------------------------------------------------
  // Tool: get_dataset_metadata
  // -------------------------------------------------------------------------

  server.tool(
    'get_dataset_metadata',
    'Return metadata for a specific Zarr dataset: available variables, coordinate dimensions, ' +
    'spatial bounding box, vertical axis, time bounds and raw Zarr attributes.',
    {
      dataset: z.string().describe(
        'Dataset identifier as returned by list_datasets, e.g. "gfs/wind" or "era5/temperature". ' +
        'Forward slashes are allowed for nested paths.'
      )
    },
    async ({ dataset }) => {
      try {
        return ok(await apiFetch(`/datasets/${dataset}/metadata`))
      } catch (e) {
        return err(e)
      }
    }
  )

  // -------------------------------------------------------------------------
  // Tool: extract_data
  // -------------------------------------------------------------------------

  server.tool(
    'extract_data',
    'Extract a spatial/temporal subset of a Zarr dataset for a given variable. ' +
    'Supports bounding box, vertical range, time selection, resolution limiting, ' +
    'variable/spatial interpolation and multiple output formats (raw, GeoJSON, mesh).',
    {
      dataset: z.string().describe('Dataset identifier, e.g. "gfs/wind".'),
      variable: z.string().describe('Name of the variable to extract, e.g. "u" or "temperature".'),
      lon_min: z.number().optional().describe('Western edge of the bounding box (degrees).'),
      lat_min: z.number().optional().describe('Southern edge of the bounding box (degrees).'),
      lon_max: z.number().optional().describe('Eastern edge of the bounding box (degrees).'),
      lat_max: z.number().optional().describe('Northern edge of the bounding box (degrees).'),
      level_min: z.number().optional().describe('Minimum vertical coordinate (altitude or depth).'),
      level_max: z.number().optional().describe('Maximum vertical coordinate.'),
      level: z.number().optional().describe(
        'Single vertical level to extract. Use level_min/level_max for a range.'
      ),
      time: z.string().optional().describe(
        'Time instant or interval for selection. ISO 8601 format, e.g. "2024-01-15T12:00:00Z".'
      ),
      interp_time: z.boolean().optional().describe(
        'If true, interpolate along the time axis to the requested instant instead of snapping to the nearest step.'
      ),
      resolution_limit: z.number().optional().describe(
        'Maximum spatial resolution factor. Values > 1 downsample the data before returning.'
      ),
      format: z.enum(['raw', 'geojson', 'mesh']).optional().describe(
        '"raw" (default): multi-dimensional arrays. "geojson": GeoJSON FeatureCollection. "mesh": triangle mesh.'
      ),
      mesh_tile_size: z.number().int().optional().describe(
        '[format=mesh] Size (in grid cells) of each mesh tile.'
      ),
      mesh_data_mapping: z.enum(['vertices', 'cells']).optional().describe(
        '[format=mesh] Whether data values are associated with mesh vertices or cell centres.'
      ),
      is_3d: z.boolean().optional().describe(
        'If true, perform a full 3D volume extraction. When false and the dataset is 3D, a vertical coordinate must be provided.'
      ),
      interp_vars: z.string().optional().describe(
        'Comma-separated list of coordinate variables to interpolate to the requested time/level, ' +
        'e.g. "u,v". Useful for datasets whose variables change with time.'
      ),
      interp_vars_method: z.enum([
        'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
        'quintic', 'polynomial', 'pchip', 'barycentric', 'krogh', 'akima', 'makima'
      ]).optional().describe('Interpolation method for coordinate variable interpolation (default: nearest).'),
      interp_vars_params: z.string().optional().describe(
        'Extra parameters for variable interpolation, colon-separated key-value pairs, ' +
        'e.g. "optparam1:VALUE1,optparam2:VALUE2".'
      ),
      interp_spatial_method: z.enum(['nearest', 'linear', 'cubic', 'idw', 'rbf']).optional().describe(
        'Spatial interpolation method applied after extraction (default: nearest).'
      ),
      interp_spatial_params: z.string().optional().describe(
        'Extra parameters for spatial interpolation, e.g. "padding:1.5".'
      ),
      as_dims: z.string().optional().describe(
        'Comma-separated list of variable names that share a name with a dimension and ' +
        'should be treated as dimension coordinates rather than data variables.'
      )
    },
    async ({
      dataset, variable, lon_min, lat_min, lon_max, lat_max,
      level_min, level_max, level, time, interp_time, resolution_limit,
      format, mesh_tile_size, mesh_data_mapping, is_3d,
      interp_vars, interp_vars_method, interp_vars_params,
      interp_spatial_method, interp_spatial_params, as_dims
    }) => {
      try {
        const params = {
          variable,
          lon_min, lat_min, lon_max, lat_max,
          level_min, level_max, level,
          time,
          interp_time: interp_time ? 'true' : undefined,
          resolution_limit,
          format,
          mesh_tile_size,
          mesh_data_mapping,
          is_3d: is_3d ? 'true' : undefined,
          interp_vars: splitList(interp_vars),
          interp_vars_method,
          interp_vars_params,
          interp_spatial_method,
          interp_spatial_params,
          as_dims: splitList(as_dims)
        }
        return ok(await apiFetch(`/datasets/${dataset}/extract`, params))
      } catch (e) {
        return err(e)
      }
    }
  )

  // -------------------------------------------------------------------------
  // Tool: probe_point
  // -------------------------------------------------------------------------

  server.tool(
    'probe_point',
    'Query the value of one or more variables at a single geographic coordinate. ' +
    'Returns values for every time step in the dataset or at a specific time.',
    {
      dataset: z.string().describe('Dataset identifier.'),
      variables: z.string().describe(
        'Comma-separated list of variables to probe, e.g. "u,v" or "temperature".'
      ),
      lon: z.number().describe('Longitude of the probe point (degrees).'),
      lat: z.number().describe('Latitude of the probe point (degrees).'),
      level: z.number().optional().describe('Vertical level coordinate for 3D datasets.'),
      time: z.string().optional().describe('ISO 8601 time instant or interval.'),
      interp_time: z.boolean().optional().describe('Interpolate along the time axis.'),
      format: z.enum(['raw', 'geojson']).optional().describe(
        '"raw" (default): structured arrays. "geojson": GeoJSON Feature with values as properties.'
      ),
      interp_vars: z.string().optional().describe(
        'Comma-separated coordinate variables to interpolate.'
      ),
      interp_vars_method: z.enum([
        'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
        'quintic', 'polynomial', 'pchip', 'barycentric', 'krogh', 'akima', 'makima'
      ]).optional().describe('Variable interpolation method.'),
      interp_vars_params: z.string().optional().describe('Variable interpolation extra params.'),
      interp_spatial_method: z.enum(['nearest', 'linear', 'cubic', 'idw', 'rbf']).optional().describe(
        'Spatial interpolation method used to compute the value at the exact probe location.'
      ),
      interp_spatial_params: z.string().optional().describe('Spatial interpolation extra params.'),
      as_dims: z.string().optional().describe('Comma-separated dimension variable names.')
    },
    async ({
      dataset, variables, lon, lat, level, time, interp_time, format,
      interp_vars, interp_vars_method, interp_vars_params,
      interp_spatial_method, interp_spatial_params, as_dims
    }) => {
      try {
        const params = {
          variables: splitList(variables),
          lon, lat, level,
          time,
          interp_time: interp_time ? 'true' : undefined,
          format,
          interp_vars: splitList(interp_vars),
          interp_vars_method,
          interp_vars_params,
          interp_spatial_method,
          interp_spatial_params,
          as_dims: splitList(as_dims)
        }
        return ok(await apiFetch(`/datasets/${dataset}/probe`, params))
      } catch (e) {
        return err(e)
      }
    }
  )

  // -------------------------------------------------------------------------
  // Tool: probe_points
  // -------------------------------------------------------------------------

  server.tool(
    'probe_points',
    'Query one or more variables at multiple geographic coordinates in a single request (batch probe). ' +
    'More efficient than calling probe_point repeatedly.',
    {
      dataset: z.string().describe('Dataset identifier.'),
      variables: z.string().describe('Comma-separated list of variables to probe, e.g. "u,v".'),
      points: z.string().describe(
        'JSON array of probe points. Each point must have "lon" and "lat" (numbers) and optionally "level". ' +
        'Example: [{"lon": 2.35, "lat": 48.85}, {"lon": 5.37, "lat": 43.30, "level": 500}]'
      ),
      time: z.string().optional().describe('ISO 8601 time instant or interval.'),
      interp_time: z.boolean().optional().describe('Interpolate along the time axis.'),
      format: z.enum(['raw', 'geojson']).optional().describe('"raw" or "geojson".'),
      interp_vars: z.string().optional().describe('Comma-separated coordinate variables to interpolate.'),
      interp_vars_method: z.enum([
        'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
        'quintic', 'polynomial', 'pchip', 'barycentric', 'krogh', 'akima', 'makima'
      ]).optional().describe('Variable interpolation method.'),
      interp_vars_params: z.string().optional().describe('Variable interpolation extra params.'),
      interp_spatial_method: z.enum(['nearest', 'linear', 'cubic', 'idw', 'rbf']).optional().describe(
        'Spatial interpolation method.'
      ),
      interp_spatial_params: z.string().optional().describe('Spatial interpolation extra params.'),
      as_dims: z.string().optional().describe('Comma-separated dimension variable names.')
    },
    async ({
      dataset, variables, points, time, interp_time, format,
      interp_vars, interp_vars_method, interp_vars_params,
      interp_spatial_method, interp_spatial_params, as_dims
    }) => {
      try {
        let parsedPoints
        try {
          parsedPoints = JSON.parse(points)
        } catch {
          return err(new Error('points must be a valid JSON array of {lon, lat, level?} objects'))
        }
        if (!Array.isArray(parsedPoints)) {
          return err(new Error('points must be a JSON array'))
        }

        const params = {
          variables: splitList(variables),
          time,
          interp_time: interp_time ? 'true' : undefined,
          format,
          interp_vars: splitList(interp_vars),
          interp_vars_method,
          interp_vars_params,
          interp_spatial_method,
          interp_spatial_params,
          as_dims: splitList(as_dims)
        }
        const body = { points: parsedPoints }
        return ok(await apiFetch(`/datasets/${dataset}/probes`, params, body))
      } catch (e) {
        return err(e)
      }
    }
  )

  // -------------------------------------------------------------------------
  // Tool: get_isoline
  // -------------------------------------------------------------------------

  server.tool(
    'get_isoline',
    'Compute contour lines (isolines) for a variable at specified threshold values. ' +
    'Returns a GeoJSON FeatureCollection of LineString features, each with a "level" property.',
    {
      dataset: z.string().describe('Dataset identifier.'),
      variable: z.string().describe('Variable for which to compute isolines, e.g. "temperature".'),
      levels: z.string().describe(
        'Comma-separated list of isoline threshold values, e.g. "0,5,10,15,20". ' +
        'Each value produces a set of contour lines at that threshold.'
      ),
      time: z.string().optional().describe('ISO 8601 time instant or interval.'),
      interp_time: z.boolean().optional().describe('Interpolate along the time axis.'),
      format: z.enum(['raw', 'geojson']).optional().describe(
        '"geojson" (default for isolines): GeoJSON FeatureCollection of LineStrings. "raw": raw contour data.'
      ),
      interp_vars: z.string().optional().describe('Comma-separated coordinate variables to interpolate.'),
      interp_vars_method: z.enum([
        'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
        'quintic', 'polynomial', 'pchip', 'barycentric', 'krogh', 'akima', 'makima'
      ]).optional().describe('Variable interpolation method.'),
      interp_vars_params: z.string().optional().describe('Variable interpolation extra params.'),
      as_dims: z.string().optional().describe('Comma-separated dimension variable names.')
    },
    async ({
      dataset, variable, levels, time, interp_time, format,
      interp_vars, interp_vars_method, interp_vars_params, as_dims
    }) => {
      try {
        const levelList = splitList(levels).map(Number).filter(v => !isNaN(v))
        if (levelList.length === 0) {
          return err(new Error('levels must be a non-empty comma-separated list of numbers'))
        }
        const params = {
          variable,
          levels: levelList,
          time,
          interp_time: interp_time ? 'true' : undefined,
          format,
          interp_vars: splitList(interp_vars),
          interp_vars_method,
          interp_vars_params,
          as_dims: splitList(as_dims)
        }
        return ok(await apiFetch(`/datasets/${dataset}/isoline`, params))
      } catch (e) {
        return err(e)
      }
    }
  )

  // -------------------------------------------------------------------------
  // Tool: select_data
  // -------------------------------------------------------------------------

  server.tool(
    'select_data',
    'Free-form dimension selection on a dataset variable. ' +
    'Returns the raw multi-dimensional array without spatial subsetting. ' +
    'Useful for exploring the full range of a variable or selecting along non-spatial dimensions.',
    {
      dataset: z.string().describe('Dataset identifier.'),
      variable: z.string().describe('Variable to select, e.g. "pressure".'),
      time: z.string().optional().describe('ISO 8601 time instant or interval.'),
      interp_time: z.boolean().optional().describe('Interpolate along the time axis.'),
      interp_vars: z.string().optional().describe('Comma-separated coordinate variables to interpolate.'),
      interp_vars_method: z.enum([
        'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
        'quintic', 'polynomial', 'pchip', 'barycentric', 'krogh', 'akima', 'makima'
      ]).optional().describe('Variable interpolation method.'),
      interp_vars_params: z.string().optional().describe('Variable interpolation extra params.'),
      as_dims: z.string().optional().describe('Comma-separated dimension variable names.')
    },
    async ({ dataset, variable, time, interp_time, interp_vars, interp_vars_method, interp_vars_params, as_dims }) => {
      try {
        const params = {
          variable,
          time,
          interp_time: interp_time ? 'true' : undefined,
          interp_vars: splitList(interp_vars),
          interp_vars_method,
          interp_vars_params,
          as_dims: splitList(as_dims)
        }
        return ok(await apiFetch(`/datasets/${dataset}/select`, params))
      } catch (e) {
        return err(e)
      }
    }
  )

  // -------------------------------------------------------------------------
  // Tool: get_mesh
  // -------------------------------------------------------------------------

  server.tool(
    'get_mesh',
    'Retrieve the support mesh (geometry) of a dataset. ' +
    'Returns the triangulated mesh structure that underlies the dataset grid, ' +
    'optionally including per-vertex data values.',
    {
      dataset: z.string().describe('Dataset identifier.'),
      format: z.enum(['mesh', 'geojson']).optional().describe(
        '"mesh" (default): compact triangle mesh with vertices/indices/values arrays. ' +
        '"geojson": GeoJSON representation.'
      ),
      mesh_data_mapping: z.enum(['vertices', 'cells']).optional().describe(
        'Whether data values are associated with mesh vertices or cell centres. ' +
        'Overrides the dataset configuration.'
      ),
      is_3d: z.boolean().optional().describe(
        'If true, generate a 3D volumetric mesh. Requires a vertical coordinate variable.'
      ),
      variable: z.string().optional().describe(
        'Variable to use as the mesh geometry basis (for 3D datasets with multiple vertical variables).'
      ),
      level_variable: z.string().optional().describe(
        'Variable to use as the level (vertical) coordinate. Overrides dataset configuration and "variable".'
      )
    },
    async ({ dataset, format, mesh_data_mapping, is_3d, variable, level_variable }) => {
      try {
        const params = {
          format,
          mesh_data_mapping,
          is_3d: is_3d ? 'true' : undefined,
          variable,
          level_variable
        }
        return ok(await apiFetch(`/datasets/${dataset}/mesh`, params))
      } catch (e) {
        return err(e)
      }
    }
  )

  return server
}

// ---------------------------------------------------------------------------
// Start server — stdio (default) or HTTP (when MCP_PORT is set).
// Guarded so that importing this module for testing does not auto-start.
// ---------------------------------------------------------------------------

const isMain = process.argv[1] === fileURLToPath(import.meta.url)

if (isMain && MCP_PORT) {
  // HTTP mode: Streamable HTTP transport, one server instance per request (stateless).
  // Connect via: claude mcp add kazarr --transport http http://<host>:<MCP_PORT>/mcp
  const httpServer = createServer(async (req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*')
    res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS')
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Mcp-Session-Id')

    if (req.method === 'OPTIONS') {
      res.writeHead(204)
      res.end()
      return
    }

    const { pathname } = new URL(req.url, `http://localhost:${MCP_PORT}`)
    if (pathname !== '/mcp') {
      res.writeHead(404).end('Not found')
      return
    }

    if (req.method !== 'POST') {
      res.writeHead(405).end(JSON.stringify({
        jsonrpc: '2.0',
        error: { code: -32000, message: 'Method not allowed' },
        id: null
      }))
      return
    }

    const mcpServer = buildServer()
    const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined })
    try {
      await mcpServer.connect(transport)
      await transport.handleRequest(req, res)
      res.on('close', () => {
        transport.close()
        mcpServer.close()
      })
    } catch (e) {
      process.stderr.write(`MCP request error: ${e}\n`)
      if (!res.headersSent) {
        res.writeHead(500).end(JSON.stringify({
          jsonrpc: '2.0',
          error: { code: -32603, message: 'Internal server error' },
          id: null
        }))
      }
    }
  })

  httpServer.listen(MCP_PORT, () => {
    process.stderr.write(`Kazarr MCP server (HTTP) listening on http://0.0.0.0:${MCP_PORT}/mcp\n`)
  })

  process.on('SIGINT', () => {
    httpServer.close()
    process.exit(0)
  })
} else if (isMain) {
  // Stdio mode (default): spawned as a local subprocess by Claude Desktop / Claude Code.
  const transport = new StdioServerTransport()
  await buildServer().connect(transport)
}

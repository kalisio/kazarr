/**
 * Tests for the Kazarr MCP Server.
 *
 * Strategy: spin up a lightweight mock Kazarr HTTP server that returns canned
 * responses, then connect an MCP Client via an in-memory transport pair.
 * Each test calls a tool through the client and asserts on the returned data
 * and/or on the request that the mock server received (path, query params,
 * method, body).
 */

import assert from 'node:assert'
import { createServer } from 'node:http'
import { Client } from '@modelcontextprotocol/sdk/client/index.js'
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js'
import { buildServer } from '../server.js'

// ---------------------------------------------------------------------------
// Fixture data
// ---------------------------------------------------------------------------

const HEALTH = { status: 'ok' }

const ROOT = {
  name: 'kazarr API',
  version: '0.1.0',
  description: 'A lightweight FastAPI service for Zarr datasets',
  endpoints: [
    '/health', '/datasets', '/datasets/{dataset}/metadata',
    '/datasets/{dataset}/extract', '/datasets/{dataset}/probe',
    '/datasets/{dataset}/isoline', '/datasets/{dataset}/select',
    '/datasets/{dataset}/mesh'
  ]
}

const DATASETS = {
  datasets: ['gfs/wind', 'era5/temperature']
}

const METADATA = {
  id: 'gfs/wind',
  description: 'GFS wind fields',
  variables: {
    u: { dimensions: ['time', 'lat', 'lon'], shape: [5, 10, 15] },
    v: { dimensions: ['time', 'lat', 'lon'], shape: [5, 10, 15] }
  },
  coordinates: {
    time: { dimensions: ['time'], shape: [5] },
    lat: { dimensions: ['lat'], shape: [10] },
    lon: { dimensions: ['lon'], shape: [15] }
  },
  bounding_box: { lon: { min: -180, max: 180 }, lat: { min: -90, max: 90 } },
  time_bounds: { min: '2024-01-01T00:00:00Z', max: '2024-01-05T00:00:00Z' }
}

const EXTRACT_RESULT = {
  shape: [2, 3],
  longitudes: [0, 10, 20],
  latitudes: [0, 10],
  variables: { u: { bounds: { min: -5.2, max: 12.8 } } },
  values: { u: [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]] }
}

const PROBE_RESULT = {
  type: 'FeatureCollection',
  features: [{
    type: 'Feature',
    geometry: { type: 'Point', coordinates: [2.35, 48.85] },
    properties: { u: 5.2, v: -1.3 }
  }]
}

const PROBES_RESULT = {
  type: 'FeatureCollection',
  features: [
    {
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [2.35, 48.85] },
      properties: { u: 5.2, v: -1.3 }
    },
    {
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [5.37, 43.30] },
      properties: { u: 3.1, v: 2.7 }
    }
  ]
}

const ISOLINE_RESULT = {
  type: 'FeatureCollection',
  features: [
    {
      type: 'Feature',
      geometry: { type: 'LineString', coordinates: [[0, 10], [5, 10], [10, 10]] },
      properties: { level: 5 }
    },
    {
      type: 'Feature',
      geometry: { type: 'LineString', coordinates: [[0, 20], [5, 20], [10, 20]] },
      properties: { level: 10 }
    }
  ]
}

const SELECT_RESULT = {
  shape: [5, 10, 15],
  values: { u: [[[1, 2, 3]]]}
}

const MESH_RESULT = {
  bounds: { min: -5.2, max: 12.8 },
  resolution_factor: { row: 1, col: 1 },
  vertices: [0, 0, 0, 10, 0, 0, 10, 10, 0],
  indices: [0, 1, 2],
  values: [1.1, 2.2, 3.3]
}

// ---------------------------------------------------------------------------
// Mock Kazarr server
// ---------------------------------------------------------------------------

// Routes keyed by pathname — dataset "gfs/wind" becomes path segment "gfs/wind"
const routes = {
  '/health': { status: 200, body: HEALTH },
  '/': { status: 200, body: ROOT },
  '/datasets': { status: 200, body: DATASETS },
  '/datasets/gfs/wind/metadata': { status: 200, body: METADATA },
  '/datasets/gfs/wind/extract': { status: 200, body: EXTRACT_RESULT },
  '/datasets/gfs/wind/probe': { status: 200, body: PROBE_RESULT },
  '/datasets/gfs/wind/probes': { status: 200, body: PROBES_RESULT },
  '/datasets/gfs/wind/isoline': { status: 200, body: ISOLINE_RESULT },
  '/datasets/gfs/wind/select': { status: 200, body: SELECT_RESULT },
  '/datasets/gfs/wind/mesh': { status: 200, body: MESH_RESULT }
}

let mockServer
let mockPort
const requests = []

function startMockServer () {
  return new Promise((resolve) => {
    mockServer = createServer((req, res) => {
      const parsedUrl = new URL(req.url, 'http://localhost')
      const chunks = []
      req.on('data', chunk => chunks.push(chunk))
      req.on('end', () => {
        const bodyText = Buffer.concat(chunks).toString()
        let body = null
        try { if (bodyText) body = JSON.parse(bodyText) } catch {}
        requests.push({
          pathname: parsedUrl.pathname,
          searchParams: parsedUrl.searchParams,
          method: req.method,
          body,
          authorization: req.headers.authorization || null
        })
        const route = routes[parsedUrl.pathname]
        if (route) {
          res.writeHead(route.status, { 'Content-Type': 'application/json' })
          res.end(JSON.stringify(route.body))
        } else {
          res.writeHead(404, { 'Content-Type': 'application/json' })
          res.end(JSON.stringify({ detail: `${parsedUrl.pathname} not found` }))
        }
      })
    })
    mockServer.listen(0, () => {
      mockPort = mockServer.address().port
      resolve()
    })
  })
}

// ---------------------------------------------------------------------------
// MCP client helpers
// ---------------------------------------------------------------------------

let mcpClient

async function createMcpClient (options = {}) {
  const server = buildServer({ url: `http://localhost:${mockPort}`, ...options })
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair()
  await server.connect(serverTransport)
  const client = new Client({ name: 'test-client', version: '1.0.0' })
  await client.connect(clientTransport)
  return { client, server }
}

async function callTool (name, args = {}) {
  const result = await mcpClient.callTool({ name, arguments: args })
  if (result.isError) throw new Error(result.content[0].text)
  return JSON.parse(result.content[0].text)
}

async function callToolExpectError (name, args = {}) {
  const result = await mcpClient.callTool({ name, arguments: args })
  assert.ok(result.isError, 'Expected tool to return an error')
  return result.content[0].text
}

function lastRequest () {
  return requests[requests.length - 1]
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

describe('Kazarr MCP Server', () => {
  before(async () => {
    await startMockServer()
    const { client } = await createMcpClient()
    mcpClient = client
  })

  after(async () => {
    await mcpClient.close()
    await new Promise(resolve => mockServer.close(resolve))
  })

  beforeEach(() => {
    requests.length = 0
  })

  // -------------------------------------------------------------------------
  // Tool listing
  // -------------------------------------------------------------------------

  it('exposes all 10 expected tools', async () => {
    const { tools } = await mcpClient.listTools()
    const names = tools.map(t => t.name).sort()
    assert.deepStrictEqual(names, [
      'extract_data',
      'get_dataset_metadata',
      'get_info',
      'get_isoline',
      'get_mesh',
      'healthcheck',
      'list_datasets',
      'probe_point',
      'probe_points',
      'select_data'
    ].sort())
  })

  // -------------------------------------------------------------------------
  // JWT authentication
  // -------------------------------------------------------------------------

  it('sends Authorization header when jwt option is provided', async () => {
    const { client: authedClient } = await createMcpClient({ jwt: 'test-token-abc' })
    await authedClient.callTool({ name: 'healthcheck', arguments: {} })
    await authedClient.close()
    assert.strictEqual(lastRequest().authorization, 'Bearer test-token-abc')
  })

  it('omits Authorization header when no jwt is configured', async () => {
    await callTool('healthcheck')
    assert.strictEqual(lastRequest().authorization, null)
  })

  // -------------------------------------------------------------------------
  // healthcheck
  // -------------------------------------------------------------------------

  it('healthcheck — calls /health and returns status', async () => {
    const data = await callTool('healthcheck')
    assert.strictEqual(data.status, 'ok')
    assert.strictEqual(lastRequest().pathname, '/health')
    assert.strictEqual(lastRequest().method, 'GET')
  })

  // -------------------------------------------------------------------------
  // get_info
  // -------------------------------------------------------------------------

  it('get_info — calls / and returns API metadata', async () => {
    const data = await callTool('get_info')
    assert.strictEqual(data.name, 'kazarr API')
    assert.ok(Array.isArray(data.endpoints))
    assert.strictEqual(lastRequest().pathname, '/')
  })

  // -------------------------------------------------------------------------
  // list_datasets
  // -------------------------------------------------------------------------

  it('list_datasets — returns datasets array', async () => {
    const data = await callTool('list_datasets')
    assert.ok(Array.isArray(data.datasets))
    assert.strictEqual(data.datasets.length, 2)
    assert.strictEqual(lastRequest().pathname, '/datasets')
  })

  it('list_datasets — passes search_path as query param', async () => {
    await callTool('list_datasets', { search_path: 'gfs' })
    assert.strictEqual(lastRequest().searchParams.get('search_path'), 'gfs')
  })

  // -------------------------------------------------------------------------
  // get_dataset_metadata
  // -------------------------------------------------------------------------

  it('get_dataset_metadata — returns metadata for dataset', async () => {
    const data = await callTool('get_dataset_metadata', { dataset: 'gfs/wind' })
    assert.strictEqual(data.id, 'gfs/wind')
    assert.ok(data.variables)
    assert.strictEqual(lastRequest().pathname, '/datasets/gfs/wind/metadata')
  })

  it('get_dataset_metadata — nonexistent dataset returns isError', async () => {
    const msg = await callToolExpectError('get_dataset_metadata', { dataset: 'nonexistent' })
    assert.ok(msg.includes('404'))
  })

  // -------------------------------------------------------------------------
  // extract_data
  // -------------------------------------------------------------------------

  it('extract_data — returns extraction result', async () => {
    const data = await callTool('extract_data', { dataset: 'gfs/wind', variable: 'u' })
    assert.ok(data.values)
    assert.ok(data.shape)
    assert.strictEqual(lastRequest().pathname, '/datasets/gfs/wind/extract')
    assert.strictEqual(lastRequest().method, 'GET')
  })

  it('extract_data — passes variable as query param', async () => {
    await callTool('extract_data', { dataset: 'gfs/wind', variable: 'u' })
    assert.strictEqual(lastRequest().searchParams.get('variable'), 'u')
  })

  it('extract_data — passes bbox params', async () => {
    await callTool('extract_data', {
      dataset: 'gfs/wind', variable: 'u',
      lon_min: -10, lat_min: 40, lon_max: 30, lat_max: 60
    })
    const { searchParams } = lastRequest()
    assert.strictEqual(searchParams.get('lon_min'), '-10')
    assert.strictEqual(searchParams.get('lat_min'), '40')
    assert.strictEqual(searchParams.get('lon_max'), '30')
    assert.strictEqual(searchParams.get('lat_max'), '60')
  })

  it('extract_data — passes time and format', async () => {
    await callTool('extract_data', {
      dataset: 'gfs/wind', variable: 'u',
      time: '2024-01-15T12:00:00Z', format: 'geojson'
    })
    const { searchParams } = lastRequest()
    assert.strictEqual(searchParams.get('time'), '2024-01-15T12:00:00Z')
    assert.strictEqual(searchParams.get('format'), 'geojson')
  })

  it('extract_data — passes interp_time as boolean string', async () => {
    await callTool('extract_data', { dataset: 'gfs/wind', variable: 'u', interp_time: true })
    assert.strictEqual(lastRequest().searchParams.get('interp_time'), 'true')
  })

  it('extract_data — splits interp_vars into repeated query params', async () => {
    await callTool('extract_data', { dataset: 'gfs/wind', variable: 'u', interp_vars: 'u,v' })
    const all = lastRequest().searchParams.getAll('interp_vars')
    assert.deepStrictEqual(all, ['u', 'v'])
  })

  it('extract_data — splits as_dims into repeated query params', async () => {
    await callTool('extract_data', { dataset: 'gfs/wind', variable: 'u', as_dims: 'lon,lat' })
    const all = lastRequest().searchParams.getAll('as_dims')
    assert.deepStrictEqual(all, ['lon', 'lat'])
  })

  it('extract_data — omits undefined/null params from query string', async () => {
    await callTool('extract_data', { dataset: 'gfs/wind', variable: 'u' })
    const { searchParams } = lastRequest()
    assert.strictEqual(searchParams.get('lon_min'), null)
    assert.strictEqual(searchParams.get('time'), null)
    assert.strictEqual(searchParams.get('format'), null)
    assert.strictEqual(searchParams.get('interp_time'), null)
  })

  // -------------------------------------------------------------------------
  // probe_point
  // -------------------------------------------------------------------------

  it('probe_point — returns probe result', async () => {
    const data = await callTool('probe_point', {
      dataset: 'gfs/wind', variables: 'u,v', lon: 2.35, lat: 48.85
    })
    assert.strictEqual(data.type, 'FeatureCollection')
    assert.strictEqual(lastRequest().pathname, '/datasets/gfs/wind/probe')
    assert.strictEqual(lastRequest().method, 'GET')
  })

  it('probe_point — passes variables as repeated params and lon/lat', async () => {
    await callTool('probe_point', {
      dataset: 'gfs/wind', variables: 'u,v', lon: 2.35, lat: 48.85
    })
    const { searchParams } = lastRequest()
    assert.deepStrictEqual(searchParams.getAll('variables'), ['u', 'v'])
    assert.strictEqual(searchParams.get('lon'), '2.35')
    assert.strictEqual(searchParams.get('lat'), '48.85')
  })

  it('probe_point — passes optional level and time', async () => {
    await callTool('probe_point', {
      dataset: 'gfs/wind', variables: 'u', lon: 2.35, lat: 48.85,
      level: 500, time: '2024-01-15T00:00:00Z'
    })
    const { searchParams } = lastRequest()
    assert.strictEqual(searchParams.get('level'), '500')
    assert.strictEqual(searchParams.get('time'), '2024-01-15T00:00:00Z')
  })

  // -------------------------------------------------------------------------
  // probe_points
  // -------------------------------------------------------------------------

  it('probe_points — sends POST request with points body', async () => {
    const points = [{ lon: 2.35, lat: 48.85 }, { lon: 5.37, lat: 43.30 }]
    const data = await callTool('probe_points', {
      dataset: 'gfs/wind',
      variables: 'u,v',
      points: JSON.stringify(points)
    })
    assert.strictEqual(data.type, 'FeatureCollection')
    assert.strictEqual(data.features.length, 2)
    const req = lastRequest()
    assert.strictEqual(req.pathname, '/datasets/gfs/wind/probes')
    assert.strictEqual(req.method, 'POST')
    assert.deepStrictEqual(req.body, { points })
  })

  it('probe_points — passes variables as repeated query params', async () => {
    const points = [{ lon: 2.35, lat: 48.85 }]
    await callTool('probe_points', {
      dataset: 'gfs/wind', variables: 'u,v', points: JSON.stringify(points)
    })
    assert.deepStrictEqual(lastRequest().searchParams.getAll('variables'), ['u', 'v'])
  })

  it('probe_points — returns isError for invalid points JSON', async () => {
    const msg = await callToolExpectError('probe_points', {
      dataset: 'gfs/wind', variables: 'u', points: 'not-json'
    })
    assert.ok(msg.includes('valid JSON'))
  })

  it('probe_points — returns isError when points is not an array', async () => {
    const msg = await callToolExpectError('probe_points', {
      dataset: 'gfs/wind', variables: 'u', points: '{"lon": 2.35, "lat": 48.85}'
    })
    assert.ok(msg.includes('array'))
  })

  // -------------------------------------------------------------------------
  // get_isoline
  // -------------------------------------------------------------------------

  it('get_isoline — returns GeoJSON isoline result', async () => {
    const data = await callTool('get_isoline', {
      dataset: 'gfs/wind', variable: 'u', levels: '5,10'
    })
    assert.strictEqual(data.type, 'FeatureCollection')
    assert.ok(Array.isArray(data.features))
    assert.strictEqual(lastRequest().pathname, '/datasets/gfs/wind/isoline')
    assert.strictEqual(lastRequest().method, 'GET')
  })

  it('get_isoline — passes levels as repeated query params', async () => {
    await callTool('get_isoline', { dataset: 'gfs/wind', variable: 'u', levels: '5,10,15' })
    const all = lastRequest().searchParams.getAll('levels')
    assert.deepStrictEqual(all, ['5', '10', '15'])
  })

  it('get_isoline — passes variable and optional time', async () => {
    await callTool('get_isoline', {
      dataset: 'gfs/wind', variable: 'u', levels: '5',
      time: '2024-01-15T00:00:00Z'
    })
    const { searchParams } = lastRequest()
    assert.strictEqual(searchParams.get('variable'), 'u')
    assert.strictEqual(searchParams.get('time'), '2024-01-15T00:00:00Z')
  })

  it('get_isoline — returns isError for empty levels string', async () => {
    const msg = await callToolExpectError('get_isoline', {
      dataset: 'gfs/wind', variable: 'u', levels: ''
    })
    assert.ok(msg.includes('levels'))
  })

  // -------------------------------------------------------------------------
  // select_data
  // -------------------------------------------------------------------------

  it('select_data — returns selection result', async () => {
    const data = await callTool('select_data', { dataset: 'gfs/wind', variable: 'u' })
    assert.ok(data.shape)
    assert.ok(data.values)
    assert.strictEqual(lastRequest().pathname, '/datasets/gfs/wind/select')
    assert.strictEqual(lastRequest().method, 'GET')
  })

  it('select_data — passes variable and time', async () => {
    await callTool('select_data', {
      dataset: 'gfs/wind', variable: 'u', time: '2024-01-15T00:00:00Z'
    })
    const { searchParams } = lastRequest()
    assert.strictEqual(searchParams.get('variable'), 'u')
    assert.strictEqual(searchParams.get('time'), '2024-01-15T00:00:00Z')
  })

  // -------------------------------------------------------------------------
  // get_mesh
  // -------------------------------------------------------------------------

  it('get_mesh — returns mesh result', async () => {
    const data = await callTool('get_mesh', { dataset: 'gfs/wind' })
    assert.ok(Array.isArray(data.vertices))
    assert.ok(Array.isArray(data.indices))
    assert.strictEqual(lastRequest().pathname, '/datasets/gfs/wind/mesh')
    assert.strictEqual(lastRequest().method, 'GET')
  })

  it('get_mesh — passes format and mesh_data_mapping', async () => {
    await callTool('get_mesh', {
      dataset: 'gfs/wind', format: 'geojson', mesh_data_mapping: 'cells'
    })
    const { searchParams } = lastRequest()
    assert.strictEqual(searchParams.get('format'), 'geojson')
    assert.strictEqual(searchParams.get('mesh_data_mapping'), 'cells')
  })

  it('get_mesh — passes is_3d as boolean string', async () => {
    await callTool('get_mesh', { dataset: 'gfs/wind', is_3d: true })
    assert.strictEqual(lastRequest().searchParams.get('is_3d'), 'true')
  })

  it('get_mesh — passes optional variable and level_variable', async () => {
    await callTool('get_mesh', {
      dataset: 'gfs/wind', is_3d: true, variable: 'u', level_variable: 'pressure'
    })
    const { searchParams } = lastRequest()
    assert.strictEqual(searchParams.get('variable'), 'u')
    assert.strictEqual(searchParams.get('level_variable'), 'pressure')
  })

  it('get_mesh — omits undefined params', async () => {
    await callTool('get_mesh', { dataset: 'gfs/wind' })
    const { searchParams } = lastRequest()
    assert.strictEqual(searchParams.get('format'), null)
    assert.strictEqual(searchParams.get('is_3d'), null)
    assert.strictEqual(searchParams.get('variable'), null)
  })
})

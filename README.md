# kazarr

A lightweight **FastAPI** service that exposes endpoints to interact with **Zarr datasets** stored in a **Simple Storage Service (S3)**:

  - a **datasets** endpoint to explore available multi-dimensional arrays,
  - an **extraction** endpoint to slice and dice data,
  - a **probe** endpoint to query specific values at given coordinates,
  - an **isoline** endpoint to compute contour lines dynamically,
  - a **mesh** endpoint to get support mesh

## API

> [!TIP]
> You can find auto-generated documentation about API at endpoints `/docs` or `/redoc`

### /health (GET)

Check for service's health, return a json object with a single member `status`.

### /datasets (GET)

Return a list of all available Zarr datasets with their id and description.

The `datasets` endpoint accepts the following query parameters:

| Name          | Description                                                             | Optional | Default |
| ------------- | ----------------------------------------------------------------------- | :------: | ------- |
| `search_path` | Path to search for datasets. Paths are always returned from root folder |    ✓     |         |

#### /datasets/{dataset}/metadata (GET)

Return metadata (dimensions, variables, attributes) for a specific Zarr dataset.
The `dataset` parameter is expected to be the dataset id, that can be found with the previous endpoint.

### /datasets/{dataset}/extract (GET)

Extracts a subset of the data based on a bounding box and a specific variable.

> [!WARNING]
> Large extractions may impact performance. Be mindful of the bounding box size for high-resolution datasets.

The `extract` endpoint accepts the following query parameters:

| Name                    | Description                                                                                                                                    | Optional | Default       |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | :------: | ------------- |
| `variable`              | The variable to extract.                                                                                                                       |    ✗     |               |
| `lon_min`               | Minimum longitude of the bounding box.                                                                                                         |    ✓     | `None`        |
| `lat_min`               | Minimum latitude of the bounding box.                                                                                                          |    ✓     | `None`        |
| `lon_max`               | Maximum longitude of the bounding box.                                                                                                         |    ✓     | `None`        |
| `lat_max`               | Maximum latitude of the bounding box.                                                                                                          |    ✓     | `None`        |
| `time`                  | The time value/slice to extract.                                                                                                               |    ✓     | `None`        |
| `level`                 | Value of the level to extract.                                                                                                                 |    ✓     | `None`        |
| `resolution_limit`      | Limit the amount of data for lat/lon axis (decimate)                                                                                           |    ✓     | `None`        |
| `format`                | Format of the extracted data (Supported: `raw`, `geojson`, `mesh`).                                                                            |    ✓     | `raw`         |
| `mesh_tile_size`        | When `format=mesh`, resample data with a grid of `mesh_tile_size`x`mesh_tile_size`                                                             |    ✓     | `None`        |
| `mesh_data_mapping`     | Whether the data of the mesh is on cells or on vertices. This will override the dataset configuration. (Supported values: 'vertices', 'cells') |    ✓     | `vertices`    |
| `is_3d`                 | If True, performs a full 3D volume extraction. If False and the dataset is 3D, a vertical coordinate must be provided.                         |    ✓     | `False`       |
| `level_min`             | Minimum vertical coordinate (altitude/depth) of the bounding box                                                                               |    ✓     | `None`        |
| `level_max`             | Maximum vertical coordinate (altitude/depth) of the bounding box                                                                               |    ✓     | `None`        |
| `interp_vars`           | Variables to interpolate during extraction                                                                                                     |    ✓     | `[]`          |
| `interp_vars_method`    | Method for variable/time interpolation (e.g. `linear`, `cubic`, ...)                                                                           |    ✓     | `nearest`     |
| `interp_vars_params`    | Parameters for variable interpolation (e.g. `method:linear`)                                                                                   |    ✓     | `None`        |
| `interp_time`           | Whether to interpolate values on time dimension or to get the closest time step. Shortcut to `interp_vars=YOUR_TIME_DIMENSION`                 |    ✓     | `False`       |
| `interp_spatial_method` | The method to use for spatial interpolation. (Supported: `nearest`, `linear`, `cubic`, `idw`, `rbf`)                                           |    ✓     | `nearest`     |
| `interp_spatial_params` | Parameters for spatial interpolation (e.g. `padding:1.0`)                                                                                      |    ✓     | `padding:1.0` |
| `as_dims`               | If a variable has the same name as a dim, force query parameters in this list to be treated as dimensions                                      |    ✓     | `[]`          |

> [!IMPORTANT]
> You may need to specify additional non-generic variables or dimensions according to your dataset. To do so, you can add query parameters with `&my_additional_variable={VALUE}`

### /datasets/{dataset}/probe (GET)

Retrieves the values of specified variables at a specific geographical location (point query).

The `probe` endpoint accepts the following query parameters:

| Name                    | Description                                                                                               | Optional | Default       |
| ----------------------- | --------------------------------------------------------------------------------------------------------- | :------: | ------------- |
| `variables`             | The list of variables to probe.                                                                           |    ✗     |               |
| `lon`                   | The longitude coordinate to probe.                                                                        |    ✗     |               |
| `lat`                   | The latitude coordinate to probe.                                                                         |    ✗     |               |
| `level`                 | The level coordinate to probe (if 3D data).                                                               |    ✓     | `None`        |
| `time`                  | The time value/slice to probe.                                                                            |    ✓     | `None`        |
| `interp_time`           | Whether to interpolate values on time dimension                                                           |    ✓     | `False`       |
| `interp_spatial_method` | The method to use for spatial interpolation. (Supported: `nearest`, `linear`, `cubic`, `idw`, `rbf`)      |    ✓     | `nearest`     |
| `interp_spatial_params` | Parameters for spatial interpolation (e.g. `padding:1.0`)                                                 |    ✓     | `padding:1.0` |
| `interp_vars`           | Variables to interpolate during probe                                                                     |    ✓     | `[]`          |
| `interp_vars_method`    | Method for variable/time interpolation                                                                    |    ✓     | `nearest`     |
| `format`                | Format of the extracted data (Supported: `raw`, `geojson`).                                               |    ✓     | `raw`         |
| `as_dims`               | If a variable has the same name as a dim, force query parameters in this list to be treated as dimensions |    ✓     | `[]`          |

> [!IMPORTANT]
> You may need to specify additional non-generic variables or dimensions according to your dataset. To do so, you can add query parameters with `&my_additional_variable={VALUE}`

> [!TIP]
> You can request multiple variables at once by repeating the `variables` parameter in the query string (e.g., `?variables=temp&variables=wind`).

### /datasets/{dataset}/probes (POST)

Retrieves the values of specified variables at multiple geographical locations.

The `probes` endpoint accepts the following query parameters:

| Name                    | Description                                                                                               | Optional | Default       |
| ----------------------- | --------------------------------------------------------------------------------------------------------- | :------: | ------------- |
| `variables`             | The list of variables to probe.                                                                           |    ✗     |               |
| `time`                  | The time value/slice to probe.                                                                            |    ✓     | `None`        |
| `interp_time`           | Whether to interpolate values on time dimension                                                           |    ✓     | `False`       |
| `interp_spatial_method` | The method to use for spatial interpolation. (Supported: `nearest`, `linear`, `cubic`, `idw`, `rbf`)      |    ✓     | `nearest`     |
| `interp_spatial_params` | Parameters for spatial interpolation (e.g. `padding:1.0`)                                                 |    ✓     | `padding:1.0` |
| `interp_vars`           | Variables to interpolate during probe                                                                     |    ✓     | `[]`          |
| `interp_vars_method`    | Method for variable/time interpolation                                                                    |    ✓     | `nearest`     |
| `format`                | Format of the extracted data (Supported: `raw`, `geojson`).                                               |    ✓     | `raw`         |
| `as_dims`               | If a variable has the same name as a dim, force query parameters in this list to be treated as dimensions |    ✓     | `[]`          |

The `probes` endpoint accepts the following body structures:

- List of points (return values for each requested time) with ad hoc structure:

```json
{
   "points": [{"lon": 5.35964198232827, "lat": 45.01486461788593, "level": 100.0}, {"lon": 7.79425497419718, "lat": 48.45134263206789, "level": 50.0}, ...],
   "times": ["2026-07-18T12:00:00", "2026-07-20T08:00:00/2026-07-20T22:00:00"]
}
```

- List of points with GeoJSON structure:

```json
{
   "type": "FeatureCollection",
   "times": ["2026-07-18T12:00:00", "2026-07-20T08:00:00/2026-07-20T22:00:00"],
   "features": [
      {
         "type": "Feature",
         "geometry": {
            "type": "Point",
            "coordinates": [5.35964198232827, 45.01486461788593]
         }
      },
      ...
   ]
}
```

- Path (one time per point) with ad hoc structure:

```json
{
   "path": [{"lon": 5.35964198232827, "lat": 45.01486461788593, "level": 100.0}, {"lon": 7.79425497419718, "lat": 48.45134263206789, "level": 50.0}],
   "times": ["2026-07-18T12:00:00", "2026-07-20T22:00:00"]
}
```

- Path with GeoJSON structure:

```json
{
   "type": "FeatureCollection",
   "features": [
      {
         "type": "Feature",
         "geometry": {
            "type": "LineString",
            "coordinates": [
               [5.35964198232827, 45.01486461788593],
               [7.7942597419718, 48.45134263206789]
            ]
         },
         "properties": {
            "times": ["2026-07-18T12:00:00", "2026-07-20T22:00:00"]
         }
      },
      ...
   ]
}
```

> [!TIP]
> For each point, the `level` property is optional.

> [!TIP]
> With "list of point" mode, if `times` is not provided, API will return every times

> [!IMPORTANT]
> Time ranges can't be used with "path" mode


### /datasets/{dataset}/isoline (GET)

Computes isolines (contour lines) for a given variable and specific thresholds.

The `isoline` endpoint accepts the following query parameters:

| Name                 | Description                                                                                               | Optional | Default   |
| -------------------- | --------------------------------------------------------------------------------------------------------- | :------: | --------- |
| `variable`           | The variable to generate isolines for.                                                                    |    ✗     |           |
| `thresholds`         | Thresholds for isoline generation: repeated values (`thresholds=0&thresholds=5`), or a range `min:max:step` (see below). |    ✗     |           |
| `time`               | The time value to use for isoline generation.                                                             |    ✓     | `None`    |
| `format`             | Format of the output (Supported: `raw`, `geojson`).                                                       |    ✓     | `raw`     |
| `lon_min`            | Minimum longitude of the bounding box.                                                                    |    ✓     | `None`    |
| `lat_min`            | Minimum latitude of the bounding box.                                                                     |    ✓     | `None`    |
| `lon_max`            | Maximum longitude of the bounding box.                                                                    |    ✓     | `None`    |
| `lat_max`            | Maximum latitude of the bounding box.                                                                     |    ✓     | `None`    |
| `interp_time`        | Whether to interpolate values on time dimension                                                           |    ✓     | `False`   |
| `interp_vars_method` | Method for variable/time interpolation                                                                    |    ✓     | `nearest` |
| `as_dims`            | If a variable has the same name as a dim, force query parameters in this list to be treated as dimensions |    ✓     | `[]`      |

Thresholds can be defined as a range `min:max:step` (e.g. `thresholds=0:30:5`), where `min` and/or `max` can be omitted:
- `min` omitted: the first threshold is the first multiple of `step` greater than or equal to the minimum value of the selected data (bounding box included), so that thresholds are round values (e.g. data in [273.4, 296.2] with `thresholds=::5` gives 275, 280, ..., 295).
- `max` omitted: the maximum value of the selected data is used.
- `max` is inclusive (`thresholds=0:10:5` gives 0, 5, 10) and `step` is required and must be positive. A range can produce up to 1000 thresholds.

Output formats:
- `raw`: `{"<threshold>": [line, ...], ...}` where each line is a list of `[lon, lat]` points.
- `geojson`: a `FeatureCollection` with one feature per requested threshold, with `id` (index of the threshold) and `threshold` properties. The geometry is a `LineString` when the threshold has a single line, a `MultiLineString` when it has several. Thresholds without any isoline get an empty `MultiLineString`.

With a bounding box, isolines are computed on the smallest block of grid cells covering it (plus one cell of padding): on curvilinear grids, lines may exceed the bounding box.

> [!IMPORTANT]
> You may need to specify additional non-generic variables or dimensions according to your dataset. To do so, you can add query parameters with `&my_additional_variable={VALUE}`

### /datasets/{dataset}/select (GET)

You can select data in a generic way with this endpoint, as raw multi-dimensional arrays. If you just specify the `variable` parameter, you will get all the data, but you are free to add additional parameters to fix some variables/dimensions

The `select` endpoint accepts the following query parameters:
| Name                 | Description                                                                                               | Optional | Default   |
| -------------------- | --------------------------------------------------------------------------------------------------------- | :------: | --------- |
| `variable`           | The variable from which you want to select the data.                                                      |    ✗     |           |
| `interp_vars`        | Variables to interpolate during selection                                                                 |    ✓     | `[]`      |
| `interp_vars_method` | Method for variable/time interpolation                                                                    |    ✓     | `nearest` |
| `as_dims`            | If a variable has the same name as a dim, force query parameters in this list to be treated as dimensions |    ✓     | `[]`      |

### /datasets/{dataset}/mesh

Get only the support mesh of the dataset

The `mesh` endpoint accepts the following query parameters:
| Name                | Description                                                                                                                                                                                            | Optional | Default    |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------: | ---------- |
| `format`            | The format of the extracted data (Currently supported: 'mesh', 'geojson'. Default to `mesh`)                                                                                                           |    ✓     | `mesh`     |
| `mesh_data_mapping` | Whether the data of the mesh is on cells or on vertices. This will override the dataset configuration. (Supported values: 'vertices', 'cells')                                                         |    ✓     | `vertices` |
| `is_3d`             | If True, generates a 3D volumetric mesh using the vertical coordinate defined in the dataset configuration if the dataset use a unique one, otherwise, see 'variable' and 'level_variable' parameters. | `False`  |
| `variable`          | The variable to base the mesh geometry on. Not mandatory if the dataset use a unique vertical coordinate.                                                                                              |  `None`  |
| `level_variable`    | The variable to use as level coordinate for the mesh geometry. This will override the dataset configuration and the 'variable' parameter.                                                              |  `None`  |

## Interpolation Overview

Interpolation is applied in four different scenarios:

1. **Xarray Variable/Time Interpolation**: Used when specifying `interp_vars` or `interp_time`.
   - *Supported methods*: `linear`, `nearest`, `zero`, `slinear`, `quadratic`, `cubic`, `quintic`, `polynomial`, `pchip`, `barycentric`, `krogh`, `akima`, `makima`.
   - *Parameters*: See the [Xarray documentation](https://docs.xarray.dev/en/latest/generated/xarray.DataArray.interp.html).
2. **Regular Grid Mesh Extraction**: Triggered by the `extract` endpoint on regular grid datasets, utilizing SciPy's `RegularGridInterpolator`.
   - *Supported methods*: `linear`, `nearest`, `slinear`, `cubic`, `quintic`, `pchip`.
   - *Parameters*: See the [SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html#scipy.interpolate.RegularGridInterpolator).
3. **Irregular Grid Mesh Extraction**: Triggered by the `extract` endpoint on irregular grid datasets using SciPy or custom methods.
   - *Supported methods*: `linear`, `nearest`, `cubic`, `RBF`, `IDW`.
   - *Parameters*: For RBF, see the [SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html).
   - **Special Cases for Level Interpolation:**
     - *1D Level Variables*: Falls back to Xarray interpolation. Ensure `interp_spatial_method` is among the supported methods listed in Scenario 1.
     - *Multi-dimensional Level Variables*: Currently restricted to `linear` or `nearest` methods.
4. **Point Probing**: Used by the `probe` endpoint to retrieve values over time.
   - *Supported methods*: Currently, only `IDW` (Inverse Distance Weighting) is supported.
   - *Parameters*:
     - `k`: Number of nearest neighbors to use (k-nearest-neighbors selection). **Default strategy** when neither `k` nor `radius` is given.
     - `radius`: Maximum search radius for neighbors (radius-based selection). Only used when explicitly given; ignored if `k` is also given.
     - `level_scale`: Number of native level units treated as equivalent to 1 degree of horizontal distance, for datasets with an irregular (multi-dimensional) level variable. See [Probe Spatial Interpolation: Grid Cases](#probe-spatial-interpolation-grid-cases) below.
     - `power`: Distance weighting power.

### Probe Spatial Interpolation: Grid Cases

The `probe`/`probes` endpoints pick a different spatial interpolation strategy depending on how the dataset's grid and level are structured. This determines which parameters are relevant and why.

| Grid case                                                    | Horizontal resolution                                                                                        | Level resolution                                                                                                                                                                            | KD-tree? | Relevant `interp_spatial_params`       |
| ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------: | ---------------------------------------- |
| **Regular grid** (1D `lon`/`lat` axes)                        | Direct axis-wise selection/interpolation (`nearest` or `linear`) along the `lon`/`lat` axes, independently.      | Same: direct axis-wise selection/interpolation along the `level` axis, independently of `lon`/`lat`.                                                                                          |    No    | *(none — IDW params don't apply)*        |
| **Irregular horizontal grid + regular (1D) level**             | IDW over a KD-tree built from the horizontal points only.                                                       | Resolved separately: linear/nearest interpolation along the shared `level` axis (like the regular grid case), applied per neighbor, independently of the horizontal IDW weights.              |  Yes (2D)  | `k` / `radius`, `power`                  |
| **Irregular horizontal grid + irregular (multi-dim) level**    | IDW over a KD-tree built jointly from horizontal *and* level coordinates.                                        | Resolved jointly with the horizontal position: there is no shared level axis to interpolate along separately, since each grid point carries its own level value.                              |  Yes (4D)  | `k` / `radius`, `power`, `level_scale`   |

> [!TIP]
> A **regular grid** never needs a KD-tree at all: `lon` and `lat` (and `level`, if present) are independent 1D axes, so xarray resolves each one on its own, without ever computing a joint 2D/3D distance. This also means the pole-compression and antimeridian issues below simply don't apply to it.

For the two irregular-grid cases, a few implementation details worth knowing:

- **Cartographic (cartesian) conversion**: horizontal coordinates are projected onto the unit sphere as 3D cartesian `(x, y, z)` points before being fed to the KD-tree. This avoids two issues inherent to searching a raw `(lon, lat)` point cloud with a generic distance-based algorithm: the antimeridian discontinuity (-180°/180°) and the compression of longitude distances near the poles. `deg_to_chord_distance`/`chord_to_deg_distance` convert between real angular degrees and the tree's native (chord) distance — an exact conversion, valid at any angle, not an approximation.
- **`level_scale` and radians (irregular level only)**: combining horizontal degrees with a vertical axis in raw physical units (meters, hPa, ...) in a single distance metric is meaningless without first making them comparable. `level_scale` converts the level axis into "degree-equivalents" (how many native level units count as 1 horizontal degree). That degree-equivalent value is then converted to radians before being stored in the KD-tree, to match the scale of the cartesian `(x, y, z)` chord coordinates (which are themselves radian-scale, not degree-scale) — otherwise the level axis would still dominate (or be dominated by) the spatial axes in the tree's own distance ordering, biasing which neighbors get selected. This bakes `level_scale` into neighbor *selection* itself (both `k` and `radius` modes), not just the final IDW weighting.
- **`k` vs `radius`**: `k` selects a fixed number of nearest neighbors (via `cKDTree.query`); `radius` instead selects every neighbor within a given distance (via `cKDTree.query_ball_point`), which can vary per point. `k` is the default, since it never risks an empty neighbor search on a sparse patch of the grid; passing `radius` opts into distance-bounded selection instead. If both are given, `k` takes precedence.

### Supplying Interpolation Parameters

For the `extract` and `probe` endpoints, you can pass interpolation options using the `interp_spatial_params` or `interp_vars_params` query parameters. Here is an example:

`interp_spatial_params=padding:0.5,neighbors:5,smoothing:0.0,kernel:thin_plate_spline`

- `padding`: A coefficient extending the requested bounding box to include contextual data for interpolation. This helps prevent boundary artifacts near tiles.
- *Other parameters*: Specific to the chosen interpolation method.

## Configuring

### Environment variables

| Variable              | Description                                                                                               | Default value |
| --------------------- | --------------------------------------------------------------------------------------------------------- | ------------- |
| PORT                  | The port to be used when exposing the service                                                             | 8000          |
| HOSTNAME              | The hostname to be used when exposing the service                                                         | localhost     |
| AWS_ACCESS_KEY_ID     | Access key ID of the S3 in which zarr data is stored                                                      |               |
| AWS_SECRET_ACCESS_KEY | Secret access key of the S3 in which zarr data is stored                                                  |               |
| AWS_DEFAULT_REGION    | Region of the S3 in which zarr data is stored                                                             |               |
| AWS_ENDPOINT_URL      | Endpoint URL of the S3 in which zarr data is stored                                                       |               |
| BUCKET_NAME           | The name of the bucket in which zarr data is stored                                                       |               |
| DATASETS_PATH         | Path to the JSON file containing datasets description                                                     | datasets.json |
| CACHE_DIR             | Path to the directory where cache will be stored. Cache will not be used if this variable is not provided |               |
| CACHE_SIZE            | Max size of cache folder (e.g. 1024KB, 512MB, 4GB)                                                        | 512MB         |
| LRU_CACHE_SIZE        | Max number of datasets kept in the in-memory `lru_cache`                                                  | 5             |
| KDTREE_MAX_CACHE_SIZE | Max number of cKDTree cached                                                                              | 10            |
| GZIP_COMPRESSION_LEVEL | Compression level of the responses, from 1 (fastest) to 9 (smallest). `0` disables compression (e.g. when a reverse proxy already compresses) | 1             |

> [!IMPORTANT]
> With some S3 provider, some errors about checksum calculation can occur (error: `botocore.exceptions.ClientError: An error occurred (InvalidArgument) when calling the PutObject operation: x-amz-content-sha256 must be UNSIGNED-PAYLOAD, or a valid sha256 value.`). In that case, you should set `AWS_REQUEST_CHECKSUM_CALCULATION` environment variable to `when_required`

## Usage

### Manual build

You can build the image with the following command:

```bash
docker build -t <your-image-name> .
```

And then start the service with:

```bash
docker run -p 8000:8000 <your-image-name>
```

### Run locally

#### With `uv`

To create an environment with `uv`, run these commands :

```bash
uv sync

uv run main.py
```

> [!TIP]
> If you also want to be able to execute tests, you will need to add `--group test` to the `uv sync` command

#### With `conda` (Anaconda)

To create an environment with Anaconda, run these commands :

```bash
conda create -y -n kazarr_env python=3.11
```

```bash
conda install -y -n kazarr_env -c conda-forge \
  fastapi \
  uvicorn \
  xarray \
  zarr \
  numpy \
  pyproj \
  dask \
  s3fs \
  matplotlib \
  pyvista=0.47.1 \
  vtk-base=9.5.2 \
  scipy \
  uvloop \
  diskcache \
  loguru
```

```bash
conda activate kazarr_env
```

```bash
python main.py
```

#### Local S3

You can run a local object storage with S3-compliant API using [garage](https://garagehq.deuxfleurs.fr/) with CLI access using [s3cmd](https://s3tools.org/s3cmd) (`pipx install s3cmd`).

First, generate a secret with `openssl rand -base64 32` and create a garage configuration file:
```toml
metadata_dir = "/home/luc/Development/GeoData/s3-meta"
data_dir = "/home/luc/Development/GeoData/s3"
db_engine = "sqlite"

replication_factor = 1

rpc_bind_addr = "[::]:3901"
rpc_public_addr = "127.0.0.1:3901"
rpc_secret = "your secret"

[s3_api]
s3_region = "localhost"
api_bind_addr = "[::]:3900"
root_domain = ".s3.garage.localhost"

[s3_web]
bind_addr = "[::]:3902"
root_domain = ".web.garage.localhost"
index = "index.html"
```
Then launch the server with `garage -c ./garage.toml server` in a terminal and get your node ID in another terminal with `garage -c ./garage.toml status`.

Create the layout of your cluster with `garage -c ./garage.toml layout assign -z localhost -c 500G nodeID && garage -c ./garage.toml layout apply --version 1`.

Create a bucket with `garage -c ./garage.toml bucket create zarr-data`.

Create an access key with `garage -c ./garage.toml key create zarr-data-key`.

Allow the key to access your bucket `garage -c ./garage.toml bucket allow --read --write --owner zarr-data --key zarr-data-key`.

Create a s3cmd configuration file:
```
[default]
access_key = your-key-id
secret_key = your-key-secret
host_base = http://localhost:3900
host_bucket = http://localhost:3900
use_https = False
```
Then synchronize any data from your local file system to garage with `s3cmd -c ./s3cmd.cfg sync ./zarr-data/ s3://zarr-data`.

## Contributing

Please read the [Contributing file](https://github.com/kalisio/k2/blob/master/.github/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](https://semver.org/) for versioning. For the versions available, see the tags on this repository.

## Preparing Datasets

An extra tool allow you to generate Zarr datasets from NetCDF or GRIB2 files. For more detail, check the [conversion tool](./conversion_tool/README.md)

## Authors

This project is sponsored by

![Kalisio](https://s3.eu-central-1.amazonaws.com/kalisioscope/kalisio/kalisio-logo-black-256x84.png)

## License

This project is licensed under the MIT License - see the [license file](./LICENSE.md) for details.

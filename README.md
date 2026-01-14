# kazarr

A lightweight **FastAPI** service that exposes endpoints to interact with **Zarr datasets** stored in a **Simple Storage Service (S3)**:

  - a **datasets** endpoint to explore available multi-dimensional arrays,
  - an **extraction** endpoint to slice and dice data,
  - a **probe** endpoint to query specific values at given coordinates,
  - an **isoline** endpoint to compute contour lines dynamically.

## API

> [!TIP]
> You can find auto-generated documentation about API at endpoints `/docs` or `/redoc`

### /health (GET)

Check for service's health, return a json object with a single member `status`.

### /datasets (GET)

Return a list of all available Zarr datasets with their id and description.

### /datasets/{dataset} (GET)

Return metadata (dimensions, variables, attributes) for a specific Zarr dataset.
The `dataset` parameter is expected to be the dataset id, that can be found with the previous endpoint.

### /datasets/{dataset}/extract (GET)

Extracts a subset of the data based on a bounding box and a specific variable.

> [!WARNING]
> Large extractions may impact performance. Be mindful of the bounding box size for high-resolution datasets.

The `extract` endpoint accepts the following query parameters:

| Name               | Description                                                                                               | Optional |
|--------------------|-----------------------------------------------------------------------------------------------------------|:--------:|
| `variable`         | The variable to extract.                                                                                  |    ✗     |
| `lon_min`          | Minimum longitude of the bounding box.                                                                    |    ✓     |
| `lat_min`          | Minimum latitude of the bounding box.                                                                     |    ✓     |
| `lon_max`          | Maximum longitude of the bounding box.                                                                    |    ✓     |
| `lat_max`          | Maximum latitude of the bounding box.                                                                     |    ✓     |
| `time`             | The time value/slice to extract.                                                                          |    ✓     |
| `resolution_limit` | Limit the amount of data for lat/lon axis (decimate)                                                      |    ✓     |
| `format`           | Format of the extracted data (Supported: `raw`, `geojson`). Ignored when `mesh_tile_size` is defined      |    ✓     |
| `mesh_tile_size`   | Return data as mesh, with a grid of `mesh_tile_size`x`mesh_tile_size`                                     |    ✓     |
| `mesh_interpolate` | Apply interpolation to mesh values                                                                        |    ✓     |
| `as_dims`          | If a variable has the same name as a dim, force query parameters in this list to be treated as dimensions |    ✓     |

> [!IMPORTANT]
> You may need to specify additional non-generic variables or dimensions according to your dataset. To do so, you can add query parameters with `&my_additional_variable={VALUE}`

### /datasets/{dataset}/probe (GET)

Retrieves the values of specified variables at a specific geographical location (point query).

The `probe` endpoint accepts the following query parameters:

| Name        | Description                                                                                               | Optional |
|-------------|-----------------------------------------------------------------------------------------------------------|:--------:|
| `variables` | The list of variables to probe.                                                                           |    ✗     |
| `lon`       | The longitude coordinate to probe.                                                                        |    ✗     |
| `lat`       | The latitude coordinate to probe.                                                                         |    ✗     |
| `height`    | The height coordinate to probe (if 3D data).                                                              |    ✓     |
| `as_dims`   | If a variable has the same name as a dim, force query parameters in this list to be treated as dimensions |    ✓     |

> [!IMPORTANT]
> You may need to specify additional non-generic variables or dimensions according to your dataset. To do so, you can add query parameters with `&my_additional_variable={VALUE}`

> [!TIP]
> You can request multiple variables at once by repeating the `variables` parameter in the query string (e.g., `?variables=temp&variables=wind`).

### /datasets/{dataset}/isoline (GET)

Computes isolines (contour lines) for a given variable and specific levels.

The `isoline` endpoint accepts the following query parameters:

| Name       | Description                                                                                               | Optional |
|------------|-----------------------------------------------------------------------------------------------------------|:--------:|
| `variable` | The variable to generate isolines for.                                                                    |    ✗     |
| `levels`   | Comma-separated list of levels for isoline generation.                                                    |    ✗     |
| `time`     | The time value to use for isoline generation.                                                             |    ✓     |
| `format`   | Format of the extracted data (Supported: `raw`, `geojson`). Ignored when `mesh_tile_size` is defined      |    ✓     |
| `as_dims`  | If a variable has the same name as a dim, force query parameters in this list to be treated as dimensions |    ✓     |

> [!IMPORTANT]
> You may need to specify additional non-generic variables or dimensions according to your dataset. To do so, you can add query parameters with `&my_additional_variable={VALUE}`

### /datasets/{dataset}/select (GET)

You can select data in a generic way with this endpoint, as raw multi-dimensional arrays. If you just specify the `variable` parameter, you will get all the data, but you are free to add additional parameters to fix some variables/dimensions

The `select` endpoint accepts the following query parameters:
| Name       | Description                                                                                               | Optional |
| ---------- | --------------------------------------------------------------------------------------------------------- | :------: |
| `variable` | The variable from which you want to select the data.                                                      |    ✗     |
| `as_dims`  | If a variable has the same name as a dim, force query parameters in this list to be treated as dimensions |    ✓     |

## Configuring

### Environment variables

| Variable              | Description                                                                                               | Default value |
|-----------------------|-----------------------------------------------------------------------------------------------------------|---------------|
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

You will need to install multiple Python packages to run this app.
To simplify, you can install Anaconda (or micromamba) and run these commands :

```bash
conda create -y -n kazarr_env python=3.11
```

```bash
conda install -y -n kazarr_env -c conda-forge \
  fastapi \
  uvicorn \
  xarray \
  zarr \
  cfgrib \
  numpy \
  pyproj \
  dask \
  s3fs \
  matplotlib \
  pyvista \
  scipy
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
Then launch the server with `garage -c ./garage.toml server` in a terminal and get your node ID in another terminal with `garage -c garage.toml status`.
Create the layout of your cluster with `garage -c garage.toml layout assign -z localhost -c 500G nodeID && garage -c garage.toml layout apply --version 1`.
Create a bucket with `garage -c garage.toml bucket create zarr-data`.
Create an access key with `garage -c garage.toml key create zarr-data-key`.
Allow the key to access your bucket `garage -c garage.toml bucket allow --read --write --owner zarr-data --key zarr-data-key`.
Create a s3cmd configuration file:
```
[default]
access_key = your-key-id
secret_key = your-key-secret
host_base = http://localhost:3900
host_bucket = http://localhost:3900
use_https = False
```
Then synchronize any data from your local file system to garage with `s3cmd -c s3cmd.cfg sync ./zarr-data/ s3://zarr-data`.

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

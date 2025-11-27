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

| Name       | Description                                  | Optional |
|------------|----------------------------------------------|:--------:|
| `variable` | The variable to extract.                     |    ✗     |
| `lon_min`  | Minimum longitude of the bounding box.       |    ✓     |
| `lat_min`  | Minimum latitude of the bounding box.        |    ✓     |
| `lon_max`  | Maximum longitude of the bounding box.       |    ✓     |
| `lat_max`  | Maximum latitude of the bounding box.        |    ✓     |
| `time`     | The time value/slice to extract.             |    ✓     |

> [!IMPORTANT]
> You may need to specify additional non-generic variables or dimensions according to your dataset. To do so, you can add query parameters with `&my_additional_variable={VALUE}`

### /datasets/{dataset}/probe (GET)

Retrieves the values of specified variables at a specific geographical location (point query).

The `probe` endpoint accepts the following query parameters:

| Name        | Description                                  | Optional |
|-------------|----------------------------------------------|:--------:|
| `variables` | The list of variables to probe.              |    ✗     |
| `lon`       | The longitude coordinate to probe.           |    ✗     |
| `lat`       | The latitude coordinate to probe.            |    ✗     |
| `height`    | The height coordinate to probe (if 3D data). |    ✓     |

> [!IMPORTANT]
> You may need to specify additional non-generic variables or dimensions according to your dataset. To do so, you can add query parameters with `&my_additional_variable={VALUE}`

> [!TIP]
> You can request multiple variables at once by repeating the `variables` parameter in the query string (e.g., `?variables=temp&variables=wind`).

### /datasets/{dataset}/isoline (GET)

Computes isolines (contour lines) for a given variable and specific levels.

The `isoline` endpoint accepts the following query parameters:

| Name       | Description                                                   | Optional |
|------------|---------------------------------------------------------------|:--------:|
| `variable` | The variable to generate isolines for.                        |    ✗     |
| `levels`   | Comma-separated list of levels for isoline generation.        |    ✗     |
| `time`     | The time value to use for isoline generation.                 |    ✓     |

> [!IMPORTANT]
> You may need to specify additional non-generic variables or dimensions according to your dataset. To do so, you can add query parameters with `&my_additional_variable={VALUE}`

## Building

### Manual build

You can build the image with the following command:

```bash
docker build -t <your-image-name> .
```

## Contributing

Please read the [Contributing file](https://github.com/kalisio/k2/blob/master/.github/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](https://semver.org/) for versioning. For the versions available, see the tags on this repository.

## Authors

This project is sponsored by

![Kalisio](https://s3.eu-central-1.amazonaws.com/kalisioscope/kalisio/kalisio-logo-black-256x84.png)

## License

This project is licensed under the MIT License - see the [license file](./LICENSE.md) for details.
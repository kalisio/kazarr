# kazarr - conversion tool

A tool to create Zarr datasets from NetCDF or GRIB2 files.

## Installation

### Docker

You can build the Docker image using the provided `Dockerfile`.

```bash
docker build -t kazarr-conversion-tool .
```

To run the container, make sure to mount your data volumes.

```bash
docker run -rm -v /path/to/data:/data -e BUCKET_NAME=my-bucket kazarr-conversion-tool [COMMAND]
```

In this image, you can find the conversion tool in `/app/dist/conversion_tool`

### Python

The tool requires Python 3.11+. You can install the dependencies using pip:

```bash
pip install python-dotenv xarray zarr cfgrib h5netcdf numpy pyproj dask s3fs matplotlib pyvista pyinstaller
```

And you can build the executable using PyInstaller as done in the Dockerfile:

```bash
pyinstaller -F -n conversion_tool main.py
```

### S3 Configuration

The tool supports reading and writing from S3. To enable this, you must set the `BUCKET_NAME` environment variable.
Standard AWS environment variables are used for authentication (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL`, `AWS_DEFAULT_REGION`).

## Usage

The tool is a command-line interface with the following subcommands.

### `new-dataset`

Create a new zarr dataset from an input source.

```bash
conversion_tool new-dataset [OPTIONS] DATASET_NAME INPUT_PATH
```

**Arguments:**

*   `DATASET_NAME`: Name of the new dataset.
*   `INPUT_PATH`: Path of the input file or folder used for generating this new dataset (local or `s3://`).

**Options:**

*   `-t, --template TEXT`: [Template](#templates) to use for the configuration of the new dataset.
*   `-c, --config JSON`: Additional [configuration](#config) as JSON string.
*   `-f, --config-file PATH`: Path to a JSON file containing additional configuration.
*   `-d, --description TEXT`: Description of the new dataset.
*   `-o, --output PATH`: Output path for the processed dataset (local or `s3://`).
*   `-p, --pipeline TEXT`: [Pipeline](#pipelines) to use for processing the new dataset [default: preprocess].
*   `--rgstr-endpoint URL`: Endpoint URL for dataset registration service.
*   `--templates-path PATH`: Path to [templates](#templates) configuration file [default: templates.json].

### `list-templates`

List available templates for new datasets.

```bash
conversion_tool list-templates [--templates-path PATH]
```

## Config

The configuration is a JSON object that defines the parameters for the pipeline processes.
It can be constructed from:
1.  A **Template**: Pre-defined configuration in a JSON file (defaults: `templates.json`).
2.  A **Config File**: A JSON file provided via `--config-file`.
3.  **Inline Config**: JSON string provided via `--config`.

These sources are merged in that order (Inline overrides File, which overrides Template).
Parameters can be defined in **snake_case** or **camelCase**.

## Pipelines

Pipelines are a list of processes to apply to your dataset sequentially.
The default pipeline is `preprocess`, but you can define others in the configuration.

Each process in the pipeline operates on the current state of the dataset (Xarray Dataset) and the configuration object.

## Templates

Templates allow you to define reusable configurations for dataset structures.
This lets you avoid repeating parameters for datasets with the same structure.

> [!TIP]
> You can find an example of a template file in the [examples folder](./examples/template.json).

> [!IMPORTANT]
> Paths pointing to [S3](#s3-configuration) should be prefixed with `s3://`.

To use a template, reference keys in `templates.json` (or your custom templates file) using the `-t` or `--template` argument.

## Processes

These are the processes you can use in your pipelines.
Some processes require specific parameters in the configuration.

You can define those parameters globally at the root of the config object, or directly in the pipeline as following:
```json
{
  "version": 2, // Global parameter used in save preprocess to define Zarr version
  "pipelines": {
    "preprocess": [
      "save", // Here, "save" preprocess will use global parameter

      // But as this parameter is only used once in this pipeline, 
      // it can be defined directly in process params
      { "type": "process", "name": "save", "params": { "version": 2 } } 
    ]
  }
}
```

### `load_from_netcdf`

Load a dataset from NetCDF(s) file(s) from local storage or S3.

**Parameters:**

| Name         | Type   | Description                                                                 |
| ------------ | ------ | --------------------------------------------------------------------------- |
| `load_path`  | String | Path to file or folder (Defaults to `INPUT_PATH` from command line)         |
| `concat_dim` | String | Dimension on which to merge NetCDF files, if `load_path` is a directory     |

### `load_from_grib`

Load a dataset from GRIB(s) file(s) from local storage or S3.

**Parameters:**

| Name         | Type   | Description                                                                 |
| ------------ | ------ | --------------------------------------------------------------------------- |
| `load_path`  | String | Path to file or folder (Defaults to `INPUT_PATH` from command line)         |
| `concat_dim` | String | Dimension on which to merge GRIB files, if `load_path` is a directory       |

### `load_from_zarr`

Load a Zarr dataset from local storage or S3.

**Parameters:**

| Name        | Type   | Description                                                                 |
| ----------- | ------ | --------------------------------------------------------------------------- |
| `load_path` | String | Path to Zarr store (Defaults to `INPUT_PATH` from command line)             |

### `assign_coords`

Create Xarray coordinates, which allow to select data with a variable value (physical quantity) instead of the corresponding dimension (index).

**Parameters:**

| Name            | Type   | Description                                                                 |
| --------------- | ------ | --------------------------------------------------------------------------- |
| `assign_coords` | Object | Dictionary mapping variable names to dimension names (or templates)         |

**Format:**

The `assign_coords` parameter is a dictionary.
*   **Simple case:** `{ "variable_name": "dimension_name" }`
*   **Template case:** Use this when variables are spread across multiple indices and follow a naming pattern.

```json
"assign_coords": {
  "prefix_{i}_suffix": {
    "dim": "dim_{i}",
    "variables": {
      "i": { "min": 1, "max": 10 }
    }
  }
}
```
This will assign coordinates for `prefix_1_suffix` to `dim_1`, etc.

### `unify_chunks`

Re-chunk the dataset to ensure chunks are unified.
This process is recommended before saving to avoid performance issues, especially when loading from multiple files.

**No parameters.**

### `rename_variables`

Rename existing variables in the dataset.

**Parameters:**

| Name         | Type   | Description                                                                 |
| ------------ | ------ | --------------------------------------------------------------------------- |
| `rename_map` | Object | Dictionary where keys are current names and values are new names            |

### `delta_time_to_datetime`

If your dataset uses a time dimension with an offset from a reference date (e.g., "hours since ..."), this process converts those offsets into actual datetime objects.

**Parameters:**

| Name                         | Type    | Description                                                                                                       |
| ---------------------------- | ------- | ----------------------------------------------------------------------------------------------------------------- |
| `referenceTime.variable`     | String  | Name of the variable containing the reference time string                                                         |
| `referenceTime.format`       | String  | Format string of the reference time (e.g. "%Y-%m-%d %H:%M:%S")                                                     |
| `referenceTime.delta_unit`   | String  | Unit of the offset values (e.g., 'h', 'D'). Optional if it can be deduced from unit attribute                     |
| `variables.time`             | String  | Name of the variable containing the time offsets                                                                  |
| `dimensions.time`            | String  | Name of the dimension associated with time (Optional, deduced if possible)                                        |
| `updateTimeVar`              | Boolean | If true (default), replaces the `variables.time` variable with the computed datetimes                             |


### `reproject_coordinates`

Perform reprojection on geographic coordinates (e.g., Lambert 93 to WGS84).

**Parameters:**

| Name                      | Type   | Description                                           |
| ------------------------- | ------ | ----------------------------------------------------- |
| `reprojection.from_crs`   | String | Source CRS (e.g., "EPSG:2154")                        |
| `reprojection.to_crs`     | String | Target CRS (e.g., "EPSG:4326")                        |
| `variables.lon`           | String | Name of the longitude variable                        |
| `variables.lat`           | String | Name of the latitude variable                         |
| `variables.height`        | String | Name of the height variable (Optional)                |

### `save`

Save the resulting Zarr dataset to local storage or S3.

**Parameters:**

| Name                 | Type    | Description                                                                                                       |
| -------------------- | ------- | ----------------------------------------------------------------------------------------------------------------- |
| `save_path`          | String  | Path to save the dataset. Defaults to `INPUT_PATH` with `.zarr` extension. Prefix with `s3://` for [S3](#s3-configuration).            |
| `version`            | Integer | Zarr format version (2 or 3). Default: 3                                                                          |
| `float64_to_float32` | Boolean | Convert float64 data to float32 to save space. Default: false                                                     |

### `register_on_api`

Send minimal metadata concerning the newly created dataset to a kazarr API

Parameters:
| Name                      | Type   | Description                                                                                                                                  |
| ------------------------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| registration_endpoint_url | String | URL to API endpoint on which dataset metadata will be posted. You can use `--rgstr-endpoint` parameter in command line to define this value. |
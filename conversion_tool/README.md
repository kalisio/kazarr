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
pip install python-dotenv xarray zarr cfgrib h5py h5netcdf numpy pyproj dask distributed s3fs matplotlib pyvista pyinstaller
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
conversion_tool new-dataset [OPTIONS] INPUT_PATH
```

**Arguments:**

*   `INPUT_PATH`: Path of the input file or folder used for generating this new dataset (local or `s3://`).

**Options:**

*   `-t, --template TEXT`: [Template](#templates) to use for the configuration of the new dataset.
*   `-c, --config JSON`: Additional [configuration](#config) as JSON string.
*   `-f, --config-file PATH`: Path to a JSON file containing additional configuration.
*   `-a, --args TEXT`: Additional arg to pass to templates. Expected format: key=value. Can be used multiple times. Can be accessed from templates by using "ARGS.key" as a value for any field.
*   `-d, --description TEXT`: Description of the new dataset.
*   `-o, --output PATH`: Output path for the processed dataset (local or `s3://`).
*   `-p, --pipeline TEXT`: [Pipeline](#pipelines) to use for processing the new dataset [default: preprocess].
*   `--templates-path PATH`: Path to [templates](#templates) configuration file [default: templates.json].
*   `--data-mapping`: Whether to map data on mesh vertices or celles [default: vertices].
*   `--mesh-type`: Type of mesh to generate (default: auto, which infers from data between regular and rectilinear but not able to handle radial meshes)
*   `--custom-eccodes-path`: Path to a folder containing extra ecCodes. This path must be set to the parent folder containing the grib1, grib2, etc. subdirectories.
*   `--dask-dashboard`: Start a Dask dashboard for monitoring the processing
*   `--s3-storage-class`: Define which class to use for saving Zarr dataset to S3 [default: `STANDARD`]  
choices: `STANDARD`, `REDUCED_REDUNDANCY`, `STANDARD_IA`, `ONEZONE_IA`, `INTELLIGENT_TIERING`, `GLACIER`, `DEEP_ARCHIVE`, `OUTPOSTS`, `GLACIER_IR`, `SNOW`, `EXPRESS_ONEZONE`, `FSX_OPENZFS`

> [!TIP]
> You can also define the path to custom ecCodes definitions folder with the `CUSTOM_ECCODES_PATH` environment variable

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

> [!TIP]
> You can add variables in command line with `-a key=value` and use it in templates with `"ARGS.key"` (only strings and int and float values are supported)

To use a template, reference keys in `templates.json` (or your custom templates file) using the `-t` or `--template` argument.

### Tips to create a template file

The best way to write a template is to first inspect your raw data using Python and xarray. This allows you to see the dataset structure exactly as the Kazarr conversion tool will see it, making it easier to choose which processes to apply.

```python
import xarray as xr

# Open the dataset (use engine="cfgrib" for GRIB or "h5netcdf" for NetCDF)
dataset = xr.open_dataset("path/to/dataset", engine="cfgrib", chunks="auto")

# 1. Check Dimensions and Coordinates
dataset.dims   # Identify dimensions (useful for 'concat_dim' in load processes)
dataset.coords # If some variables act as coordinates but are listed as data_vars, you will need the `assign_coords` process.

# 2. List Variables
dataset.data_vars.keys() # Use these names to configure `keep_variables`, `exclude_variables`, or `rename_variables`.

# 3. Inspect Attributes
dataset.attrs                # Global metadata
dataset["myVariable"].attrs  # Variable metadata. Check 'units' here to see if you need the `delta_time_to_datetime` process (e.g., "hours since...").

# 4. Check Data Types and Chunks
dataset["myVariable"].dtype  # If float64, you might want to enable `float64_to_float32` in the `save` process to reduce file size.
```

By exploring the dataset, you can quickly deduce your pipeline needs:
  - Time offset? If the time is an integer with a unit like "hours since 1970-01-01", add delta_time_to_datetime.
  - Local projection? If X/Y coordinates are in a specific CRS (like Lambert 93), note their exact names to configure reproject_coordinates.
  - Cluttered data? If dataset.data_vars shows 50 variables but you only need 3, use keep_variables.

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

### `init_dask_dashboard`

Start a Dask client to access a dashboard from a browser, allowing to view Dask tasks during processing. This process will automatically be added to the list if `--dask-dashboard` is provided on the command line.

### `load_from_netcdf`

Load a dataset from NetCDF(s) file(s) from local storage or S3.

**Parameters:**

| Name                 | Type    | Description                                                                                                                                         |
| -------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `load_path`          | String  | Path to file or folder (Defaults to `INPUT_PATH` from command line)                                                                                 |
| `concat_dim`         | String  | Dimension on which to merge NetCDF files, if `load_path` is a directory                                                                             |
| `store_as_secondary` | Boolean | Whether or not store this dataset as a secondary dataset. Usefull for processes that need multiples datasets. Default: `false`                      |
| `secondary_tag`      | String  | When `store_as_secondary` is true, define a key to this newly loaded dataset. Default: `secondary_X` where X is the xth loaded dataset as secondary |

### `load_from_grib`

Load a dataset from GRIB(s) file(s) from local storage or S3.

**Parameters:**

| Name                 | Type    | Description                                                                                                                                         |
| -------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `load_path`          | String  | Path to file or folder (Defaults to `INPUT_PATH` from command line)                                                                                 |
| `concat_dim`         | String  | Dimension on which to merge GRIB files, if `load_path` is a directory                                                                               |
| `store_as_secondary` | Boolean | Whether or not store this dataset as a secondary dataset. Usefull for processes that need multiples datasets. Default: `false`                      |
| `secondary_tag`      | String  | When `store_as_secondary` is true, define a key to this newly loaded dataset. Default: `secondary_X` where X is the xth loaded dataset as secondary |

### `load_from_zarr`

Load a Zarr dataset from local storage or S3.

**Parameters:**

| Name                 | Type    | Description                                                                                                                                         |
| -------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `load_path`          | String  | Path to Zarr store (Defaults to `INPUT_PATH` from command line)                                                                                     |
| `store_as_secondary` | Boolean | Whether or not store this dataset as a secondary dataset. Usefull for processes that need multiples datasets. Default: `false`                      |
| `secondary_tag`      | String  | When `store_as_secondary` is true, define a key to this newly loaded dataset. Default: `secondary_X` where X is the xth loaded dataset as secondary |

### `load_and_merge_from_grib`

For a folder of GRIB files, allow to concatenate some files (filtered by a string discriminator) with a binary concatenation into a new temporary GRIB dataset, and then, merge all of thoses into a XArray dataset.
For example, this process can be used with files splitted over time and packages. This process will concatenate each package over time and then merge all packages.

**Parameters:**

| Name                     | Type          | Description                                                                                                                                                                                            |
| ------------------------ | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `discriminator`          | List<String\> | String to find in each file name you want to group to be concatenated                                                                                                                                  |
| `rename_before_merge`    | List<Object\> | In some cases, two datasets can have variables with same name, and that will cause merge to fail. For each group, you should define an object with original variable name as key and new one as value  |
| `dataset_backend_kwargs` | List<Object\> | For each group of file that will be concatenated, you can provide args to cfgrib engine (e.g. `{"filter_by_keys": {"shortName": ["MyVariable"]}` will only consider `MyVariable` when loading dataset) |
| `merge_in_place`         | Boolean       | If true, will use Xarray to concatenate files, instead of creating a temporary GRIB file. This may increase needed memory                                                                              |
| `path`                   | String        | Path to the folder where GRIB files are stored                                                                                                                                                         |
| `store_as_secondary`     | Boolean       | Whether or not store this dataset as a secondary dataset. Usefull for processes that need multiples datasets. Default: `false`                                                                         |
| `secondary_tag`          | String        | When `store_as_secondary` is true, define a key to this newly loaded dataset. Default: `secondary_X` where X is the xth loaded dataset as secondary                                                    |

### `combine_at_time`

Merges two datasets sharing the same base by splitting them at a given point in time: values from the primary dataset are retained up to (inclusive) that point, and values from the secondary dataset are retained from that point onward (exclusive).

> Typical use case: a study is recomputed following parameter changes, but historical values from the original run must be preserved as the ground truth.

**Prerequisites**:

This operation requires two datasets loaded in your pipeline:

- The primary dataset, loaded normally
- The secondary dataset, must be loaded with `"store_as_secondary": true`

__Example__:

```json
{
  // ...
  "pipelines": {
    "myPipeline": [
      "load_from_zarr", // This process will use INPUT_PATH from the command line
      { "type": "process", "name": "load_from_zarr", "params": { "load_path": "/path/to/my/secondary/zarr/dataset.zarr", "store_as_secondary": true, "secondary_tag": "mySecondaryDataset" } },
      { "type": "proces", "name": "combine_at_time", "params": { "combine_time": "2026-06-12 16:00", "combine_time_format": "%Y-%m-%d %H:%M", "combine_dataset_tag": "mySecondaryDataset" } },
      // ...
    ]
  }
}
```

**Parameters:**

| Name                                         | Type   | Description                                                                                                                                                                                                                                                                                      |
| -------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `combine_time`                               | String | The split time. Data before this value is taken from the primary dataset; data from this value onward is taken from the secondary dataset.                                                                                                                                                       |
| `combine_time_format`                        | String | Format of the `combine_time` ([in Python format](https://docs.python.org/3.11/library/datetime.html#strftime-and-strptime-format-codes)). Default: `%Y-%m-%dT%H:%M:%S`                                                                                                                           |
| `combine_dataset_tag`                        | String | Tag identifying the secondary dataset to merge with. Default: `secondary_1`                                                                                                                                                                                                                      |
| `variables.time`                             | String | Name of the time variable in the dataset.                                                                                                                                                                                                                                                        |
| `variables.lon`                              | String | Name of the longitude variable. Optional — used to detect point-list datasets and align spatial dimensions between the two datasets before concatenation.                                                                                                                                        |
| `variables.lat`                              | String | Name of the latitude variable. Optional — same role as `variables.lon`.                                                                                                                                                                                                                          |
| `variables.level`                            | String | Name of the level variable. Optional — used to align the level dimension between the two datasets before concatenation.                                                                                                                                                                          |
| `combine_point_discriminator_var`            | String | For point-list datasets: name of the variable used to discriminate individual points (e.g. a station name variable). If omitted, the process auto-detects a string variable sharing the same dimension as `variables.lon`/`lat`.                                                                 |
| `attr_indexed_var_renaming`                  | Object | Optional. When the two datasets share variables whose names embed a numeric index (e.g. `MyVar-0`, `MyVar-1`), and that index is described in the dataset attributes, use this option to re-align those indices before concatenation. See the dedicated section below.                           |
| `attr_indexed_var_renaming.attr_pattern`     | String | Regex (Python `re.fullmatch`) applied to every **attribute key** of the dataset. Must contain two named groups: `(?P<name>...)` to capture the attribute name prefix, and `(?P<index>...)` to capture the numeric index.                                                                         |
| `attr_indexed_var_renaming.target_attr_name` | String | The exact value that `(?P<name>...)` must match to select the attributes used as the index-to-entity mapping (e.g. `"MyVarName"`). All other matching attributes are still renamed if their index changes, but only attributes whose name prefix equals this value are used to build the mapping. |
| `attr_indexed_var_renaming.var_pattern`      | String | Regex (Python `re.fullmatch`) applied to every **variable name**. Must contain at least `(?P<index>...)` to identify the index embedded in the name. Any other named group is passed to `var_template`.                                                                                          |
| `attr_indexed_var_renaming.var_template`     | String | Python format string used to reconstruct a variable name after its index has been updated. Uses the named groups from `var_pattern` (e.g. `"{name}-{index}"`).                                                                                                                                   |

#### `attr_indexed_var_renaming` — detailed explanation

Some datasets encode multiple entities (e.g. measurement stations) as a set of parallel variables, each suffixed with a numeric index, where the mapping between index and entity name lives in the dataset **attributes**. When combining two such datasets, the same entity may carry a different index in each, causing variable name conflicts or data mix-ups.

`attr_indexed_var_renaming` resolves this automatically:

1. It scans both datasets' attributes with `attr_pattern` and `target_attr_name` to build an **index ↔ entity name** map for each dataset.
2. For each entity in the secondary dataset it finds the correct index from the primary, and renames all secondary variables (matched by `var_pattern`) and attributes accordingly.
3. Entities that appear only in the secondary dataset are assigned a new index that continues from the highest index found in the primary dataset.

> [!WARNING]
> The `attr_pattern` regex must use a **non-greedy** quantifier for the `name` group (e.g. `.+?` instead of `.+`) to correctly capture multi-digit indices. With a greedy `.+`, `MyVarName10` would be parsed as `name=MyVarName1, index=0`, causing the entity at index 10 to be silently ignored.

**Example** — attributes are named `MyVarName0`, `MyVarName1`... (values are entity names) and variables are named `MyVar-0`, `MyVar-1`...:

```json
"attr_indexed_var_renaming": {
  "attr_pattern": "^(?P<name>.+?)(?P<index>\\d+)$",
  "target_attr_name": "MyVarName",
  "var_pattern": "^(?P<name>.+)-(?P<index>\\d+)$",
  "var_template": "{name}-{index}"
}
```

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

### `exclude_variables`

Exclude some variables from the dataset

**Parameters:**

| Name           | Type         | Description                                   |
| -------------- | ------------ | --------------------------------------------- |
| `exclude_vars` | List<String\> | List of variables to exclude from the dataset |

### `keep_variables`

Opposite of the previous process: keep only provided variables

**Parameters:**

| Name        | Type         | Description               |
| ----------- | ------------ | ------------------------- |
| `keep_vars` | List<String\> | List of variables to keep |

### `delta_time_to_datetime`

If your dataset uses a time dimension with an offset from a reference date (e.g., "hours since ..."), this process converts those offsets into actual datetime objects.

**Parameters:**

| Name                       | Type    | Description                                                                                   |
| -------------------------- | ------- | --------------------------------------------------------------------------------------------- |
| `referenceTime.variable`   | String  | Name of the variable containing the reference time string                                     |
| `referenceTime.format`     | String  | Format string of the reference time (e.g. "%Y-%m-%d %H:%M:%S", or "timestamp")                |
| `referenceTime.delta_unit` | String  | Unit of the offset values (e.g., 'h', 'D'). Optional if it can be deduced from unit attribute |
| `variables.time`           | String  | Name of the variable containing the time offsets                                              |
| `dimensions.time`          | String  | Name of the dimension associated with time (Optional, deduced if possible)                    |
| `updateTimeVar`            | Boolean | If true (default), replaces the `variables.time` variable with the computed datetimes         |


### `reproject_coordinates`

Perform reprojection on geographic coordinates (e.g., Lambert 93 to WGS84).

**Parameters:**

| Name                    | Type   | Description                           |
| ----------------------- | ------ | ------------------------------------- |
| `reprojection.from_crs` | String | Source CRS (e.g., "EPSG:2154")        |
| `reprojection.to_crs`   | String | Target CRS (e.g., "EPSG:4326")        |
| `variables.lon`         | String | Name of the longitude variable        |
| `variables.lat`         | String | Name of the latitude variable         |
| `variables.level`       | String | Name of the level variable (Optional) |


### `simplify_grid`

Attempts to convert an irregular grid (where `lat`, `lon`, and `level` are 2D or 3D arrays) into a regular one (1D arrays). 
This process can result in a fully regular grid, a hybrid grid (e.g., 1D `level` with 2D `lat`/`lon`), or remain unchanged if simplification is not possible.

**Parameters:**

| Name                    | Type   | Description                           |
| ----------------------- | ------ | ------------------------------------- |
| `variables.lon`         | String | Name of the longitude variable        |
| `variables.lat`         | String | Name of the latitude variable         |
| `variables.level`       | String | Name of the level variable (Optional) |

### `save`

Save the resulting Zarr dataset to local storage or S3.

**Parameters:**

| Name                 | Type    | Description                                                                                                       |
| -------------------- | ------- | ----------------------------------------------------------------------------------------------------------------- |
| `save_path`          | String  | Path to save the dataset. Defaults to `INPUT_PATH` with `.zarr` extension. Prefix with `s3://` for [S3](#s3-configuration).            |
| `version`            | Integer | Zarr format version (2 or 3). Default: 3                                                                          |
| `float64_to_float32` | Boolean | Convert float64 data to float32 to save space. Default: false                                                     |

### `clean`

Delete all files generated or used during any of thoses processes

**Parameters:**

| Name    | Type   | Description                                                                                                                                                                                                                                                                                                                                                                                                           |
| ------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `clean` | Object | Define which files should be deleted. Thoses keys can be defined in this object: "used", "generated", "idx", and must have a boolean value. "used" (default: False) correspond to files that have been used for any process. "generated" (default: True) correspond to files that have been generated by any process. "idx" (default: True) correspond to index files generated by Xarray for some formats like GRIB. |
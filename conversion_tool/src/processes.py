import os
import itertools
import re
import shutil
from pathlib import Path
from datetime import datetime

import xarray as xr
import numpy as np
import s3fs
from pyproj import Transformer
import cfgrib
from platformdirs import user_cache_dir

from dask.distributed import Client, performance_report

from src.utils import (
    get_dataset_config_value,
    get_ci,
    merge,
    rechunk_if_needed,
    merge_grib,
    get_s3_storage_options,
    get_s3_filesystem,
    parse_datetime,
    get_redundant_dimensions,
    get_spatial_variables,
    add_to_clean_config,
    init_store_as_secondary,
)


S3_PREFIX = "s3://"
BUCKET_NAME_ENV_VAR = "BUCKET_NAME"
LON_VARIABLE_KEY = "variables.lon"
LAT_VARIABLE_KEY = "variables.lat"
LEVEL_VARIABLE_KEY = "variables.level"


def init_dask_dashboard(dataset, config):
    """Initialize the Dask dashboard to monitor computation progress and load."""
    client = Client()
    link = client.dashboard_link
    print("======================================" + (len(link) * "="))
    print(f"[KAZARR] Dask dashboard available at: {link}")
    print("======================================" + (len(link) * "="))
    config["dask_dashboard_initialised"] = True

    return dataset, config


def load_from_netcdf(dataset, config):
    """Load a dataset from local or S3 NetCDF files."""
    ds_count = 0
    total_count = 0

    def progress_callback(ds):
        nonlocal ds_count, total_count
        ds_count += 1
        percentage = ds_count / total_count * 100
        print(f"Progress: {ds_count} / {total_count} ({percentage:.2f}%)")
        return ds

    path = get_dataset_config_value(
        dataset,
        config,
        "load_path",
        default=get_ci(config, "path"),
        error_message="Missing 'load_path' or 'path' config parameters for load_from_netcdf process.",
    )

    new_dataset = None
    if path.startswith(S3_PREFIX):
        bucket = os.getenv(BUCKET_NAME_ENV_VAR)
        if bucket is None:
            raise ValueError(f"{BUCKET_NAME_ENV_VAR} environment variable not set.")
        path = path.replace(S3_PREFIX, "")
        path = os.path.join(bucket, path)

        # Check if path is folder or file
        fs = get_s3_filesystem(config, path)
        if fs.isfile(path):
            new_dataset = xr.open_dataset(
                fs.open(path, mode="rb", cache_type="bytes"),
                engine="h5netcdf",
                chunks="auto",
            )
        elif fs.isdir(path):
            concat_dim = get_dataset_config_value(
                dataset,
                config,
                "concat_dim",
                error_message="Missing 'concat_dim' config parameter for loading multiple NetCDF files from S3 folder.",
            )
            files = fs.glob(os.path.join(path, "*.nc"))
            files = sorted(files)
            s3_files = [fs.open(f) for f in files]
            total_count = len(s3_files)

            print(f"Loading {total_count} NetCDF files from S3 folder {path}...")

            new_dataset = xr.open_mfdataset(
                s3_files,
                engine="h5netcdf",
                combine="nested",
                compat="broadcast_equals",
                concat_dim=concat_dim,
                data_vars="minimal",
                coords="minimal",
                parallel=True,
                chunks="auto",
                preprocess=progress_callback,
            )
    else:
        if os.path.isfile(path):
            new_dataset = xr.open_dataset(path, chunks="auto", engine="h5netcdf")
        elif os.path.isdir(path):
            concat_dim = get_dataset_config_value(
                dataset,
                config,
                "concat_dim",
                error_message="Missing 'concat_dim' for loading multiple NetCDF files from folder.",
            )
            files = [
                os.path.join(path, f) for f in os.listdir(path) if f.endswith(".nc")
            ]
            files = sorted(files)
            total_count = len(files)

            new_dataset = xr.open_mfdataset(
                files,
                engine="h5netcdf",
                combine="nested",
                compat="broadcast_equals",
                concat_dim=concat_dim,
                data_vars="minimal",
                coords="minimal",
                parallel=True,
                chunks="auto",
                preprocess=progress_callback,
            )

    if new_dataset is None:
        raise ValueError(f"Unable to load NetCDF dataset from path: {path}")
    return init_store_as_secondary(dataset, new_dataset, config)


def load_from_grib(dataset, config):
    """Load a dataset from local or S3 GRIB files."""
    ds_count = 0
    total_count = 0

    def progress_callback(ds):
        nonlocal ds_count, total_count
        ds_count += 1
        percentage = ds_count / total_count * 100
        print(f"Progress: {ds_count} / {total_count} ({percentage:.2f}%)")
        return ds

    path = get_dataset_config_value(
        dataset,
        config,
        "load_path",
        default=get_ci(config, "path"),
        error_message="Missing 'load_path' or 'path' config parameters for load_from_grib process.",
    )
    file_pattern = get_dataset_config_value(
        dataset, config, "file_regex", default="*.grib2"
    )
    backend_kwargs = get_dataset_config_value(
        dataset, config, "backend_kwargs", default={}
    )

    new_dataset = None
    if path.startswith(S3_PREFIX):
        bucket = os.getenv(BUCKET_NAME_ENV_VAR)
        if bucket is None:
            raise ValueError(f"{BUCKET_NAME_ENV_VAR} environment variable not set.")
        path = path.replace(S3_PREFIX, "")
        path = os.path.join(bucket, path)

        # Check if path is folder or file
        fs = get_s3_filesystem(config, path)
        target_tmp_dir = "/tmp/kazarr_grib/"
        if fs.isfile(path):
            target_tmp_dir = user_cache_dir(
                appname="kazarr", appauthor=False, version="gribs"
            )
            os.makedirs(target_tmp_dir, exist_ok=True)
            if not os.path.exists(os.path.join(target_tmp_dir, os.path.basename(path))):
                fs.get(path, os.path.join(target_tmp_dir, os.path.basename(path)))
                config = add_to_clean_config(config, "used_paths", os.path.join(target_tmp_dir, os.path.basename(path)))
            new_dataset = xr.open_dataset(
                os.path.join(target_tmp_dir, os.path.basename(path)),
                engine="cfgrib",
                chunks="auto",
                backend_kwargs=backend_kwargs,
            )
        elif fs.isdir(path):
            concat_dim = get_dataset_config_value(
                dataset,
                config,
                "concat_dim",
                error_message="Missing 'concat_dim' config parameter for loading multiple GRIB files from S3 folder.",
            )

            all_files = fs.ls(os.path)
            files = [
                f for f in all_files if re.search(file_pattern, os.path.basename(f))
            ]

            if not files:
                raise ValueError(
                    f"No files matching regex '{file_pattern}' found in S3 path: {path}"
                )

            files = sorted(files)
            s3_files = [fs.open(f) for f in files]
            total_count = len(s3_files)

            new_dataset = xr.open_mfdataset(
                s3_files,
                engine="cfgrib",
                combine="nested",
                compat="broadcast_equals",
                concat_dim=concat_dim,
                data_vars="minimal",
                coords="minimal",
                parallel=False,
                chunks="auto",
                preprocess=progress_callback,
                **backend_kwargs,
            )
    else:
        if os.path.isfile(path):
            backend_kwargs["errors"] = "raise"
            try:
                new_dataset = xr.open_dataset(
                    path, engine="cfgrib", chunks="auto", backend_kwargs=backend_kwargs
                )
            except cfgrib.dataset.DatasetBuildError:
                print(
                    "[KAZARR] Failed to load GRIB file with Xarray, trying to open as multiple datasets... This may cause higher memory usage and slower performance."
                )
                datasets = cfgrib.open_datasets(path, backend_kwargs=backend_kwargs)
                full_variables = set()
                final_variables = set()
                for ds in datasets:
                    full_variables.update(ds.data_vars)
                    full_variables.update(ds.coords)
                if len(datasets) == 1:
                    new_dataset = datasets[0].chunk("auto")
                else:
                    try:
                        print(
                            "[KAZARR] Multiple datasets found in GRIB file, attempting to merge them..."
                        )
                        # Use chunk("auto") on each dataset to enable Dask parallelism during merge, which can help reduce memory usage and speed up the process
                        for i, ds in enumerate(datasets):
                            datasets[i] = ds.chunk("auto")
                        new_dataset = xr.merge(datasets, compat="minimal", join="outer")
                        final_variables.update(new_dataset.data_vars)
                        final_variables.update(new_dataset.coords)
                        removed_variables = full_variables - final_variables
                        if removed_variables:
                            print(
                                f"         ! The following variables were removed during merge due to conflicts: {removed_variables}"
                            )
                    except Exception as e:
                        print(f"Error occurred while merging datasets: {e}")
        elif os.path.isdir(path):
            concat_dim = get_dataset_config_value(
                dataset,
                config,
                "concat_dim",
                error_message="Missing 'concat_dim' for loading multiple GRIB files from folder.",
            )

            files = [
                os.path.join(path, f)
                for f in os.listdir(path)
                if re.search(file_pattern, f)
            ]

            if not files:
                raise ValueError(
                    f"No files matching regex '{file_pattern}' found in local path: {path}"
                )

            files = sorted(files)
            total_count = len(files)

            new_dataset = xr.open_mfdataset(
                files,
                engine="cfgrib",
                combine="nested",
                compat="broadcast_equals",
                concat_dim=concat_dim,
                data_vars="minimal",
                coords="minimal",
                parallel=False,
                chunks="auto",
                preprocess=progress_callback,
                **backend_kwargs,
            )
    if new_dataset is None:
        raise ValueError(f"Unable to load GRIB dataset from path: {path}")
    return init_store_as_secondary(dataset, new_dataset, config)


def load_from_zarr(dataset, config):
    """Load a dataset from a local or S3 Zarr store."""
    path = get_dataset_config_value(
        dataset,
        config,
        "load_path",
        default=get_ci(config, "path"),
        error_message="Missing 'load_path' or 'path' config parameters for load_from_zarr process.",
    )
    if path.startswith(S3_PREFIX):
        bucket = os.getenv(BUCKET_NAME_ENV_VAR)
        if bucket is None:
            raise ValueError(f"{BUCKET_NAME_ENV_VAR} environment variable not set.")
        path = path.replace(S3_PREFIX, "")
        path = os.path.join(bucket, path)

        # check=false to avoid checking (1 more request) for existence of root in the store
        s3_store = s3fs.S3Map(
            root=path, s3=get_s3_filesystem(config, path), check=False
        )
        new_dataset = xr.open_zarr(s3_store, chunks="auto")
    else:
        new_dataset = xr.open_zarr(path, chunks="auto")
    return init_store_as_secondary(dataset, new_dataset, config)


def load_and_merge_from_grib(dataset, config):
    """Load and merge multiple GRIB files based on the specified discriminators."""

    # List of strings that can be used to discriminate files to merge together
    # TODO: improve this by allowing to extract discriminator values with regex capture groups, to support more complex cases
    discriminator = get_dataset_config_value(
        dataset,
        config,
        "discriminator",
        error_message="Missing 'discriminator' config parameter for load_and_merge_from_grib process.",
    )
    rename_before_merge = get_dataset_config_value(
        dataset, config, "rename_before_merge", default=[]
    )
    backend_kwargs = get_dataset_config_value(
        dataset, config, "dataset_backend_kwargs", default=[]
    )
    in_place = get_dataset_config_value(
        dataset, config, "merge_in_place", default=False
    )
    path = get_dataset_config_value(
        dataset,
        config,
        "load_path",
        default=get_ci(config, "path"),
        error_message="Missing 'load_path' or 'path' config parameters for load_and merge_from_grib process.",
    )

    if isinstance(discriminator, str):
        discriminators = [discriminator]
    elif isinstance(discriminator, list):
        discriminators = discriminator

    sub_datasets = []
    for index, discriminator in enumerate(discriminators):
        try:
            if index < len(backend_kwargs):
                config["backend_kwargs"] = backend_kwargs[index]
            if in_place:  # TODO use only glob or regex, not both
                sub_dataset, _ = load_from_grib(
                    dataset,
                    merge({"file_regex": rf"^.*{discriminator}.*\.grib2$"}, config),
                )
            else:
                concat_filename = f"concatenated_{discriminator}_{index}.grib2"
                config = merge_grib(
                    path, concat_filename, config, f"*{discriminator}*.grib2"
                )
                sub_dataset, _ = load_from_grib(
                    dataset,
                    merge({"load_path": os.path.join(path, concat_filename)}, config),
                )
        except Exception:
            # Don't fail if a discriminator doesn't match any file, just skip it
            continue

        if index < len(rename_before_merge) and rename_before_merge[index]:
            sub_dataset = sub_dataset.rename(rename_before_merge[index])

        sub_datasets.append(sub_dataset)

    if sub_datasets:
        merged_dataset = xr.merge(sub_datasets)
        new_dataset = rechunk_if_needed(
            merged_dataset
        )  # Re-chunk usually needed after merge
    else:
        new_dataset = None

    if new_dataset is None:
        raise ValueError(
            f"Unable to find any files matching discriminators {discriminators} in path: {path} for load_and_merge_from_grib process."
        )

    return init_store_as_secondary(dataset, new_dataset, config)


def combine_at_time(dataset, config):
    """Combine the main dataset with another dataset along the time dimension."""
    combine_time = get_dataset_config_value(
        dataset,
        config,
        "combine_time",
        error_message="Missing 'combine_time' config parameter for combine_at_time process.",
    )
    combine_time_format = get_dataset_config_value(
        dataset,
        config,
        "combine_time_format",
        default="%Y-%m-%dT%H:%M:%S",
    )
    combine_dataset_tag = get_dataset_config_value(
        dataset, config, "combine_dataset_tag", default="secondary_1"
    )
    time_var = get_dataset_config_value(
        dataset,
        config,
        "variables.time",
        error_message="Missing 'variables.time' config parameter for combine_at_time process.",
    )

    # Those variables will be used to check if spatial dimensions have changed between the two datasets,
    # and to pad the first dataset if needed to align dimensions before concatenation.
    # They are not mandatory, as the process can still work without them,
    # but if they are present in the secondary dataset they should be checked against the primary dataset
    # to ensure proper alignment during concatenation.
    # TODO: Those variables must be static and not ATTRS dependent (e.g. ATTRS.level_type)
    lon_var = get_ci(config, LON_VARIABLE_KEY)
    lat_var = get_ci(config, LAT_VARIABLE_KEY)
    level_var = get_ci(config, LEVEL_VARIABLE_KEY)

    if (
        "secondary_datasets" not in config
        or combine_dataset_tag not in config["secondary_datasets"]
    ):
        raise ValueError(
            f"Dataset with tag '{combine_dataset_tag}' not found in config 'secondary_datasets' for combine_at_time process."
        )

    combine_dataset = config["secondary_datasets"][combine_dataset_tag]

    if time_var not in dataset:
        raise ValueError(
            f"Time variable '{time_var}' not found in the main dataset for combine_at_time process."
        )
    if time_var not in combine_dataset:
        raise ValueError(
            f"Time variable '{time_var}' not found in the secondary dataset '{combine_dataset_tag}' for combine_at_time process."
        )
    
    if dataset[time_var].ndim != 1:
        raise ValueError(
            f"Time variable '{time_var}' in the main dataset has {dataset[time_var].ndims} dimensions, but a 1-dimensional time variable is required for combine_at_time process."
        )
    
    time_dim = dataset[time_var].dims[0]

    if np.issubdtype(dataset[time_var].dtype, np.datetime64) or np.issubdtype(
        combine_dataset[time_var].dtype, np.datetime64
    ):
        try:
            combine_time = parse_datetime(combine_time, combine_time_format)
        except Exception as e:
            raise ValueError(
                f"Failed to parse 'combine_time' value '{combine_time}' with format '{combine_time_format}' for combine_at_time process. Original error: {e}"
            ) from e

    try:
        primary_before = dataset.sel({time_var: dataset[time_var] <= combine_time})
        secondary_after = combine_dataset.sel(
            {time_var: combine_dataset[time_var] > combine_time}
        )
    except Exception as e:
        raise ValueError(
            f"Failed to split datasets at 'combine_time' value '{combine_time}' "
            f"(type: {type(combine_time).__name__}) on time variable '{time_var}' "
            f"(dtype: {dataset[time_var].dtype}). Check that 'combine_time' is compatible "
            f"with the time variable's dtype. Original error: {e}"
        ) from e

    if len(primary_before[time_var]) == 0:
        raise ValueError(
            (
                f"'combine_time' value '{combine_time}' is earlier than or equal to all timestamps "
                f"in the primary dataset (earliest: {dataset[time_var].values[0] if len(dataset[time_var]) > 0 else 'N/A'}). "
                f"The primary dataset contributes no data to the combined result."
            )
        )
    if len(secondary_after[time_var]) == 0:
        raise ValueError(
            (
                f"'combine_time' value '{combine_time}' is later than or equal to all timestamps "
                f"in the secondary dataset '{combine_dataset_tag}' "
                f"(latest: {combine_dataset[time_var].values[-1] if len(combine_dataset[time_var]) > 0 else 'N/A'}). "
                f"The secondary dataset contributes no data to the combined result."
            )
        )

    # For list point case, we need to find a variable to discriminate points, and set index of this variable
    spatial_vars_defined = lon_var is not None and lat_var is not None
    spatial_vars_in_dataset = (
        spatial_vars_defined
        and lon_var in primary_before
        and lat_var in primary_before
        and lon_var in secondary_after
        and lat_var in secondary_after
    )
    lons, lats = None, None
    if spatial_vars_in_dataset:
        lons = secondary_after[lon_var]
        lats = secondary_after[lat_var]
    is_point_list = (
        lons is not None
        and lats is not None
        and (
            (lons.ndim == 1 and lats.ndim == 1 and lons.dims == lats.dims)
            or (lons.ndim == 0 and lats.ndim == 0)
        )
    )
    if is_point_list:
        point_discriminator_var = get_dataset_config_value(
            dataset,
            config,
            "combine_point_discriminator_var",
        )

        if point_discriminator_var is None:
            spatial_variable_dims = lons.dims
            corresponding_vars = []
            for var in list(primary_before.data_vars) + list(primary_before.coords):
                have_same_dims = primary_before[var].dims == spatial_variable_dims
                is_string_type = primary_before[var].dtype.kind in ("U", "S", "O")
                if have_same_dims and is_string_type:
                    corresponding_vars.append(var)

            if len(corresponding_vars) != 1:
                if len(corresponding_vars) > 1:
                    message_end = (
                        f"{corresponding_vars} were found. Please specify 'combine_point_discriminator_var' "
                        "config parameter to select which one to use."
                    )
                else:
                    message_end = (
                        "no variable with the same dimensions as the spatial variables and of string type was found. "
                        "Please specify 'combine_point_discriminator_var' config parameter to select a variable to use, "
                        "or check that the dataset contains a string variable with the same dimensions as the spatial variables."
                    )
                raise ValueError(
                    (
                        "This dataset was detected as a point list. To combine two datasets"
                        ", this process need a variable to discriminate points (names), "
                        f"but {message_end}"
                    )
                )

            index_target_var = corresponding_vars[0]
        else:
            index_target_var = point_discriminator_var

        if index_target_var not in secondary_after:
            raise ValueError(
                f"Variable '{index_target_var}' not found in the secondary dataset. Please specify a valid variable to discriminate points ('combine_point_discriminator_var' config parameter)."
            )
        
        if len(primary_before[index_target_var].dims) == 0:
            raise ValueError(
                (
                    f"Variable '{index_target_var}' selected as point discriminator "
                    "is a scalar variable in the primary dataset, but it needs to be"
                    " an array variable with the same dimensions as the spatial "
                    "variables to be used as an index for combining the datasets. "
                    "Please specify a valid variable to discriminate points "
                    "('combine_point_discriminator_var' config parameter) that is "
                    "an array variable with the same dimensions as the spatial "
                    "variables."
                )
            )

        index_dim_name = primary_before[index_target_var].dims[0]

        for label, ds in [("primary", primary_before), ("secondary", secondary_after)]:
            if index_dim_name not in ds.dims:
                raise ValueError(
                    f"Dimension '{index_dim_name}' (derived from discriminator variable "
                    f"'{index_target_var}') was not found among the {label} dataset dimensions: "
                    f"{list(ds.dims)}. Cannot set index for point-list combination."
                )

        primary_before = primary_before.set_index({index_dim_name: index_target_var})
        secondary_after = secondary_after.set_index({index_dim_name: index_target_var})

    if len(primary_before[time_var]) == 0:
        combined_dataset = secondary_after
    elif len(secondary_after[time_var]) == 0:
        combined_dataset = primary_before
    else:
        # Get dimensions used with spatial variables
        spatial_dimensions = []
        for var in [lon_var, lat_var, level_var]:
            if var and var in secondary_after:
                for dim in secondary_after[var].dims:
                    if dim not in spatial_dimensions:
                        spatial_dimensions.append(dim)

        # Check if spatial dimensions have changed between the two datasets
        for dim in spatial_dimensions:
            if dim in primary_before.dims:
                diff = secondary_after.sizes[dim] - primary_before.sizes[dim]
                # If so, pad the first dataset to align dimensions
                if diff > 0:
                    try:
                        primary_before = primary_before.pad(
                            {dim: (0, diff)}, constant_values=np.nan
                        )
                    except Exception as e:
                        raise ValueError(
                            f"Failed to pad primary dataset along dimension '{dim}' by {diff} element(s) "
                            f"to align with secondary dataset. This can happen if the dimension is an index "
                            f"or has an incompatible dtype. Original error: {e}"
                        ) from e

        try:
            combined_dataset = xr.concat(
                [primary_before, secondary_after], dim=time_dim
            )
        except Exception as e:
            raise ValueError(
                f"Failed to concatenate primary and secondary datasets along '{time_var}'. "
                f"This can happen if both datasets have variables with incompatible shapes or dtypes. "
                f"Primary variables: {list(primary_before.data_vars)}, "
                f"Secondary variables: {list(secondary_after.data_vars)}. "
                f"Original error: {e}"
            ) from e
        
    # Try to remove useless time dimension created by concat
    # 1. Remove time dimension from spatial variables:
    #   As grid can change between the two datasets, but the combined one
    # should have the shape of the two combined, we will keep the last version (last time step)
    # and fill NaN values with the first version (first time step) for the time steps where 
    # the grid is different, to avoid having NaN values for spatial variables in the 
    # combined dataset.
    def merge_vars(ds, var):
        if ds[var].ndim > 1 and time_dim in ds[var].dims:
            ds[var] = ds[var][-1].fillna(ds[var][0])
        return ds
    
    for var in [lon_var, lat_var, level_var]:
        if time_dim in combined_dataset[var].dims:
            combined_dataset = merge_vars(combined_dataset, var)

    # 2. Remove time dimension from scalar variables that have not changed
    #   Scalar variables will also have been concatenated along the 
    # time dimension, but if their value is the same in both 
    # datasets, we can just keep one version, and so, remove
    # the time dimension added by concat
    for var in combined_dataset.data_vars:
        if combined_dataset[var].dims != (time_dim,):
            continue
        if var not in primary_before or var not in secondary_after:
            continue
        if primary_before[var].ndim != 0 and secondary_after[var].ndim != 0:
            continue
        if primary_before[var].values == secondary_after[var].values:
            combined_dataset[var] = primary_before[var]

    if is_point_list:
        combined_dataset = combined_dataset.rename_vars(
            {index_dim_name: index_target_var}
        )

    combined_dataset = combined_dataset.sortby(time_var)
    return combined_dataset, config


def assign_coords(dataset, config):
    """Assign coordinates to dataset dimensions based on the provided configuration."""
    coords = get_dataset_config_value(
        dataset,
        config,
        "assign_coords",
        error_message="Missing 'assign_coords' config parameter for assign_coords process.",
    )
    if not isinstance(coords, dict):
        raise TypeError(
            "'assign_coords' parameter must be a dictionary mapping variable names to dimension names."
        )

    # Parse possible templates in coords
    expanded_coords = {}
    for var, dim in coords.items():
        if isinstance(dim, dict) and "variables" in dim:
            var_ranges = dim["variables"]
            dim_template = dim["dim"]

            keys = []
            ranges = []
            for key, rng in var_ranges.items():
                keys.append(key)
                n_min = rng.get("min", 0)
                n_max = rng.get("max", 0)
                ranges.append(range(n_min, n_max + 1))

            for values in itertools.product(*ranges):
                var_name = var
                dim_name = dim_template

                for key, value in zip(keys, values):
                    target = f"{{{key}}}"
                    var_name = var_name.replace(target, str(value))
                    dim_name = dim_name.replace(target, str(value))

                expanded_coords[var_name] = dim_name
        else:
            expanded_coords[var] = dim

    assign_dict = {}
    for var, dim in expanded_coords.items():
        if var not in dataset or dim not in dataset.dims:
            continue
        elif var not in dataset.coords:
            values = dataset[var].values
            # Decode binary strings if needed
            if values.dtype.kind == "S":  # Fixed-length bytes
                values = values.astype(str)
            elif values.dtype == object:  # Variable-length bytes or other objects
                if values.size > 0 and isinstance(values.flat[0], bytes):
                    values = np.array(
                        [
                            v.decode("utf-8") if isinstance(v, bytes) else v
                            for v in values.ravel()
                        ]
                    ).reshape(values.shape)
            assign_dict[var] = (dim, values, dataset[var].attrs)

    dataset = dataset.assign_coords(assign_dict)

    return dataset, config


def unify_chunks(dataset, config):
    """Unify chunk sizes across variables in the dataset to optimize Dask performance."""
    (dataset,) = xr.unify_chunks(dataset)
    return dataset, config


def rename_variables(dataset, config):
    """Rename dataset variables using the provided renaming map."""
    rename_map = get_dataset_config_value(
        dataset,
        config,
        "rename_map",
        error_message="Missing 'rename_map' config parameter for rename_variables process.",
    )
    if not isinstance(rename_map, dict):
        raise TypeError(
            "'rename_map' parameter must be a dictionary mapping old variable names to new variable names."
        )

    rename_map = {old: new for old, new in rename_map.items() if old in dataset}
    for old_name, new_name in rename_map.items():
        if old_name not in dataset:
            print(
                f"[KAZARR] Warning: Variable '{old_name}' not found in dataset for rename_variables process. Skipping."
            )
        elif new_name in dataset:
            print(
                f"[KAZARR] Warning: Variable '{new_name}' already exists in dataset. Cannot rename '{old_name}' to '{new_name}'. Skipping."
            )
        else:
            print(f"[KAZARR] Renaming variable '{old_name}' to '{new_name}'")

    dataset = dataset.rename(rename_map)
    return dataset, config


def exclude_variables(dataset, config):
    """Drop specified variables from the dataset."""
    exclude_vars = get_dataset_config_value(
        dataset,
        config,
        "exclude_vars",
        error_message="Missing 'exclude_vars' config parameter for exclude_variables process.",
    )
    if not isinstance(exclude_vars, list):
        raise TypeError(
            "'exclude_vars' parameter must be a list of variable names to exclude."
        )

    dataset = dataset.drop_vars(exclude_vars, errors="ignore")
    return dataset, config


def keep_variables(dataset, config):
    """Keep only the specified variables and drop all others from the dataset."""
    keep_vars = get_dataset_config_value(
        dataset,
        config,
        "keep_vars",
        error_message="Missing 'keep_vars' config parameter for keep_variables process.",
    )
    if not isinstance(keep_vars, list):
        raise TypeError(
            "'keep_vars' parameter must be a list of variable names to keep."
        )

    vars_to_drop = [var for var in dataset.data_vars if var not in keep_vars]
    dataset = dataset.drop_vars(vars_to_drop, errors="ignore")
    return dataset, config


def delta_time_to_datetime(dataset, config):
    """Convert a relative time delta variable into absolute datetime values based on a reference time."""
    time_ref_var = get_dataset_config_value(
        dataset,
        config,
        "referenceTime.variable",
        error_message="Missing 'referenceTime.variable' config parameter for delta_time_to_datetime process.",
    )
    time_ref_format = get_dataset_config_value(dataset, config, "referenceTime.format")
    delta_unit = get_dataset_config_value(dataset, config, "referenceTime.delta_unit")
    time_var = get_dataset_config_value(
        dataset,
        config,
        "variables.time",
        error_message="Missing 'variables.time' config parameter for delta_time_to_datetime process.",
    )

    time_dim = get_dataset_config_value(dataset, config, "dimensions.time")

    # Whether or not to update the original time variable in the config to point to the new datetime variable created by this process (default: True). If False, the new datetime variable will be created but the config will still point to the original time variable, which may cause issues for downstream processes that rely on it being a datetime variable.
    update_time_var = get_dataset_config_value(
        dataset, config, "updateTimeVar", default=True
    )

    units = {
        "years": "Y",
        "year": "Y",
        "months": "M",
        "month": "M",
        "days": "D",
        "day": "D",
        "hours": "h",
        "hour": "h",
        "minutes": "m",
        "minute": "m",
        "min": "m",
        "seconds": "s",
        "second": "s",
        "sec": "s",
    }
    if (
        delta_unit is None
        and time_var is not None
        and time_var in dataset
        and hasattr(dataset[time_var], "units")
    ):
        delta_unit = dataset[time_var].units
    if (
        delta_unit is not None
        and delta_unit not in ["Y", "M", "D", "h", "m", "s"]
        and delta_unit.lower() in units
    ):
        delta_unit = units[delta_unit.lower()]
    if delta_unit is None or delta_unit not in ["Y", "M", "D", "h", "m", "s"]:
        delta_unit = "h"  # Default to hours

    try:
        var_data = dataset[time_ref_var].values
        if var_data.shape == ():
            time_ref = var_data[()]
        elif var_data.size == 1:
            time_ref = var_data[0]
        else:
            raise ValueError(
                f"Reference time variable '{time_ref_var}' must be a scalar, a string, or a datetime64 variable."
            )
        try:
            time_ref = parse_datetime(time_ref, time_ref_format)
        except Exception as e:
            raise ValueError(
                "Missing 'referenceTime.format' config parameter for delta_time_to_datetime process."
            ) from e
    except ValueError as e:
        raise ValueError(f"Error parsing reference time: {e}") from e

    try:
        # Try to deduce time dimension from time variable if time dimension is not provided
        # If time dimension is found, we create a coordinate "datetimes" along that dimension
        # If not, we create a new variable "datetimes"
        if time_dim is None and time_var is not None and time_var in dataset and dataset[time_var].ndim >= 1:
            time_dim = dataset[time_var].dims[0]

        time_deltas = dataset[time_var].values
        # Convert to timedelta64 if not already
        if not np.issubdtype(time_deltas.dtype, np.timedelta64):
            time_deltas = np.array(time_deltas, dtype=f"timedelta64[{delta_unit}]")

        time_values = time_ref + time_deltas
    except Exception as e:
        raise ValueError(f"Error calculating datetime values: {e}") from e

    if time_dim is not None:
        dataset = dataset.assign_coords(datetimes=(time_dim, time_values))
    else:
        dataset["datetimes"] = (dataset[time_var].dims, time_values)

    if update_time_var:
        config["variables"]["time"] = (
            "datetimes"  # Update config to use new datetime variable
        )

    return dataset, config


def reproject_coordinates(dataset, config):
    """Reproject dataset coordinates (longitude, latitude, altitude) to another CRS."""
    from_crs = get_dataset_config_value(
        dataset,
        config,
        "reprojection.from_crs",
        error_message="Missing 'from_crs' config parameter for reproject_coordinates process.",
    )
    to_crs = get_dataset_config_value(
        dataset,
        config,
        "reprojection.to_crs",
        error_message="Missing 'to_crs' config parameter for reproject_coordinates process.",
    )

    lon_var = get_ci(
        config,
        LON_VARIABLE_KEY,
        message=f"Missing '{LON_VARIABLE_KEY}' config parameter for reproject_coordinates process.",
    )
    lat_var = get_ci(
        config,
        LAT_VARIABLE_KEY,
        message=f"Missing '{LAT_VARIABLE_KEY}' config parameter for reproject_coordinates process.",
    )
    if lon_var not in dataset or lat_var not in dataset:
        raise ValueError(
            "Longitude or latitude variable not found in dataset for reproject_coordinates process."
        )

    transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)

    def reproject(*args):
        return transformer.transform(*args)

    vars_to_reproject = [lon_var, lat_var]
    values = [dataset[var].values for var in vars_to_reproject]

    # Use xarray apply_ufunc to reproject coordinates with Dask arrays (parallelized)
    output = xr.apply_ufunc(
        reproject,
        *values,
        output_core_dims=[[]] * len(vars_to_reproject),
        vectorize=True,
        dask="parallelized",
    )

    dataset[lon_var] = (dataset[lon_var].dims, output[0], dataset[lon_var].attrs)
    dataset[lat_var] = (dataset[lat_var].dims, output[1], dataset[lat_var].attrs)

    return dataset, config


def simplify_grid(dataset, config):
    spatial_variables = get_spatial_variables(dataset, config)

    new_coords = {}
    original_dims = set()
    simplified_dims = set()
    for var in spatial_variables:
        if var is None:
            continue
        original_dims.update(dataset[var].dims)
        redundant_dims = get_redundant_dimensions(dataset, var)
        if redundant_dims:
            dataset[var] = dataset[var].isel(dict.fromkeys(redundant_dims, 0))
            simplified_dims.update(dataset[var].dims)
            if dataset[var].ndim == 1:
                new_coords[var] = dataset[var].dims[0]

    # Check now if any dim is not used anymore
    # If so, we can safely remove it from the dataset and from the config dimensions fixed list if needed
    unused_dims = original_dims - simplified_dims
    if unused_dims:
        for dim in unused_dims:
            print(
                f"[KAZARR] Removing unused dimension '{dim}' after grid simplification."
            )
            dataset = dataset.squeeze(dim)
            if (
                "dimensions" in config
                and "fixed" in config["dimensions"]
                and dim in config["dimensions"]["fixed"]
            ):
                del config["dimensions"]["fixed"][dim]

    # Check if grid is now regular
    is_regular = True
    for var in spatial_variables:
        if var is None:
            continue
        if dataset[var].ndim != 1:
            is_regular = False
            break
    if is_regular:
        config["mesh_type"] = "regular"
        print(
            "[KAZARR] Grid simplified to regular grid. Mesh type set to 'regular' in config."
        )

    dataset, _ = assign_coords(dataset, {"assign_coords": new_coords})
    return dataset, config


def save(dataset, config):
    """Save the final dataset to Zarr format locally or on S3."""
    path = get_dataset_config_value(dataset, config, "save_path")
    if path is None:
        path = get_dataset_config_value(
            dataset,
            config,
            "path",
            error_message="Missing 'save_path' or 'path' config parameter for save process.",
        )
        path = path.replace(".nc", "").replace(".grib2", "") + ".zarr"
    config["save_path"] = path  # Update config with actual save path
    version = get_dataset_config_value(dataset, config, "version", default=3)
    float64_to_float32 = get_dataset_config_value(
        dataset, config, "float64_to_float32", default=False
    )

    if float64_to_float32:
        for var in dataset.data_vars:
            if dataset[var].dtype == np.float64:
                dataset[var] = dataset[var].astype(np.float32)

    keep_keys = ["variables", "dimensions", "mesh_data_on_cells", "mesh_type"]
    kazarr_metadata = {k: v for k, v in config.items() if k in keep_keys}
    dataset.attrs["kazarr"] = kazarr_metadata

    final_path = path
    if path.startswith(S3_PREFIX):
        bucket = os.getenv(BUCKET_NAME_ENV_VAR)
        if bucket is None:
            raise ValueError(f"{BUCKET_NAME_ENV_VAR} environment variable not set.")
        final_path = path.replace(S3_PREFIX, S3_PREFIX + bucket + "/")

    # if os.getenv("AWS_ENDPOINT_URL", "").endswith("cloud.ovh.net"):
    #   # Special case for OVH S3 to disable payload signing
    #   # (error: botocore.exceptions.ClientError: An error occurred (InvalidArgument) when calling the PutObject operation: x-amz-content-sha256 must be UNSIGNED-PAYLOAD, or a valid sha256 value.)
    #   # Either define the environment variables (only the first one seems necessary)
    #   os.environ["AWS_REQUEST_CHECKSUM_CALCULATION"] = "when_required"
    #   os.environ["AWS_RESPONSE_CHECKSUM_VALIDATION"] = "when_required"
    #   # Either create a custom s3fs S3FileSystem with config kwargs
    #   fs = s3fs.S3FileSystem(
    #     config_kwargs={
    #       'signature_version': 's3v4',
    #       's3': {
    #         'payload_signing_enabled': False,
    #         'addressing_style': 'path',
    #       },
    #       'request_checksum_calculation': 'when_required',
    #       'response_checksum_validation': 'when_required',
    #     }
    #   )
    #   store = fs.get_mapper(final_path)
    #   dataset.to_zarr(store=store, mode="w", consolidated=(version == 2), zarr_format=version)
    # else:

    print("[KAZARR] Saving dataset to Zarr format...")

    dataset = rechunk_if_needed(dataset)

    zarr_kwargs = {
        "mode": "w",
        "consolidated": version == 2,
        "zarr_format": version,
    }
    storage_options = get_s3_storage_options(config, final_path)
    if storage_options:
        zarr_kwargs["storage_options"] = storage_options
    if config.get("dask_dashboard_initialised", False):
        with performance_report(filename="dask-performance-report.html"):
            dataset.to_zarr(final_path, **zarr_kwargs)
            path = Path(__file__).parent.resolve()
            print("===============================================")
            print(
                f"[KAZARR] Dask performance report available at: file://{path}/dask-performance-report.html"
            )
            print("===============================================")
    else:
        dataset.to_zarr(final_path, **zarr_kwargs)
    return dataset, config


def clean(dataset, config):
    """Clean temporary, generated, or index files."""
    clean = get_dataset_config_value(
        dataset,
        config,
        "clean",
        default={"used": False, "generated": True, "idx": True},
    )

    def clean_proc(paths):
        for file in paths:
            if os.path.exists(file):
                if os.path.isfile(file):
                    os.remove(file)
                elif os.path.isdir(file):
                    shutil.rmtree(file)

    def clean_idx(folders):
        for folder in folders:
            if os.path.exists(folder) and os.path.isdir(folder):
                for file in os.listdir(folder):
                    if file.endswith(".idx"):
                        os.remove(os.path.join(folder, file))

    if clean.get("used", False):
        used_paths = clean.get("used_paths", [])
        clean_proc(used_paths)
    if clean.get("generated", True):
        generated_paths = clean.get("generated_paths", [])
        clean_proc(generated_paths)
    if clean.get("idx", True):
        idx_paths = clean.get("idx_folders", [])
        clean_idx(idx_paths)
    return dataset, config

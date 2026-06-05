import numpy as np
import pandas as pd
from loguru import logger as log

import src.exceptions as exceptions
from src.processing.contexts import TimeRange


# Ensure xindex is set for a variable in the dataset
def set_xindex(dataset, var_name):
    if var_name not in dataset.xindexes:
        if dataset[var_name].ndim == 1:
            dataset = dataset.set_xindex(var_name)
        else:
            raise exceptions.BadConfigurationVariable(
                f"Variable '{var_name}' has more than 1 dimension, cannot set xindex."
            )
    return dataset


# Check if a variable is monotonic
def is_monotonic_var(dataset, var_name):
    try:
        var_data = dataset[var_name].values
        is_monotonic = np.all(np.diff(var_data) >= 0) or np.all(np.diff(var_data) <= 0)
    except Exception:
        is_monotonic = False
    return is_monotonic


# Get dimensions and coordinates that must be provided for a selection, and not already defined
def get_required_dims_and_coords(
    dataset,
    variables,
    fixed_coords,
    fixed_dims,
    request,
    interp_vars=None,
    optional_coords=None,
    optional_dims=None,
    coords_keep_dims=None,
    as_dims=None,
    greedy=True,
):
    if interp_vars is None:
        interp_vars = []
    if optional_coords is None:
        optional_coords = []
    if optional_dims is None:
        optional_dims = []
    if coords_keep_dims is None:
        coords_keep_dims = []
    if as_dims is None:
        as_dims = []

    if greedy:
        for key, value in request.query_params.items():
            if key in dataset.coords and key not in as_dims and key not in fixed_coords:
                fixed_coords[key] = value
            elif key in dataset.dims and key not in fixed_dims:
                try:
                    fixed_dims[key] = int(value)
                except ValueError:
                    pass

    if optional_coords == "*" or optional_dims == "*":
        return fixed_coords, fixed_dims

    # Expand optional dimensions from optional coordinates
    optional_coords_list = (
        optional_coords if isinstance(optional_coords, list) else [optional_coords]
    )
    for coord in optional_coords_list:
        if coord in dataset:
            for dim in dataset[coord].dims:
                if dim not in optional_dims:
                    optional_dims.append(dim)
    # Identify missing dimensions
    missing_dims = {}
    variables_list = variables if isinstance(variables, list) else [variables]

    for variable in variables_list:
        if variable not in dataset:
            continue
        for dim in dataset[variable].dims:
            if dim in fixed_dims or dim in optional_dims:
                continue

            # Check if any coordinate for this dimension is provided
            assigned_coords = [
                coord
                for coord in dataset.coords
                if dim in dataset[coord].dims and coord not in as_dims
            ]

            is_satisfied = any(
                coord in fixed_coords
                or coord in optional_coords
                or coord in interp_vars
                for coord in assigned_coords
            )

            if not is_satisfied:
                missing_dims[dim] = assigned_coords
    if missing_dims:
        raise exceptions.MissingDimensionsOrCoordinates(missing_dims)

    # Convert coords_keep_dims coordinates to list if they are in fixed_coords, so that they are not squeezed during selection
    # This allows to keep the dimension of these coordinates even if only one value is selected
    # example: longitude or latitude value is fixed for an extract => we want to keep the longitude and latitude dimensions in the output, even if only one value is selected for each of them
    dims_to_keep = set()
    for coord in coords_keep_dims:
        dims_to_keep.update(dataset[coord].dims)
        if coord in fixed_coords:
            fixed_coords[coord] = [fixed_coords[coord]]
    for dim in dims_to_keep:
        if dim in fixed_dims:
            fixed_dims[dim] = [fixed_dims[dim]]

    return fixed_coords, fixed_dims


def get_bounded_time(dataset, time_var, time_range: TimeRange) -> TimeRange:
    if time_var not in dataset or not is_monotonic_var(dataset, time_var):
        raise exceptions.GenericInternalError(
            f"Time variable '{time_var}' not found or not monotonic in dataset."
        )

    time_data = dataset[time_var].values
    min_time = time_data[0]
    max_time = time_data[-1]

    def bound_value(t_val: str | None) -> str | None:
        if t_val is None:
            return None

        try:
            t_arr = np.array(t_val, dtype=time_data.dtype)
            clipped = np.clip(t_arr, min_time, max_time)

            if np.issubdtype(time_data.dtype, np.datetime64):
                return pd.Timestamp(clipped).isoformat()
            else:
                return str(clipped)

        except Exception:
            raise exceptions.GenericInternalError(
                f"Unable to convert time value '{t_val}' to the correct type."
            )

    bounded_start = bound_value(time_range.start)
    bounded_end = bound_value(time_range.end)

    return TimeRange(
        start=bounded_start, end=bounded_end, has_time_range=time_range.has_time_range
    )


def get_times_in_range(dataset, time_var, time_range: TimeRange):
    if time_var not in dataset:
        return []

    if time_range.start is None and not time_range.has_time_range:
        time_data = dataset[time_var].values
    else:
        try:
            sliced_data = dataset[time_var].sel({time_var: time_range.get_indexer()})
            time_data = sliced_data.values
        except KeyError:
            return []

    time_data = np.atleast_1d(time_data)

    if np.issubdtype(time_data.dtype, np.datetime64):
        return [str(np.datetime_as_string(t)) for t in time_data]
    else:
        return time_data.tolist()


# Smart selection on a dataset variable with coordinates and dimensions
def sel(
    dataset,
    variable,
    fixed_coords,
    fixed_dims,
    interp_vars=None,
    interp_method="linear",
    interp_methods=None, # Use when different interpolation methods are needed for different variables
    interp_config=None,
):
    if interp_vars is None:
        interp_vars = []
    if interp_config is None:
        interp_config = {}

    for var in interp_vars:
        if var in dataset.coords and var not in dataset.dims:
            if len(dataset[var].dims) == 1:
                current_dim = dataset[var].dims[0]
                dataset = dataset.swap_dims({current_dim: var})

    # Ensure xindexes are set for all fixed coords
    for coord in fixed_coords:
        try:
            dataset = set_xindex(dataset, coord)
        except exceptions.BadConfigurationVariable:
            raise exceptions.VariableCannotBeUsedForSelection(coord)

    # Convert coords values to target dtype
    for var, val in fixed_coords.items():
        # Avoid to convert slice
        if not isinstance(val, slice):
            fixed_coords[var] = np.array(val, dtype=dataset[var].dtype)

    # Only monotonic variables can be used with sel(method='nearest')
    monotonic_fixed_vars = {
        var: val
        for var, val in fixed_coords.items()
        if is_monotonic_var(dataset, var) and var not in interp_vars and not isinstance(val, slice)
    }
    non_monotonic_fixed_vars = {
        var: val
        for var, val in fixed_coords.items()
        if (not is_monotonic_var(dataset, var) or isinstance(val, slice)) and var not in interp_vars
    }

    try:
        data = (
            dataset.sel(monotonic_fixed_vars, method="nearest")
            .sel(non_monotonic_fixed_vars)
            .isel(fixed_dims)[variable]
        )
    except ValueError:
        raise exceptions.BadSelection(
            "Data selection failed. Please check your query parameters and dataset configuration. This can happen if you have specified a coordinate and its corresponding dimension at the same time."
        )
    except IndexError:
        raise exceptions.BadSelection(
            "Data selection failed due to an index error. Please check your query parameters and dataset configuration."
        )
    except Exception:
        raise exceptions.GenericInternalError(
            "Data selection failed. Please check your query parameters and dataset configuration."
        )
    # Interpolate if needed
    if len(interp_vars) > 0:
        interpolated_vars = {
            var: fixed_coords[var] for var in interp_vars if var in fixed_coords
        }
        if len(interpolated_vars) > 0:
            try:
                if interp_methods is not None:
                    method_groups = {}
                    for var, coord in interpolated_vars.items():
                        method = interp_methods.get(var, interp_method)
                        method_groups.setdefault(method, {})[var] = coord
                else:
                    method_groups = {interp_method: interpolated_vars}

                for method, vars_to_interp in method_groups.items():
                    if method not in [
                        "linear",
                        "nearest",
                        "zero",
                        "slinear",
                        "quadratic",
                        "cubic",
                        "quintic",
                        "polynomial",
                        "pchip",
                        "barycentric",
                        "krogh",
                        "akima",
                        "makima",
                    ]:
                        log.warning(
                            "[Kazarr] Unsupported interpolation method: {interp_method}. Falling back to linear.",
                            interp_method=method,
                        )
                        method = "linear"
                    data = data.interp(
                        vars_to_interp,
                        method=method,
                        assume_sorted=True,
                        **interp_config,
                    )
            except ValueError:
                raise exceptions.BadSelection(
                    "Data interpolation failed. Please check your query parameters and dataset configuration."
                )
    return data


# Deep get from nested dict (mimic lodash get)
def dget(d, key, default=None):
    keys = key.split(".")
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default.copy() if isinstance(default, dict) else default
    if d is None:
        return default.copy() if isinstance(default, dict) else default
    return d


# Deep get multiple keys from nested dict
def dgets(d, keys, default=None):
    values = []
    for key in keys if isinstance(keys, list) else [keys]:
        values.append(dget(d, key, default=default))
    return tuple(values)


def get_dataset_level_vars(dataset, config):
    level_var = dget(config, "variables.level")
    if level_var is not None and (
        level_var.startswith("ATTRS.") or level_var.startswith("ATTRIBUTES.")
    ):
        level_var_name = level_var.replace("ATTRS.", "").replace("ATTRIBUTES.", "")
        level_vars = {}
        for var in dataset.data_vars:
            if level_var_name in dataset[var].attrs:
                target_level_var = dataset[var].attrs[level_var_name]
                if target_level_var not in level_vars:
                    level_vars[target_level_var] = [var]
                else:
                    level_vars[target_level_var].append(var)
        if len(level_vars) == 0:
            return None
        elif len(level_vars) == 1:
            return list(level_vars.keys())[0]
        return level_vars
    return level_var


def get_level_var(dataset, config, variable):
    level_var = dget(config, "variables.level")
    if level_var is not None and (
        level_var.startswith("ATTRS.") or level_var.startswith("ATTRIBUTES.")
    ):
        level_var = dataset[variable].attrs.get(
            level_var.replace("ATTRS.", "").replace("ATTRIBUTES.", ""), None
        )
    return level_var


def parse_query_dict(query_string):
    # Check if string match key:value pairs separated by commas
    if not all(":" in part for part in query_string.split(",")):
        raise exceptions.GenericInternalError(
            "Invalid interpolation query format. Expected format: 'key1:value1,key2:value2,...'"
        )

    def cast_value(value):
        for cast in (int, float):
            try:
                return cast(value)
            except ValueError:
                continue
        return value

    params = {}
    for part in query_string.split(","):
        if ":" in part:
            key, value = part.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not key or not value:
                raise exceptions.GenericInternalError(
                    "Invalid interpolation query format. Expected format: 'key1:value1,key2:value2,...'"
                )
            params[key] = cast_value(value)
    return params

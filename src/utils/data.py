import numpy as np

import src.exceptions as exceptions


# Ensure xindex is set for a variable in the dataset
def set_xindex(dataset, var_name):
    if var_name not in dataset.xindexes:
        dataset = dataset.set_xindex(var_name)
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
    as_dims=None,
    greedy=True,
):
    if interp_vars is None:
        interp_vars = []
    if optional_coords is None:
        optional_coords = []
    if optional_dims is None:
        optional_dims = []
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

    return fixed_coords, fixed_dims


def get_bounded_time(dataset, time_var, time):
    if time_var not in dataset or not is_monotonic_var(dataset, time_var):
        raise exceptions.GenericInternalError(
            f"Time variable '{time_var}' not found or not monotonic in dataset."
        )

    try:
        time_data = dataset[time_var].values
        # Try to cast time to the same dtype as time_data
        time = np.array(time, dtype=time_data.dtype)
        return np.clip(time, time_data[0], time_data[-1])
    except Exception:
        raise exceptions.GenericInternalError(
            f"Unable to convert time value '{time}' to the correct type."
        )


# Smart selection on a dataset variable with coordinates and dimensions
def sel(
    dataset,
    variable,
    fixed_coords,
    fixed_dims,
    interp_vars=None,
    interp_method="linear",
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
        dataset = set_xindex(dataset, coord)

    # Convert coords values to target dtype
    for var, val in fixed_coords.items():
        fixed_coords[var] = np.array(val, dtype=dataset[var].dtype)

    # Only monotonic variables can be used with sel(method='nearest')
    monotonic_fixed_vars = {
        var: val
        for var, val in fixed_coords.items()
        if is_monotonic_var(dataset, var) and var not in interp_vars
    }
    non_monotonic_fixed_vars = {
        var: val
        for var, val in fixed_coords.items()
        if not is_monotonic_var(dataset, var) and var not in interp_vars
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
    except Exception as e:
        raise exceptions.GenericInternalError("Data selection failed. Please check your query parameters and dataset configuration.")
    # Interpolate if needed
    if len(interp_vars) > 0:
        interpolated_vars = {
            var: fixed_coords[var] for var in interp_vars if var in fixed_coords
        }
        if len(interpolated_vars) > 0:
            try:
                if interp_method not in [
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
                    print(
                        f"[Kazarr - Warning] Unsupported interpolation method: {interp_method}. Falling back to linear."
                    )
                    interp_method = "linear"
                data = data.interp(
                    interpolated_vars,
                    method=interp_method,
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
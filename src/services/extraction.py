import numpy as np
from scipy.spatial import cKDTree
import xarray as xr
from fastapi import Request
from typing import Any, Dict, List, Optional, Union

from src import exceptions
from src.schemas.config import ExtractionConfig
from src.utils.data import (
    dget,
    dgets,
    sel,
    get_required_dims_and_coords,
    get_bounded_time,
)
from src.utils.file import load_dataset
from src.utils.logging import StepDurationLogger
from src.processing import bbox, interpolation, output
from src.processing.contexts import BBoxContext


def extract(
    request: Request,
    dataset_id: str,
    variable: str,
    time: Optional[str] = None,
    format: str = "raw",
    config: Union[Dict[str, Any], ExtractionConfig, None] = None,
) -> Dict[str, Any]:
    if not isinstance(config, ExtractionConfig):
        config = ExtractionConfig.model_validate(config or {})

    step_logger = StepDurationLogger(
        "extract", parameters=(dataset_id, variable, time, format, config)
    )

    bounding_box = BBoxContext.from_tuple(config.bbox)
    has_bb = bounding_box.has_bb
    has_bb_lon = bounding_box.has_bb_lon
    has_bb_lat = bounding_box.has_bb_lat

    step_logger.step_start("Load dataset and config")
    dataset, dataset_config = load_dataset(dataset_id)
    fixed_coords, fixed_dims = dgets(
        dataset_config, ["variables.fixed", "dimensions.fixed"], {}
    )
    interp_vars = config.interpolation.vars.items

    if variable not in dataset:
        raise exceptions.VariableNotFound([variable])

    lon_var, lat_var = dgets(dataset_config, ["variables.lon", "variables.lat"])
    missing_vars = []
    if has_bb_lon and lon_var is None:
        raise exceptions.MissingConfigurationElement("variables.lon")
    if lon_var not in dataset:
        missing_vars.append(f"lon ({lon_var})")
    if has_bb_lat and lat_var is None:
        raise exceptions.MissingConfigurationElement("variables.lat")
    if lat_var not in dataset:
        missing_vars.append(f"lat ({lat_var})")

    time_var = dget(dataset_config, "variables.time")
    if time is not None and time_var is not None:
        if time_var not in dataset:
            missing_vars.append(f"time ({time_var})")
        else:
            fixed_coords[time_var] = get_bounded_time(dataset, time_var, time)
            if config.interpolation.vars.time and time_var not in interp_vars:
                interp_vars.append(time_var)
            # Remove "time" from request parameters to avoid confusion in later steps (case where time is a variable in the dataset)
            request.query_params._dict.pop("time", None)

    mesh_type = dget(dataset_config, "mesh_type", "auto")
    if len(missing_vars) > 0:
        raise exceptions.BadConfigurationVariable(missing_vars)

    mesh_tile_shape = config.mesh.tile_shape
    force_data_mapping = config.mesh.data_mapping

    interp_vars_method = config.interpolation.vars.method
    interp_vars_params = config.interpolation.vars.params or {}
    interp_spatial_method = config.interpolation.spatial.method
    interp_spatial_params = config.interpolation.spatial.params or {}
    index_padding = interp_spatial_params.pop("index_padding", 2)
    spatial_padding = interp_spatial_params.pop("padding", 1.0)
    if not has_bb:
        index_padding, spatial_padding = 0, 0.0

    fixed_coords, fixed_dims = get_required_dims_and_coords(
        dataset,
        variable,
        fixed_coords,
        fixed_dims,
        request,
        interp_vars=interp_vars,
        optional_coords=[lon_var, lat_var],
        as_dims=config.as_dims or [],
    )

    lons = sel(dataset, lon_var, fixed_coords, fixed_dims)
    lats = sel(dataset, lat_var, fixed_coords, fixed_dims)

    is_regular_grid = lons.ndim == 1 and lats.ndim == 1 and lons.dims != lats.dims
    is_point_list = (lons.ndim == 1 and lats.ndim == 1 and lons.dims == lats.dims) or (
        lons.ndim == 0 and lats.ndim == 0
    )
    pad = index_padding if format == "mesh" else 0

    lons_vals_in = np.atleast_1d(lons.values)
    lats_vals_in = np.atleast_1d(lats.values)

    if is_point_list:
        step_logger.step_start("Point list: apply bounding box")
        indices = bbox.apply_point_list_bounding_box(
            lons_vals_in, lats_vals_in, bounding_box
        )
        point_indices, n_points = indices.point_indices, indices.n_points
        height_raw, width_raw = indices.height_raw, indices.width_raw
    elif is_regular_grid and has_bb:
        step_logger.step_start("Regular grid: apply bounding box")
        lons_1d = lons.values
        lats_1d = lats.values
        indices = bbox.apply_regular_grid_bounding_box(
            lons_1d, lats_1d, bounding_box, pad
        )
        col_min, col_max = indices.col_min, indices.col_max
        row_min, row_max = indices.row_min, indices.row_max
        width_raw, height_raw = indices.width_raw, indices.height_raw
    else:
        step_logger.step_start("Unstructured grid: apply bounding box")
        lons_vals, lats_vals, indices = bbox.apply_unstructured_bounding_box(
            lons_vals_in,
            lats_vals_in,
            bounding_box,
            is_regular_grid,
            spatial_padding,
            pad,
        )
        col_min, col_max = indices.col_min, indices.col_max
        row_min, row_max = indices.row_min, indices.row_max
        width_raw, height_raw = indices.width_raw, indices.height_raw

    if not is_point_list and (height_raw < 2 or width_raw < 2):
        raise exceptions.TooFewPoints()

    resolution_limit = config.resolution_limit
    step_row, step_col = bbox.apply_resolution_limit(
        height_raw,
        width_raw,
        resolution_limit,
        is_point_list,
        n_points if is_point_list else 0,
    )

    step_logger.step_start("Load variable values")
    vals_da = sel(
        dataset,
        variable,
        fixed_coords,
        fixed_dims,
        interp_vars=interp_vars,
        interp_method=interp_vars_method,
        interp_config=interp_vars_params,
    )
    if is_point_list:
        if resolution_limit is not None and n_points > resolution_limit:
            point_indices = point_indices[::step_row]
        vals = np.atleast_1d(vals_da.values)[point_indices]
    else:
        vals = vals_da[
            ..., row_min : row_max + 1 : step_row, col_min : col_max + 1 : step_col
        ].values
    vals = vals.astype(float)

    step_logger.step_start("Crop latitude and longitude")
    lons_1d, lats_1d = None, None
    if is_point_list:
        lons = lons_vals_in[point_indices]
        lats = lats_vals_in[point_indices]
    elif is_regular_grid:
        lons_1d = lons_vals_in[col_min : col_max + 1 : step_col]
        lats_1d = lats_vals_in[row_min : row_max + 1 : step_row]
        lons, lats = np.meshgrid(lons_1d, lats_1d)
    else:
        lons = lons_vals[
            ..., row_min : row_max + 1 : step_row, col_min : col_max + 1 : step_col
        ]
        lats = lats_vals[
            ..., row_min : row_max + 1 : step_row, col_min : col_max + 1 : step_col
        ]

    if has_bb:
        mask_cropped = np.ones(lons.shape, dtype=bool)
        if has_bb_lon:
            bb_lon_min = (
                bounding_box.lon_min if bounding_box.lon_min is not None else -np.inf
            )
            bb_lon_max = (
                bounding_box.lon_max if bounding_box.lon_max is not None else np.inf
            )
            mask_cropped &= (lons >= bb_lon_min) & (lons <= bb_lon_max)
        if has_bb_lat:
            bb_lat_min = (
                bounding_box.lat_min if bounding_box.lat_min is not None else -np.inf
            )
            bb_lat_max = (
                bounding_box.lat_max if bounding_box.lat_max is not None else np.inf
            )
            mask_cropped &= (lats >= bb_lat_min) & (lats <= bb_lat_max)

        if format != "mesh":
            vals[~mask_cropped] = np.nan
    else:
        mask_cropped = None

    cell_data = force_data_mapping != "vertices" and (
        force_data_mapping == "cells"
        or dget(dataset_config, "mesh_data_on_cells", False)
    )
    if cell_data and not is_point_list:
        step_logger.step_start("Cell to point data conversion")
        lons, lats, vals, lons_1d, lats_1d = interpolation.cell_to_point_conversion(
            lons, lats, vals, variable, mesh_type, is_regular_grid
        )

    if mesh_tile_shape is not None:
        step_logger.step_start("Generate meshgrid")
        target_h, target_w = mesh_tile_shape
        lons, lats, vals, mask_cropped = interpolation.generate_meshgrid_and_interpolate(
            lons,
            lats,
            vals,
            lons_1d,
            lats_1d,
            bounding_box,
            target_w,
            target_h,
            is_regular_grid,
            is_point_list,
            interp_spatial_method,
            interp_spatial_params,
        )

    if format == "mesh":
        step_logger.step_start("Clean data for PyVista")
        out = output.prepare_mesh_output(
            lons, lats, vals, variable, mask_cropped, step_row, step_col
        )
    elif format == "raw":
        step_logger.step_start("Prepare output (raw)")
        out = output.prepare_raw_output(vals, lons, lats, step_row, step_col)
    elif format == "geojson":
        step_logger.step_start("Prepare output (GeoJSON)")
        out = output.prepare_geojson_output(
            vals, lons, lats, step_row, step_col
        )
    else:
        raise exceptions.BadConfigurationVariable(f"Unsupported format: {format}")

    step_logger.end()
    return out


def probe(
    request: Request,
    dataset_id: str,
    variables: Union[str, List[str]],
    lon: float,
    lat: float,
    height: Optional[float] = None,
    time: Optional[str] = None,
    format: str = "raw",
    config: Union[Dict[str, Any], ExtractionConfig, None] = None,
) -> Dict[str, Any]:
    if not isinstance(config, ExtractionConfig):
        config = ExtractionConfig.model_validate(config or {})
    step_logger = StepDurationLogger(
        "probe",
        parameters=(dataset_id, variables, lon, lat, height, time, format, config),
    )
    step_logger.step_start("Load dataset and config")
    variables = variables if isinstance(variables, list) else [variables]
    dataset, dataset_config = load_dataset(dataset_id)
    with_height = height is not None
    fixed_coords, fixed_dims = dgets(
        dataset_config, ["variables.fixed", "dimensions.fixed"], {}
    )
    interp_vars = config.interpolation.vars.items
    lon_var, lat_var, height_var, time_var = dgets(
        dataset_config,
        ["variables.lon", "variables.lat", "variables.height", "variables.time"],
    )
    time_dim = dget(dataset_config, "dimensions.time")

    not_found_vars = []
    for var in variables:
        if var not in dataset:
            not_found_vars.append(var)
    if len(not_found_vars) > 0:
        raise exceptions.VariableNotFound(not_found_vars)

    missing_vars = []
    if lon_var is None or lon_var not in dataset:
        missing_vars.append(f"lon ({lon_var})")
    if lat_var is None or lat_var not in dataset:
        missing_vars.append(f"lat ({lat_var})")
    if with_height and (height_var is None or height_var not in dataset):
        missing_vars.append(f"height ({height_var})")
    if time is not None and (time_var is None or time_var not in dataset):
        missing_vars.append(f"time ({time_var})")
    if len(missing_vars) > 0:
        raise exceptions.BadConfigurationVariable(missing_vars)

    if time is not None and time_var is not None:
        fixed_coords[time_var] = get_bounded_time(dataset, time_var, time)
        if config.interpolation.vars.time:
            interp_vars.append(time_var)

    interp_spatial_method = config.interpolation.spatial.method
    interp_spatial_params = config.interpolation.spatial.params or {}
    interp_vars_method = config.interpolation.vars.method
    interp_vars_params = config.interpolation.vars.params or {}

    longitudes = dataset[lon_var]
    latitudes = dataset[lat_var]
    is_regular_grid = longitudes.ndim == 1 and latitudes.ndim == 1 and longitudes.dims != latitudes.dims
    if is_regular_grid:
        fixed_coords[lon_var] = lon
        fixed_coords[lat_var] = lat
        if interp_spatial_method != "nearest":
            interp_vars.extend([lon_var, lat_var])
    else:
        if with_height:
            heights = dataset[height_var].values
            points = np.column_stack(
                (longitudes.values.ravel(), latitudes.values.ravel(), heights.ravel())
            )
            target_pt = np.array([lon, lat, height])
        else:
            points = np.column_stack(
                (longitudes.values.ravel(), latitudes.values.ravel())
            )
            target_pt = np.array([lon, lat])

        step_logger.step_start("Build spatial index for probe")
        tree = cKDTree(points)

        if interp_spatial_method != "nearest":
            step_logger.step_start("Probe interpolation (IDW)")
            max_radius = interp_spatial_params.get("radius", 0.05)
            power = interp_spatial_params.get("power", 2.0)
            indices = tree.query_ball_point(target_pt, r=max_radius)

            if not indices:
                raise exceptions.NoDataInSelection(
                    "Try increasing interpolation radius"
                )

            neighbors_coords = points[indices]
            dists = np.linalg.norm(neighbors_coords - target_pt, axis=1)

            zero_dist = dists < 1e-12
            if np.any(zero_dist):
                weights = np.zeros(len(indices))
                weights[np.argmax(zero_dist)] = 1.0
            else:
                weights = (1.0 / (dists**power)) / np.sum(1.0 / (dists**power))

            neighbor_indices = [
                np.unravel_index(idx, longitudes.shape) for idx in indices
            ]
            nearest_idx = neighbor_indices[np.argmax(weights)]
            for dim_name, indice in zip(longitudes.dims, nearest_idx):
                fixed_dims[dim_name] = indice
        else:
            step_logger.step_start("Probe interpolation (nearest neighbor)")
            _, flat_index = tree.query(np.array([target_pt]), k=1)
            flat_index = np.atleast_1d(flat_index)[0]
            indices = np.unravel_index(flat_index, longitudes.shape)
            for dim_name, indice in zip(longitudes.dims, indices):
                fixed_dims[dim_name] = indice

    optional_coords = [time_var] if time_var is not None else []
    optional_dims = [time_dim] if time_dim is not None else []
    fixed_coords, fixed_dims = get_required_dims_and_coords(
        dataset,
        variables,
        fixed_coords,
        fixed_dims,
        request,
        optional_coords=optional_coords,
        optional_dims=optional_dims,
        as_dims=config.as_dims or [],
    )

    step_logger.step_start("Extract variable values at probe location")
    if not is_regular_grid and interp_spatial_method != "nearest":
        dim_indices = {dim: [] for dim in longitudes.dims}
        for idx in neighbor_indices:
            for dim_name, indice in zip(longitudes.dims, idx):
                dim_indices[dim_name].append(indice)

        spatial_indexers = {
            dim: xr.DataArray(vals, dims=["neighbor"])
            for dim, vals in dim_indices.items()
        }
        weights_da = xr.DataArray(weights, dims=["neighbor"])
        base_fixed_dims = {
            k: v for k, v in fixed_dims.items() if k not in longitudes.dims
        }

    data = {}
    for var in variables:
        if not is_regular_grid and interp_spatial_method != "nearest":
            filtered_da = sel(
                dataset,
                var,
                fixed_coords,
                base_fixed_dims,
                interp_vars=interp_vars,
                interp_method=interp_vars_method,
                interp_config=interp_vars_params,
            )
            neighbor_data = filtered_da.isel(**spatial_indexers)
            interpolated_values = (
                (neighbor_data * weights_da).sum(dim="neighbor").values
            )
            data[var] = {
                "values": interpolated_values.tolist(),
                "attrs": dataset[var].attrs,
            }
        else:
            data[var] = {
                "values": sel(
                    dataset,
                    var,
                    fixed_coords,
                    fixed_dims,
                    interp_vars=interp_vars,
                    interp_method=interp_vars_method,
                    interp_config=interp_vars_params,
                ).values.tolist(),
                "attrs": dataset[var].attrs,
            }

    step_logger.step_start(
        "Get time values for probe location if time variable is defined"
    )
    times = None
    if time_var is not None:
        if time is not None and interp_spatial_method != "nearest":
            times = [time]
        elif time_var in dataset:
            times = sel(
                dataset,
                time_var,
                fixed_coords,
                fixed_dims,
                interp_method=interp_vars_method,
                interp_config=interp_vars_params,
            ).values
            times = np.atleast_1d(times)
            if np.issubdtype(times.dtype, np.datetime64):
                times = [str(np.datetime_as_string(t)) for t in times]
            else:
                times = times.tolist()

    out = {"variables": data}
    if times is not None:
        out["times"] = times

    if format == "geojson" and times is not None:
        out = {"type": "FeatureCollection", "features": []}
        variables_props = {}
        for var in data:
            variables_props[var] = {"values": data[var]["values"], **data[var]["attrs"]}
        feature = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
            "properties": {"times": times, **variables_props},
        }
        out["features"].append(feature)

    step_logger.end()
    return out


def free_selection(
    request: Request,
    dataset_id: str,
    variable: str,
    config: Union[Dict[str, Any], ExtractionConfig, None] = None,
) -> Dict[str, Any]:
    if not isinstance(config, ExtractionConfig):
        config = ExtractionConfig.model_validate(config or {})
    step_logger = StepDurationLogger(
        "free_selection", parameters=(dataset_id, variable, config)
    )

    step_logger.step_start("Load dataset and config")
    dataset, dataset_config = load_dataset(dataset_id)

    if variable not in dataset:
        raise exceptions.VariableNotFound([variable])

    interp_vars = config.interpolation.vars.items
    for var in interp_vars:
        if var not in dataset:
            raise exceptions.VariableNotFound([var])

    fixed_coords, fixed_dims = dgets(
        dataset_config, ["variables.fixed", "dimensions.fixed"], {}
    )
    as_dims = config.as_dims or []
    fixed_coords, fixed_dims = get_required_dims_and_coords(
        dataset,
        variable,
        fixed_coords,
        fixed_dims,
        request,
        interp_vars=interp_vars,
        optional_dims="*",
        as_dims=as_dims,
    )

    step_logger.step_start("Extract variable values for free selection")
    data = sel(dataset, variable, fixed_coords, fixed_dims, interp_vars).values.tolist()

    step_logger.end()
    return {"data": data}

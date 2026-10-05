import threading
from typing import Any

import numpy as np
import xarray as xr
from fastapi import Request
from loguru import logger as log

from src import exceptions
from src.processing import bbox, interpolation, output
from src.processing.contexts import BBoxContext, MultiTimeRange, TimeRange
from src.schemas.config import ExtractionConfig
from src.utils import serialization
from src.utils.data import (
    dget,
    dgets,
    get_aliased_variable,
    get_bounded_time,
    get_dataset_level_vars,
    get_level_var,
    get_required_dims_and_coords,
    get_times_in_range,
    is_time_out_of_bounds_data,
    sel,
)
from src.utils.file import load_dataset
from src.utils.logging import StepLoggerAndAborter
from src.utils.requests import get_from_query
from src.utils.spatial import (
    chord_to_deg_distance,
    deg_to_chord_distance,
    get_cached_ckdtree,
    get_cached_structured_grid_geometry,
    lonlat_to_sphere_points,
    lonlat_to_xyz,
)

FIXED_DIMENSIONS_KEY = "dimensions.fixed"
FIXED_VARIABLES_KEY = "variables.fixed"
LAT_VARIABLE_KEY = "variables.lat"
LON_VARIABLE_KEY = "variables.lon"
LEVEL_VARIABLE_KEY = "variables.level"


def extract(
    request: Request,
    dataset_id: str,
    variable: str,
    time_range: str | None = None,
    level: float | None = None,
    format: str = "raw",
    config: dict[str, Any] | ExtractionConfig | None = None,
    cancel_event: threading.Event | None = None,
) -> bytes:
    """Extract data and return it serialized as JSON bytes."""
    if not isinstance(config, ExtractionConfig):
        config = ExtractionConfig.model_validate(config or {})

    step_logger = StepLoggerAndAborter(
        "extract",
        parameters=(dataset_id, variable, time_range, level, format, config),
        cancel_event=cancel_event,
    )

    bounding_box = BBoxContext.from_tuple(config.bbox)
    has_bb = bounding_box.has_bb
    has_bb_lon = bounding_box.has_bb_lon
    has_bb_lat = bounding_box.has_bb_lat
    has_bb_level = bounding_box.has_bb_level

    step_logger.step_start("Load dataset and config")
    dataset, dataset_config = load_dataset(dataset_id)
    fixed_coords, fixed_dims = dgets(
        dataset_config, [FIXED_VARIABLES_KEY, FIXED_DIMENSIONS_KEY], {}
    )
    interp_vars = config.interpolation.vars.items

    if variable not in dataset:
        raise exceptions.VariableNotFound([variable])

    lon_var, lat_var = dgets(dataset_config, [LON_VARIABLE_KEY, LAT_VARIABLE_KEY])
    level_var = get_level_var(dataset, dataset_config, variable)
    missing_vars = []
    if has_bb_lon and lon_var is None:
        raise exceptions.MissingConfigurationElement(LON_VARIABLE_KEY)
    if lon_var not in dataset:
        missing_vars.append(f"lon ({lon_var})")
    if has_bb_lat and lat_var is None:
        raise exceptions.MissingConfigurationElement(LAT_VARIABLE_KEY)
    if lat_var not in dataset:
        missing_vars.append(f"lat ({lat_var})")

    time_var = dget(dataset_config, "variables.time")
    time_range = get_from_query(time_var, time_range, request)
    time_range = TimeRange.from_string(time_range)
    if time_range.has_time() and time_var is not None:
        if time_var not in dataset:
            missing_vars.append(f"time ({time_var})")
        else:
            time_info = is_time_out_of_bounds_data(dataset, time_var, time_range)
            if time_info["out_of_bounds"]:
                raise exceptions.TimeOutOfBounds(
                    time_str=time_info["time"],
                    min_time=time_info["min_time"],
                    max_time=time_info["max_time"],
                )
            bounded_time_range = get_bounded_time(dataset, time_var, time_range)
            time_range_indexer = bounded_time_range.get_indexer()
            if time_range_indexer is not None:
                fixed_coords[time_var] = time_range_indexer
                if config.interpolation.vars.time and time_var not in interp_vars:
                    interp_vars.append(time_var)
                # Remove "time" from request parameters to avoid confusion in later steps (case where time is a variable in the dataset)
                request.query_params._dict.pop("time", None)

    if len(missing_vars) > 0:
        raise exceptions.BadConfigurationVariable(missing_vars)

    # Normalize longitudes if the dataset is in [0, 360] and the bbox uses negative values
    dataset, _ = bbox.normalize_dataset_longitudes(
        dataset, lon_var, lat_var, bbox=bounding_box
    )

    # Detect multi-timestep mode: when time is not specified but time_var is defined,
    # we will return data for all timesteps.
    time_values = get_times_in_range(dataset, time_var, time_range)
    is_multi_time = len(time_values) > 1

    # Irregular grids store levels as a 3D variable (DimK, DimJ, DimI), which
    # cannot be used as an Xarray coordinate for interpolation. Detect this case
    # upfront so we can route to the custom vertical interpolation path.
    has_levels = level_var is not None and level_var in dataset
    has_irregular_level = has_levels and dataset[level_var].ndim > 1
    has_regular_level = has_levels and dataset[level_var].ndim == 1
    # For irregular grids the level variable is 3D and cannot be set as a
    # fixed coordinate — the custom vertical interpolation step handles it later.
    spatial_interp_vars = None
    query_level = get_from_query(level_var, level, request)
    level = float(query_level) if query_level is not None else None
    if level is not None and level_var is not None and has_regular_level:
        fixed_coords[level_var] = level
        config.is_3d = False
        spatial_interp_vars = [level_var] if level_var not in interp_vars else []

    mesh_type = dget(dataset_config, "mesh_type", "auto")
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

    # Also treat as 3D when a specific level is requested on an irregular grid:
    # we need to load all levels so the custom interpolation can do its work.
    is_3d_grid = config.is_3d or has_irregular_level

    if is_3d_grid:
        # In 3D mode, the vertical dimension is not required to be fixed — we want all levels
        optional_coords = [lon_var, lat_var, level_var]
        coords_keep_dims = [lon_var, lat_var, level_var]
    else:
        # In 2D mode, level_var is NOT optional: the user must provide a vertical coordinate
        optional_coords = [lon_var, lat_var]
        coords_keep_dims = [lon_var, lat_var]

    if is_multi_time and time_var is not None and time_var not in optional_coords:
        optional_coords.append(time_var)

    fixed_coords, fixed_dims = get_required_dims_and_coords(
        dataset,
        variable,
        fixed_coords,
        fixed_dims,
        request,
        interp_vars=interp_vars,
        optional_coords=optional_coords,
        coords_keep_dims=coords_keep_dims,
        as_dims=config.as_dims or [],
    )

    lons = sel(dataset, lon_var, fixed_coords, fixed_dims)
    lats = sel(dataset, lat_var, fixed_coords, fixed_dims)
    levels_da = (
        sel(dataset, level_var, fixed_coords, fixed_dims) if has_levels else None
    )

    if not is_3d_grid and (
        (is_multi_time and lons.ndim > 3 and lons.shape[-3] != 1)
        or (not is_multi_time and lons.ndim > 2 and lons.shape[-3] != 1)
    ):
        raise exceptions.TooManyDimensions(lons.ndim)

    is_regular_grid = lons.ndim == 1 and lats.ndim == 1 and lons.dims != lats.dims
    is_point_list = (lons.ndim == 1 and lats.ndim == 1 and lons.dims == lats.dims) or (
        lons.ndim == 0 and lats.ndim == 0
    )
    pad = index_padding if format == "mesh" else 0

    lons_vals_in = np.atleast_1d(lons.values)
    lats_vals_in = np.atleast_1d(lats.values)

    levels_1d = None
    level_min, level_max, step_level = 0, 0, 1
    if is_3d_grid and has_regular_level:
        # Regular 3D: level is a 1-D coordinate vector
        levels_1d = levels_da.values
        if has_bb_level:
            level_min, level_max, levels_1d = (
                bbox.apply_level_bounding_box_regular_grid(levels_1d, bounding_box)
            )
        else:
            level_min, level_max = 0, len(levels_1d) - 1
            levels_1d = levels_1d[level_min : level_max + 1 : step_level]

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
    elif is_3d_grid and not is_regular_grid:
        step_logger.step_start("Irregular 3D grid: apply bounding box")
        lons_2d_slice = lons_vals_in[0] if lons_vals_in.ndim == 3 else lons_vals_in
        lats_2d_slice = lats_vals_in[0] if lats_vals_in.ndim == 3 else lats_vals_in
        lons_vals, lats_vals, indices = bbox.apply_irregular_bounding_box(
            lons_2d_slice,
            lats_2d_slice,
            bounding_box,
            False,
            spatial_padding,
            pad,
        )
        col_min, col_max = indices.col_min, indices.col_max
        row_min, row_max = indices.row_min, indices.row_max
        width_raw, height_raw = indices.width_raw, indices.height_raw
    else:
        step_logger.step_start("Unstructured grid: apply bounding box")
        lons_vals, lats_vals, indices = bbox.apply_irregular_bounding_box(
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

    resolution_limit = config.resolution_limit
    step_row, step_col = bbox.apply_resolution_limit(
        height_raw,
        width_raw,
        resolution_limit,
        is_point_list,
        n_points if is_point_list else 0,
    )

    step_logger.step_start("Load variable values")
    methods = {}
    for var in spatial_interp_vars if spatial_interp_vars is not None else []:
        methods[var] = interp_spatial_method
    for var in interp_vars:
        methods[var] = interp_vars_method
    vals_da = sel(
        dataset,
        variable,
        fixed_coords,
        fixed_dims,
        interp_vars=interp_vars
        + (spatial_interp_vars if spatial_interp_vars is not None else []),
        interp_methods=methods,
        interp_config=interp_vars_params,
    )
    if is_point_list:
        if resolution_limit is not None and n_points > resolution_limit:
            point_indices = point_indices[::step_row]
        vals = np.atleast_1d(vals_da.values)
        if is_multi_time:
            if vals.ndim == 1:
                # Point selection collapsed spatial dimension for a single point.
                vals = vals[:, np.newaxis]
            else:
                vals = vals[:, point_indices]
        else:
            vals = vals[point_indices]
    elif is_3d_grid and not is_regular_grid:
        if has_regular_level:
            vals = vals_da[
                ...,
                level_min : level_max + 1 : step_level,
                row_min : row_max + 1 : step_row,
                col_min : col_max + 1 : step_col,
            ].values
        else:
            vals = vals_da[
                ...,
                :,
                row_min : row_max + 1 : step_row,
                col_min : col_max + 1 : step_col,
            ].values
    elif is_3d_grid and is_regular_grid:
        if vals_da.ndim >= 3:
            vals = vals_da[
                level_min : level_max + 1 : step_level,
                row_min : row_max + 1 : step_row,
                col_min : col_max + 1 : step_col,
            ].values
        else:
            vals = vals_da[
                ..., row_min : row_max + 1 : step_row, col_min : col_max + 1 : step_col
            ].values
        # Transpose from (nz, ny, nx) → (nx, ny, nz) to match meshgrid(lons, lats, levels, indexing="ij")
        vals = vals.transpose(2, 1, 0)
    else:
        vals = vals_da[
            ..., row_min : row_max + 1 : step_row, col_min : col_max + 1 : step_col
        ].values
        # Squeeze out a trailing size-1 level when doing 2D extraction from a dataset that has levels
        if not config.is_3d and vals.ndim >= 3 and vals.shape[-3] == 1:
            vals = vals.squeeze(axis=-3)
            lons_vals_in = lons_vals_in.squeeze(axis=-3)
            lats_vals_in = lats_vals_in.squeeze(axis=-3)
            lons_vals = lons_vals.squeeze(axis=-3)
            lats_vals = lats_vals.squeeze(axis=-3)
            levels_da = levels_da.squeeze(axis=-3) if levels_da is not None else None
    vals = vals.astype(float)

    step_logger.step_start("Crop latitude and longitude")
    lons_1d, lats_1d = None, None
    levels = None
    if is_point_list:
        lons = lons_vals_in[point_indices]
        lats = lats_vals_in[point_indices]
    elif is_3d_grid and not is_regular_grid:
        # Crop the full 3D coordinate arrays along spatial axes; keep all levels
        lons = lons_vals_in[
            ...,
            row_min : row_max + 1 : step_row,
            col_min : col_max + 1 : step_col,
        ]
        lats = lats_vals_in[
            ...,
            row_min : row_max + 1 : step_row,
            col_min : col_max + 1 : step_col,
        ]
        if has_regular_level:
            levels = levels_1d
        else:
            levels = levels_da.values[
                ...,
                row_min : row_max + 1 : step_row,
                col_min : col_max + 1 : step_col,
            ]
    elif is_regular_grid:
        lons_1d = lons_vals_in[col_min : col_max + 1 : step_col]
        lats_1d = lats_vals_in[row_min : row_max + 1 : step_row]
        if config.is_3d and levels_1d is not None:
            lons, lats, levels = np.meshgrid(lons_1d, lats_1d, levels_1d, indexing="ij")
        else:
            lons, lats = np.meshgrid(lons_1d, lats_1d)
    else:
        lons = lons_vals[
            ..., row_min : row_max + 1 : step_row, col_min : col_max + 1 : step_col
        ]
        lats = lats_vals[
            ..., row_min : row_max + 1 : step_row, col_min : col_max + 1 : step_col
        ]

    if has_bb:  # Only lat/lon
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

        # Apply level bounding box if provided
        if (
            is_3d_grid
            and not is_regular_grid
            and has_bb_level
            and levels is not None
            and has_irregular_level
        ):
            mask_cropped &= bbox.apply_level_bounding_box_irregular_grid(
                levels, bounding_box
            )

        if format != "mesh":
            if is_multi_time:
                vals[:, ~mask_cropped] = np.nan
            else:
                vals[~mask_cropped] = np.nan
    elif (
        is_3d_grid
        and not is_regular_grid
        and has_bb_level
        and levels is not None
        and has_irregular_level
    ):
        # No spatial bbox but a Z bbox is present
        mask_cropped = bbox.apply_level_bounding_box_irregular_grid(
            levels, bounding_box
        )
        if is_multi_time:
            vals[:, ~mask_cropped] = np.nan
        else:
            vals[~mask_cropped] = np.nan
    else:
        mask_cropped = None

    # For irregular grids the level variable is 3D and Xarray cannot interpolate
    # along it. So we need a custom interpolation step that handles this case
    if has_irregular_level and level is not None:
        irregular_level_method = (
            "linear"
            if interp_spatial_method is not None and interp_spatial_method != "nearest"
            else "nearest"
        )
        step_logger.step_start(
            f"Irregular 3D grid: vertical level {irregular_level_method} selection"
        )
        vals = interpolation.interpolate_level_irregular_grid(
            vals, levels, level, method=irregular_level_method
        )
        lons = lons[0]
        lats = lats[0]
        levels = None
        mask_cropped = np.isfinite(vals)
        is_3d_grid = False

    cell_data = force_data_mapping != "vertices" and (
        force_data_mapping == "cells"
        or dget(dataset_config, "mesh_data_on_cells", False)
    )
    if cell_data and not is_point_list:
        step_logger.step_start("Cell to point data conversion")
        lons, lats, levels, vals, lons_1d, lats_1d, levels_1d = (
            interpolation.cell_to_point_conversion(
                lons,
                lats,
                levels,
                vals,
                variable,
                mesh_type,
                is_regular_grid,
                is_3d=config.is_3d,
            )
        )

    if mesh_tile_shape is not None:
        step_logger.step_start("Generate meshgrid")
        target_h, target_w = mesh_tile_shape
        if is_3d_grid and is_regular_grid and levels_1d is not None:
            target_d = levels_1d.shape[0]
        elif is_3d_grid and not is_regular_grid and levels is not None:
            target_d = levels.shape[0]
        else:
            target_d = 1
        lons, lats, levels, vals, mask_cropped = (
            interpolation.generate_meshgrid_and_interpolate(
                lons,
                lats,
                levels,
                vals,
                lons_1d,
                lats_1d,
                levels_1d,
                bounding_box,
                target_w,
                target_h,
                target_d,
                is_regular_grid,
                is_point_list,
                interp_spatial_method,
                interp_spatial_params,
                is_3d=config.is_3d,
            )
        )

    # For irregular grids with a regular vertical coordinate,
    # we need to broadcast the 2D lon/lat arrays and 1D level array to 3D arrays
    # so they can be used together in the output generation step
    if not is_regular_grid and has_regular_level and levels is not None:
        nk = len(levels)
        nj, ni = lons.shape
        lons = np.broadcast_to(lons[np.newaxis, :, :], (nk, nj, ni))
        lats = np.broadcast_to(lats[np.newaxis, :, :], (nk, nj, ni))
        levels = np.broadcast_to(levels[:, np.newaxis, np.newaxis], (nk, nj, ni))
        if mask_cropped is not None:
            mask_cropped = np.broadcast_to(mask_cropped[np.newaxis, :, :], (nk, nj, ni))

    # Crop irregular grids to the bounding box after interpolation to avoid returning huge arrays with mostly NaNs when a tight bbox is applied on a sparse grid (e.g. Z-bounding box on an irregular grid with few vertical levels)
    if not is_regular_grid and mask_cropped is not None:
        # As vals can have been squeezed to 2D if "level" was only a single level
        # we need to do the same with lons and lats
        lons = lons.squeeze()
        lats = lats.squeeze()
        vals = vals.squeeze()
        levels = levels.squeeze() if levels is not None else None
        mask_cropped = mask_cropped.squeeze()

        lons_cropped = lons[mask_cropped]
        lats_cropped = lats[mask_cropped]
        levels_cropped = levels[mask_cropped] if levels is not None else None
        if is_multi_time and vals.ndim > 1:
            vals_cropped = vals[:, mask_cropped]
        else:
            vals_cropped = vals[mask_cropped]
    else:
        lons_cropped, lats_cropped, levels_cropped, vals_cropped = (
            lons,
            lats,
            levels,
            vals,
        )

    if levels_cropped is None and level is not None:
        levels_cropped = level

    global_props = {"resolution_factor": {"row": step_row, "col": step_col}}
    if is_multi_time:
        global_props["times"] = time_values

    if format == "mesh":
        step_logger.step_start("Prepare output (mesh)")
        out = output.prepare_mesh_output(
            lons, lats, levels, vals, variable, mask_cropped, step_row, step_col
        )
    elif format == "raw":
        step_logger.step_start("Prepare output (raw)")
        out = output.prepare_raw_output(
            [variable],
            [vals_cropped],
            lons_cropped,
            lats_cropped,
            levels=levels_cropped,
            global_props=global_props,
            var_props={variable: dataset[variable].attrs},
            has_time_dimension=is_multi_time,
            as_numpy=True,
        )
    elif format == "geojson":
        step_logger.step_start("Prepare output (GeoJSON)")

        out = output.prepare_geojson_output(
            [variable],
            [vals_cropped],
            lons_cropped,
            lats_cropped,
            levels=levels_cropped,
            collection_props=global_props,
            var_props={variable: dataset[variable].attrs},
            has_time_dimension=is_multi_time,
        )
    else:
        raise exceptions.BadConfigurationVariable(f"Unsupported format: {format}")

    # Serialized here (in the worker thread) with orjson, which handles numpy
    # arrays natively, rather than by FastAPI in the event loop
    step_logger.step_start("Serialize output")
    content = serialization.dumps(out)

    step_logger.end()
    return content


def _is_uniform_time_grid(flat_times: list, group_sizes: list[int]) -> bool:
    """True when every point was asked for the exact same list of times (same
    values, same order). In that case, a batch of per-point/time samples is
    really a regular (time, point) grid rather than independent per-point
    series, and can be reshaped back into that grid for the response.
    """
    if not group_sizes or len(set(group_sizes)) != 1:
        return False
    size = group_sizes[0]
    if size == 0:
        return False
    reference = flat_times[:size]
    offset = 0
    for group_size in group_sizes:
        if flat_times[offset : offset + group_size] != reference:
            return False
        offset += group_size
    return True


def _expand_time_range_strings(dataset, time_var, times):
    """Expand any 'start/end' range string in `times` into the concrete
    dataset timestamps it covers, de-duplicating and sorting the whole list
    exactly as a single combined time-range query would (matching the
    historical, pre-flatten behaviour of `get_times_in_range`). A list with
    no range string is returned unchanged, so this is a no-op for the common
    case of exact timestamps (trajectories and per-point `times` already
    forbid ranges at the schema level).
    """
    if not times or not any(isinstance(t, str) and "/" in t for t in times):
        return times
    if time_var is None or time_var not in dataset:
        return times
    expanded = get_times_in_range(dataset, time_var, MultiTimeRange.from_strings(times))
    return expanded if expanded else times


def _idw_weights(dists: np.ndarray, power: float) -> np.ndarray:
    """Inverse-distance-weighting weights for one point's neighbors, from
    their already-computed distances. An exact (zero-distance) match gets
    weight 1.0 instead of dividing by zero. Shared by both IDW
    neighbor-selection strategies ("radius" and "k") in `probe()`, since
    weighting only depends on the resulting distances, not on how the
    neighbors were found.
    """
    dists = np.asarray(dists)
    zero_dist = dists < 1e-12
    if np.any(zero_dist):
        weights = np.zeros(len(dists))
        weights[np.argmax(zero_dist)] = 1.0
    else:
        weights = (1.0 / (dists**power)) / np.sum(1.0 / (dists**power))
    return weights


def probe(
    request: Request,
    dataset_id: str,
    variables: str | list[str],
    points: list[dict[str, float]],
    time_range: str | list[str] | None = None,
    point_times: list[list[str] | None] | None = None,
    is_path: bool = False,
    is_single_probe: bool = False,
    format: str = "raw",
    config: dict[str, Any] | ExtractionConfig | None = None,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    if not isinstance(config, ExtractionConfig):
        config = ExtractionConfig.model_validate(config or {})

    step_logger = StepLoggerAndAborter(
        "probe",
        parameters=(
            dataset_id,
            variables,
            f"{len(points)} points",
            f"{len(time_range) if time_range else 0} times",
            is_path,
            format,
            config,
        ),
        cancel_event=cancel_event,
    )
    step_logger.step_start("Load dataset and config")

    variables = variables if isinstance(variables, list) else [variables]

    with_level = any(point.level is not None for point in points)
    dataset, dataset_config = load_dataset(dataset_id)
    fixed_coords, fixed_dims = dgets(
        dataset_config, [FIXED_VARIABLES_KEY, FIXED_DIMENSIONS_KEY], {}
    )
    interp_vars = config.interpolation.vars.items
    spatial_interp_vars = []
    lon_var, lat_var, time_var = dgets(
        dataset_config,
        [LON_VARIABLE_KEY, LAT_VARIABLE_KEY, "variables.time"],
    )
    time_dim = dget(dataset_config, "dimensions.time")
    if time_var is not None:
        request.query_params._dict.pop("time", None)

    # Flatten (points, times) into the canonical engine input: one sample per
    # (point, time) pair, all sharing a single "point" dimension.
    # `point_times`, when given, wins per point; any entry left unset (or the
    # whole thing, for the legacy ad hoc "points" + shared "times" request)
    # falls back to `time_range`, and finally — if there is no time
    # information at all for that point — to every timestep the dataset has,
    # matching the historical "no time filter" probe behaviour.
    fallback_times = (
        time_range
        if isinstance(time_range, list)
        else ([time_range] if time_range else None)
    )
    if point_times is None:
        point_times = [None] * len(points)

    resolved_point_times = []
    for pt_times in point_times:
        if pt_times:
            resolved_point_times.append(
                _expand_time_range_strings(dataset, time_var, list(pt_times))
            )
        elif fallback_times:
            resolved_point_times.append(
                _expand_time_range_strings(dataset, time_var, list(fallback_times))
            )
        elif time_var is not None and time_var in dataset:
            time_vals = dataset[time_var].values
            if np.issubdtype(time_vals.dtype, np.datetime64):
                resolved_point_times.append(
                    [str(np.datetime_as_string(t)) for t in time_vals]
                )
            else:
                resolved_point_times.append(time_vals.tolist())
        else:
            resolved_point_times.append([None])

    group_sizes = [len(t) for t in resolved_point_times]
    flat_time_range = []
    for times_for_point in resolved_point_times:
        flat_time_range.extend(times_for_point)
    time_range = flat_time_range

    is_uniform_grid = _is_uniform_time_grid(time_range, group_sizes)
    time_range = MultiTimeRange.from_strings(time_range)
    level_vars = get_dataset_level_vars(dataset, dataset_config)
    level_var = None
    if level_vars is None or isinstance(level_vars, str):
        level_var = level_vars
    else:
        for lv in level_vars:
            if all(var in level_vars[lv] for var in variables):
                level_var = lv
                break
        if level_var is None:
            raise exceptions.DifferentTypesOfLevel()

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
    if with_level and (level_var is None or level_var not in dataset):
        missing_vars.append(f"level ({level_var})")
    if time_range.has_time() and (time_var is None or time_var not in dataset):
        missing_vars.append(f"time ({time_var})")
    if len(missing_vars) > 0:
        raise exceptions.BadConfigurationVariable(missing_vars)

    if time_range.has_time() and time_var is not None:
        time_info = is_time_out_of_bounds_data(dataset, time_var, time_range)
        if time_info["out_of_bounds"]:
            raise exceptions.TimeOutOfBounds(
                time_str=time_info["time"],
                min_time=time_info["min_time"],
                max_time=time_info["max_time"],
            )
        # Bounded time ranges are also used for paths
        time_range = get_bounded_time(dataset, time_var, time_range)
        if config.interpolation.vars.time and time_var not in interp_vars:
            interp_vars.append(time_var)

    interp_spatial_method = config.interpolation.spatial.method
    interp_spatial_params = config.interpolation.spatial.params or {}
    interp_vars_method = config.interpolation.vars.method
    interp_vars_params = config.interpolation.vars.params or {}

    longitudes = dataset[lon_var]
    latitudes = dataset[lat_var]
    is_regular_grid = (
        longitudes.ndim == 1
        and latitudes.ndim == 1
        and longitudes.dims != latitudes.dims
    )
    has_regular_level = level_var is not None and dataset[level_var].ndim == 1
    if with_level and has_regular_level and interp_spatial_method != "nearest":
        spatial_interp_vars.append(level_var)
    if is_regular_grid and interp_spatial_method != "nearest":
        spatial_interp_vars.extend([lon_var, lat_var])

    # Pre-parse levels to determine global with_level correctly
    parsed_levels = []
    for point in points:
        query_level = get_from_query(level_var, point.level, request)
        parsed_levels.append(float(query_level) if query_level is not None else None)
    with_level = any(lvl is not None for lvl in parsed_levels)

    knn_k = interp_spatial_params.get("k")
    use_knn = knn_k is not None or "radius" not in interp_spatial_params
    if use_knn:
        knn_k = knn_k if knn_k is not None else 4
        try:
            knn_k = int(knn_k)
        except (TypeError, ValueError):
            knn_k = None
        if not knn_k or knn_k <= 0:
            raise exceptions.BadSelection(
                "interp_spatial_params.k must be a positive integer."
            )
    else:
        max_radius = interp_spatial_params.get("radius", 0.05)
    power = interp_spatial_params.get("power", 2.0)
    # Scale the vertical axis to match horizontal degrees for 3D IDW.
    # Combining horizontal degrees (~111 km/deg) with raw vertical units
    # (meters, hPa) directly is physically meaningless and causes one axis
    # to dwarf the other.
    # `level_scale` defines how many vertical units equal 1 horizontal degree,
    # converting the vertical coordinate into degree-equivalents. It is baked
    # directly into the tree's own coordinates below, so that neighbor *selection*
    # -- for both "radius" and "k" -- already respects it, not only the final 
    # weighting.
    # Caveat: Assumes a roughly linear relationship to physical distance (e.g., meters).
    # For nonlinear coordinates like pressure levels (logarithmic with altitude),
    # this is only a local approximation; consider pre-transforming the data.
    level_scale = interp_spatial_params.get("level_scale", 1.0)
    if not isinstance(level_scale, (int, float)) or level_scale <= 0:
        raise exceptions.BadSelection(
            "interp_spatial_params.level_scale must be a positive number."
        )

    tree = None
    if not is_regular_grid:
        # Project (lon, lat) to 3D Cartesian (x, y, z) on the unit sphere.
        # Raw degrees are geometrically invalid: they distort distances outside
        # the equator, compress near the poles, and break across the antimeridian.
        # In 3D, Euclidean chord distance is monotonic with true great-circle distance.
        # Applied systematically to all irregular grids to avoid these distortions.
        sphere_points = lonlat_to_sphere_points(
            longitudes.values.ravel(), latitudes.values.ravel()
        )
        if with_level and not has_regular_level:
            # Fold `level_scale` in here, converting the raw level axis into
            # degree-equivalents and then into radians -- the same "small
            # angle" scale the (x, y, z) chord coordinates above are
            # naturally in. Without this second radians conversion, the
            # level axis (degree-scale) would still dwarf the chord axis
            # (radian-scale, ~57x smaller for an equal angular size), which
            # would silently bias the tree's own distance ordering again,
            # just along a different axis.
            levels_arr = dataset[level_var].values
            level_col = np.radians(levels_arr.ravel() / level_scale)
            grid_points = np.column_stack((sphere_points, level_col))
        else:
            grid_points = sphere_points
        coord_vars = (
            (lon_var, lat_var, level_var, f"level_scale={level_scale}")
            if with_level and not has_regular_level
            else (lon_var, lat_var, level_var)
            if with_level
            else (lon_var, lat_var)
        )
        tree = get_cached_ckdtree(
            grid_points, dataset_id=dataset_id, coord_vars=coord_vars
        )

    points_data = []
    target_pts = []

    # Calculate dataset longitude convention once to avoid calling .min() in a loop, as bbox.normalize_dataset_longitudes would do
    lon_min_val = float(dataset[lon_var].min()) if lon_var and lon_var in dataset else 0
    is_0_360 = lon_min_val >= 0

    for i, point in enumerate(points):
        level = parsed_levels[i]

        # Normalize longitudes inline
        lon = point.lon
        if lon_var and lon_var in dataset:
            if is_0_360:
                if lon < 0:
                    lon = lon + 360
                elif lon > 360:
                    lon = lon - 360
            else:
                if lon > 180:
                    lon = lon - 360
                elif lon < -180:
                    lon = lon + 360

        p_data = {
            "lon": lon,
            "lat": point.lat,
            "level": level,
            "fixed_coords": {},
            "fixed_dims": {},
            "spatial_indexers": None,
            "weights_da": None,
        }

        if with_level and has_regular_level:
            p_data["fixed_coords"][level_var] = level
        if is_regular_grid:
            p_data["fixed_coords"][lon_var] = lon
            p_data["fixed_coords"][lat_var] = point.lat
        else:
            px, py, pz = lonlat_to_xyz(lon, point.lat)
            if with_level and not has_regular_level:
                # Same level_scale + radians pre-scaling as `grid_points`
                # above, so the target point lands in the tree's own
                # coordinate space.
                target_pts.append([px, py, pz, np.radians(level / level_scale)])
            else:
                target_pts.append([px, py, pz])

        points_data.append(p_data)

    # Structured irregular grids (lon/lat arrays of 2+ dimensions sharing the
    # same dims, e.g. curvilinear or radial grids): flag the probes falling
    # outside the grid. Nearest / IDW would otherwise silently return the
    # value of the closest node, however far it is. Their values are set to
    # NaN below, as a regular grid does when interpolating outside its extent.
    # Point lists (1D lon/lat) have no cell topology and are not checked.
    outside_mask = np.zeros(len(points), dtype=bool)
    is_structured_irregular_grid = (
        not is_regular_grid
        and longitudes.ndim >= 2
        and longitudes.dims == latitudes.dims
    )
    if is_structured_irregular_grid and target_pts:
        step_logger.step_start("Check probes against grid extent")
        query_xyz = np.array(target_pts)[:, :3]
        if grid_points.shape[1] == 3:
            horizontal_tree = tree
        else:
            horizontal_tree = get_cached_ckdtree(
                sphere_points, dataset_id=dataset_id, coord_vars=(lon_var, lat_var)
            )
        nearest_dist, nearest_idx = horizontal_tree.query(query_xyz, k=1)
        geometry = get_cached_structured_grid_geometry(
            longitudes.values,
            latitudes.values,
            dataset_id=dataset_id,
            coord_vars=(lon_var, lat_var),
            data_on_cells=bool(dget(dataset_config, "mesh_data_on_cells", False)),
        )
        outside_mask = ~geometry.contains(query_xyz, nearest_idx, nearest_dist)
        if outside_mask.any():
            log.info(
                "[KAZARR] {n}/{total} probe(s) outside the grid of dataset {dataset_id}",
                n=int(outside_mask.sum()),
                total=len(points),
                dataset_id=dataset_id,
            )

    # Query all target points at once instead of in a Python loop for massive speedups.
    if not is_regular_grid and target_pts:
        target_pts_arr = np.array(target_pts)
        if interp_spatial_method != "nearest":
            # IDW Interpolation on unstructured grid.
            if use_knn:
                knn_k_eff = min(knn_k, len(grid_points))
                query_dists, query_indices = tree.query(target_pts_arr, k=knn_k_eff)
                if knn_k_eff == 1:
                    # scipy collapses the neighbor axis when k=1; restore it
                    # so every downstream index/loop can assume 2D arrays.
                    query_dists = query_dists[:, np.newaxis]
                    query_indices = query_indices[:, np.newaxis]
                max_neighbors = knn_k_eff
            else:
                if with_level and not has_regular_level:
                    # We want to match points inside an anisotropic ellipsoid:
                    #     sqrt(spatial_deg**2 + (level_diff / level_scale)**2) <= max_radius
                    # The tree's own 4D coordinates already combine a chord
                    # distance (spatial) with a radians(level_diff / level_scale)
                    # value (level) -- both on the same small-angle scale. Since
                    # chord(x) <= radians(x) for any x >= 0, every point that
                    # truly satisfies the ellipsoid bound above also satisfies
                    # the tree's own native
                    #     sqrt(chord**2 + radians(level_diff/level_scale)**2) <= radians(max_radius)
                    # so querying with radians(max_radius) is a safe,
                    # conservative superset -- tight, not just "over-inclusive":
                    # it differs from the exact ellipsoid only by the same
                    # negligible chord-vs-radians gap already accepted for the
                    # pure spatial case below. We still filter exactly below to
                    # correct that tiny gap.
                    query_radius = np.radians(max_radius)
                else:
                    # Pure spatial case: convert the radius from degrees of true
                    # angular distance to the equivalent unit-sphere chord
                    # distance, so the query threshold matches the tree's space.
                    query_radius = deg_to_chord_distance(max_radius)
                indices_list = tree.query_ball_point(target_pts_arr, r=query_radius)
                max_neighbors = (
                    max(len(idx) for idx in indices_list) if len(indices_list) > 0 else 0
                )
                # Probes outside the grid may have no neighbor at all: they get
                # a single placeholder neighbor (their value is discarded).
                max_neighbors = max(max_neighbors, 1)

            for i in range(len(target_pts_arr)):
                if use_knn:
                    indices = query_indices[i]
                else:
                    indices = indices_list[i]
                    if not indices:
                        if not outside_mask[i]:
                            raise exceptions.NoDataInSelection(
                                f"Try increasing interpolation radius (point {i + 1})"
                            )
                        # Outside the grid: placeholder neighbor, value
                        # replaced by NaN after extraction.
                        indices = [0]

                target_pt = target_pts_arr[i]
                if with_level and not has_regular_level:
                    # Keep the spatial (chord) and level components separate:
                    # convert the spatial part back to true angular degrees,
                    # and convert the (already level_scale-scaled) level
                    # component back out of radians into the same
                    # degree-equivalent unit, so `power` applies to a single
                    # physically-consistent anisotropic distance instead of
                    # mixing degrees with a different scale.
                    neighbors_coords = grid_points[indices]
                    spatial_chord = np.linalg.norm(
                        neighbors_coords[:, :3] - target_pt[:3], axis=1
                    )
                    spatial_deg = chord_to_deg_distance(spatial_chord)
                    # neighbors_coords[:, 3] is already level_diff pre-scaled
                    # by level_scale and converted to radians (see grid_points
                    # construction above) -- undo the radians conversion to
                    # get back to the same degree-equivalent unit as
                    # spatial_deg. No division by level_scale here: that was
                    # already applied when the tree was built.
                    level_diff_deg = np.degrees(neighbors_coords[:, 3] - target_pt[3])
                    dists = np.sqrt(spatial_deg**2 + level_diff_deg**2)

                    if not use_knn:
                        # The query above is a tight, conservative bound (see
                        # comment there) -- filter down to the exact requested
                        # ellipsoid before weighting.
                        within_radius = dists <= max_radius
                        if not np.any(within_radius):
                            if not outside_mask[i]:
                                raise exceptions.NoDataInSelection(
                                    f"Try increasing interpolation radius (point {i + 1})"
                                )
                            within_radius = np.ones_like(within_radius)
                        indices = [
                            idx for idx, keep in zip(indices, within_radius) if keep
                        ]
                        dists = dists[within_radius]
                    # In "k" mode there is no radius to filter against: these
                    # are simply the `knn_k` closest neighbors under the
                    # tree's own native metric. Since that metric is now the
                    # pre-scaled ellipsoid distance itself (not the raw,
                    # unscaled one), neighbor *selection* here is consistent
                    # with `level_scale`, just like "radius" mode -- both are
                    # limited only by the same negligible chord-vs-radians
                    # approximation, not by an anisotropy bias.
                elif use_knn:
                    # tree.query already returns the exact chord distance —
                    # no need to look the neighbor coordinates back up.
                    dists = chord_to_deg_distance(query_dists[i])
                else:
                    neighbors_coords = grid_points[indices]
                    spatial_chord = np.linalg.norm(neighbors_coords - target_pt, axis=1)
                    dists = chord_to_deg_distance(spatial_chord)

                weights = _idw_weights(dists, power)

                # As each point may have a different number of neighbors
                # ("radius" mode only — "k" mode always has exactly
                # `max_neighbors`), we need to pad the indices and weights
                # arrays to the same length for batch processing later.
                pad_width = max_neighbors - len(indices)
                if pad_width > 0:
                    indices = list(indices) + [indices[0]] * pad_width
                    weights = np.pad(
                        weights, (0, pad_width), mode="constant", constant_values=0.0
                    )

                neighbor_indices = [
                    np.unravel_index(idx, longitudes.shape) for idx in indices
                ]
                dim_indices = {dim: [] for dim in longitudes.dims}
                for idx in neighbor_indices:
                    for dim_name, indice in zip(longitudes.dims, idx):
                        dim_indices[dim_name].append(indice)

                points_data[i]["spatial_indexers"] = {
                    dim: xr.DataArray(vals, dims=["neighbor"])
                    for dim, vals in dim_indices.items()
                }
                points_data[i]["weights_da"] = xr.DataArray(weights, dims=["neighbor"])
        else:
            # Nearest neighbor on unstructured grid
            _, flat_indices = tree.query(target_pts_arr, k=1)
            flat_indices = np.atleast_1d(flat_indices)
            for i, flat_index in enumerate(flat_indices):
                indices = np.unravel_index(flat_index, longitudes.shape)
                for dim_name, indice in zip(longitudes.dims, indices):
                    points_data[i]["fixed_dims"][dim_name] = indice

    # Broadcast the deduplicated per-point spatial resolution (one entry per
    # requested point) back onto the full flattened (point, time) batch:
    # every sample sharing a point gets an identical copy of that point's
    # already-resolved spatial data, so the temporal extraction below still
    # sees one row per (point, time) sample exactly as before.
    repeat_idx = np.repeat(np.arange(len(points)), group_sizes)
    points_data = [points_data[j] for j in repeat_idx]
    sample_outside_mask = outside_mask[repeat_idx]

    optional_coords = [time_var] if time_var is not None else []
    optional_dims = [time_dim] if time_dim is not None else []

    if points_data:
        p0 = points_data[0]
        for key in ["fixed_coords", "fixed_dims", "spatial_indexers"]:
            if p0.get(key):
                if key == "fixed_coords":
                    optional_coords.extend(p0[key].keys())
                else:
                    optional_dims.extend(p0[key].keys())

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
    step_logger.step_start("Extract variable values at probe locations")
    lats, lons, levels = [], [], []
    # `times` is filled in below, one entry per flattened (point, time)
    # sample — unless the dataset has no time dimension at all, in which case
    # it stays None throughout.
    times = None if time_var is None else []
    var_props = {}

    lats = [p["lat"] for p in points_data]
    lons = [p["lon"] for p in points_data]
    levels = [p["level"] for p in points_data]

    batch_fixed_coords = {**fixed_coords}
    batch_fixed_dims = {**fixed_dims}

    # Aggregate spatial coords/dims mapped along the new `point` dimension
    if points_data:
        if points_data[0]["fixed_coords"]:
            for k in points_data[0]["fixed_coords"]:
                batch_fixed_coords[k] = xr.DataArray(
                    [p["fixed_coords"][k] for p in points_data], dims=["point"]
                )
        if points_data[0]["fixed_dims"]:
            for k in points_data[0]["fixed_dims"]:
                batch_fixed_dims[k] = xr.DataArray(
                    [p["fixed_dims"][k] for p in points_data], dims=["point"]
                )

    # Aggregate spatial indexers and weights for IDW interpolation on unstructured grids
    batch_spatial_indexers = None
    batch_weights_da = None
    if points_data and points_data[0]["spatial_indexers"]:
        batch_spatial_indexers = {}
        for k in points_data[0]["spatial_indexers"]:
            vals_k = np.stack([p["spatial_indexers"][k].values for p in points_data])
            batch_spatial_indexers[k] = xr.DataArray(vals_k, dims=["point", "neighbor"])
        weights_vals = np.stack([p["weights_da"].values for p in points_data])
        batch_weights_da = xr.DataArray(weights_vals, dims=["point", "neighbor"])

    # Extract time points: one requested timestamp per flattened sample.
    # Samples whose requested time is outside the dataset time extent are
    # ignored: like probes outside the grid, they still go through the batch
    # selection with a valid placeholder time (their start was already clipped
    # to the nearest bound by get_bounded_time), then get NaN values and a
    # None time in the output. Appending None here instead would break the
    # vectorized selection (None -> NaT in a datetime64 array).
    sample_time_oob_mask = np.array(
        [tr.is_out_of_bounds for tr in time_range.ranges], dtype=bool
    )
    req_times = []
    for tr in time_range.ranges:
        if tr.start is not None:
            req_times.append(tr.start)
        elif time_var is not None and time_var in dataset:
            req_times.append(dataset[time_var].values[0])
        else:
            req_times.append(None)

    if time_var is not None:
        if time_range.has_time() and interp_vars_method != "nearest":
            times = []
            for t in req_times:
                try:
                    t_parsed = np.datetime64(t)
                    times.append(str(np.datetime_as_string(t_parsed)))
                except ValueError:
                    times.append(str(t))
        elif time_var in dataset:
            req_times_arr = np.array(req_times, dtype=dataset[time_var].dtype)
            temp_fixed_coords = {
                **batch_fixed_coords,
                time_var: xr.DataArray(req_times_arr, dims=["point"]),
            }

            time_da = sel(
                dataset,
                time_var,
                temp_fixed_coords,
                batch_fixed_dims,
                interp_method=interp_vars_method,
                interp_config=interp_vars_params,
            )
            time_vals = time_da.values
            if time_vals.ndim > 1:
                time_vals = time_vals.squeeze()

            if np.issubdtype(time_vals.dtype, np.datetime64):
                times = [str(np.datetime_as_string(t)) for t in time_vals]
            else:
                times = time_vals.tolist()
        else:
            times = req_times

    if times and time_var:
        time_dtype = dataset[time_var].dtype
        time_arr = np.array(times, dtype=time_dtype)
        batch_fixed_coords[time_var] = xr.DataArray(time_arr, dims=["point"])

    # Out-of-bounds times are reported as None in the output (only after the
    # batch coords above were built with the valid placeholder times).
    if times and sample_time_oob_mask.any():
        times = [
            None if oob else t for t, oob in zip(times, sample_time_oob_mask)
        ]

    # Samples to blank out: probe outside the grid, or time out of bounds
    sample_ignored_mask = sample_outside_mask.copy()
    if sample_time_oob_mask.size == sample_ignored_mask.size:
        sample_ignored_mask |= sample_time_oob_mask

    # One error code (or None) per sample, i.e. per (point, time), reported in
    # the output "errors" field aligned with the times. A sample that is both
    # outside the grid and out of the time bounds gets the combined code.
    sample_location_oob = np.asarray(sample_outside_mask, dtype=bool)
    sample_time_oob = (
        sample_time_oob_mask
        if sample_time_oob_mask.size == len(points_data)
        else np.zeros(len(points_data), dtype=bool)
    )
    sample_errors = [None] * len(points_data)
    for i in np.flatnonzero(sample_location_oob | sample_time_oob):
        if sample_location_oob[i] and sample_time_oob[i]:
            sample_errors[i] = exceptions.TIME_AND_LOCATION_OUT_OF_BOUNDS
        elif sample_location_oob[i]:
            sample_errors[i] = exceptions.LOCATION_OUT_OF_BOUNDS
        else:
            sample_errors[i] = exceptions.TIME_OUT_OF_BOUNDS

    interp_methods = None
    if spatial_interp_vars:
        interp_spatial_method_pt = "linear"
        interp_vars_pt = list(dict.fromkeys(spatial_interp_vars + interp_vars))
        interp_methods = dict.fromkeys(spatial_interp_vars, interp_spatial_method_pt)
    else:
        interp_vars_pt = interp_vars
    data = []
    for var in variables:
        if not is_regular_grid and interp_spatial_method != "nearest":
            # For IDW on unstructured grid: filter base dimensions then multiply by spatial indexers
            base_fixed_dims = {
                k: v for k, v in batch_fixed_dims.items() if k not in longitudes.dims
            }
            filtered_da = sel(
                dataset,
                var,
                batch_fixed_coords,
                base_fixed_dims,
                interp_vars=interp_vars_pt,
                interp_method=interp_vars_method,
                interp_methods=interp_methods,
                interp_config=interp_vars_params,
            )
            neighbor_data = filtered_da.isel(**batch_spatial_indexers)
            all_nan_mask = neighbor_data.isnull().all(dim="neighbor")
            interpolated_values = (neighbor_data * batch_weights_da).sum(dim="neighbor")
            interpolated_values = interpolated_values.where(~all_nan_mask)
            da_result = interpolated_values
        else:
            # Native pointwise selection for regular grids or nearest neighbors
            da_result = sel(
                dataset,
                var,
                batch_fixed_coords,
                batch_fixed_dims,
                interp_vars=interp_vars_pt,
                interp_method=interp_vars_method,
                interp_methods=interp_methods,
                interp_config=interp_vars_params,
            )

        # Force 'point' dimension as the last dimension so that .values is returned
        # as [time, points] instead of [points, time].
        dims = list(da_result.dims)
        if "point" in dims:
            dims.remove("point")
            dims.append("point")
            da_result = da_result.transpose(*dims)

        var_data = da_result.values
        if var_data.ndim == 1:
            var_data = var_data[np.newaxis, :]
        if sample_ignored_mask.any():
            # Probes outside the grid or with an out-of-bounds time: no value
            # (null in the output, and a NO_DATA_IN_SELECTION error if no
            # probe at all is inside).
            if not np.issubdtype(var_data.dtype, np.floating):
                var_data = var_data.astype(float)
            else:
                var_data = var_data.copy()
            var_data[..., sample_ignored_mask] = np.nan

        data.append(var_data)
        var_props[var] = dataset[var].attrs

    # Decide how to shape the response: a single connected trajectory line, a
    # (time, point) grid, or one independent series per requested point. The
    # computation above is always identical (the flat, vectorized batch) —
    # only this final step differs.
    if is_path:
        output_mode = "path"
    elif is_uniform_grid:
        output_mode = "grid"
    else:
        output_mode = "series"

    if output_mode == "series":
        step_logger.step_start("Prepare output (per-point series)")
        n_groups = len(group_sizes)
        cum = np.concatenate(([0], np.cumsum(group_sizes)))
        group_lons = [lons[cum[i]] for i in range(n_groups)]
        group_lats = [lats[cum[i]] for i in range(n_groups)]
        group_levels = (
            [levels[cum[i]] for i in range(n_groups)] if with_level else None
        )
        group_times = (
            [list(times[cum[i] : cum[i + 1]]) for i in range(n_groups)]
            if times is not None
            else None
        )
        group_data = {
            var: [
                data[vi][0, cum[i] : cum[i + 1]].tolist() for i in range(n_groups)
            ]
            for vi, var in enumerate(variables)
        }
        # One error list per point, aligned with that point's times
        group_errors = [sample_errors[cum[i] : cum[i + 1]] for i in range(n_groups)]
        out = output.prepare_series_output(
            variables,
            group_data,
            group_lons,
            group_lats,
            levels=group_levels,
            times=group_times,
            var_props=var_props,
            format=format,
            errors=group_errors,
        )
        step_logger.end()
        return out

    # Path: one error code (or None) per vertex, aligned with the path times.
    # Grid: overridden below with one error list per point, aligned with times.
    errors = sample_errors

    if output_mode == "grid":
        # The engine always computes a flat, per-sample batch. When every
        # point shares the exact same requested times, fold it back into the
        # (time, point) grid the "raw"/"geojson" grid output expects, instead
        # of ever building it via a slower orthogonal broadcast.
        n_groups = len(group_sizes)
        n_times = group_sizes[0]
        cum = np.concatenate(([0], np.cumsum(group_sizes)))
        errors = [sample_errors[cum[i] : cum[i + 1]] for i in range(n_groups)]
        lons = [lons[cum[i]] for i in range(n_groups)]
        lats = [lats[cum[i]] for i in range(n_groups)]
        if with_level:
            levels = [levels[cum[i]] for i in range(n_groups)]
        if times is not None:
            times = list(times[:n_times])
        data = [d.reshape(n_groups, n_times).T for d in data]

    output_has_time = (times is not None) if not is_single_probe else False

    if format == "raw":
        step_logger.step_start("Prepare output (raw)")
        out = output.prepare_raw_output(
            variables,
            data,
            np.array(lons),
            np.array(lats),
            levels=np.array(levels) if with_level else None,
            global_props={"times": times} if times is not None else None,
            var_props=var_props,
            has_time_dimension=output_has_time,
            is_path=is_path,
            errors=errors,
        )
    elif format == "geojson":
        step_logger.step_start("Prepare output (GeoJSON)")
        out = output.prepare_geojson_output(
            variables,
            data,
            np.array(lons),
            np.array(lats),
            levels=np.array(levels) if with_level else None,
            collection_props={"times": times} if times is not None else {},
            var_props=var_props,
            has_time_dimension=output_has_time,
            is_path=is_path,
            line_string_props={"times": times}
            if (is_path and times is not None)
            else None,
            errors=errors,
        )
    else:
        raise exceptions.BadConfigurationVariable(f"Unsupported format: {format}")

    step_logger.end()
    return out


def free_selection(
    request: Request,
    dataset_id: str,
    variable: str,
    config: dict[str, Any] | ExtractionConfig | None = None,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    if not isinstance(config, ExtractionConfig):
        config = ExtractionConfig.model_validate(config or {})
    step_logger = StepLoggerAndAborter(
        "free_selection",
        parameters=(dataset_id, variable, config),
        cancel_event=cancel_event,
    )

    step_logger.step_start("Load dataset and config")
    dataset, dataset_config = load_dataset(dataset_id)

    variable = get_aliased_variable(dataset, variable, dataset_config)
    if variable not in dataset:
        raise exceptions.VariableNotFound([variable])

    interp_vars = config.interpolation.vars.items
    for var in interp_vars:
        if var not in dataset:
            raise exceptions.VariableNotFound([var])

    fixed_coords, fixed_dims = dgets(
        dataset_config, [FIXED_VARIABLES_KEY, FIXED_DIMENSIONS_KEY], {}
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
    data = sel(dataset, variable, fixed_coords, fixed_dims, interp_vars)

    data_values = data.values if hasattr(data, "values") else np.asarray(data)
    if np.issubdtype(data_values.dtype, np.number):
        data = np.where(np.isnan(data_values), None, data_values).tolist()
    else:
        data = data_values.tolist()

    step_logger.end()
    return {"data": data}

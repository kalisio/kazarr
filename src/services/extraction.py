import numpy as np
import xarray as xr
from fastapi import Request
from typing import Any, Dict, List, Optional, Union
import threading

from src import exceptions
from src.schemas.config import ExtractionConfig
from src.utils.data import (
    dget,
    dgets,
    get_level_var,
    get_dataset_level_vars,
    sel,
    get_required_dims_and_coords,
    get_bounded_time,
    get_times_in_range,
)
from src.utils.file import load_dataset
from src.utils.logging import StepLoggerAndAborter
from src.utils.spatial import get_cached_ckdtree
from src.processing import bbox, interpolation, output
from src.processing.contexts import BBoxContext, TimeRange


FIXED_DIMENSIONS_KEY = "dimensions.fixed"
FIXED_VARIABLES_KEY = "variables.fixed"
LAT_VARIABLE_KEY = "variables.lat"
LON_VARIABLE_KEY = "variables.lon"
LEVEL_VARIABLE_KEY = "variables.level"


def extract(
    request: Request,
    dataset_id: str,
    variable: str,
    time_range: Optional[str] = None,
    level: Optional[float] = None,
    format: str = "raw",
    config: Union[Dict[str, Any], ExtractionConfig, None] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    if not isinstance(config, ExtractionConfig):
        config = ExtractionConfig.model_validate(config or {})

    step_logger = StepLoggerAndAborter(
        "extract",
        parameters=(dataset_id, variable, time_range, format, config),
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
    time_range = TimeRange.from_string(time_range)
    if time_range.has_time() and time_var is not None:
        if time_var not in dataset:
            missing_vars.append(f"time ({time_var})")
        else:
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
            vals = vals_da.values[
                ...,
                level_min : level_max + 1 : step_level,
                row_min : row_max + 1 : step_row,
                col_min : col_max + 1 : step_col,
            ]
        else:
            vals = vals_da.values[
                ...,
                :,
                row_min : row_max + 1 : step_row,
                col_min : col_max + 1 : step_col,
            ]
    elif is_3d_grid and is_regular_grid:
        if vals_da.ndim >= 3:
            vals = vals_da.values[
                level_min : level_max + 1 : step_level,
                row_min : row_max + 1 : step_row,
                col_min : col_max + 1 : step_col,
            ]
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

    step_logger.end()
    return out


def probe(
    request: Request,
    dataset_id: str,
    variables: Union[str, List[str]],
    lon: float,
    lat: float,
    level: Optional[float] = None,
    time_range: Optional[str] = None,
    format: str = "raw",
    config: Union[Dict[str, Any], ExtractionConfig, None] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    if not isinstance(config, ExtractionConfig):
        config = ExtractionConfig.model_validate(config or {})
    step_logger = StepLoggerAndAborter(
        "probe",
        parameters=(dataset_id, variables, lon, lat, level, time_range, format, config),
        cancel_event=cancel_event,
    )
    step_logger.step_start("Load dataset and config")
    variables = variables if isinstance(variables, list) else [variables]
    dataset, dataset_config = load_dataset(dataset_id)
    with_level = level is not None
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
    time_range = TimeRange.from_string(time_range)
    level_vars = get_dataset_level_vars(dataset, dataset_config)
    level_var = None
    if level_vars is None or isinstance(level_vars, str):
        level_var = level_vars
    else:
        for lv in level_vars:
            if all([var in level_vars[lv] for var in variables]):
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
        bounded_time_range = get_bounded_time(dataset, time_var, time_range)
        time_range_indexer = bounded_time_range.get_indexer()
        if time_range_indexer is not None:
            fixed_coords[time_var] = time_range_indexer
            if config.interpolation.vars.time:
                interp_vars.append(time_var)
            # Remove "time" from request parameters to avoid confusion in later steps (case where time is a variable in the dataset)
            request.query_params._dict.pop("time", None)

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
    if with_level and has_regular_level:
        fixed_coords[level_var] = level
        if interp_spatial_method != "nearest":
            spatial_interp_vars.append(level_var)
    if is_regular_grid:
        fixed_coords[lon_var] = lon
        fixed_coords[lat_var] = lat
        if interp_spatial_method != "nearest":
            spatial_interp_vars.extend([lon_var, lat_var])
    else:
        if with_level and not has_regular_level:
            levels = dataset[level_var].values
            points = np.column_stack(
                (longitudes.values.ravel(), latitudes.values.ravel(), levels.ravel())
            )
            target_pt = np.array([lon, lat, level])
        else:
            points = np.column_stack(
                (longitudes.values.ravel(), latitudes.values.ravel())
            )
            target_pt = np.array([lon, lat])

        step_logger.step_start("Build or retrieve spatial index for probe")
        coord_vars = (lon_var, lat_var, level_var) if with_level else (lon_var, lat_var)
        tree = get_cached_ckdtree(points, dataset_id=dataset_id, coord_vars=coord_vars)

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

    data = []
    var_props = {}
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
            var_data = interpolated_values.tolist()
        else:
            interp_methods = None
            if is_regular_grid and interp_spatial_method != "nearest":
                # Force to linear interpolation as IDW (only other method supported for probe) doesn't make sense on a regular grid
                interp_spatial_method = "linear"
                interp_vars = list(dict.fromkeys(spatial_interp_vars + interp_vars))
                interp_methods = dict.fromkeys(
                    spatial_interp_vars, interp_spatial_method
                )
            var_data = sel(
                dataset,
                var,
                fixed_coords,
                fixed_dims,
                interp_vars=interp_vars,
                interp_method=interp_vars_method,
                interp_methods=interp_methods,
                interp_config=interp_vars_params,
            ).values.tolist()
        data.append(np.atleast_1d(var_data))
        var_props[var] = dataset[var].attrs

    step_logger.step_start(
        "Get time values for probe location if time variable is defined"
    )
    times = None
    if time_var is not None:
        if time_range.has_time() and interp_spatial_method != "nearest":
            times = get_times_in_range(dataset, time_var, time_range)
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

    if format == "raw":
        step_logger.step_start("Prepare output (raw)")
        out = output.prepare_raw_output(
            variables,
            data,
            np.atleast_1d(float(lon)),
            np.atleast_1d(float(lat)),
            levels=np.atleast_1d(float(level)) if level is not None else None,
            global_props={"times": times} if times is not None else None,
            var_props=var_props,
        )
    elif format == "geojson" and times is not None:
        step_logger.step_start("Prepare output (GeoJSON)")
        out = output.prepare_geojson_output(
            variables,
            data,
            np.atleast_1d(float(lon)),
            np.atleast_1d(float(lat)),
            levels=np.atleast_1d(float(level)) if level is not None else None,
            collection_props={"times": times},
            var_props=var_props,
        )
    else:
        raise exceptions.BadConfigurationVariable(f"Unsupported format: {format}")

    step_logger.end()
    return out


def multi_probe(
    request: Request,
    dataset_id: str,
    variables: Union[str, List[str]],
    points: List[Dict[str, float]],
    time_range: Optional[str] = None,
    format: str = "raw",
    config: Union[Dict[str, Any], ExtractionConfig, None] = None,
    cancel_event: Optional[threading.Event] = None,
):
    variables = variables if isinstance(variables, list) else [variables]

    lats, lons, levels, vals = [], [], [], {}
    times, var_props = None, None
    for point in points:
        result = probe(
            request,
            dataset_id,
            variables,
            point.lon,
            point.lat,
            point.level,
            time_range,
            "raw",
            config,
            cancel_event,
        )
        lats.append(point.lat)
        lons.append(point.lon)
        levels.append(point.level)
        for var in variables:
            if var not in vals:
                vals[var] = []
            vals[var].append(result["values"][var])
        if times is None:
            times = result.get("times")
        if var_props is None:
            var_props = {var: result["variables"][var] for var in variables}
    vals = [vals[var] for var in variables]
    # As probe returns values with dims (points, times) and output formatter will expect (times, points), we need to transpose the first two dimensions here
    # Here, dim 0 is the variable dimension
    vals = np.array(vals).transpose(0, 2, 1)
    if format == "raw":
        out = output.prepare_raw_output(
            variables,
            vals,
            np.asarray(lons),
            np.asarray(lats),
            levels=np.asarray(levels) if levels and any(levels) else None,
            global_props={"times": times} if times is not None else None,
            has_time_dimension=times is not None,
        )
    elif format == "geojson":
        out = output.prepare_geojson_output(
            variables,
            vals,
            np.asarray(lons),
            np.asarray(lats),
            levels=np.asarray(levels) if levels and any(levels) else None,
            collection_props={"times": times} if times is not None else None,
            var_props=var_props,
            has_time_dimension=times is not None,
        )
    else:
        raise exceptions.BadConfigurationVariable(f"Unsupported format: {format}")

    return out


def free_selection(
    request: Request,
    dataset_id: str,
    variable: str,
    config: Union[Dict[str, Any], ExtractionConfig, None] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    if not isinstance(config, ExtractionConfig):
        config = ExtractionConfig.model_validate(config or {})
    step_logger = StepLoggerAndAborter(
        "free_selection",
        parameters=(dataset_id, variable, config),
        cancel_event=cancel_event,
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

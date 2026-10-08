from typing import Any

from fastapi import Request

from src import exceptions
from src.processing import bbox
from src.processing.contexts import BBoxContext, TimeRange
from src.processing.isoline import (
    ThresholdRange,
    format_isoline_geojson,
    format_isoline_raw,
    generate_isolines,
    resolve_thresholds,
    serialize_isolines,
)
from src.schemas.config import ExtractionConfig
from src.utils.data import (
    dget,
    dgets,
    get_bounded_time,
    get_required_dims_and_coords,
    is_time_out_of_bounds_data,
    sel,
)
from src.utils.file import load_dataset
from src.utils.logging import StepDurationLogger
from src.utils.requests import get_from_query

# Number of grid cells added around the bounding box, so that isolines reach its edges
BBOX_INDEX_PADDING = 1


def isoline(
    request: Request,
    dataset_id: str,
    variable: str,
    thresholds: list[float] | ThresholdRange,
    time: str | None = None,
    format: str = "raw",
    config: dict[str, Any] | ExtractionConfig | None = None,
) -> bytes:
    """Compute isolines and return them serialized as JSON bytes."""
    if not isinstance(config, ExtractionConfig):
        config = ExtractionConfig.model_validate(config or {})
    step_logger = StepDurationLogger(
        "isoline", parameters=(dataset_id, variable, thresholds, time, format, config)
    )

    step_logger.step_start("Load dataset and config")
    dataset, dataset_config = load_dataset(dataset_id)
    fixed_coords, fixed_dims = dgets(
        dataset_config, ["variables.fixed", "dimensions.fixed"], {}
    )
    interp_vars = []

    if variable not in dataset:
        raise exceptions.VariableNotFound([variable])

    lon_var, lat_var = dgets(dataset_config, ["variables.lon", "variables.lat"])
    missing_vars = []
    if lon_var is None or lon_var not in dataset:
        missing_vars.append(f"lon ({lon_var})")
    if lat_var is None or lat_var not in dataset:
        missing_vars.append(f"lat ({lat_var})")
    # The time is received as a string: it must be parsed into a TimeRange (as
    # in extract) before being checked and used for the selection
    time_var = dget(dataset_config, "variables.time")
    time_range = TimeRange.from_string(get_from_query(time_var, time, request))
    if time_range.has_time_range:
        raise exceptions.InvalidTimeRange(
            "Isolines require a single time value, not a time range."
        )
    if time_range.has_time():
        if time_var is None or time_var not in dataset:
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
            fixed_coords[time_var] = bounded_time_range.get_indexer()
            if config.interpolation.vars.time and time_var not in interp_vars:
                interp_vars.append(time_var)
            # Remove "time" from request parameters so that it is not used
            # again by the greedy lookup of get_required_dims_and_coords
            request.query_params._dict.pop("time", None)
    if len(missing_vars) > 0:
        raise exceptions.BadConfigurationVariable(missing_vars)

    # Handle [0, 360] datasets and bounding boxes crossing the antimeridian
    bounding_box = BBoxContext.from_tuple(config.bbox)
    dataset, _ = bbox.normalize_dataset_longitudes(
        dataset, lon_var, lat_var, bbox=bounding_box
    )

    fixed_coords, fixed_dims = get_required_dims_and_coords(
        dataset,
        variable,
        fixed_coords,
        fixed_dims,
        request,
        optional_coords=[lon_var, lat_var],
        coords_keep_dims=[lon_var, lat_var],
        as_dims=config.as_dims or [],
    )

    step_logger.step_start("Load coordinates")
    lon_da = sel(dataset, lon_var, fixed_coords, fixed_dims)
    lat_da = sel(dataset, lat_var, fixed_coords, fixed_dims)
    if lon_da.ndim == 0 or lat_da.ndim == 0 or (
        lon_da.ndim == 1 and lat_da.ndim == 1 and lon_da.dims == lat_da.dims
    ):
        raise exceptions.BadSelection(
            "Isolines can only be computed on gridded datasets (point lists are not supported)."
        )
    is_regular_grid = lon_da.ndim == 1 and lat_da.ndim == 1
    if not is_regular_grid:
        # Remove extra leading dimensions of size 1 (e.g. a single vertical level)
        lon_da = lon_da.squeeze([d for d in lon_da.dims[:-2] if lon_da.sizes[d] == 1])
        lat_da = lat_da.squeeze([d for d in lat_da.dims[:-2] if lat_da.sizes[d] == 1])
        if lon_da.ndim != 2 or lon_da.dims != lat_da.dims:
            raise exceptions.TooManyDimensions(lon_da.ndim)
    lon = lon_da.values
    lat = lat_da.values

    step_logger.step_start("Apply bounding box")
    if is_regular_grid:
        row_dim, col_dim = lat_da.dims[0], lon_da.dims[0]
        n_rows, n_cols = len(lat), len(lon)
    else:
        row_dim, col_dim = lon_da.dims
        n_rows, n_cols = lon.shape
    row_min, row_max, col_min, col_max = 0, n_rows - 1, 0, n_cols - 1
    if bounding_box.has_bb:
        if is_regular_grid:
            indices = bbox.apply_regular_grid_bounding_box(
                lon, lat, bounding_box, BBOX_INDEX_PADDING
            )
        else:
            _, _, indices = bbox.apply_irregular_bounding_box(
                lon, lat, bounding_box, False, 0.0, 0
            )
            indices.row_min = max(0, indices.row_min - BBOX_INDEX_PADDING)
            indices.row_max = min(n_rows - 1, indices.row_max + BBOX_INDEX_PADDING)
            indices.col_min = max(0, indices.col_min - BBOX_INDEX_PADDING)
            indices.col_max = min(n_cols - 1, indices.col_max + BBOX_INDEX_PADDING)
        row_min, row_max = indices.row_min, indices.row_max
        col_min, col_max = indices.col_min, indices.col_max
    rows = slice(row_min, row_max + 1)
    cols = slice(col_min, col_max + 1)
    if is_regular_grid:
        lon, lat = lon[cols], lat[rows]
    else:
        lon, lat = lon[rows, cols], lat[rows, cols]

    step_logger.step_start("Load variable values")
    val_da = sel(
        dataset,
        variable,
        fixed_coords,
        fixed_dims,
        interp_vars=interp_vars,
        interp_method=config.interpolation.vars.method,
        interp_config=config.interpolation.vars.params or {},
    )
    if row_dim not in val_da.dims or col_dim not in val_da.dims:
        raise exceptions.BadSelection(
            f"Variable '{variable}' is not defined on the ({row_dim}, {col_dim}) grid."
        )
    # Only the part of the field covering the bounding box is read
    val = (
        val_da.isel({row_dim: rows, col_dim: cols})
        .squeeze([d for d in val_da.dims if d not in (row_dim, col_dim) and val_da.sizes[d] == 1])
        .transpose(row_dim, col_dim)
        .values
    )
    if val.shape[0] < 2 or val.shape[1] < 2:
        raise exceptions.NoDataInSelection(
            "At least 2x2 grid points are required to compute isolines."
        )

    # Thresholds given as a range are computed from the selected data (bbox included)
    thresholds = resolve_thresholds(thresholds, val)

    step_logger.step_start("Extract isolines")
    isolines = generate_isolines(lon, lat, val, thresholds)

    step_logger.step_start("Prepare output")
    if format == "raw":
        out = format_isoline_raw(isolines, thresholds)
    elif format == "geojson":
        out = format_isoline_geojson(isolines, thresholds)
    else:
        raise exceptions.BadConfigurationVariable(f"Unsupported format: {format}")
    # Serialized here (in the worker thread) rather than by FastAPI in the event loop
    content = serialize_isolines(out)

    step_logger.end()
    return content

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
from src.processing.isoline import (
    generate_isolines,
    format_isoline_raw,
    format_isoline_geojson,
)


def isoline(
    request: Request,
    dataset_id: str,
    variable: str,
    levels: List[float],
    time: Optional[str] = None,
    format: str = "raw",
    config: Union[Dict[str, Any], ExtractionConfig, None] = None,
) -> Dict[str, Any]:
    if not isinstance(config, ExtractionConfig):
        config = ExtractionConfig.model_validate(config or {})
    step_logger = StepDurationLogger(
        "isoline", parameters=(dataset_id, variable, levels, time, format, config)
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
    if time is not None:
        time_var = dget(dataset_config, "variables.time")
        if time_var is None or time_var not in dataset:
            missing_vars.append(f"time ({time_var})")
        else:
            fixed_coords[time_var] = get_bounded_time(dataset, time_var, time)
            if config.interpolation.vars.time:
                interp_vars.append(time_var)
    if len(missing_vars) > 0:
        raise exceptions.BadConfigurationVariable(missing_vars)

    fixed_coords, fixed_dims = get_required_dims_and_coords(
        dataset,
        variable,
        fixed_coords,
        fixed_dims,
        request,
        optional_coords=[lon_var, lat_var],
        as_dims=config.as_dims or [],
    )

    lon = sel(dataset, lon_var, fixed_coords, fixed_dims)
    lat = sel(dataset, lat_var, fixed_coords, fixed_dims)
    val = sel(dataset, variable, fixed_coords, fixed_dims, interp_vars=interp_vars)

    step_logger.step_start("Extract isolines")
    isolines = generate_isolines(lon, lat, val, levels)

    step_logger.step_start("Prepare output")
    if format == "raw":
        out = format_isoline_raw(isolines, levels)
    elif format == "geojson":
        out = format_isoline_geojson(isolines, levels)
    else:
        raise exceptions.BadConfigurationVariable(f"Unsupported format: {format}")

    step_logger.end()
    return out

import numpy as np
from typing import Any, Dict, Optional

from src.utils.data import dgets, dget
from src.utils.file import load_datasets, load_dataset


def list_datasets(search_path: Optional[str] = None) -> Dict[str, Any]:
    datasets = load_datasets(search_path)
    return {"datasets": datasets}


def dataset_infos(dataset_id: str) -> Dict[str, Any]:
    dataset, config = load_dataset(dataset_id)
    variables = {}
    coords = {}
    for var in dataset.data_vars:
        variables[var] = {
            "dimensions": dataset[var].dims,
            "shape": dataset[var].shape,
            "attributes": dataset[var].attrs,
        }
        if len(dataset[var].dims) == 0:
            variables[var]["value"] = dataset[var].values.item()
    for coord in dataset.coords:
        coords[coord] = {
            "dimensions": dataset[coord].dims,
            "shape": dataset[coord].shape,
            "attributes": dataset[coord].attrs,
        }
        if len(dataset[coord].dims) == 0:
            coords[coord]["value"] = dataset[coord].values.item()

    lon_var, lat_var, height_var = dgets(
        config, ["variables.lon", "variables.lat", "variables.height"]
    )
    bounding_box = False
    if lon_var in dataset and lat_var in dataset:
        bounding_box = {
            "lon": {
                "min": float(dataset[lon_var].min()),
                "max": float(dataset[lon_var].max()),
            },
            "lat": {
                "min": float(dataset[lat_var].min()),
                "max": float(dataset[lat_var].max()),
            },
        }
        if height_var in dataset:
            bounding_box["height"] = {
                "min": float(dataset[height_var].min()),
                "max": float(dataset[height_var].max()),
            }

    time_var = dget(config, "variables.time")
    time_bounds = False
    if (
        time_var in dataset
        and np.issubdtype(dataset[time_var].dtype, np.datetime64)
        and dataset[time_var].ndim == 1
    ):
        t_dim = dataset[time_var].dims[0]
        t_min = dataset[time_var].isel({t_dim: 0}).values
        t_max = dataset[time_var].isel({t_dim: -1}).values
        time_bounds = {
            "min": str(np.datetime_as_string(t_min)),
            "max": str(np.datetime_as_string(t_max)),
        }

    return {
        "id": dataset_id,
        "description": config.get("description", ""),
        "variables": variables,
        "coordinates": coords,
        "bounding_box": bounding_box if bounding_box else None,
        "time_bounds": time_bounds if time_bounds else None,
        "attributes": dataset.attrs,
    }

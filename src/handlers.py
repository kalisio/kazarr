from fastapi import Request, HTTPException
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from src.utils import dget, dgets, sel, load_datasets, load_dataset, get_required_dims_and_coords, is_monotonic_var

def list_datasets():
  datasets = load_datasets()
  out = []
  for dataset in datasets.keys():
    out.append({ "id": dataset, "description": datasets[dataset].get("description", "") })
  return {"datasets": out}

def dataset_infos(dataset_id):
  dataset, config = load_dataset(dataset_id)
  variables = {}
  for var in dataset.data_vars:
    variables[var] = {
      "dims": dataset[var].dims,
      "shape": dataset[var].shape,
      "attrs": dataset[var].attrs
    }
    if len(dataset[var].dims) == 0:
      variables[var]["value"] = dataset[var].values.item()

  return {
    "id": dataset_id,
    "description": config.get("description", ""),
    "variables": variables,
    "attrs": dataset.attrs
  }

def extract(dataset, variable, request, time = None, bounding_box = None):
  lon_min, lat_min, lon_max, lat_max = (None, None, None, None) if bounding_box is None else bounding_box
  has_bb_lon = lon_min is not None or lon_max is not None
  has_bb_lat = lat_min is not None or lat_max is not None

  dataset, config = load_dataset(dataset)

  fixed_coords, fixed_dims = dgets(config, ['variables.fixed', 'dimensions.fixed'], {})

  lon_dim, lat_dim = dgets(config, ['dimensions.lon', 'dimensions.lat'])
  lon_var, lat_var = dgets(config, ['variables.lon', 'variables.lat'])
  missing_vars = []
  if has_bb_lon and lon_var is None:
    raise exceptions.MissingConfigurationElement("variables.lon")
    if lon_var not in dataset:
      missing_vars.append(f"lon ({lon_var})")
  if has_bb_lat and lat_var is None:
    raise exceptions.MissingConfigurationElement("variables.lat")
    if lat_var not in dataset:
      missing_vars.append(f"lat ({lat_var})")
  if len(missing_vars) > 0:
    raise exceptions.BadConfigurationVariable(missing_vars)

  time_var = dget(config, 'variables.time')
  if time is not None and time_var is not None:
    fixed_coords[time_var] = time

  fixed_coords, fixed_dims = get_required_dims_and_coords(dataset, config, variable, fixed_coords, fixed_dims, request, [lon_dim, lat_dim])

  data = None
  if has_bb_lon or has_bb_lat:
    # Apply bounding box selection if lon/lat bounds are provided
    lons = dataset[lon_var]
    lats = dataset[lat_var]
    mask = xr.ones_like(lons, dtype=bool)
    if has_bb_lon:
      mask = mask & (lons >= lon_min) & (lons <= lon_max)
    if has_bb_lat:
      mask = mask & (lats >= lat_min) & (lats <= lat_max)

    if has_bb_lon or has_bb_lat:
      mask = mask.compute()
      data = dataset.where(mask)

  data = sel(dataset if data is None else data, variable, fixed_coords, fixed_dims)

  lons_sel = sel(dataset, lon_var, fixed_coords, fixed_dims)
  lats_sel = sel(dataset, lat_var, fixed_coords, fixed_dims)
  out = []
  for lon_val, lat_val, data_val in zip(lons_sel.values.flatten(), lats_sel.values.flatten(), data.values.flatten()):
    if np.isnan(data_val):
      continue
    out.append({
      "lon": lon_val.item(),
      "lat": lat_val.item(),
      "value": data_val.item()
    })
  return out

def probe(dataset, variables, lon, lat, request, height = None):
  variables = variables if isinstance(variables, list) else [variables]

  dataset, config = load_dataset(dataset)
  with_height = height is not None

  fixed_coords, fixed_dims = dgets(config, ['variables.fixed', 'dimensions.fixed'], {})

  lon_var, lat_var, height_var, time_var = dgets(config, ['variables.lon', 'variables.lat', 'variables.height', 'variables.time'])
  lon_dim, lat_dim, height_dim, time_dim = dgets(config, ['dimensions.lon', 'dimensions.lat', 'dimensions.height', 'dimensions.time'])

  missing_vars = []
  if lon_var is None or lon_var not in dataset:
    missing_vars.append(f"lon ({lon_var})")
  if lat_var is None or lat_var not in dataset:
    missing_vars.append(f"lat ({lat_var})")
  if with_height and height_var is None or height_var not in dataset:
    missing_vars.append(f"height ({height_var})")
  if len(missing_vars) > 0:
    raise exceptions.BadConfigurationVariable(missing_vars)

  longitudes = dataset[lon_var].values
  latitudes = dataset[lat_var].values
  if with_height:
    heights = dataset[height_var].values
    dist = np.sqrt((longitudes - lon)**2 + (latitudes - lat)**2 + (heights - height)**2)
  else:
    dist = np.sqrt((longitudes - lon)**2 + (latitudes - lat)**2)
  k, j, i = np.unravel_index(np.argmin(dist), dist.shape)
  fixed_dims[lon_dim] = i
  fixed_dims[lat_dim] = j
  if with_height:
    fixed_dims[height_dim] = k

  optional_dims = [time_dim] if time_dim is not None else []
  fixed_coords, fixed_dims = get_required_dims_and_coords(dataset, config, variables, fixed_coords, fixed_dims, request, optional_dims)

  # Get values for each variable
  data = {}
  for var in variables:
    data[var] = {
      "values": sel(dataset, var, fixed_coords, fixed_dims).values.tolist(),
      "attrs": dataset[var].attrs
    }
  
  # Get times list
  if time_var is not None and time_var in dataset:
    times = sel(dataset, time_var, fixed_coords, fixed_dims).values
    if np.issubdtype(times.dtype, np.datetime64):
      times = [str(np.datetime_as_string(t)) for t in times]
    
  out = { "variables": data }
  if times is not None:
    out["times"] = times
  return out

def isoline(dataset, variable, levels, request, time = None):
  dataset, config = load_dataset(dataset)

  fixed_coords, fixed_dims = dgets(config, ['variables.fixed', 'dimensions.fixed'], {})

  lon_var, lat_var = dgets(config, ['variables.lon', 'variables.lat'])
  missing_vars = []
  if lon_var is None or lon_var not in dataset:
    missing_vars.append(f"lon ({lon_var})")
  if lat_var is None or lat_var not in dataset:
    missing_vars.append(f"lat ({lat_var})")
  if time is not None:
    time_var = dget(config, 'variables.time')
    if time_var is None or time_var not in dataset:
      missing_vars.append(f"time ({time_var})")
    else:
      fixed_coords[time_var] = time
  if len(missing_vars) > 0:
    raise exceptions.BadConfigurationVariable(missing_vars)

  lon = sel(dataset, lon_var, fixed_coords, fixed_dims)
  lat = sel(dataset, lat_var, fixed_coords, fixed_dims)
  val = sel(dataset, variable, fixed_coords, fixed_dims)

  contours = plt.contour(lon, lat, val, levels=levels)
  isolines = []
  for paths in contours.get_paths():
    isolines.append(paths.vertices.tolist())

  out = {}
  for i, level in enumerate(levels):
    out[level] = isolines[i]

  return out

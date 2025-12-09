import math, re

from fastapi import Request, HTTPException
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pyvista as pv

from src.utils import dget, dgets, sel, load_datasets, load_dataset, save_datasets, get_required_dims_and_coords, is_monotonic_var
from src import exceptions

def list_datasets():
  datasets = load_datasets()
  out = []
  for dataset in datasets.keys():
    out.append({ "id": dataset, "description": datasets[dataset].get("description", "") })
  return {"datasets": out}

def dataset_infos(dataset_id):
  dataset, config = load_dataset(dataset_id)
  variables = {}
  coords = {}
  for var in dataset.data_vars:
    variables[var] = {
      "dims": dataset[var].dims,
      "shape": dataset[var].shape,
      "attrs": dataset[var].attrs
    }
    if len(dataset[var].dims) == 0:
      variables[var]["value"] = dataset[var].values.item()
  for coord in dataset.coords:
    coords[coord] = {
      "dims": dataset[coord].dims,
      "shape": dataset[coord].shape,
      "attrs": dataset[coord].attrs
    }
    if len(dataset[coord].dims) == 0:
      variables[coord]["value"] = dataset[coord].values.item()

  # Check if bounding box can be defined
  lon_var, lat_var, height_var = dgets(config, ['variables.lon', 'variables.lat', 'variables.height'])
  bounding_box = False
  if lon_var in dataset and lat_var in dataset:
    bounding_box = {
      "lon": {
        "min": float(dataset[lon_var].min()),
        "max": float(dataset[lon_var].max())
      },
      "lat": {
        "min": float(dataset[lat_var].min()),
        "max": float(dataset[lat_var].max())
      }
    }
    if height_var in dataset:
      bounding_box["height"] = {
        "min": float(dataset[height_var].min()),
        "max": float(dataset[height_var].max())
      }
    
  # Check if time bounds can be defined
  time_var = dget(config, 'variables.time')
  time_bounds = False
  if time_var in dataset:
    time_data = dataset[time_var].values
    if np.issubdtype(time_data.dtype, np.datetime64):
      time_bounds = { "min": str(np.datetime_as_string(np.min(time_data))), "max": str(np.datetime_as_string(np.max(time_data))) }

  return {
    "id": dataset_id,
    "description": config.get("description", ""),
    "variables": variables,
    "coords": coords,
    "bounding_box": bounding_box if bounding_box else None,
    "time_bounds": time_bounds if time_bounds else None,
    "attrs": dataset.attrs
  }

def extract(dataset, variable, request, time = None, bounding_box = None, resolution_limit = None, as_mesh = False, as_dims = []):
  lon_min, lat_min, lon_max, lat_max = (None, None, None, None) if bounding_box is None else bounding_box
  has_bb_lon = lon_min is not None or lon_max is not None
  has_bb_lat = lat_min is not None or lat_max is not None
  has_bb = has_bb_lon or has_bb_lat

  dataset, config = load_dataset(dataset)

  fixed_coords, fixed_dims = dgets(config, ['variables.fixed', 'dimensions.fixed'], {})

  if variable not in dataset:
    raise exceptions.VariableNotFound([variable])

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

  fixed_coords, fixed_dims = get_required_dims_and_coords(dataset, config, variable, fixed_coords, fixed_dims, request, optional_coords=[lon_var, lat_var], as_dims=as_dims)

  vals = sel(dataset, variable, fixed_coords, fixed_dims).values
  lons = sel(dataset, lon_var, fixed_coords, fixed_dims).values
  lats = sel(dataset, lat_var, fixed_coords, fixed_dims).values

  if lons.ndim == 1 and lats.ndim == 1:
    lons, lats = np.meshgrid(lons, lats)

  if has_bb:
    mask = np.ones(lons.shape, dtype=bool) 
    if has_bb_lon:
      valid_lon_min = lon_min if lon_min is not None else -np.inf
      valid_lon_max = lon_max if lon_max is not None else np.inf
      mask &= (lons >= valid_lon_min) & (lons <= valid_lon_max)
    if has_bb_lat:
      valid_lat_min = lat_min if lat_min is not None else -np.inf
      valid_lat_max = lat_max if lat_max is not None else np.inf
      mask &= (lats >= valid_lat_min) & (lats <= valid_lat_max)

    if not np.any(mask):
      # No data in bounding box
      raise exceptions.NoDataInSelection()

    # Crop to bounding box
    # Don't use slice as it will not work with non regular grids
    rows, cols = np.where(mask)
    row_min, row_max = rows.min(), rows.max()
    col_min, col_max = cols.min(), cols.max()

    height_raw = row_max - row_min + 1
    width_raw = col_max - col_min + 1
  else:
    row_min, row_max = 0, lons.shape[0] - 1
    col_min, col_max = 0, lons.shape[1] - 1
    height_raw, width_raw = lons.shape

  if height_raw < 2 or width_raw < 2:
    # Not enough data to extract
    raise exceptions.TooFewPoints()

  step_row, step_col = 1, 1
  if resolution_limit is not None:
    if height_raw > resolution_limit:
      step_row = math.ceil(height_raw / resolution_limit)
    if width_raw > resolution_limit:
      step_col = math.ceil(width_raw / resolution_limit)

  vals = vals[row_min:row_max+1:step_row, col_min:col_max+1:step_col]
  lons = lons[row_min:row_max+1:step_row, col_min:col_max+1:step_col]
  lats = lats[row_min:row_max+1:step_row, col_min:col_max+1:step_col]

  if has_bb:
    mask_cropped = mask[row_min:row_max+1:step_row, col_min:col_max+1:step_col]
  else:
    mask_cropped = None

  # Convert to float so we can assign NaN
  # Exclude values outside bounding box
  vals = vals.astype(float)
  if mask_cropped is not None:
    vals[~mask_cropped] = np.nan

  if as_mesh:
    # TODO Handle 3D
    z_zeros = np.zeros_like(lons)
    grid = pv.StructuredGrid(lons, lats, z_zeros)
    grid.point_data[variable] = vals.ravel()
    if mask_cropped is not None:
      # Create a valid mask for thresholding (bool to float)
      grid.point_data["valid_mask"] = mask_cropped.ravel().astype(float)
      # Set NaN values to 0 in valid_mask
      grid.point_data["valid_mask"][np.isnan(vals.ravel())] = 0
      try:
        thresholded = grid.threshold(0.5, scalars="valid_mask")
      except Exception as e:
        raise exceptions.GenericInternalError(str(e))
    else:
      thresholded = grid

    if thresholded.n_points == 0:
      raise exceptions.NoDataInSelection()

    tri_grid = thresholded.triangulate()
    vertices = tri_grid.points.flatten()
    # As cells are stored as (N, 4) with first value being number of points per cell (3 for triangle)
    # we need to reshape and skip first values
    indices = tri_grid.cells.reshape((-1, 4))[:, 1:].flatten()
    values = tri_grid.point_data[variable]
    clean_values = [v if np.isfinite(v) else None for v in values]

    # As bounding can
    valid_numbers = values[np.isfinite(values)]
    if valid_numbers.size == 0:
      val_min, val_max = None, None
    else:
      val_min, val_max = float(valid_numbers.min()), float(valid_numbers.max())

    return {
      "bounds": { "min": val_min, "max": val_max },
      "resolution_factor": { "row": step_row, "col": step_col },
      "vertices": vertices.tolist(),
      "indices": indices.tolist(),
      "values": clean_values
    }
  else:
    flat_vals = vals.flatten()
    flat_lons = lons.flatten()
    flat_lats = lats.flatten()

    valid_vals = flat_vals[~np.isnan(flat_vals)]
    if valid_vals.size == 0:
      raise exceptions.NoDataInSelection()

    return {
      "shape": vals.shape,
      "bounds": {
        "min": np.min(valid_vals).item(),
        "max": np.max(valid_vals).item()
      },
      "resolution_factor": { "row": step_row, "col": step_col },
      "data": {
        "longitudes": flat_lons.tolist(),
        "latitudes": flat_lats.tolist(),
        "values": [None if np.isnan(v) else v.item() for v in flat_vals]
      }
    }

def probe(dataset, variables, lon, lat, request, height = None, as_dims = []):
  variables = variables if isinstance(variables, list) else [variables]

  dataset, config = load_dataset(dataset)
  with_height = height is not None

  fixed_coords, fixed_dims = dgets(config, ['variables.fixed', 'dimensions.fixed'], {})

  lon_var, lat_var, height_var, time_var = dgets(config, ['variables.lon', 'variables.lat', 'variables.height', 'variables.time'])
  time_dim = dget(config, 'dimensions.time')

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
  if with_height and height_var is None or height_var not in dataset:
    missing_vars.append(f"height ({height_var})")
  if len(missing_vars) > 0:
    raise exceptions.BadConfigurationVariable(missing_vars)

  longitudes = dataset[lon_var]
  latitudes = dataset[lat_var]
  if longitudes.ndim == 1 and latitudes.ndim == 1:
    fixed_coords[lon_var] = lon
    fixed_coords[lat_var] = lat
  else:
    if with_height:
      heights = dataset[height_var].values
      dist = np.sqrt((longitudes - lon)**2 + (latitudes - lat)**2 + (heights - height)**2)
    else:
      dist = np.sqrt((longitudes - lon)**2 + (latitudes - lat)**2)
    dist_values = dist.values
    min_idx_flat = np.argmin(dist_values)
    indices = np.unravel_index(min_idx_flat, dist_values.shape)
    for dim_name, indice in zip(dist.dims, indices):
      fixed_dims[dim_name] = indice

  # Time is optional, so add to both optional coords and dims, as one or the other may be defined in config
  optional_coords = [time_var] if time_var is not None else []
  optional_dims = [time_dim] if time_dim is not None else []
  fixed_coords, fixed_dims = get_required_dims_and_coords(dataset, config, variables, fixed_coords, fixed_dims, request, optional_coords=optional_coords, optional_dims=optional_dims, as_dims=as_dims)

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

def isoline(dataset, variable, levels, request, time = None, as_dims = []):
  dataset, config = load_dataset(dataset)

  fixed_coords, fixed_dims = dgets(config, ['variables.fixed', 'dimensions.fixed'], {})

  if variable not in dataset:
    raise exceptions.VariableNotFound([variable])

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

  fixed_coords, fixed_dims = get_required_dims_and_coords(dataset, config, variable, fixed_coords, fixed_dims, request, as_dims=as_dims)

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

def free_selection(dataset, variable, request, as_dims = []):
  dataset, config = load_dataset(dataset)

  if variable not in dataset:
    raise exceptions.VariableNotFound([variable])

  # TODO Even remove fixed vars/dims ?
  fixed_coords, fixed_dims = dgets(config, ['variables.fixed', 'dimensions.fixed'], {})
  fixed_coords, fixed_dims = get_required_dims_and_coords(dataset, config, variable, fixed_coords, fixed_dims, request, optional_dims="*", as_dims=as_dims)

  data = sel(dataset, variable, fixed_coords, fixed_dims).values.tolist()
  return { "data": data }

def register_dataset(name, path, description="", config = {}):
  datasets = load_datasets()
  # Ensure unique name
  while name in datasets:
    match = re.search(r'-(\d+)$', name)
    if match:
      index = int(match.group(1))
      base_name = name[:match.start()]
      name = f"{base_name}-{index + 1}"
    else:
      name = f"{name}-1"
    
  # Merge with existing datasets configurations
  datasets[name] = {
    "path": path,
    "description": description,
    **config
  }
  save_datasets(datasets)

  return {
    "id": name
  }

import os, json, time
from functools import lru_cache

import s3fs
import xarray as xr
import numpy as np
from botocore.exceptions import NoCredentialsError

import src.exceptions as exceptions

# Open Zarr dataset as XArray dataset from S3
@lru_cache(maxsize=5)
def load(path):
  bucket = os.getenv("BUCKET_NAME")
  if bucket is None:
    raise exceptions.GenericInternalError("BUCKET_NAME environment variable not set.")
  try:
    s3_store = s3fs.S3Map(root=os.path.join(bucket, path.replace('s3://', '')), s3=s3fs.S3FileSystem(anon=False))
    # Will try to open consolidated metadata first (https://docs.xarray.dev/en/latest/generated/xarray.open_zarr.html#xarray.open_zarr)
    # Will try to determine zarr_format (v2 or v3) automatically
    # chunks must be defined to enable dask lazy loading
    dataset = xr.open_zarr(s3_store, chunks="auto")
  except NoCredentialsError as e:
    raise exceptions.GenericInternalError("S3 credentials not found.")
  except Exception as e:
    raise exceptions.GenericInternalError("Unable to access S3: " + str(e))
  return dataset

# Load datasets config file from S3
@lru_cache(maxsize=1)
def load_datasets(path = "datasets.json"):
  s3_store = s3fs.S3FileSystem(anon=False)
  bucket = os.getenv("BUCKET_NAME")
  if bucket is None:
    raise exceptions.GenericInternalError("BUCKET_NAME environment variable not set.")
  try:
    with s3_store.open(os.path.join(bucket, path), 'r') as f:
      datasets = json.load(f)
  except NoCredentialsError as e:
    raise exceptions.GenericInternalError("S3 credentials not found.")
  except Exception as e:
    raise exceptions.GenericInternalError("Unable to access S3: " + str(e))
  return datasets

def save_datasets(datasets, path = "datasets.json"):
  s3_store = s3fs.S3FileSystem(anon=False)
  bucket = os.getenv("BUCKET_NAME")
  if bucket is None:
    raise exceptions.GenericInternalError("BUCKET_NAME environment variable not set.")
  try:
    with s3_store.open(os.path.join(bucket, path), 'w') as f:
      json.dump(datasets, f, indent=2)
  except NoCredentialsError as e:
    raise exceptions.GenericInternalError("S3 credentials not found.")
  except Exception as e:
    raise exceptions.GenericInternalError("Unable to access S3: " + str(e))

# Load a dataset and its configuration from its ID
def load_dataset(dataset_id):
  datasets = load_datasets()
  if dataset_id not in datasets:
    raise exceptions.DatasetNotFound(dataset_id)
  config = datasets[dataset_id]
  dataset_path = config.get("path")
  if dataset_path is None:
    raise exceptions.MissingConfigurationElement("path")
  dataset = load(dataset_path)
  return dataset, config

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
  except Exception as e:
    is_monotonic = False
  return is_monotonic

# Get dimensions and coordinates that must be provided for a selection, and not already defined
def get_required_dims_and_coords(dataset, config, variables, fixed_coords, fixed_dims, request, optional_coords=[], optional_dims=[], as_dims=[]):
  # Find optional dimensions from optional coordinates
  for coord in optional_coords if isinstance(optional_coords, list) else [optional_coords]:
    if coord in dataset:
      for dim in dataset[coord].dims:
        if dim not in optional_dims:
          optional_dims.append(dim) 

  needed_dims = {}
  for variable in variables if isinstance(variables, list) else [variables]:
    for dim in dataset[variable].dims:
      if dim not in fixed_dims and dim not in optional_dims:
        needed = True
        assigned_coords = []
        # Find coordinates related to this dimension
        for coord in dataset.coords:
          if dim in dataset[coord].dims and coord not in as_dims and coord not in assigned_coords:
            assigned_coords.append(coord)
            if coord in fixed_coords or coord in optional_coords:
              needed = False
        if needed:
          needed_dims[dim] = assigned_coords

  # Check if all needed dimensions/coordinates are provided in query params
  missing_dims = {}
  for dim, coords in needed_dims.items():
    dim_value = request.query_params.get(dim)
    if dim_value is not None:
      fixed_dims[dim] = int(dim_value)
    else:
      coord_found = False
      for coord in coords:
        coord_value = request.query_params.get(coord)
        if coord_value is not None:
          fixed_coords[coord] = coord_value
          coord_found = True
          break
      if not coord_found:
        missing_dims[dim] = coords

  if len(missing_dims) > 0:
    raise exceptions.MissingDimensionsOrCoordinates(missing_dims)
  return fixed_coords, fixed_dims

# Smart selection on a dataset variable with coordinates and dimensions
def sel(dataset, variable, fixed_coords, fixed_dims):
  # Ensure xindexes are set for all fixed coords
  for coord in fixed_coords:
    dataset = set_xindex(dataset, coord)

  # Convert coords values to target dtype
  for var, val in fixed_coords.items():
    fixed_coords[var] = np.array(val, dtype=dataset[var].dtype)

  # Only monotonic variables can be used with sel(method='nearest')
  monotonic_fixed_vars = {var: val for var, val in fixed_coords.items() if is_monotonic_var(dataset, var)}
  non_monotonic_fixed_vars = {var: val for var, val in fixed_coords.items() if not is_monotonic_var(dataset, var)}

  data = dataset.sel(monotonic_fixed_vars, method='nearest').sel(non_monotonic_fixed_vars).isel(fixed_dims)[variable]
  return data  

# Deep get from nested dict (mimic lodash get)
def dget(d, key, default=None):
  keys = key.split('.')
  for k in keys:
    if isinstance(d, dict) and k in d:
      d = d[k]
    else:
      return default.copy() if isinstance(default, dict) else default
  return d

# Deep get multiple keys from nested dict
def dgets(d, keys, default=None):
  values = []
  for key in keys if isinstance(keys, list) else [keys]:
    values.append(dget(d, key, default=default))
  return tuple(values)

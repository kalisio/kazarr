import os, json, copy, logging, time
from functools import lru_cache
import shutil
from pathlib import Path

import s3fs
import xarray as xr
import numpy as np
import fsspec
from botocore.exceptions import NoCredentialsError
from loguru import logger as log

import src.exceptions as exceptions

log.add("logs/app_fastapi.log", rotation="10 MB", compression="zip", level="INFO")

class KazarrLoggerHandler(logging.Handler):
  def __init__(self):
    super().__init__()

  def emit(self, record):
    formatted_record = self.format(record)
    if formatted_record.startswith("CALL: get_object"):
      try:
        data = json.loads(formatted_record.split(" - ")[-1].replace("'", '"'))
        print(f"[Kazarr - S3FS] Try downloading {data.get('Bucket')}/{data.get('Key')}")
      except Exception as e:
        pass

def enable_s3fs_debug_logging():
  handler = KazarrLoggerHandler()
  handler.setLevel(logging.DEBUG)
  logger = logging.getLogger("s3fs")
  logger.setLevel(logging.DEBUG)
  logger.addHandler(handler)

def get_datasets_path():
  return os.getenv("DATASETS_PATH", "/")

# Open Zarr dataset as XArray dataset from S3
@lru_cache(maxsize=5)
def load(path):
  bucket = os.getenv("BUCKET_NAME")
  if bucket is None:
    raise exceptions.GenericInternalError("BUCKET_NAME environment variable not set.")
  try:
    cache_size = os.getenv("CACHE_SIZE", "512MB")
    cache_path = os.getenv("CACHE_DIR")
    use_cache = cache_path is not None and get_cache_size_bytes(cache_size) is not None
    if not use_cache:
      store = s3fs.S3Map(root=os.path.join(bucket, get_datasets_path(), path.replace('s3://', '')), s3=s3fs.S3FileSystem(anon=False))
    else:
      fs = fsspec.filesystem(
        "simplecache", 
        target_protocol='s3',
        cache_storage=cache_path,
        target_options={'anon': False},
        expiry_time=60 * 60 * 24 * 7 * 4 # 4 weeks
      )
      store = fs.get_mapper(os.path.join(bucket, get_datasets_path(), path.replace('s3://', '')))

    # Will try to open consolidated metadata first (https://docs.xarray.dev/en/latest/generated/xarray.open_zarr.html#xarray.open_zarr)
    # Will try to determine zarr_format (v2 or v3) automatically
    # chunks must be defined to enable dask lazy loading
    dataset = xr.open_zarr(store, chunks="auto")
  except NoCredentialsError as e:
    raise exceptions.GenericInternalError("S3 credentials not found.")
  except Exception as e:
    raise exceptions.GenericInternalError("Unable to access S3: " + str(e))
  return dataset

# Load JSON file from S3
def load_json(path):
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

def find_datasets(path = "/"):
  # Search all folders recursively that ends with .zarr
  s3_store = s3fs.S3FileSystem(anon=False)
  bucket = os.getenv("BUCKET_NAME")
  if bucket is None:
    raise exceptions.GenericInternalError("BUCKET_NAME environment variable not set.")
  datasets = []
  try:
    for root, dirs, _ in s3_store.walk(os.path.join(bucket, get_datasets_path().lstrip('/'), path.lstrip('/'))):
      for dirname in dirs[:]:
        print("CHECKING", dirname)
        if dirname.endswith('.zarr'):
          found_path = f"{root}/{dirname}"
          datasets.append(found_path.replace(os.path.join(bucket, get_datasets_path().lstrip('/')), '').replace('.zarr', ''))
          dirs.remove(dirname)
  except NoCredentialsError as e:
    raise exceptions.GenericInternalError("S3 credentials not found.")
  except Exception as e:
    raise exceptions.GenericInternalError("Unable to access S3: " + str(e))
  return datasets

def load_datasets(search_path = None):
  search_path = search_path if search_path is not None else ""
  return find_datasets(search_path)

def save_datasets(datasets):
  s3_store = s3fs.S3FileSystem(anon=False)
  bucket = os.getenv("BUCKET_NAME")
  if bucket is None:
    raise exceptions.GenericInternalError("BUCKET_NAME environment variable not set.")
  try:
    with s3_store.open(os.path.join(bucket, get_datasets_path()), 'w') as f:
      json.dump(datasets, f, indent=2)
  except NoCredentialsError as e:
    raise exceptions.GenericInternalError("S3 credentials not found.")
  except Exception as e:
    raise exceptions.GenericInternalError("Unable to access S3: " + str(e))

# Load a dataset and its configuration from its ID
def load_dataset(dataset_path):
  dataset = load(dataset_path + ".zarr")
  cache_size = os.getenv("CACHE_SIZE", "512MB")
  cache_path = os.getenv("CACHE_DIR")
  use_cache = cache_path is not None and get_cache_size_bytes(cache_size) is not None
  if use_cache:
    # Enforce cache size limit after loading
    enforce_cache_limit(cache_path, max_size=cache_size)
  config = copy.deepcopy(dataset.attrs.get('kazarr', {}))
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
def get_required_dims_and_coords(dataset, variables, fixed_coords, fixed_dims, interp_vars, request, optional_coords=[], optional_dims=[], as_dims=[]):
  needed_dims = {}

  if optional_coords != "*" and optional_dims != "*":
    # Find optional dimensions from optional coordinates
    for coord in optional_coords if isinstance(optional_coords, list) else [optional_coords]:
      if coord in dataset:
        for dim in dataset[coord].dims:
          if dim not in optional_dims:
            optional_dims.append(dim)

    for variable in variables if isinstance(variables, list) else [variables]:
      if variable not in dataset:
        continue
      for dim in dataset[variable].dims:
        if dim not in fixed_dims and dim not in optional_dims:
          needed = True
          assigned_coords = []
          # Find coordinates related to this dimension
          for coord in dataset.coords:
            if dim in dataset[coord].dims and coord not in as_dims and coord not in assigned_coords:
              assigned_coords.append(coord)
              if coord in fixed_coords or coord in optional_coords or coord in interp_vars:
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

def get_bounded_time(dataset, time_var, time):
  if time_var not in dataset or not is_monotonic_var(dataset, time_var):
    raise exceptions.GenericInternalError(f"Time variable '{time_var}' not found or not monotonic in dataset.")
  
  try:
    time_data = dataset[time_var].values
    # Try to cast time to the same dtype as time_data
    time = np.array(time, dtype=time_data.dtype)
    return np.clip(time, time_data[0], time_data[-1])
  except Exception:
    raise exceptions.GenericInternalError(f"Unable to convert time value '{time}' to the correct type.")

def extrapolate_edges_from_cell_data(lons, lats, heights=None, mesh_type="rectilinear", periodic_axes=None): # mesh_type="regular"|"rectilinear"|"radial"
  if periodic_axes is None:
    periodic_axes = []

  if mesh_type == "radial" and len(periodic_axes) == 0:
    periodic_axes = [1]

  def expand_axis(arr, axis):
    ndim = arr.ndim
    dim_size = arr.shape[axis]
    if dim_size == 1:
      return arr

    slice_left = [slice(None)] * ndim
    slice_left[axis] = slice(0, -1)
    slice_right = [slice(None)] * ndim
    slice_right[axis] = slice(1, None)

    midpoints = 0.5 * (arr[tuple(slice_left)] + arr[tuple(slice_right)])

    slice_first = [slice(None)] * ndim
    slice_first[axis] = slice(0, 1)

    slice_last = [slice(None)] * ndim
    slice_last[axis] = slice(-1, None)

    slice_mid_first = list(slice_first)
    slice_mid_last = list(slice_last)

    first_edge = arr[tuple(slice_first)] - (
      midpoints[tuple(slice_mid_first)] - arr[tuple(slice_first)]
    )
    last_edge = arr[tuple(slice_last)] + (
      arr[tuple(slice_last)] - midpoints[tuple(slice_mid_last)]
    )

    if axis in periodic_axes:
      return np.concatenate([first_edge, midpoints, first_edge], axis=axis)
    else:
      return np.concatenate([first_edge, midpoints, last_edge], axis=axis)

  x_cells = lons
  y_cells = lats
  z_cells = heights

  x_bounds = expand_axis(x_cells, 0)
  y_bounds = expand_axis(y_cells, 0)
  if heights is not None:
    z_bounds = expand_axis(z_cells, 0)
  else:
    z_bounds = np.zeros_like(x_bounds)

  if mesh_type != "regular":
    if heights is not None:
      x_bounds = expand_axis(x_bounds, 2)
      y_bounds = expand_axis(y_bounds, 2)
      z_bounds = expand_axis(z_bounds, 2)

    x_bounds = expand_axis(x_bounds, 1)
    y_bounds = expand_axis(y_bounds, 1)
    z_bounds = expand_axis(z_bounds, 1)

  return x_bounds, y_bounds, z_bounds

# Smart selection on a dataset variable with coordinates and dimensions
def sel(dataset, variable, fixed_coords, fixed_dims, interp_vars=[]):
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
  monotonic_fixed_vars = {var: val for var, val in fixed_coords.items() if is_monotonic_var(dataset, var) and var not in interp_vars}
  non_monotonic_fixed_vars = {var: val for var, val in fixed_coords.items() if not is_monotonic_var(dataset, var) and var not in interp_vars}

  data = dataset.sel(monotonic_fixed_vars, method='nearest').sel(non_monotonic_fixed_vars).isel(fixed_dims)[variable]
  # Interpolate if needed
  if len(interp_vars) > 0:
    interpolated_vars = {var: fixed_coords[var] for var in interp_vars if var in fixed_coords}
    if len(interpolated_vars) > 0:
      data = data.interp(interpolated_vars, method="linear", assume_sorted=True)
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

# Convert cache size string (e.g., "1024MB") to bytes
def get_cache_size_bytes(cache_size_str):
  size_unit = cache_size_str[-2:].upper()
  size_value = cache_size_str[:-2]
  try:
    size = int(size_value)
  except ValueError:
    return None
  if size_unit == "KB":
    size_bytes = size * 1024
  elif size_unit == "MB":
    size_bytes = size * 1024 * 1024
  elif size_unit == "GB":
    size_bytes = size * 1024 * 1024 * 1024
  else:
    return None
  return size_bytes

# Enforce cache size limit by deleting old files
def enforce_cache_limit(cache_dir, max_size="512MB"):
  cache_path = Path(cache_dir)
  if not cache_path.exists():
    return
  max_size_bytes = get_cache_size_bytes(max_size)
  if max_size_bytes is None:
    return
  
  # 1. Compute total cache size
  total_size = sum(f.stat().st_size for f in cache_path.glob('**/*') if f.is_file())
  if total_size < max_size_bytes:
    return

  if os.getenv("DEBUG") == "1":
    print(f"[Kazarr - Cache] Cache exceeding max size of {max_size_bytes / 1e6:.2f} MB. Starting cleanup...")

  # 2. Retrieve all files with their modification date
  files = []
  for f in cache_path.glob('**/*'):
    if f.is_file():
      files.append((f, f.stat().st_mtime, f.stat().st_size))
  
  # 3. Sort by date (oldest to newest)
  files.sort(key=lambda x: x[1])

  # 4. Delete old files until under limit
  for f_path, _, f_size in files:
    try:
      os.remove(f_path)
      total_size -= f_size
      if total_size < max_size_bytes:
          break
    except OSError:
      pass # File may be in use or already deleted


class StepDurationLogger:
  def __init__(self, method_name, parameters=None):
    self.method_name = method_name
    self.parameters = parameters if parameters is not None else ()
    self.start_time = time.perf_counter()
    self.steps = []

  def step_start(self, step_name, auto_end_previous=True):
    if len(self.steps) > 0 and auto_end_previous:
      self.step_end()

    self.steps.append({
      "name": step_name,
      "start_time": time.perf_counter(),
      "end_time": None
    })

  def step_end(self):
    if len(self.steps) == 0:
      return

    current_step = self.steps[-1]
    if current_step["end_time"] is None:
      current_step["end_time"] = time.perf_counter()

  def log(self):
    if os.getenv("DEBUG") != "1":
      return
    
    total_end_time = time.perf_counter()
    total_duration = total_end_time - self.start_time

    log_message = f"[Kazarr - Performance] | Method:{self.method_name} | Parameters: {self.parameters} | Total duration: {total_duration:.4f} seconds"
    for step in self.steps:
      if step["end_time"] is not None:
        step_duration = step["end_time"] - step["start_time"]
        log_message += f" | Step {step['name']}: {step_duration:.4f}s"
      else:
        log_message += f" | Step '{step['name']}': not ended"

    log.info(log_message.strip())

  def end(self):
    self.step_end()
    self.log()
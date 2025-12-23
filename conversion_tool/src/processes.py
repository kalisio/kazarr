import os, json, requests, itertools

import xarray as xr
import numpy as np
import s3fs
from pyproj import Transformer
from datetime import datetime

from src.utils import get_ci

def load_from_netcdf(dataset, config):
  ds_count = 0
  total_count = 0
  def progress_callback(ds):
    nonlocal ds_count, total_count
    ds_count += 1
    percentage = ds_count / total_count * 100
    print(f"Progress: {ds_count} / {total_count} ({percentage:.2f}%)")
    return ds

  path = get_ci(config, 'load_path', get_ci(config, 'path'), message="Missing 'load_path' or 'path' config parameters for load_from_netcdf process.")

  if path.startswith("s3://"):
    bucket = os.getenv("BUCKET_NAME")
    if bucket is None:
      raise ValueError("BUCKET_NAME environment variable not set.")
    path = path.replace("s3://", "")
    path = os.path.join(bucket, path)

    # Check if path is folder or file
    fs = s3fs.S3FileSystem(anon=False)
    if fs.isfile(path):
      dataset = xr.open_dataset(fs.open(path, mode='rb', cache_type="bytes"), engine="h5netcdf", chunks="auto")
      return dataset, config
    elif fs.isdir(path):
      concat_dim = get_ci(config, 'concat_dim', message="Missing 'concat_dim' config parameter for loading multiple NetCDF files from S3 folder.")
      files = fs.glob(os.path.join(path, '*.nc'))
      files = sorted(files)
      s3_files = [fs.open(f) for f in files]
      total_count = len(s3_files)

      print(f"Loading {total_count} NetCDF files from S3 folder {path}...")

      dataset = xr.open_mfdataset(
        s3_files,
        engine="h5netcdf",
        combine="nested",
        compat="broadcast_equals",
        concat_dim=concat_dim,
        data_vars="minimal",
        coords="minimal",
        parallel=True,
        chunks="auto",
        preprocess=progress_callback
      )
  else:
    if os.path.isfile(path):
      dataset = xr.open_dataset(path, chunks="auto", engine="h5netcdf")
    elif os.path.isdir(path):
      concat_dim = get_ci(config, 'concat_dim', message="Missing 'concat_dim' for loading multiple NetCDF files from folder.")
      files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".nc")]
      files = sorted(files)
      total_count = len(files)

      dataset = xr.open_mfdataset(
        files,
        engine="h5netcdf",
        combine="nested",
        compat="broadcast_equals",
        concat_dim=concat_dim,
        data_vars="minimal",
        coords="minimal",
        parallel=True,
        chunks="auto",
        preprocess=progress_callback
      )

  if dataset is None:
    raise ValueError(f"Unable to load NetCDF dataset from path: {path}")
  return dataset, config

def load_from_grib(dataset, config):
  ds_count = 0
  total_count = 0
  def progress_callback(ds):
    nonlocal ds_count, total_count
    ds_count += 1
    percentage = ds_count / total_count * 100
    print(f"Progress: {ds_count} / {total_count} ({percentage:.2f}%)")
    return ds

  path = get_ci(config, 'load_path', get_ci(config, 'path'), message="Missing 'load_path' or 'path' config parameters for load_from_grib process.")

  dataset = None
  if path.startswith("s3://"):
    bucket = os.getenv("BUCKET_NAME")
    if bucket is None:
      raise ValueError("BUCKET_NAME environment variable not set.")
    path = path.replace("s3://", "")
    path = os.path.join(bucket, path)

    # Check if path is folder or file
    fs = s3fs.S3FileSystem(anon=False)
    target_tmp_dir = "/tmp/kazarr_grib/"
    if fs.isfile(path):
      os.makedirs(target_tmp_dir, exist_ok=True)
      if not os.path.exists(os.path.join(target_tmp_dir, os.path.basename(path))):
        fs.get(path, os.path.join(target_tmp_dir, os.path.basename(path)))
      dataset = xr.open_dataset(os.path.join(target_tmp_dir, os.path.basename(path)), engine="cfgrib", chunks="auto")
    elif fs.isdir(path):
      concat_dim = get_ci(config, 'concat_dim', message="Missing 'concat_dim' config parameter for loading multiple GRIB files from S3 folder.")
      files = fs.glob(os.path.join(path, "*.grib2"))
      files = sorted(files)
      s3_files = [fs.open(f) for f in files]
      total_count = len(s3_files)

      dataset = xr.open_mfdataset(
        s3_files,
        engine="cfgrib",
        combine="nested",
        compat="broadcast_equals",
        concat_dim=concat_dim,
        data_vars="minimal",
        coords="minimal",
        parallel=True,
        chunks="auto",
        preprocess=progress_callback
      )
  else:
    if os.path.isfile(path):
      dataset = xr.open_dataset(path, chunks="auto", engine="cfgrib")
    elif os.path.isdir(path):
      concat_dim = get_ci(config, 'concat_dim', message="Missing 'concat_dim' for loading multiple GRIB files from folder.")
      files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".grib2")]
      files = sorted(files)
      total_count = len(files)

      dataset = xr.open_mfdataset(
        files,
        engine="cfgrib",
        combine="nested",
        compat="broadcast_equals",
        concat_dim=concat_dim,
        data_vars="minimal",
        coords="minimal",
        parallel=True,
        chunks="auto",
        preprocess=progress_callback
      )
  if dataset is None:
    raise ValueError(f"Unable to load GRIB dataset from path: {path}")
  return dataset, config

def load_from_zarr(dataset, config):
  path = get_ci(config, 'load_path', get_ci(config, 'path'), message="Missing 'load_path' or 'path' config parameters for load_from_zarr process.")
  if path.startswith("s3://"):
    bucket = os.getenv("BUCKET_NAME")
    if bucket is None:
      raise ValueError("BUCKET_NAME environment variable not set.")
    path = path.replace("s3://", "")
    path = os.path.join(bucket, path)

    # check=false to avoid checking (1 more request) for existence of root in the store
    s3_store = s3fs.S3Map(root=path, s3=s3fs.S3FileSystem(anon=False), check=False)
    dataset = xr.open_zarr(s3_store, chunks="auto")
  else:
    dataset = xr.open_zarr(path, chunks="auto")
  return dataset, config

def assign_coords(dataset, config):
  coords = get_ci(config, 'assign_coords', message="Missing 'assign_coords' config parameter for assign_coords process.")
  if not isinstance(coords, dict):
    raise TypeError("'assign_coords' parameter must be a dictionary mapping variable names to dimension names.")

  # Parse possible templates in coords
  expanded_coords = {}
  for var, dim in coords.items():
    if isinstance(dim, dict) and 'variables' in dim:
      var_ranges = dim['variables']
      dim_template = dim['dim']
      
      keys = []
      ranges = []
      for key, rng in var_ranges.items():
        keys.append(key)
        n_min = rng.get('min', 0)
        n_max = rng.get('max', 0)
        ranges.append(range(n_min, n_max + 1))
          
      for values in itertools.product(*ranges):
        var_name = var
        dim_name = dim_template
        
        for key, value in zip(keys, values):
          target = f"{{{key}}}"
          var_name = var_name.replace(target, str(value))
          dim_name = dim_name.replace(target, str(value))
            
        expanded_coords[var_name] = dim_name 
    else:
      expanded_coords[var] = dim
    # if isinstance(dim, dict) and 'variables' in dim:
    #   var_ranges = dim['variables']
    #   dim_template = dim['dim']
    #   for n_key, n_range in var_ranges.items():
    #     n_min = n_range.get('min', 0)
    #     n_max = n_range.get('max', 0)
    #     for n in range(n_min, n_max + 1):
    #       var_name = var.replace(f"{{{n_key}}}", str(n))
    #       dim_name = dim_template.replace(f"{{{n_key}}}", str(n))
    #       expanded_coords[var_name] = dim_name
    # else:
    #   expanded_coords[var] = dim

  assign_dict = {}
  for var, dim in expanded_coords.items():
    if var not in dataset:
      print(f"Warning: Variable '{var}' not found in dataset for assign_coords process. Skipping.")
    elif dim not in dataset.dims:
      print(f"Warning: Dimension '{dim}' not found in dataset for assign_coords process. Skipping.")
    elif var not in dataset.coords:
      assign_dict[var] = (dim, dataset[var].values)
    
  dataset = dataset.assign_coords(assign_dict)

  return dataset, config

def unify_chunks(dataset, config):
  dataset, = xr.unify_chunks(dataset)
  return dataset, config

def rename_variables(dataset, config):
  rename_map = get_ci(config, 'rename_map', message="Missing 'rename_map' config parameter for rename_variables process.")
  if not isinstance(rename_map, dict):
    raise TypeError("'rename_map' parameter must be a dictionary mapping old variable names to new variable names.")
  
  dataset = dataset.rename(rename_map)
  return dataset, config

def delta_time_to_datetime(dataset, config):
  time_ref_var = get_ci(config, 'referenceTime.variable', message="Missing 'referenceTime.variable' config parameter for delta_time_to_datetime process.")
  time_ref_format = get_ci(config, 'referenceTime.format', message="Missing 'referenceTime.format' config parameter for delta_time_to_datetime process.")
  delta_unit = get_ci(config, 'referenceTime.delta_unit')
  time_var = get_ci(config, 'variables.time', message="Missing 'variables.time' config parameter for delta_time_to_datetime process.")
  
  time_dim = get_ci(config, 'dimensions.time')

  units = {
    'years': 'Y', 'year': 'Y',
    'months': 'M', 'month': 'M',
    'days': 'D', 'day': 'D',
    'hours': 'h', 'hour': 'h',
    'minutes': 'm', 'minute': 'm', 'min': 'm',
    'seconds': 's', 'second': 's', 'sec': 's',
  }
  if delta_unit is None and time_var is not None and time_var in dataset and hasattr(dataset[time_var], 'units'):
    delta_unit = dataset[time_var].units
  if delta_unit is not None and delta_unit not in ['Y', 'M', 'D', 'h', 'm', 's'] and delta_unit.lower() in units:
    delta_unit = units[delta_unit.lower()]
  if delta_unit is None or delta_unit not in ['Y', 'M', 'D', 'h', 'm', 's']:
    delta_unit = 'h'  # Default to hours

  try:
    time_ref = dataset[time_ref_var].values.item().decode('utf-8')
    time_ref = datetime.strptime(time_ref, time_ref_format)
  except ValueError as e:
    raise ValueError(f"Error parsing reference time: {e}")

  # Try to deduce time dimension from time variable if time dimension is not provided
  # If time dimension is found, we create a coordinate "datetimes" along that dimension
  # If not, we create a new variable "datetimes"
  if time_dim is None and time_var is not None and time_var in dataset:
    time_dim = dataset[time_var].dims[0]
  time_deltas = dataset[time_var].values
  time_values = np.array([np.datetime64(time_ref) + np.timedelta64(int(td), delta_unit) for td in time_deltas])
  if time_dim is not None:
    dataset = dataset.assign_coords({"datetimes": time_values})
  else:
    dataset["datetimes"] = (dataset[time_var].dims, time_values)

  return dataset, config

def reproject_coordinates(dataset, config):
  from_crs = get_ci(config, 'reprojection.from_crs', message="Missing 'from_crs' config parameter for reproject_coordinates process.")
  to_crs = get_ci(config, 'reprojection.to_crs', message="Missing 'to_crs' config parameter for reproject_coordinates process.")
  
  lon_var = get_ci(config, 'variables.lon', message="Missing 'variables.lon' config parameter for reproject_coordinates process.")
  lat_var = get_ci(config, 'variables.lat', message="Missing 'variables.lat' config parameter for reproject_coordinates process.")
  if lon_var not in dataset or lat_var not in dataset:
    raise ValueError(f"Longitude or latitude variable not found in dataset for reproject_coordinates process.")
  height_var = get_ci(config, 'variables.height')
  if height_var is not None and height_var in dataset:
    has_height = True

  transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
  def reproject(*args):
    return transformer.transform(*args)

  vars_to_reproject = [lon_var, lat_var]
  if has_height:
    vars_to_reproject.append(height_var)
  values = [dataset[var].values for var in vars_to_reproject]

  # Use xarray apply_ufunc to reproject coordinates with Dask arrays (parallelized)
  output = xr.apply_ufunc(
    reproject,
    *values,
    output_core_dims=[[]] * len(vars_to_reproject),
    vectorize=True,
    dask="parallelized"
  )

  dataset[lon_var] = (dataset[lon_var].dims, output[0])
  dataset[lat_var] = (dataset[lat_var].dims, output[1])
  if has_height:
    dataset[height_var] = (dataset[height_var].dims, output[2])

  return dataset, config

def save(dataset, config):
  path = get_ci(config, 'save_path')
  if path is None:
    path = get_ci(config, 'path', message="Missing 'save_path' or 'path' config parameter for save process.")
    path = path.replace(".nc", "").replace(".grib2", "") + ".zarr"
  config['save_path'] = path  # Update config with actual save path
  version = get_ci(config, 'version', default=3)
  float64_to_float32 = get_ci(config, 'float64_to_float32', default=False)

  if float64_to_float32:
    for var in dataset.data_vars:
      if dataset[var].dtype == np.float64:
        dataset[var] = dataset[var].astype(np.float32)

  final_path = path
  if path.startswith("s3://"):
    bucket = os.getenv("BUCKET_NAME")
    if bucket is None:
      raise ValueError("BUCKET_NAME environment variable not set.")
    final_path = path.replace("s3://", "s3://" + bucket + "/")

  dataset.to_zarr(final_path, mode="w", consolidated=(version==2), zarr_format=version)
  return dataset, config

def save_config(dataset, config):
  path = get_ci(config, 'config_save_path', message="Missing 'config_save_path' config parameter for save_config process.")
  with open(path, 'w') as f:
    json.dump(config, f, indent=2)
  return dataset, config

def register_on_api(dataset, config):
  url = os.getenv("API_ENDPOINT_URL")
  if url is None:
    url = get_ci(config, 'registration_endpoint_url', message="Missing 'registration_endpoint_url' config parameter for post_dataset process and API_ENDPOINT_URL environment variable not set.")

  keep_keys = ['variables', 'dimensions']
  post_config = {
    "name": get_ci(config, 'name', message="Missing 'name' config parameter for post_dataset process."),
    "description": get_ci(config, 'description', ""),
    "path": get_ci(config, 'save_path', message="Missing 'save_path' config parameter for post_dataset process."),
    "config": {
      k: v for k, v in config.items() if k in keep_keys
    }
  }
  result = requests.post(url, json = post_config)
  try:
    result_json = result.json()
    if result.status_code != 200:
      raise ValueError(f"Error registering dataset on API: {result_json.get('detail', 'Unknown error')}")
    if "id" in result_json:
      config['api_id'] = result_json.get('id')
  except Exception as e:
    raise ValueError(f"Error processing response from API: {str(e)}")
  return dataset, config
import os, time, json

import s3fs
from botocore.exceptions import NoCredentialsError

# Load datasets config file
def load_datasets(path="datasets.json"):
  if path.startswith("s3://"):
    path = path[5:]
    s3_store = s3fs.S3FileSystem(anon=False)
    bucket = os.getenv("BUCKET_NAME")
    if bucket is None:
      raise ValueError("BUCKET_NAME environment variable not set.")
    try:
      with s3_store.open(os.path.join(bucket, path), 'r') as f:
        datasets = json.load(f)
    except NoCredentialsError as e:
      raise ValueError("S3 credentials not found.")
    except Exception as e:
      raise ValueError("Unable to access S3: " + str(e))
  else:
    try:
      with open(path, 'r') as f:
        datasets = json.load(f)
    except Exception as e:
      raise ValueError("Unable to access local file: " + str(e))
  return datasets

def load_dataset_config(dataset_name, datasets_path="datasets.json"):
  datasets = load_datasets(datasets_path)

  if dataset_name not in datasets:
    raise ValueError(f"Dataset {dataset_name} not found.")

  template = datasets[dataset_name].get("template")
  if template:
    templates = load_datasets("templates.json")
    if template in templates:
      datasets[dataset_name] = merge(templates[template], datasets[dataset_name])
  return datasets[dataset_name]

# Deep merge two dicts
def merge(src, dest):
  for key, value in src.items():
    if isinstance(value, dict):
      # get node or create one
      node = dest.setdefault(key, {})
      merge(value, node)
    else:
      dest[key] = value
  return dest

# Convert camelCase to snake_case
def camel_to_snake(string):
  if not string:
    return string
  return ''.join(['_' + l.lower() if l.isupper() else l for l in string])

# Convert snake_case to camelCase
def snake_to_camel(string):
  if not string:
    return string
  components = string.split('_')
  return components[0] + ''.join(x.title() for x in components[1:])

# Deep get from nested dict (mimic lodash get)
def dget(d, key, default=None):
  keys = key.split('.')
  for k in keys:
    if isinstance(d, dict) and k in d:
      d = d[k]
    else:
      return default.copy() if isinstance(default, dict) else default
  return d

# Get value from dict with camelCase or snake_case key
def get_ci(d, key, default=None, message=None):
  value = dget(d, key)
  if value is not None:
    return value
  value = dget(d, snake_to_camel(key))
  if value is not None:
    return value
  if message is not None and default is None:
    raise ValueError(message)
  return default

# Print duration since start_time with message
def print_duration(start_time, message):
  duration = time.time() - start_time
  print("[KAZARR]{" + f"{duration:.2f}s" + "} " + message)

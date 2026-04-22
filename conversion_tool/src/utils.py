import os
import time
import json
from pathlib import Path
import shutil
from datetime import datetime

import s3fs
import numpy as np
from botocore.exceptions import NoCredentialsError
from dask import array as da


def get_s3_storage_options(config, path=None):
    if path and not path.startswith("s3://"):
        return {}

    storage_options = get_ci(config, "storage_options", default={})

    # Default timeouts
    connect_timeout = get_ci(config, "s3.connect_timeout", 300)
    read_timeout = get_ci(config, "s3.read_timeout", 300)

    # Ensure client_kwargs exists
    config_kwargs = storage_options.get("config_kwargs", {})
    config_kwargs.setdefault("connect_timeout", connect_timeout)
    config_kwargs.setdefault("read_timeout", read_timeout)
    storage_options["config_kwargs"] = config_kwargs

    # Handle OVH specific configuration
    # endpoint_url = os.getenv("AWS_ENDPOINT_URL", "")
    # if "cloud.ovh.net" in endpoint_url or "cloud.ovh.net" in (path or ""):
    #     config_kwargs = storage_options.get("config_kwargs", {})
    #     config_kwargs.setdefault("signature_version", "s3v4")

    #     s3_config = config_kwargs.get("s3", {})
    #     s3_config.setdefault("addressing_style", "path")
    #     s3_config.setdefault("payload_signing_enabled", False)
    #     config_kwargs["s3"] = s3_config

    #     storage_options["config_kwargs"] = config_kwargs

    # Ensure authentication
    if "anon" not in storage_options:
        storage_options["anon"] = False

    return storage_options


def get_s3_filesystem(config, path=None):
    return s3fs.S3FileSystem(**get_s3_storage_options(config, path))


# Load JSON file
def load_json(path="datasets.json", config={}):
    if path.startswith("s3://"):
        path = path[5:]
        fs = get_s3_filesystem(config, path)
        bucket = os.getenv("BUCKET_NAME")
        if bucket is None:
            raise ValueError("BUCKET_NAME environment variable not set.")
        try:
            with fs.open(os.path.join(bucket, path), "r") as f:
                datasets = json.load(f)
        except NoCredentialsError:
            raise ValueError("S3 credentials not found.")
        except Exception as e:
            raise ValueError("Unable to access S3: " + str(e))
    else:
        try:
            with open(path, "r") as f:
                datasets = json.load(f)
        except Exception as e:
            raise ValueError("Unable to access local file: " + str(e))
    return datasets


# Load dataset configuration by name
def load_dataset_config(
    dataset_name, datasets_path="datasets.json", template_path="templates.json"
):
    datasets = load_json(datasets_path)

    if dataset_name not in datasets:
        raise ValueError(f"Dataset {dataset_name} not found.")

    template = datasets[dataset_name].get("template")
    if template:
        templates = load_json(template_path)
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
    return "".join(["_" + char.lower() if char.isupper() else char for char in string])


# Convert snake_case to camelCase
def snake_to_camel(string):
    if not string:
        return string
    components = string.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


# Deep get from nested dict (mimic lodash get)
def dget(d, key, default=None):
    keys = key.split(".")
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default.copy() if isinstance(default, dict) else default
    return d


# Get value (case insensitive) from dict with camelCase or snake_case key
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


def get_optimal_chunks_for_zarr(data_array, target_size="100MB"):
    shape = data_array.shape
    dtype = data_array.dtype
    dims = data_array.dims

    dask_chunks = da.core.normalize_chunks(
        target_size, shape=shape, dtype=dtype, previous_chunks=None
    )

    uniform_chunks = {}
    for i, dim_name in enumerate(dims):
        uniform_chunks[dim_name] = dask_chunks[i][0]

    return uniform_chunks


def rechunk_if_needed(dataset, target_size_mb=100, tolerance=0.3):
    for var_name, variable in dataset.variables.items():
        if variable.chunks is None:
            continue

        for dim_chunks in variable.chunks:
            if len(dim_chunks) <= 1:
                continue

            # Check if chunks are too small compared to target size
            chunk_shape = [c[0] for c in variable.chunks]
            bytes_per_chunk = np.prod(chunk_shape) * variable.dtype.itemsize
            current_chunk_mb = bytes_per_chunk / (1024**2)
            chunks_too_small = current_chunk_mb < target_size_mb * (1 - tolerance)

            # Check if all main chunks are the same size (uniform)
            is_uniform = True
            for dim_chunks in variable.chunks:
                # Last chunk is often smaller, so we ignore it for the uniformity check
                if len(dim_chunks) > 1 and len(set(dim_chunks[:-1])) > 1:
                    is_uniform = False
                    break

            if not is_uniform or chunks_too_small:
                # Rechunk variable to uniform chunk sizes based on Dask's optimal chunking
                optimal_chunks = get_optimal_chunks_for_zarr(variable)
                dataset[var_name] = variable.chunk(optimal_chunks)

    return dataset


def merge_grib(folder_path, output_filename, config, glob_search_pattern="*.grib2"):
    print(f'[KAZARR] < Merging GRIB files in "{folder_path}" into "{output_filename}"')
    if os.path.exists(os.path.join(folder_path, output_filename)):
        os.remove(os.path.join(folder_path, output_filename))

    # Remove .idx files if they exist, as they are not needed for the merge and can cause issues
    for idx_file in Path(folder_path).glob(f"{output_filename}*.idx"):
        os.remove(idx_file)

    folder = Path(folder_path)
    files = sorted(folder.glob(glob_search_pattern))

    if not files:
        return config

    if "clean" not in config:
        config["clean"] = {"used_paths": files}
    elif "used_paths" not in config["clean"]:
        config["clean"]["used_paths"] = files
    else:
        config["clean"]["used_paths"].extend(files)

    with open(folder / output_filename, "wb") as output_file:
        for file in files:
            with open(file, "rb") as input_file:
                shutil.copyfileobj(input_file, output_file)

    if "generated_paths" not in config["clean"]:
        config["clean"]["generated_paths"] = [
            os.path.join(folder_path, output_filename)
        ]
    else:
        config["clean"]["generated_paths"].append(
            os.path.join(folder_path, output_filename)
        )

    if "idx_folders" not in config["clean"]:
        config["clean"]["idx_folders"] = [folder_path]
    elif folder_path not in config["clean"]["idx_folders"]:
        config["clean"]["idx_folders"].append(folder_path)

    print(
        f'[KAZARR] > Completed merging GRIB files ({len(files)}) into "{output_filename}"'
    )

    return config


def timestamp_to_datetime(timestamp):
    try:
        ts = float(timestamp)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid timestamp format: {timestamp}")

    if ts < 1e11:
        divisor = 1.0  # Seconds
    elif ts < 1e14:
        divisor = 1e3  # Milliseconds
    elif ts < 1e17:
        divisor = 1e6  # Microseconds
    else:
        divisor = 1e9  # Nanoseconds

    return datetime.fromtimestamp(ts / divisor)

import os
import time
import json
from pathlib import Path
import shutil

import s3fs
from botocore.exceptions import NoCredentialsError
from dask import array as da


# Load JSON file
def load_json(path="datasets.json"):
    if path.startswith("s3://"):
        path = path[5:]
        s3_store = s3fs.S3FileSystem(anon=False)
        bucket = os.getenv("BUCKET_NAME")
        if bucket is None:
            raise ValueError("BUCKET_NAME environment variable not set.")
        try:
            with s3_store.open(os.path.join(bucket, path), "r") as f:
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


def rechunk_if_needed(dataset):
    for var_name, variable in dataset.variables.items():
        if variable.chunks is None:
            continue

        for dim_chunks in variable.chunks:
            if len(dim_chunks) <= 1:
                continue

            # Last chunk is often smaller, so we ignore it for the uniformity check
            main_chunks = dim_chunks[:-1]

            # Check if all main chunks are the same size (uniform)
            if len(set(main_chunks)) > 1:
                # Rechunk variable to uniform chunk sizes based on Dask's optimal chunking
                optimal_chunks = get_optimal_chunks_for_zarr(variable)
                dataset[var_name] = variable.chunk(optimal_chunks)

    return dataset


def merge_grib(folder_path, output_filename, config, glob_search_pattern="*.grib2"):
    print(f'[KAZARR] < Merging GRIB files in "{folder_path}" into "{output_filename}"')
    if os.path.exists(os.path.join(folder_path, output_filename)):
        os.remove(os.path.join(folder_path, output_filename))

    folder = Path(folder_path)
    files = sorted(folder.glob(glob_search_pattern))

    if not files:
        return

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

    print(f'[KAZARR] > Completed merging GRIB files into "{output_filename}"')

    return config

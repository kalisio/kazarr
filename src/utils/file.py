import os
import json
import copy
from functools import lru_cache
from pathlib import Path

import s3fs
import xarray as xr
import fsspec
from botocore.exceptions import NoCredentialsError

import src.exceptions as exceptions


def get_datasets_path():
    return os.getenv("DATASETS_PATH", "/")


def s3_credentials_exists():
    bucket = os.getenv("BUCKET_NAME")
    if bucket is None:
        return False
    s3_store = s3fs.S3FileSystem(anon=False)
    try:
        s3_store.ls(os.path.join(bucket, get_datasets_path()))
        return True
    except NoCredentialsError:
        return False
    except Exception:
        return False


# Open Zarr dataset as XArray dataset from S3
@lru_cache(maxsize=5)
def load(path):
    if not s3_credentials_exists():
        try:
            return xr.open_zarr(
                os.path.join(get_datasets_path().rstrip("/"), path), chunks="auto"
            )
        except Exception:
            raise exceptions.DatasetNotFound(path.replace(".zarr", ""))

    bucket = os.getenv("BUCKET_NAME")
    if bucket is None:
        raise exceptions.GenericInternalError(
            "BUCKET_NAME environment variable not set."
        )
    try:
        cache_size = os.getenv("CACHE_SIZE", "512MB")
        cache_path = os.getenv("CACHE_DIR")
        use_cache = (
            cache_path is not None and get_cache_size_bytes(cache_size) is not None
        )
        if not use_cache:
            store = s3fs.S3Map(
                root=os.path.join(
                    bucket, get_datasets_path(), path.replace("s3://", "")
                ),
                s3=s3fs.S3FileSystem(anon=False),
            )
        else:
            fs = fsspec.filesystem(
                "simplecache",
                target_protocol="s3",
                cache_storage=cache_path,
                target_options={"anon": False},
                expiry_time=60 * 60 * 24 * 7 * 4,  # 4 weeks
            )
            store = fs.get_mapper(
                os.path.join(bucket, get_datasets_path(), path.replace("s3://", ""))
            )

        # Will try to open consolidated metadata first (https://docs.xarray.dev/en/latest/generated/xarray.open_zarr.html#xarray.open_zarr)
        # Will try to determine zarr_format (v2 or v3) automatically
        # chunks must be defined to enable dask lazy loading
        dataset = xr.open_zarr(store, chunks="auto")
    except NoCredentialsError:
        raise exceptions.GenericInternalError("S3 credentials not found.")
    except Exception as e:
        raise exceptions.GenericInternalError("Unable to access S3: " + str(e))
    return dataset


# Load JSON file from S3
def load_json(path):
    s3_store = s3fs.S3FileSystem(anon=False)
    bucket = os.getenv("BUCKET_NAME")
    if bucket is None:
        raise exceptions.GenericInternalError(
            "BUCKET_NAME environment variable not set."
        )
    try:
        with s3_store.open(os.path.join(bucket, path), "r") as f:
            datasets = json.load(f)
    except NoCredentialsError:
        raise exceptions.GenericInternalError("S3 credentials not found.")
    except Exception as e:
        raise exceptions.GenericInternalError("Unable to access S3: " + str(e))
    return datasets


def find_datasets(path="/"):
    if not s3_credentials_exists():
        datasets = []
        full_path = (
            os.path.join(get_datasets_path().rstrip("/"), path.lstrip("/")) or "."
        )
        print("[KAZARR] Searching datasets in path:", full_path)
        for root, dirs, _ in os.walk(full_path):
            for dirname in dirs:
                if dirname.endswith(".zarr"):
                    found_path = os.path.join(root, dirname)
                    datasets.append(
                        found_path.replace(get_datasets_path(), "").replace(".zarr", "")
                    )
        return datasets

    # Search all folders recursively that ends with .zarr
    s3_store = s3fs.S3FileSystem(anon=False)
    bucket = os.getenv("BUCKET_NAME")
    if bucket is None:
        raise exceptions.GenericInternalError(
            "BUCKET_NAME environment variable not set."
        )
    datasets = []
    try:
        for root, dirs, _ in s3_store.walk(
            os.path.join(bucket, get_datasets_path().rstrip("/"), path.lstrip("/"))
        ):
            for dirname in dirs[:]:
                if dirname.endswith(".zarr"):
                    found_path = f"{root}/{dirname}"
                    datasets.append(
                        found_path.replace(
                            os.path.join(bucket, get_datasets_path().rstrip("/")), ""
                        ).replace(".zarr", "")
                    )
                    dirs.remove(dirname)
    except NoCredentialsError:
        raise exceptions.GenericInternalError("S3 credentials not found.")
    except Exception as e:
        raise exceptions.GenericInternalError("Unable to access S3: " + str(e))
    return datasets


def load_datasets(search_path=None):
    search_path = search_path if search_path is not None else ""
    return find_datasets(search_path)


# Load a dataset and its configuration from its ID
def load_dataset(dataset_path):
    dataset = load(dataset_path + ".zarr")
    cache_size = os.getenv("CACHE_SIZE", "512MB")
    cache_path = os.getenv("CACHE_DIR")
    use_cache = cache_path is not None and get_cache_size_bytes(cache_size) is not None
    if use_cache:
        # Enforce cache size limit after loading
        enforce_cache_limit(cache_path, max_size=cache_size)
    config = copy.deepcopy(dataset.attrs.get("kazarr", {}))
    return dataset, config


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
    total_size = sum(f.stat().st_size for f in cache_path.glob("**/*") if f.is_file())
    if total_size < max_size_bytes:
        return

    if os.getenv("DEBUG") == "1":
        print(
            f"[Kazarr - Cache] Cache exceeding max size of {max_size_bytes / 1e6:.2f} MB. Starting cleanup..."
        )

    # 2. Retrieve all files with their modification date
    files = []
    for f in cache_path.glob("**/*"):
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
            pass  # File may be in use or already deleted
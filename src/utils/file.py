import os
import json
import copy
import datetime
from functools import lru_cache

import s3fs
from diskcache import Cache
import xarray as xr
from botocore.exceptions import NoCredentialsError
from zarr.abc.store import Store
from zarr.core.buffer import Buffer, BufferPrototype

from loguru import logger as log
import src.exceptions as exceptions


ZARR_EXTENSION = ".zarr"
S3_PREFIX = "s3://"
BUCKET_NAME_ENV_VAR = "BUCKET_NAME"
CACHE_LIMIT_MARGIN = 0.9  # 90% of the specified cache size limit


def get_datasets_path():
    return os.getenv("DATASETS_PATH", "/")


def s3_credentials_exists():
    bucket = os.getenv(BUCKET_NAME_ENV_VAR)
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


# Internal cached loader — keyed by (path, version) so that updating a dataset
# on S3 (which changes its last_modified timestamp) automatically creates a new
# lru_cache entry and discards the stale xarray Dataset object.
@lru_cache(maxsize=5)
def _load_versioned(path, version):
    if not s3_credentials_exists():
        try:
            return xr.open_zarr(
                os.path.join(get_datasets_path().rstrip("/"), path), chunks="auto"
            )
        except Exception:
            raise exceptions.DatasetNotFound(path.replace(ZARR_EXTENSION, ""))

    bucket = os.getenv(BUCKET_NAME_ENV_VAR)
    if bucket is None:
        raise exceptions.GenericInternalError(
            f"{BUCKET_NAME_ENV_VAR} environment variable not set."
        )
    try:
        cache_size = os.getenv("CACHE_SIZE", "512MB")
        cache_path = os.getenv("CACHE_DIR")
        use_cache = (
            cache_path is not None
            and cache_path != ""
            and get_cache_size_bytes(cache_size) is not None
        )
        if not use_cache:
            store = s3fs.S3Map(
                root=os.path.join(
                    bucket, get_datasets_path(), path.replace(S3_PREFIX, "")
                ),
                s3=s3fs.S3FileSystem(anon=False),
            )
        else:
            dataset_id = path.replace(S3_PREFIX, "")
            # Use version (last_modified) to salt the diskcache namespace so
            # stale chunks from a previous dataset version are never returned.
            cache_namespace = f"{dataset_id}_{version}" if version else dataset_id
            store = S3CachedStore(
                s3_root=os.path.join(bucket, get_datasets_path(), dataset_id),
                cache_dir=cache_path,
                dataset_id=cache_namespace,
                expiry_seconds=60 * 60 * 24 * 7 * 4,
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


def load(path):
    """Open a Zarr dataset, using last_modified as a version key to invalidate
    the in-memory lru_cache whenever the dataset is updated on S3."""
    dataset_id = path.replace(ZARR_EXTENSION, "").replace(S3_PREFIX, "")
    version = get_dataset_last_modified(dataset_id)
    return _load_versioned(path, version)


# Load JSON file from S3
def load_json(path):
    s3_store = s3fs.S3FileSystem(anon=False)
    bucket = os.getenv(BUCKET_NAME_ENV_VAR)
    if bucket is None:
        raise exceptions.GenericInternalError(
            f"{BUCKET_NAME_ENV_VAR} environment variable not set."
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
        log.info("[KAZARR] Searching datasets in path: {path}", path=full_path)
        for root, dirs, _ in os.walk(full_path):
            for dirname in dirs:
                if dirname.endswith(ZARR_EXTENSION):
                    found_path = os.path.join(root, dirname)
                    datasets.append(
                        found_path.replace(get_datasets_path(), "").replace(
                            ZARR_EXTENSION, ""
                        )
                    )
        return datasets

    # Search all folders recursively that ends with .zarr
    s3_store = s3fs.S3FileSystem(anon=False)
    bucket = os.getenv(BUCKET_NAME_ENV_VAR)
    if bucket is None:
        raise exceptions.GenericInternalError(
            f"{BUCKET_NAME_ENV_VAR} environment variable not set."
        )
    datasets = []
    try:
        for root, dirs, _ in s3_store.walk(
            os.path.join(bucket, get_datasets_path().rstrip("/"), path.lstrip("/"))
        ):
            for dirname in dirs[:]:
                if dirname.endswith(ZARR_EXTENSION):
                    found_path = f"{root}/{dirname}"
                    datasets.append(
                        found_path.replace(
                            os.path.join(bucket, get_datasets_path().rstrip("/")), ""
                        ).replace(ZARR_EXTENSION, "")
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
    dataset = load(dataset_path + ZARR_EXTENSION)
    config = copy.deepcopy(dataset.attrs.get("kazarr", {}))
    return dataset, config


def get_dataset_last_modified(dataset_id: str) -> os.stat_result | str | None:
    path = dataset_id + ZARR_EXTENSION
    if not s3_credentials_exists():
        full_path = os.path.join(get_datasets_path().rstrip("/"), path)
        try:
            return datetime.datetime.fromtimestamp(
                os.path.getmtime(full_path), tz=datetime.timezone.utc
            )
        except Exception:
            return None

    bucket = os.getenv(BUCKET_NAME_ENV_VAR)
    if bucket is None:
        return None
    s3_store = s3fs.S3FileSystem(anon=False)
    try:
        s3_path = os.path.join(bucket, get_datasets_path().lstrip("/"), path)
        # Often .zmetadata is updated when dataset changes
        zmetadata_path = os.path.join(s3_path, ".zmetadata")
        if s3_store.exists(zmetadata_path):
            info = s3_store.info(zmetadata_path)
            return info.get("LastModified")
        # Fallback to the directory itself
        info = s3_store.info(s3_path)
        return info.get("LastModified")
    except Exception as e:
        log.warning(f"[Kazarr] Could not get last modified date for {dataset_id}: {e}")
        return None


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


class S3CachedStore(Store):
    def __init__(
        self,
        s3_root: str,
        cache_dir: str,
        dataset_id: str = "",
        expiry_seconds: int = 60 * 60 * 24 * 7 * 4,
    ):
        super().__init__(read_only=True)
        self.s3_root = s3_root.rstrip("/")
        self.s3 = s3fs.S3FileSystem(anon=False)
        self.expiry_seconds = expiry_seconds
        self.dataset_id = dataset_id.strip("/")

        cache_size_str = os.getenv("CACHE_SIZE", "512MB")
        max_size_bytes = get_cache_size_bytes(cache_size_str) or (512 * 1024 * 1024)

        # To avoid overflow, we set the cache size limit to 90% of the specified size
        max_size_bytes = int(max_size_bytes * CACHE_LIMIT_MARGIN)

        self.cache = Cache(
            directory=cache_dir,
            size_limit=max_size_bytes,
            eviction_policy="least-recently-used",
        )

    @property
    def supports_writes(self) -> bool:
        return False

    @property
    def supports_deletes(self) -> bool:
        return False

    @property
    def supports_listing(self) -> bool:
        return True

    def _cache_key(self, key: str) -> str:
        return f"{self.dataset_id}/{key}" if self.dataset_id else key

    def _fetch(self, key: str) -> bytes:
        c_key = self._cache_key(key)
        data = self.cache.get(c_key)

        if data is None:
            log.debug("[Kazarr - Cache] MISS for key: {key}", key=c_key)
            s3_path = f"{self.s3_root}/{key}"
            try:
                data = self.s3.cat(s3_path)
            except FileNotFoundError:
                raise KeyError(key)

            self.cache.set(c_key, data, expire=self.expiry_seconds)
        else:
            log.debug("[Kazarr - Cache] HIT for key: {key}", key=c_key)

        return data

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, S3CachedStore)
            and self.s3_root == other.s3_root
            and self.cache.directory == other.cache.directory
            and self.dataset_id == getattr(other, "dataset_id", "")
        )

    async def get(
        self, key: str, prototype: BufferPrototype, byte_range=None
    ) -> Buffer | None:
        try:
            data = self._fetch(key)
            if byte_range is not None:
                start = byte_range.start or 0
                end = byte_range.stop or len(data)
                data = data[start:end]
            return prototype.buffer.from_bytes(data)
        except KeyError:
            return None

    async def get_partial_values(self, prototype: BufferPrototype, key_ranges):
        results = []
        for key, byte_range in key_ranges:
            results.append(await self.get(key, prototype, byte_range))
        return results

    async def set(self, key: str, value: Buffer, byte_range=None) -> None:
        raise NotImplementedError("Read-only store")

    async def delete(self, key: str) -> None:
        raise NotImplementedError("Read-only store")

    async def exists(self, key: str) -> bool:
        if self._cache_key(key) in self.cache:
            return True
        return self.s3.exists(f"{self.s3_root}/{key}")

    async def list(self):
        for entry in self.s3.ls(self.s3_root, detail=False):
            yield os.path.relpath(entry, self.s3_root)

    async def list_prefix(self, prefix: str):
        s3_path = f"{self.s3_root}/{prefix}".rstrip("/")
        for entry in self.s3.ls(s3_path, detail=False):
            yield os.path.relpath(entry, self.s3_root)

    async def list_dir(self, prefix: str):
        s3_path = f"{self.s3_root}/{prefix}".rstrip("/")
        for entry in self.s3.ls(s3_path, detail=False):
            yield os.path.relpath(entry, self.s3_root)

import hashlib
import os

import numpy as np
from loguru import logger as log
from scipy.spatial import cKDTree

_spatial_index_cache: dict[tuple, cKDTree] = {}
MAX_CACHE_SIZE = os.getenv("KDTREE_MAX_CACHE_SIZE", "10")


def _get_array_hash(arr: np.ndarray) -> str:
    """Compute a fast hash for a numpy array.
    We use the shape, dtype, and a sample of values to keep it fast.
    """
    sample_indices = np.linspace(0, arr.size - 1, min(100, arr.size), dtype=int)
    sample_data = arr.ravel()[sample_indices]

    hasher = hashlib.md5(usedforsecurity=False)
    hasher.update(str(arr.shape).encode())
    hasher.update(str(arr.dtype).encode())
    hasher.update(sample_data.tobytes())
    return hasher.hexdigest()


def lonlat_to_xyz(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project geographic coordinates (lon, lat in degrees) onto the unit
    sphere as 3D Cartesian coordinates.
    """
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)
    cos_lat = np.cos(lat_rad)
    x = cos_lat * np.cos(lon_rad)
    y = cos_lat * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return x, y, z


def lonlat_to_sphere_points(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Convenience wrapper: (lon, lat) arrays -> an (N, 3) array of unit-sphere
    XYZ points, ready to feed into cKDTree or to column_stack with extra
    (e.g. level) columns.
    """
    x, y, z = lonlat_to_xyz(lon, lat)
    return np.column_stack((x, y, z))


def deg_to_chord_distance(deg: np.ndarray | float) -> np.ndarray | float:
    """Convert an angular great-circle distance in degrees to the equivalent
    unit-sphere chord (straight-line) distance, so it can be used as a
    radius threshold for queries against unit-sphere-projected points.
    """
    rad = np.radians(deg)
    return 2.0 * np.sin(rad / 2.0)


def chord_to_deg_distance(chord: np.ndarray | float) -> np.ndarray | float:
    """Convert a unit-sphere chord distance back to the true angular
    great-circle distance in degrees. Inverse of `deg_to_chord_distance`.
    """
    # Clip for numerical safety: chord should be in [0, 2], but floating
    # point round-trips can push it a hair outside due to rounding.
    chord = np.clip(chord, 0.0, 2.0)
    return np.degrees(2.0 * np.arcsin(chord / 2.0))


def get_cached_ckdtree(
    points: np.ndarray,
    dataset_id: str | None = None,
    coord_vars: tuple[str, ...] | None = None,
) -> cKDTree:
    """Retrieve or build a cKDTree for the given points.

    If dataset_id and coord_vars are provided, they are used for caching.
    Otherwise, a hash of the points is used.
    """
    # Create cache key
    if dataset_id and coord_vars:
        # We still add a data hash to be safe if the file changed or subsetting happened
        data_hash = _get_array_hash(points)
        cache_key = (dataset_id, coord_vars, data_hash)
    else:
        data_hash = _get_array_hash(points)
        cache_key = (data_hash,)

    if cache_key in _spatial_index_cache:
        log.info(
            "[KAZARR] Using cached spatial index (cKDTree) for points with shape {shape}",
            shape=points.shape,
        )
        return _spatial_index_cache[cache_key]

    log.info(
        "[KAZARR] Building new spatial index (cKDTree) for points with shape {shape}",
        shape=points.shape,
    )
    tree = cKDTree(points)

    # Manage cache size
    if len(_spatial_index_cache) >= int(MAX_CACHE_SIZE):
        log.warning("[KAZARR] Cache size limit reached, evicting oldest entry")
        # Simple FIFO-ish eviction: remove a random entry or the first one
        _spatial_index_cache.pop(next(iter(_spatial_index_cache)))

    _spatial_index_cache[cache_key] = tree
    return tree

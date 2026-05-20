import hashlib
import os
import numpy as np
from scipy.spatial import cKDTree
from typing import Dict, Tuple, Optional

_spatial_index_cache: Dict[Tuple, cKDTree] = {}
MAX_CACHE_SIZE = os.getenv("KDTREE_MAX_CACHE_SIZE", 10)


def _get_array_hash(arr: np.ndarray) -> str:
    """Compute a fast hash for a numpy array.
    We use the shape, dtype, and a sample of values to keep it fast.
    """
    sample_indices = np.linspace(0, arr.size - 1, min(100, arr.size), dtype=int)
    sample_data = arr.ravel()[sample_indices]

    hasher = hashlib.md5()
    hasher.update(str(arr.shape).encode())
    hasher.update(str(arr.dtype).encode())
    hasher.update(sample_data.tobytes())
    return hasher.hexdigest()


def get_cached_ckdtree(
    points: np.ndarray,
    dataset_id: Optional[str] = None,
    coord_vars: Optional[Tuple[str, ...]] = None,
) -> cKDTree:
    """Retrieve or build a cKDTree for the given points.

    If dataset_id and coord_vars are provided, they are used for caching.
    Otherwise, a hash of the points is used.
    """
    global _spatial_index_cache

    # Create cache key
    if dataset_id and coord_vars:
        # We still add a data hash to be safe if the file changed or subsetting happened
        data_hash = _get_array_hash(points)
        cache_key = (dataset_id, coord_vars, data_hash)
    else:
        data_hash = _get_array_hash(points)
        cache_key = (data_hash,)

    if cache_key in _spatial_index_cache:
        print(
            "[KAZARR] Using cached spatial index (cKDTree) for points with shape",
            points.shape,
        )
        return _spatial_index_cache[cache_key]

    print(
        "[KAZARR] Building new spatial index (cKDTree) for points with shape",
        points.shape,
    )
    tree = cKDTree(points)

    # Manage cache size
    if len(_spatial_index_cache) >= MAX_CACHE_SIZE:
        print("[KAZARR] Cache size limit reached, evicting oldest entry")
        # Simple FIFO-ish eviction: remove a random entry or the first one
        _spatial_index_cache.pop(next(iter(_spatial_index_cache)))

    _spatial_index_cache[cache_key] = tree
    return tree

import math

import numpy as np
import xarray as xr

from src import exceptions
from src.processing.contexts import BBoxContext, GridIndices


def normalize_dataset_longitudes(dataset, lon_var, lat_var, bbox=None, lon=None):
    """Dynamically handle longitudes depending on the dataset's convention
    ([0, 360] or [-180, 180]).

    Returns:
        A tuple ``(dataset, lon)`` where *dataset* may have been sliced and
        re-indexed, and *lon* may have been wrapped.
    """
    if lon_var is None or lon_var not in dataset:
        return dataset, lon

    lon_min_val = float(dataset[lon_var].min())
    is_0_360 = lon_min_val >= 0

    # --- Single-point case (probe) ---
    if lon is not None and (bbox is None or not bbox.has_bb_lon):
        if is_0_360:
            if lon < 0:
                lon = lon + 360
            elif lon > 360:
                lon = lon - 360
        else:
            if lon > 180:
                lon = lon - 360
            elif lon < -180:
                lon = lon + 360
        return dataset, lon

    # --- Bounding-box case (extract) ---
    if not bbox.has_bb_lon:
        return dataset, lon

    longitudes = dataset[lon_var]
    latitudes = dataset[lat_var]
    is_regular_grid = (
        longitudes.ndim == 1
        and latitudes.ndim == 1
        and longitudes.dims != latitudes.dims
    )
    if not is_regular_grid:
        # We cannot slice non-dimension coordinates. The filtering will be handled
        # via boolean masks in apply_irregular_bounding_box / apply_point_list_bounding_box.
        return dataset, lon

    bb_lon_min = (
        bbox.lon_min if bbox.lon_min is not None else (-180.0 if not is_0_360 else 0.0)
    )
    bb_lon_max = (
        bbox.lon_max if bbox.lon_max is not None else (180.0 if not is_0_360 else 360.0)
    )

    # If the requested bbox wraps around mathematically (e.g. [170, -170]),
    # unwrap it (-> [170, 190]) so we can process it as a continuous range.
    if bb_lon_max < bb_lon_min:
        bb_lon_max += 360.0

    if is_0_360:
        mapped_min = bb_lon_min % 360.0
        mapped_max = bb_lon_max % 360.0
        seam_val = 360.0
        min_val = 0.0
    else:
        mapped_min = ((bb_lon_min + 180.0) % 360.0) - 180.0
        mapped_max = ((bb_lon_max + 180.0) % 360.0) - 180.0
        seam_val = 180.0
        min_val = -180.0

    crosses_seam = (
        mapped_min > mapped_max
        or (bb_lon_max - bb_lon_min) >= 360.0
        or mapped_min == mapped_max
    )

    is_native = (is_0_360 and bb_lon_min >= 0 and bb_lon_max <= 360) or (
        not is_0_360 and bb_lon_min >= -180 and bb_lon_max <= 180
    )

    if is_native and not crosses_seam:
        # Normal case: no crossing, and already in native domain
        return dataset, lon

    parts = []

    if crosses_seam:
        # Select upper part
        ds_upper = dataset.sel({lon_var: slice(mapped_min, seam_val)})
        if ds_upper.sizes.get(lon_var, 0) > 0:
            shift_upper = math.floor((bb_lon_min - mapped_min) / 360.0 + 0.5) * 360.0
            if shift_upper != 0:
                ds_upper = ds_upper.assign_coords(
                    {lon_var: ds_upper[lon_var] + shift_upper}
                )
            parts.append(ds_upper)

        # Select lower part
        ds_lower = dataset.sel({lon_var: slice(min_val, mapped_max)})
        if ds_lower.sizes.get(lon_var, 0) > 0:
            shift_lower = math.floor((bb_lon_max - mapped_max) / 360.0 + 0.5) * 360.0
            if shift_lower != 0:
                ds_lower = ds_lower.assign_coords(
                    {lon_var: ds_lower[lon_var] + shift_lower}
                )
            parts.append(ds_lower)
    else:
        # Doesn't cross seam, but shifted
        ds_chunk = dataset.sel({lon_var: slice(mapped_min, mapped_max)})
        if ds_chunk.sizes.get(lon_var, 0) > 0:
            shift = math.floor((bb_lon_min - mapped_min) / 360.0 + 0.5) * 360.0
            if shift != 0:
                ds_chunk = ds_chunk.assign_coords({lon_var: ds_chunk[lon_var] + shift})
            parts.append(ds_chunk)

    if len(parts) > 1:
        dataset = xr.concat(parts, dim=lon_var)
    elif len(parts) == 1:
        dataset = parts[0]

    # The longitude selection has already been performed.
    # Disable lon filtering downstream so we don't re-filter on the new coordinates.
    if bbox is not None:
        bbox.has_bb_lon = False
        bbox.has_bb = bbox.has_bb_lat

    return dataset, lon


def apply_point_list_bounding_box(
    lons_vals, lats_vals, bbox: BBoxContext
) -> GridIndices:
    mask = np.ones(lons_vals.shape, dtype=bool)
    if bbox.has_bb_lon:
        bb_lon_min = bbox.lon_min if bbox.lon_min is not None else -np.inf
        bb_lon_max = bbox.lon_max if bbox.lon_max is not None else np.inf

        is_0_360 = float(np.min(lons_vals)) >= 0
        if bb_lon_max < bb_lon_min:
            bb_lon_max += 360.0

        if is_0_360:
            mapped_min = bb_lon_min % 360.0
            mapped_max = bb_lon_max % 360.0
            seam_val, min_val = 360.0, 0.0
        else:
            mapped_min = ((bb_lon_min + 180.0) % 360.0) - 180.0
            mapped_max = ((bb_lon_max + 180.0) % 360.0) - 180.0
            seam_val, min_val = 180.0, -180.0

        crosses_seam = mapped_min > mapped_max or (bb_lon_max - bb_lon_min) >= 360.0

        if crosses_seam:
            mask &= ((lons_vals >= mapped_min) & (lons_vals <= seam_val)) | (
                (lons_vals >= min_val) & (lons_vals <= mapped_max)
            )
        else:
            mask &= (lons_vals >= mapped_min) & (lons_vals <= mapped_max)
    if bbox.has_bb_lat:
        bb_lat_min = bbox.lat_min if bbox.lat_min is not None else -np.inf
        bb_lat_max = bbox.lat_max if bbox.lat_max is not None else np.inf
        mask &= (lats_vals >= bb_lat_min) & (lats_vals <= bb_lat_max)

    point_indices = np.nonzero(mask)[0]
    n_points = len(point_indices)
    if n_points == 0:
        raise exceptions.NoDataInSelection()

    return GridIndices(
        point_indices=point_indices, n_points=n_points, height_raw=n_points, width_raw=1
    )


def apply_regular_grid_bounding_box(
    lons_1d, lats_1d, bbox: BBoxContext, pad
) -> GridIndices:
    if bbox.has_bb_lon:
        lon_min_val, lon_max_val = lons_1d.min(), lons_1d.max()
        bb_lon_min = bbox.lon_min if bbox.lon_min is not None else -np.inf
        bb_lon_max = bbox.lon_max if bbox.lon_max is not None else np.inf
        if bb_lon_min > lon_max_val or bb_lon_max < lon_min_val:
            raise exceptions.NoDataInSelection()
    if bbox.has_bb_lat:
        lat_min_val, lat_max_val = lats_1d.min(), lats_1d.max()
        bb_lat_min = bbox.lat_min if bbox.lat_min is not None else -np.inf
        bb_lat_max = bbox.lat_max if bbox.lat_max is not None else np.inf
        if bb_lat_min > lat_max_val or bb_lat_max < lat_min_val:
            raise exceptions.NoDataInSelection()

    if bbox.has_bb_lon:
        idx_start = (np.abs(lons_1d - bb_lon_min)).argmin()
        idx_end = (np.abs(lons_1d - bb_lon_max)).argmin()
        i_min, i_max = min(idx_start, idx_end), max(idx_start, idx_end)
    else:
        i_min, i_max = 0, lons_1d.shape[0] - 1

    if bbox.has_bb_lat:
        idx_start = (np.abs(lats_1d - bb_lat_min)).argmin()
        idx_end = (np.abs(lats_1d - bb_lat_max)).argmin()
        j_min, j_max = min(idx_start, idx_end), max(idx_start, idx_end)
    else:
        j_min, j_max = 0, lats_1d.shape[0] - 1

    col_min = max(0, i_min - pad)
    col_max = min(lons_1d.shape[0] - 1, i_max + pad)
    row_min = max(0, j_min - pad)
    row_max = min(lats_1d.shape[0] - 1, j_max + pad)
    width_raw = col_max - col_min + 1
    height_raw = row_max - row_min + 1

    return GridIndices(
        col_min=int(col_min),
        col_max=int(col_max),
        row_min=int(row_min),
        row_max=int(row_max),
        width_raw=int(width_raw),
        height_raw=int(height_raw),
    )


def apply_irregular_bounding_box(
    lons_vals, lats_vals, bbox: BBoxContext, is_regular_grid, spatial_padding, pad
) -> tuple[np.ndarray, np.ndarray, GridIndices]:
    if is_regular_grid:
        lons_vals, lats_vals = np.meshgrid(lons_vals, lats_vals)

    if bbox.has_bb:
        mask = np.ones(lons_vals.shape, dtype=bool)
        # Compute lon_padding using the effective span of the bbox.
        # For a wrapped bbox (lon_min > lon_max, e.g. [170, -170]), compute the
        # positive span after unwrapping instead of using the raw difference which
        # would be negative and produce an inverted (shrinking) padding.
        if bbox.has_bb_lon and bbox.lon_min is not None and bbox.lon_max is not None:
            lon_span = bbox.lon_max - bbox.lon_min
            if lon_span < 0:
                lon_span += 360.0
            lon_padding = spatial_padding * lon_span
        else:
            lon_padding = 0
        lat_padding = (
            (spatial_padding * (bbox.lat_max - bbox.lat_min))
            if bbox.has_bb_lat and hasattr(bbox.lat_max, "__sub__")
            else 0
        )
        if bbox.has_bb_lon:
            bb_lon_min = (
                (bbox.lon_min - lon_padding) if bbox.lon_min is not None else -np.inf
            )
            bb_lon_max = (
                (bbox.lon_max + lon_padding) if bbox.lon_max is not None else np.inf
            )
            is_0_360 = float(np.min(lons_vals)) >= 0
            if bb_lon_max < bb_lon_min:
                bb_lon_max += 360.0

            if is_0_360:
                mapped_min = bb_lon_min % 360.0
                mapped_max = bb_lon_max % 360.0
                seam_val, min_val = 360.0, 0.0
            else:
                mapped_min = ((bb_lon_min + 180.0) % 360.0) - 180.0
                mapped_max = ((bb_lon_max + 180.0) % 360.0) - 180.0
                seam_val, min_val = 180.0, -180.0

            crosses_seam = mapped_min > mapped_max or (bb_lon_max - bb_lon_min) >= 360.0

            if crosses_seam:
                mask &= ((lons_vals >= mapped_min) & (lons_vals <= seam_val)) | (
                    (lons_vals >= min_val) & (lons_vals <= mapped_max)
                )
            else:
                mask &= (lons_vals >= mapped_min) & (lons_vals <= mapped_max)
        if bbox.has_bb_lat:
            bb_lat_min = (
                (bbox.lat_min - lat_padding) if bbox.lat_min is not None else -np.inf
            )
            bb_lat_max = (
                (bbox.lat_max + lat_padding) if bbox.lat_max is not None else np.inf
            )
            mask &= (lats_vals >= bb_lat_min) & (lats_vals <= bb_lat_max)

        if not np.any(mask):
            raise exceptions.NoDataInSelection()
        else:
            where_indices = np.nonzero(mask)
            rows, cols = where_indices[-2], where_indices[-1]
            row_min, row_max = rows.min(), rows.max()
            col_min, col_max = cols.min(), cols.max()
    else:
        row_min, row_max = 0, lons_vals.shape[-2] - 1
        col_min, col_max = 0, lons_vals.shape[-1] - 1

    height_raw = row_max - row_min + 1
    width_raw = col_max - col_min + 1

    indices = GridIndices(
        col_min=int(col_min),
        col_max=int(col_max),
        row_min=int(row_min),
        row_max=int(row_max),
        width_raw=int(width_raw),
        height_raw=int(height_raw),
    )
    return lons_vals, lats_vals, indices


def apply_resolution_limit(
    height_raw, width_raw, resolution_limit, is_point_list_res, n_points_res=0
):
    step_row, step_col = 1, 1
    if resolution_limit is not None:
        if is_point_list_res and n_points_res > resolution_limit:
            step_row = math.ceil(n_points_res / resolution_limit)
        else:
            if height_raw > resolution_limit:
                step_row = math.ceil(height_raw / resolution_limit)
            if width_raw > resolution_limit:
                step_col = math.ceil(width_raw / resolution_limit)
    return step_row, step_col


def apply_level_bounding_box_regular_grid(levels_1d, bbox):
    bb_level_min = bbox.level_min if bbox.level_min is not None else -np.inf
    bb_level_max = bbox.level_max if bbox.level_max is not None else np.inf
    level_mask = (levels_1d >= bb_level_min) & (levels_1d <= bb_level_max)
    if not np.any(level_mask):
        raise exceptions.NoDataInSelection()
    level_indices = np.nonzero(level_mask)[0]
    level_min, level_max = int(level_indices[0]), int(level_indices[-1])
    return level_min, level_max, levels_1d[level_min : level_max + 1]


def apply_level_bounding_box_irregular_grid(levels, bbox):
    bb_level_min = bbox.level_min if bbox.level_min is not None else -np.inf
    bb_level_max = bbox.level_max if bbox.level_max is not None else np.inf
    level_mask = (levels >= bb_level_min) & (levels <= bb_level_max)
    if not np.any(level_mask):
        raise exceptions.NoDataInSelection()
    return level_mask

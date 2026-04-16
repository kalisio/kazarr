import math

import numpy as np

from src import exceptions
from src.processing.contexts import BBoxContext, GridIndices


def apply_point_list_bounding_box(
    lons_vals, lats_vals, bbox: BBoxContext
) -> GridIndices:
    mask = np.ones(lons_vals.shape, dtype=bool)
    if bbox.has_bb_lon:
        bb_lon_min = bbox.lon_min if bbox.lon_min is not None else -np.inf
        bb_lon_max = bbox.lon_max if bbox.lon_max is not None else np.inf
        mask &= (lons_vals >= bb_lon_min) & (lons_vals <= bb_lon_max)
    if bbox.has_bb_lat:
        bb_lat_min = bbox.lat_min if bbox.lat_min is not None else -np.inf
        bb_lat_max = bbox.lat_max if bbox.lat_max is not None else np.inf
        mask &= (lats_vals >= bb_lat_min) & (lats_vals <= bb_lat_max)

    point_indices = np.where(mask)[0]
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


def apply_unstructured_bounding_box(
    lons_vals, lats_vals, bbox: BBoxContext, is_regular_grid, spatial_padding, pad
) -> tuple[np.ndarray, np.ndarray, GridIndices]:
    if is_regular_grid:
        lons_vals, lats_vals = np.meshgrid(lons_vals, lats_vals)

    if bbox.has_bb:
        mask = np.ones(lons_vals.shape, dtype=bool)
        lon_padding = (
            (spatial_padding * (bbox.lon_max - bbox.lon_min))
            if bbox.has_bb_lon and hasattr(bbox.lon_max, "__sub__")
            else 0
        )
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
            mask &= (lons_vals >= bb_lon_min) & (lons_vals <= bb_lon_max)
        if bbox.has_bb_lat:
            bb_lat_min = (
                (bbox.lat_min - lat_padding) if bbox.lat_min is not None else -np.inf
            )
            bb_lat_max = (
                (bbox.lat_max + lat_padding) if bbox.lat_max is not None else np.inf
            )
            mask &= (lats_vals >= bb_lat_min) & (lats_vals <= bb_lat_max)

        if not np.any(mask):
            center_lon = (bb_lon_min + bb_lon_max) / 2.0
            center_lat = (bb_lat_min + bb_lat_max) / 2.0
            dist = (lons_vals - center_lon) ** 2 + (lats_vals - center_lat) ** 2
            nearest_idx = np.argmin(dist)
            indices = np.unravel_index(nearest_idx, lons_vals.shape)
            nearest_row, nearest_col = indices[-2], indices[-1]
            fallback_pad = max(pad, 1)
            row_min = max(0, nearest_row - fallback_pad)
            row_max = min(lons_vals.shape[-2] - 1, nearest_row + fallback_pad)
            col_min = max(0, nearest_col - fallback_pad)
            col_max = min(lons_vals.shape[1] - 1, nearest_col + fallback_pad)
        else:
            rows, cols = np.where(mask)
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

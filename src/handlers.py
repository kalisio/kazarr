import math

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.interpolate import RegularGridInterpolator, RBFInterpolator

from src.utils import (
    dget,
    dgets,
    sel,
    load_datasets,
    load_dataset,
    get_required_dims_and_coords,
    get_bounded_time,
    extrapolate_edges_from_cell_data,
    StepDurationLogger,
)
from src import exceptions


def list_datasets(search_path=None):
    datasets = load_datasets(search_path)
    return {"datasets": datasets}


def dataset_infos(dataset_id):
    dataset, config = load_dataset(dataset_id)
    variables = {}
    coords = {}
    for var in dataset.data_vars:
        variables[var] = {
            "dims": dataset[var].dims,
            "shape": dataset[var].shape,
            "attrs": dataset[var].attrs,
        }
        if len(dataset[var].dims) == 0:
            variables[var]["value"] = dataset[var].values.item()
    for coord in dataset.coords:
        coords[coord] = {
            "dims": dataset[coord].dims,
            "shape": dataset[coord].shape,
            "attrs": dataset[coord].attrs,
        }
        if len(dataset[coord].dims) == 0:
            coords[coord]["value"] = dataset[coord].values.item()

    # Check if bounding box can be defined
    lon_var, lat_var, height_var = dgets(
        config, ["variables.lon", "variables.lat", "variables.height"]
    )
    bounding_box = False
    if lon_var in dataset and lat_var in dataset:
        bounding_box = {
            "lon": {
                "min": float(dataset[lon_var].min()),
                "max": float(dataset[lon_var].max()),
            },
            "lat": {
                "min": float(dataset[lat_var].min()),
                "max": float(dataset[lat_var].max()),
            },
        }
        if height_var in dataset:
            bounding_box["height"] = {
                "min": float(dataset[height_var].min()),
                "max": float(dataset[height_var].max()),
            }

    # Check if time bounds can be defined
    time_var = dget(config, "variables.time")
    time_bounds = False
    if (
        time_var in dataset
        and np.issubdtype(dataset[time_var].dtype, np.datetime64)
        and dataset[time_var].ndim == 1
    ):
        # Warning: we suppose here, that time is ordered (as it will most of the time be the case)
        # dataset[time_var].min()/max() will not work as Dask tries to add values, and its not supported for datetime64
        # np.min()/max() will load all data in memory, which is not optimized
        t_dim = dataset[time_var].dims[0]
        t_min = dataset[time_var].isel({t_dim: 0}).values
        t_max = dataset[time_var].isel({t_dim: -1}).values
        time_bounds = {
            "min": str(np.datetime_as_string(t_min)),
            "max": str(np.datetime_as_string(t_max)),
        }

    return {
        "id": dataset_id,
        "description": config.get("description", ""),
        "variables": variables,
        "coords": coords,
        "bounding_box": bounding_box if bounding_box else None,
        "time_bounds": time_bounds if time_bounds else None,
        "attrs": dataset.attrs,
    }


# format = "raw" | "geojson" | { "type": "mesh", shape: [row, col], interpolate: True, force_data_on_cells: True }
def extract(
    dataset,
    variable,
    request,
    time=None,
    bounding_box=None,
    resolution_limit=None,
    format="raw",
    interp_vars=[],
    time_interpolate=False,
    as_dims=[],
):
    lon_min, lat_min, lon_max, lat_max = (
        (None, None, None, None) if bounding_box is None else bounding_box
    )
    has_bb_lon = lon_min is not None or lon_max is not None
    has_bb_lat = lat_min is not None or lat_max is not None
    has_bb = has_bb_lon or has_bb_lat

    step_logger = StepDurationLogger(
        "extract",
        parameters=(
            dataset,
            variable,
            bounding_box,
            time,
            resolution_limit,
            format,
            interp_vars,
            time_interpolate,
            as_dims,
        ),
    )

    step_logger.step_start("Load dataset and config")
    dataset, config = load_dataset(dataset)

    fixed_coords, fixed_dims = dgets(
        config, ["variables.fixed", "dimensions.fixed"], {}
    )
    interp_vars = interp_vars or []

    if variable not in dataset:
        raise exceptions.VariableNotFound([variable])

    lon_var, lat_var = dgets(config, ["variables.lon", "variables.lat"])
    missing_vars = []
    if has_bb_lon and lon_var is None:
        raise exceptions.MissingConfigurationElement("variables.lon")
    if lon_var not in dataset:
        missing_vars.append(f"lon ({lon_var})")

    if has_bb_lat and lat_var is None:
        raise exceptions.MissingConfigurationElement("variables.lat")
    if lat_var not in dataset:
        missing_vars.append(f"lat ({lat_var})")

    time_var = dget(config, "variables.time")
    if time is not None and time_var is not None:
        if time_var not in dataset:
            missing_vars.append(f"time ({time_var})")
        else:
            fixed_coords[time_var] = get_bounded_time(dataset, time_var, time)
            if time_interpolate and time_var not in interp_vars:
                interp_vars.append(time_var)

    mesh_type = dget(config, "mesh_type", "auto")

    if len(missing_vars) > 0:
        raise exceptions.BadConfigurationVariable(missing_vars)

    out_type = format.get("type") if isinstance(format, dict) else format
    mesh_tile_shape = format.get("shape") if isinstance(format, dict) else None
    mesh_interpolate = format.get("interpolate") if isinstance(format, dict) else False
    force_data_mapping = (
        format.get("force_data_mapping") if isinstance(format, dict) else None
    )

    fixed_coords, fixed_dims = get_required_dims_and_coords(
        dataset,
        variable,
        fixed_coords,
        fixed_dims,
        request,
        interp_vars=interp_vars,
        optional_coords=[lon_var, lat_var],
        as_dims=as_dims,
    )

    lons = sel(dataset, lon_var, fixed_coords, fixed_dims)
    lats = sel(dataset, lat_var, fixed_coords, fixed_dims)

    is_regular_grid = lons.ndim == 1 and lats.ndim == 1 and lons.dims != lats.dims
    is_point_list = lons.ndim == 1 and lats.ndim == 1 and lons.dims == lats.dims
    pad = 2 if out_type == "mesh" else 0

    if is_point_list:
        step_logger.step_start("Point list: apply bounding box")
        lons_vals = lons.values
        lats_vals = lats.values

        mask = np.ones(lons_vals.shape, dtype=bool)
        if has_bb_lon:
            bb_lon_min = lon_min if lon_min is not None else -np.inf
            bb_lon_max = lon_max if lon_max is not None else np.inf
            mask &= (lons_vals >= bb_lon_min) & (lons_vals <= bb_lon_max)
        if has_bb_lat:
            bb_lat_min = lat_min if lat_min is not None else -np.inf
            bb_lat_max = lat_max if lat_max is not None else np.inf
            mask &= (lats_vals >= bb_lat_min) & (lats_vals <= bb_lat_max)

        point_indices = np.where(mask)[0]

        n_points = len(point_indices)
        if n_points == 0:
            raise exceptions.NoDataInSelection()

        height_raw, width_raw = n_points, 1
    elif is_regular_grid and has_bb:
        step_logger.step_start("Regular grid: apply bounding box")
        lons_1d = lons.values
        lats_1d = lats.values
        # Check if bounding box intersects data
        if has_bb_lon:
            lon_min_val, lon_max_val = lons_1d.min(), lons_1d.max()
            bb_lon_min = lon_min if lon_min is not None else -np.inf
            bb_lon_max = lon_max if lon_max is not None else np.inf
            if bb_lon_min > lon_max_val or bb_lon_max < lon_min_val:
                raise exceptions.NoDataInSelection()
        if has_bb_lat:
            lat_min_val, lat_max_val = lats_1d.min(), lats_1d.max()
            bb_lat_min = lat_min if lat_min is not None else -np.inf
            bb_lat_max = lat_max if lat_max is not None else np.inf
            if bb_lat_min > lat_max_val or bb_lat_max < lat_min_val:
                raise exceptions.NoDataInSelection()

        # Get indices in bounding box
        if has_bb_lon:
            idx_start = (np.abs(lons_1d - bb_lon_min)).argmin()
            idx_end = (np.abs(lons_1d - bb_lon_max)).argmin()
            i_min, i_max = min(idx_start, idx_end), max(idx_start, idx_end)
        else:
            i_min, i_max = 0, lons_1d.shape[0] - 1
        if has_bb_lat:
            idx_start = (np.abs(lats_1d - bb_lat_min)).argmin()
            idx_end = (np.abs(lats_1d - bb_lat_max)).argmin()
            j_min, j_max = min(idx_start, idx_end), max(idx_start, idx_end)
        else:
            j_min, j_max = 0, lats_1d.shape[0] - 1

        # Keep some padding to avoid edge effects during interpolation
        col_min = max(0, i_min - pad)
        col_max = min(lons_1d.shape[0] - 1, i_max + pad)
        row_min = max(0, j_min - pad)
        row_max = min(lats_1d.shape[0] - 1, j_max + pad)
        width_raw = col_max - col_min + 1
        height_raw = row_max - row_min + 1
    else:
        step_logger.step_start("Unstructured grid: apply bounding box")
        lons_vals = lons.values
        lats_vals = lats.values
        if is_regular_grid:
            lons_vals, lats_vals = np.meshgrid(lons_vals, lats_vals)

        if has_bb:
            mask = np.ones(lons_vals.shape, dtype=bool)
            if has_bb_lon:
                bb_lon_min = lon_min if lon_min is not None else -np.inf
                bb_lon_max = lon_max if lon_max is not None else np.inf
                mask &= (lons_vals >= bb_lon_min) & (lons_vals <= bb_lon_max)
            if has_bb_lat:
                bb_lat_min = lat_min if lat_min is not None else -np.inf
                bb_lat_max = lat_max if lat_max is not None else np.inf
                mask &= (lats_vals >= bb_lat_min) & (lats_vals <= bb_lat_max)

            if not np.any(mask):
                # Case where bounding box is smaller than grid mesh

                # Compute center of bounding box
                center_lon = (bb_lon_min + bb_lon_max) / 2.0
                center_lat = (bb_lat_min + bb_lat_max) / 2.0

                # Find index (i, j) of nearest point (squared Euclidean distance)
                dist = (lons_vals - center_lon) ** 2 + (lats_vals - center_lat) ** 2
                nearest_idx = np.argmin(dist)
                nearest_row, nearest_col = np.unravel_index(
                    nearest_idx, lons_vals.shape
                )

                fallback_pad = max(pad, 1)

                row_min = max(0, nearest_row - fallback_pad)
                row_max = min(lons_vals.shape[0] - 1, nearest_row + fallback_pad)
                col_min = max(0, nearest_col - fallback_pad)
                col_max = min(lons_vals.shape[1] - 1, nearest_col + fallback_pad)
            else:
                rows, cols = np.where(mask)
                row_min, row_max = (
                    max(0, rows.min() - pad),
                    min(lons_vals.shape[0] - 1, rows.max() + pad),
                )
                col_min, col_max = (
                    max(0, cols.min() - pad),
                    min(lons_vals.shape[1] - 1, cols.max() + pad),
                )
        else:
            row_min, row_max = 0, lons_vals.shape[0] - 1
            col_min, col_max = 0, lons_vals.shape[1] - 1
        height_raw = row_max - row_min + 1
        width_raw = col_max - col_min + 1

    if not is_point_list and (height_raw < 2 or width_raw < 2):
        # Not enough data to extract
        raise exceptions.TooFewPoints()

    # Determine step to apply to respect resolution limit
    step_row, step_col = 1, 1
    if resolution_limit is not None:
        if is_point_list and n_points > resolution_limit:
            step_row = math.ceil(n_points / resolution_limit)
            point_indices = point_indices[::step_row]
        else:
            if height_raw > resolution_limit:
                step_row = math.ceil(height_raw / resolution_limit)
            if width_raw > resolution_limit:
                step_col = math.ceil(width_raw / resolution_limit)

    # Load values
    # Apply bounding box and resolution limit
    # Now that values are sliced, with bounding box and resolution limit, we can load them in memory
    step_logger.step_start("Load variable values")
    vals_da = sel(dataset, variable, fixed_coords, fixed_dims, interp_vars=interp_vars)
    if is_point_list:
        vals = vals_da.values[point_indices]
    else:
        vals = vals_da[
            row_min : row_max + 1 : step_row, col_min : col_max + 1 : step_col
        ].values
    # Convert to float so we can assign NaN
    vals = vals.astype(float)

    step_logger.step_start("Crop latitude and longitude")
    lons_1d, lats_1d = None, None
    if is_point_list:
        lons = lons_vals[point_indices]
        lats = lats_vals[point_indices]
    elif is_regular_grid:
        lons_1d = lons[col_min : col_max + 1 : step_col]
        lats_1d = lats[row_min : row_max + 1 : step_row]
        lons, lats = np.meshgrid(lons_1d, lats_1d)
    else:
        lons = lons_vals[
            row_min : row_max + 1 : step_row, col_min : col_max + 1 : step_col
        ]
        lats = lats_vals[
            row_min : row_max + 1 : step_row, col_min : col_max + 1 : step_col
        ]

    if has_bb:
        mask_cropped = np.ones(lons.shape, dtype=bool)
        if has_bb_lon:
            bb_lon_min = lon_min if lon_min is not None else -np.inf
            bb_lon_max = lon_max if lon_max is not None else np.inf
            mask_cropped &= (lons >= bb_lon_min) & (lons <= bb_lon_max)
        if has_bb_lat:
            bb_lat_min = lat_min if lat_min is not None else -np.inf
            bb_lat_max = lat_max if lat_max is not None else np.inf
            mask_cropped &= (lats >= bb_lat_min) & (lats <= bb_lat_max)

        if out_type != "mesh":
            # Set values outside bounding box to NaN
            vals[~mask_cropped] = np.nan
    else:
        mask_cropped = None

    # Cell to point data conversion must be done before resampling
    cell_data = force_data_mapping != "vertices" and (
        force_data_mapping == "cells" or dget(config, "mesh_data_on_cells", False)
    )
    if cell_data and not is_point_list:
        step_logger.step_start("Cell to point data conversion")

        # TODO: how to know when mesh is radial ?
        lons_points, lats_points, height_points = extrapolate_edges_from_cell_data(
            lons, lats, None, "radial" if mesh_type == "radial" else "rectilinear"
        )

        temp_grid = pv.StructuredGrid(lons_points, lats_points, height_points)
        # Use "F" order to match 2D arrays with PyVista points (column-major order)
        temp_grid.cell_data[variable] = vals.ravel(order="F")
        temp_point_grid = temp_grid.cell_data_to_point_data()

        new_shape = lons_points.shape
        vals = temp_point_grid.point_data[variable].reshape(new_shape, order="F")
        lons, lats = lons_points, lats_points

        if is_regular_grid:
            lons_1d = lons[0, :]
            lats_1d = lats[:, 0]

    # Generate meshgrid with mesh_tile_shape if specified, and interpolate values if needed
    if mesh_tile_shape is not None:
        step_logger.step_start("Generate meshgrid")
        t_lon_min = lon_min if (bounding_box and lon_min is not None) else lons.min()
        t_lon_max = lon_max if (bounding_box and lon_max is not None) else lons.max()
        t_lat_min = lat_min if (bounding_box and lat_min is not None) else lats.min()
        t_lat_max = lat_max if (bounding_box and lat_max is not None) else lats.max()

        target_h, target_w = mesh_tile_shape
        interpolation_method = "linear" if mesh_interpolate else "nearest"

        xi = np.linspace(t_lon_min, t_lon_max, target_w)
        yi = np.linspace(t_lat_min, t_lat_max, target_h)

        xi_mesh, yi_mesh = np.meshgrid(xi, yi, indexing="ij")

        if is_regular_grid and lons_1d is not None and not is_point_list:
            # Use RegularGridInterpolator for better performance on regular grids
            try:
                step_logger.step_start("Interpolate regular grid")
                rgi = RegularGridInterpolator(
                    (lats_1d, lons_1d),
                    vals,
                    bounds_error=False,
                    method=interpolation_method,
                    fill_value=np.nan,
                )
                pts = np.stack([yi_mesh.ravel(), xi_mesh.ravel()], axis=-1)
                interpolated_vals = rgi(pts).reshape(xi_mesh.shape)
            except Exception as e:
                raise exceptions.GenericInternalError(f"Interpolation failed: {str(e)}")
        else:
            lons_flat = lons.ravel()
            lats_flat = lats.ravel()
            vals_flat = vals.ravel()

            # Use griddata for non regular grids (slower)
            valid_mask = np.isfinite(vals_flat)
            points = np.column_stack((lons_flat[valid_mask], lats_flat[valid_mask]))
            values = vals_flat[valid_mask]

            if points.shape[0] < 4:
                raise exceptions.TooFewPoints()

            # Remove duplicate points which can cause issues for interpolation (especially for RBFInterpolator)
            # Mainly for radial meshes for seam where first and last columns are the same
            points_uniques, indices_uniques = np.unique(
                points, axis=0, return_index=True
            )
            points = points_uniques
            values = values[indices_uniques]

            try:
                pts_target = np.column_stack((xi_mesh.ravel(), yi_mesh.ravel()))
                rbf = RBFInterpolator(points, values, kernel="linear")
                interpolated_vals = rbf(pts_target).reshape(xi_mesh.shape)
            except Exception as e:
                raise exceptions.GenericInternalError(f"Interpolation failed: {str(e)}")

        lons, lats, vals = xi_mesh, yi_mesh, interpolated_vals
        # Recreate mask_cropped based on NaNs from griddata
        mask_cropped = np.isfinite(vals)

    if out_type == "mesh":
        step_logger.step_start("Clean data for PyVista")

        # TODO Handle 3D
        z_zeros = np.zeros_like(lons)
        grid = pv.StructuredGrid(lons, lats, z_zeros)
        grid.point_data[variable] = vals.ravel(order="F")
        valid_mask = np.ones_like(grid.point_data[variable])
        if mask_cropped is not None:
            valid_mask *= mask_cropped.ravel(order="F").astype(float)

        valid_mask[np.isnan(grid.point_data[variable])] = 0.0
        grid.point_data["valid_mask"] = valid_mask
        try:
            thresholded = grid.threshold(0.5, scalars="valid_mask")
        except Exception as e:
            raise exceptions.GenericInternalError(str(e))

        if thresholded.n_points == 0 or thresholded.n_cells == 0:
            raise exceptions.NoDataInSelection()

        step_logger.step_start("Generate mesh data")
        tri_grid = thresholded.triangulate()
        # Remove unreferenced points to avoid artifacts in the output mesh (NaN values, duplicated vertices, etc.)
        tri_grid = tri_grid.clean()

        vertices = tri_grid.points.flatten()
        # As cells are stored as (N, 4) with first value being number of points per cell (3 for triangle)
        # we need to reshape and skip first values
        indices = tri_grid.cells.reshape((-1, 4))[:, 1:].flatten()
        values = tri_grid.point_data[variable]

        step_logger.step_start("Prepare output (mesh)")
        # This line is slow but as NaN are not supported in JSON, we need to convert them to None (null in JSON)
        clean_values = [float(v) if np.isfinite(v) else None for v in values]

        valid_numbers = values[np.isfinite(values)]
        if valid_numbers.size == 0:
            val_min, val_max = None, None
        else:
            val_min, val_max = float(valid_numbers.min()), float(valid_numbers.max())

        out = {
            "bounds": {"min": val_min, "max": val_max},
            "resolution_factor": {"row": step_row, "col": step_col},
            "vertices": vertices.tolist(),
            "indices": indices.tolist(),
            "values": clean_values,
        }
    else:
        flat_vals = vals.flatten()
        flat_lons = lons.flatten()
        flat_lats = lats.flatten()

        valid_vals = flat_vals[~np.isnan(flat_vals)]
        if valid_vals.size == 0:
            raise exceptions.NoDataInSelection()

        if out_type == "raw":
            step_logger.step_start("Prepare output (raw)")
            out = {
                "shape": vals.shape,
                "bounds": {
                    "min": np.min(valid_vals).item(),
                    "max": np.max(valid_vals).item(),
                },
                "resolution_factor": {"row": step_row, "col": step_col},
                "data": {
                    "longitudes": flat_lons.tolist(),
                    "latitudes": flat_lats.tolist(),
                    "values": [None if np.isnan(v) else v.item() for v in flat_vals],
                },
            }
        elif out_type == "geojson":
            step_logger.step_start("Prepare output (GeoJSON)")
            features = []
            for i in range(flat_vals.shape[0]):
                if not np.isnan(flat_vals[i]):
                    features.append(
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "Point",
                                "coordinates": [
                                    float(flat_lons[i]),
                                    float(flat_lats[i]),
                                ],
                            },
                            "properties": {"id": i, "value": float(flat_vals[i])},
                        }
                    )
            out = {
                "type": "FeatureCollection",
                "bounds": {
                    "min": np.min(valid_vals).item(),
                    "max": np.max(valid_vals).item(),
                },
                "resolution_factor": {"row": step_row, "col": step_col},
                "features": features,
            }

    step_logger.end()
    return out


def probe(
    dataset, variables, lon, lat, request, height=None, interpolate=True, as_dims=[]
):
    variables = variables if isinstance(variables, list) else [variables]

    dataset, config = load_dataset(dataset)
    with_height = height is not None

    fixed_coords, fixed_dims = dgets(
        config, ["variables.fixed", "dimensions.fixed"], {}
    )
    interp_vars = []

    lon_var, lat_var, height_var, time_var = dgets(
        config, ["variables.lon", "variables.lat", "variables.height", "variables.time"]
    )
    time_dim = dget(config, "dimensions.time")

    not_found_vars = []
    for var in variables:
        if var not in dataset:
            not_found_vars.append(var)
    if len(not_found_vars) > 0:
        raise exceptions.VariableNotFound(not_found_vars)

    missing_vars = []
    if lon_var is None or lon_var not in dataset:
        missing_vars.append(f"lon ({lon_var})")
    if lat_var is None or lat_var not in dataset:
        missing_vars.append(f"lat ({lat_var})")
    if with_height and height_var is None or height_var not in dataset:
        missing_vars.append(f"height ({height_var})")
    if len(missing_vars) > 0:
        raise exceptions.BadConfigurationVariable(missing_vars)

    longitudes = dataset[lon_var]
    latitudes = dataset[lat_var]
    if longitudes.ndim == 1 and latitudes.ndim == 1:  # Regular grid
        fixed_coords[lon_var] = lon
        fixed_coords[lat_var] = lat
        if interpolate:
            interp_vars.extend([lon_var, lat_var])
    else:  # Irregular grid
        if with_height:
            heights = dataset[height_var].values
            dist = np.sqrt(
                (longitudes - lon) ** 2
                + (latitudes - lat) ** 2
                + (heights - height) ** 2
            )
        else:
            dist = np.sqrt((longitudes - lon) ** 2 + (latitudes - lat) ** 2)
        dist_values = dist.values
        if interpolate:  # IDW interpolation
            k_neighbors = 4  # Number of neighbors to consider for interpolation
            flat_dist = dist_values.flatten()
            k_neighbors = min(k_neighbors, len(flat_dist))

            # Find indices of the k nearest neighbors
            min_indices = np.argpartition(flat_dist, k_neighbors - 1)[:k_neighbors]
            nearest_dists = flat_dist[min_indices]

            # Exact point case
            if np.any(nearest_dists == 0):
                weights = np.zeros(k_neighbors)
                weights[np.argmin(nearest_dists)] = 1.0
            else:
                weights = (1.0 / nearest_dists) / np.sum(1.0 / nearest_dists)

            # Flat indices to multi-dimensional indices
            neighbor_indices = [
                np.unravel_index(idx, dist_values.shape) for idx in min_indices
            ]

            # Temporarily set `fixed_dims` with the heaviest point so that `get_required_dims_and_coords` works normally.
            nearest_idx = neighbor_indices[np.argmax(weights)]
            for dim_name, indice in zip(dist.dims, nearest_idx):
                fixed_dims[dim_name] = indice
        else:
            min_idx_flat = np.argmin(dist_values)
            indices = np.unravel_index(min_idx_flat, dist_values.shape)
            for dim_name, indice in zip(dist.dims, indices):
                fixed_dims[dim_name] = indice

    # Time is optional, so add to both optional coords and dims, as one or the other may be defined in config
    optional_coords = [time_var] if time_var is not None else []
    optional_dims = [time_dim] if time_dim is not None else []
    fixed_coords, fixed_dims = get_required_dims_and_coords(
        dataset,
        variables,
        fixed_coords,
        fixed_dims,
        request,
        optional_coords=optional_coords,
        optional_dims=optional_dims,
        as_dims=as_dims,
    )

    # Get values for each variable
    data = {}
    for var in variables:
        if (
            longitudes.ndim != 1 or latitudes.ndim != 1
        ) and interpolate:  # Irregular grid with interpolation
            interpolated_values = None
            for idx, weight in zip(neighbor_indices, weights):
                neighbor_dims = fixed_dims.copy()
                for dim_name, indice in zip(dist.dims, idx):
                    neighbor_dims[dim_name] = indice

                neighbor_val = sel(
                    dataset, var, fixed_coords, neighbor_dims, interp_vars=interp_vars
                ).values

                if interpolated_values is None:
                    interpolated_values = neighbor_val * weight
                else:
                    interpolated_values += neighbor_val * weight

            data[var] = {
                "values": interpolated_values.tolist(),
                "attrs": dataset[var].attrs,
            }
        else:
            data[var] = {
                "values": sel(
                    dataset, var, fixed_coords, fixed_dims, interp_vars=interp_vars
                ).values.tolist(),
                "attrs": dataset[var].attrs,
            }

    # Get times list
    times = None
    if time_var is not None and time_var in dataset:
        times = sel(dataset, time_var, fixed_coords, fixed_dims).values
        if np.issubdtype(times.dtype, np.datetime64):
            times = [str(np.datetime_as_string(t)) for t in times]

    out = {"variables": data}
    if times is not None:
        out["times"] = times
    return out


def isoline(
    dataset,
    variable,
    levels,
    request,
    time=None,
    format="raw",
    time_interpolate=False,
    as_dims=[],
):
    dataset, config = load_dataset(dataset)

    fixed_coords, fixed_dims = dgets(
        config, ["variables.fixed", "dimensions.fixed"], {}
    )
    interp_vars = []

    if variable not in dataset:
        raise exceptions.VariableNotFound([variable])

    lon_var, lat_var = dgets(config, ["variables.lon", "variables.lat"])
    missing_vars = []
    if lon_var is None or lon_var not in dataset:
        missing_vars.append(f"lon ({lon_var})")
    if lat_var is None or lat_var not in dataset:
        missing_vars.append(f"lat ({lat_var})")
    if time is not None:
        time_var = dget(config, "variables.time")
        if time_var is None or time_var not in dataset:
            missing_vars.append(f"time ({time_var})")
        else:
            fixed_coords[time_var] = get_bounded_time(dataset, time_var, time)
            if time_interpolate:
                interp_vars.append(time_var)
    if len(missing_vars) > 0:
        raise exceptions.BadConfigurationVariable(missing_vars)

    fixed_coords, fixed_dims = get_required_dims_and_coords(
        dataset,
        variable,
        fixed_coords,
        fixed_dims,
        request,
        optional_coords=[lon_var, lat_var],
        as_dims=as_dims,
    )

    lon = sel(dataset, lon_var, fixed_coords, fixed_dims)
    lat = sel(dataset, lat_var, fixed_coords, fixed_dims)
    val = sel(dataset, variable, fixed_coords, fixed_dims, interp_vars=interp_vars)

    contours = plt.contour(lon, lat, val, levels=levels)
    isolines = []
    for paths in contours.get_paths():
        isolines.append(paths.vertices.tolist())

    if format == "raw":
        out = {}
        for i, level in enumerate(levels):
            out[level] = isolines[i]

        return out
    elif format == "geojson":
        features = []
        for i, level in enumerate(levels):
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [float(coord[0]), float(coord[1])] for coord in isolines[i]
                        ],
                    },
                    "properties": {"id": i, "level": level},
                }
            )
        return {"type": "FeatureCollection", "features": features}


def free_selection(dataset, variable, request, interp_vars=[], as_dims=[]):
    dataset, config = load_dataset(dataset)

    if variable not in dataset:
        raise exceptions.VariableNotFound([variable])

    for var in interp_vars:
        if var not in dataset:
            raise exceptions.VariableNotFound([var])

    fixed_coords, fixed_dims = dgets(
        config, ["variables.fixed", "dimensions.fixed"], {}
    )
    fixed_coords, fixed_dims = get_required_dims_and_coords(
        dataset,
        variable,
        fixed_coords,
        fixed_dims,
        request,
        interp_vars=interp_vars,
        optional_dims="*",
        as_dims=as_dims,
    )

    data = sel(dataset, variable, fixed_coords, fixed_dims, interp_vars).values.tolist()
    return {"data": data}


def mesh(dataset, format="mesh", force_data_mapping=None):
    step_logger = StepDurationLogger(
        "mesh",
        parameters=(dataset, format, force_data_mapping),
    )

    step_logger.step_start("Load dataset and config")
    dataset, config = load_dataset(dataset)

    lon_var, lat_var = dgets(config, ["variables.lon", "variables.lat"])
    missing_vars = []
    if lon_var not in dataset:
        missing_vars.append(f"lon ({lon_var})")
    if lat_var not in dataset:
        missing_vars.append(f"lat ({lat_var})")
    if len(missing_vars) > 0:
        raise exceptions.BadConfigurationVariable(missing_vars)

    lons = dataset[lon_var]
    lats = dataset[lat_var]

    is_regular_grid = lons.ndim == 1 and lats.ndim == 1 and lons.dims != lats.dims
    is_point_list = lons.ndim == 1 and lats.ndim == 1 and lons.dims == lats.dims

    if is_regular_grid:
        lons, lats = np.meshgrid(lons.values, lats.values)
        height_points = np.zeros_like(lons)

    cell_data = force_data_mapping != "vertices" and (
        force_data_mapping == "cells" or dget(config, "mesh_data_on_cells", False)
    )
    if cell_data and not is_point_list:
        step_logger.step_start("Cell to point data conversion")
        mesh_type = dget(config, "mesh_type", "auto")

        # TODO: how to know when mesh is radial ?
        lons_points, lats_points, height_points = extrapolate_edges_from_cell_data(
            lons, lats, None, "radial" if mesh_type == "radial" else "rectilinear"
        )
    else:
        lons_points, lats_points = lons.values, lats.values
        # TODO: handle 3D
        height_points = np.zeros_like(lons_points)

    if format == "geojson":
        step_logger.step_start("Prepare output (GeoJSON)")
        flat_lons = lons_points.flatten()
        flat_lats = lats_points.flatten()
        flat_heights = height_points.flatten()

        features = []
        for index, _ in np.ndenumerate(lons_points):
            feature_id = "_".join(map(str, index))
            flat_idx = np.ravel_multi_index(index, lons_points.shape)
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [
                            float(flat_lons[flat_idx]),
                            float(flat_lats[flat_idx]),
                            float(flat_heights[flat_idx]),
                        ],
                    },
                    "properties": {"id": feature_id},
                }
            )

        out = {"type": "FeatureCollection", "features": features}
    else:
        if is_point_list:
            points = np.column_stack((lons_points, lats_points, height_points))

            cloud = pv.PolyData(points)
            tri_grid = cloud.delaunay_2d()
            cells = tri_grid.faces
        else:
            lons_2d = (
                lons_points.squeeze()
                if lons_points.ndim == 3 and lons_points.shape[0] == 1
                else lons_points
            )
            lats_2d = (
                lats_points.squeeze()
                if lats_points.ndim == 3 and lats_points.shape[0] == 1
                else lats_points
            )
            height_2d = (
                height_points.squeeze()
                if height_points.ndim == 3 and height_points.shape[0] == 1
                else height_points
            )
            step_logger.step_start("Prepare output (mesh)")
            grid = pv.StructuredGrid(lons_2d, lats_2d, height_2d)
            tri_grid = grid.triangulate()
            cells = tri_grid.cells

        vertices = tri_grid.points.flatten()
        # As cells are stored as (N, 4) with first value being number of points per cell (3 for triangle)
        # we need to reshape and skip first values
        indices = cells.reshape((-1, 4))[:, 1:].flatten()

        out = {"vertices": vertices.tolist(), "indices": indices.tolist()}

    step_logger.end()
    return out

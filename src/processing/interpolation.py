import numpy as np
import pyvista as pv
from scipy.interpolate import (
    griddata,
    RegularGridInterpolator,
    RBFInterpolator,
)
from scipy.spatial import cKDTree


from src import exceptions
from src.processing.contexts import BBoxContext

def cell_to_point_conversion(lons, lats, vals, variable, mesh_type, is_regular_grid):
    lons_points, lats_points, height_points = extrapolate_edges_from_cell_data(
        lons, lats, None, "radial" if mesh_type == "radial" else "rectilinear"
    )
    temp_grid = pv.StructuredGrid(lons_points, lats_points, height_points)
    temp_grid.cell_data[variable] = vals.ravel(order="F")
    temp_point_grid = temp_grid.cell_data_to_point_data()
    new_shape = lons_points.shape
    vals = temp_point_grid.point_data[variable].reshape(new_shape, order="F")
    lons, lats = lons_points, lats_points
    lons_1d, lats_1d = None, None
    if is_regular_grid:
        lons_1d = lons[0, :]
        lats_1d = lats[:, 0]
    return lons, lats, vals, lons_1d, lats_1d


def generate_meshgrid_and_interpolate(
    lons,
    lats,
    vals,
    lons_1d,
    lats_1d,
    bbox: BBoxContext,
    target_w,
    target_h,
    is_regular_grid,
    is_point_list,
    interp_spatial_method,
    interp_spatial_params,
):
    t_lon_min = (
        bbox.lon_min if (bbox.has_bb and bbox.lon_min is not None) else lons.min()
    )
    t_lon_max = (
        bbox.lon_max if (bbox.has_bb and bbox.lon_max is not None) else lons.max()
    )
    t_lat_min = (
        bbox.lat_min if (bbox.has_bb and bbox.lat_min is not None) else lats.min()
    )
    t_lat_max = (
        bbox.lat_max if (bbox.has_bb and bbox.lat_max is not None) else lats.max()
    )

    xi = np.linspace(t_lon_min, t_lon_max, target_w)
    yi = np.linspace(t_lat_min, t_lat_max, target_h)
    xi_mesh, yi_mesh = np.meshgrid(xi, yi, indexing="ij")

    if is_regular_grid and lons_1d is not None and not is_point_list:
        if interp_spatial_method not in [
            "linear",
            "nearest",
            "slinear",
            "cubic",
            "quintic",
            "pchip",
        ]:
            interp_spatial_method = "linear"
        try:
            rgi = RegularGridInterpolator(
                (lats_1d, lons_1d),
                vals,
                bounds_error=False,
                method=interp_spatial_method,
                fill_value=np.nan,
            )
            pts = np.stack([yi_mesh.ravel(), xi_mesh.ravel()], axis=-1)
            interpolated_vals = rgi(pts).reshape(xi_mesh.shape)
        except Exception as e:
            raise exceptions.GenericInternalError(f"Interpolation failed: {str(e)}")
    else:
        try:
            interpolated_vals = apply_spatial_interpolation_irregular_grid(
                source_lons=lons,
                source_lats=lats,
                source_values=vals,
                target_lon_mesh=xi_mesh,
                target_lat_mesh=yi_mesh,
                method=interp_spatial_method,
                **interp_spatial_params,
            )
        except Exception as e:
            raise exceptions.GenericInternalError(f"Interpolation failed: {str(e)}")

    mask_cropped = np.isfinite(interpolated_vals)
    return xi_mesh, yi_mesh, interpolated_vals, mask_cropped


def extrapolate_edges_from_cell_data(
    lons, lats, heights=None, mesh_type="rectilinear", periodic_axes=None
):  # mesh_type="regular"|"rectilinear"|"radial"
    if periodic_axes is None:
        periodic_axes = []

    if mesh_type == "radial" and len(periodic_axes) == 0:
        periodic_axes = [1]

    def expand_axis(arr, axis):
        ndim = arr.ndim
        dim_size = arr.shape[axis]
        if dim_size == 1:
            return arr

        slice_left = [slice(None)] * ndim
        slice_left[axis] = slice(0, -1)
        slice_right = [slice(None)] * ndim
        slice_right[axis] = slice(1, None)

        midpoints = 0.5 * (arr[tuple(slice_left)] + arr[tuple(slice_right)])

        slice_first = [slice(None)] * ndim
        slice_first[axis] = slice(0, 1)

        slice_last = [slice(None)] * ndim
        slice_last[axis] = slice(-1, None)

        slice_mid_first = list(slice_first)
        slice_mid_last = list(slice_last)

        first_edge = arr[tuple(slice_first)] - (
            midpoints[tuple(slice_mid_first)] - arr[tuple(slice_first)]
        )
        last_edge = arr[tuple(slice_last)] + (
            arr[tuple(slice_last)] - midpoints[tuple(slice_mid_last)]
        )

        if axis in periodic_axes:
            return np.concatenate([first_edge, midpoints, first_edge], axis=axis)
        else:
            return np.concatenate([first_edge, midpoints, last_edge], axis=axis)

    x_cells = lons
    y_cells = lats
    z_cells = heights

    x_bounds = expand_axis(x_cells, 0)
    y_bounds = expand_axis(y_cells, 0)
    if heights is not None:
        z_bounds = expand_axis(z_cells, 0)
    else:
        z_bounds = np.zeros_like(x_bounds)

    if mesh_type != "regular":
        if heights is not None:
            x_bounds = expand_axis(x_bounds, 2)
            y_bounds = expand_axis(y_bounds, 2)
            z_bounds = expand_axis(z_bounds, 2)

        x_bounds = expand_axis(x_bounds, 1)
        y_bounds = expand_axis(y_bounds, 1)
        z_bounds = expand_axis(z_bounds, 1)

    return x_bounds, y_bounds, z_bounds


def apply_spatial_interpolation_irregular_grid(
    source_lons,
    source_lats,
    source_values,
    target_lon_mesh,
    target_lat_mesh,
    method="linear",
    **kwargs,
):
    points = np.column_stack((source_lons.ravel(), source_lats.ravel()))
    values = source_values.ravel()

    valid_mask = np.isfinite(values)
    points = points[valid_mask]
    values = values[valid_mask]

    if points.shape[0] < 4:
        raise exceptions.TooFewPoints()

    pts_target = np.column_stack((target_lon_mesh.ravel(), target_lat_mesh.ravel()))

    if method not in ["nearest", "linear", "cubic", "idw", "rbf"]:
        print(
            f"[Kazarr - Warning] Unsupported interpolation method: {method}. Falling back to linear."
        )
        method = "linear"

    if method in ["nearest", "linear", "cubic"]:
        interpolated_flat = griddata(
            points, values, pts_target, method=method, fill_value=np.nan
        )

    elif method == "idw":
        max_radius = kwargs.get("radius", 0.02)
        power = kwargs.get("power", 2.0)

        tree = cKDTree(points)
        indices_list = tree.query_ball_point(pts_target, r=max_radius)

        interpolated_flat = np.full(pts_target.shape[0], np.nan)

        for i, indices in enumerate(indices_list):
            if not indices:
                continue

            neighbors_coords = points[indices]
            dists = np.linalg.norm(neighbors_coords - pts_target[i], axis=1)

            zero_dist = dists < 1e-12
            if np.any(
                zero_dist
            ):  # Case where target point is exactly on a source point
                interpolated_flat[i] = values[np.array(indices)[zero_dist][0]]
            else:
                weights = 1.0 / (dists**power)
                interpolated_flat[i] = np.sum(weights * values[indices]) / np.sum(
                    weights
                )
    elif method == "rbf":
        # Remove duplicate points which can cause issues for interpolation (especially for RBFInterpolator)
        # Mainly for radial meshes for seam where first and last columns are the same
        points_uniques, indices_uniques = np.unique(points, axis=0, return_index=True)
        points = points_uniques
        values = values[indices_uniques]

        kernel = kwargs.get("kernel", "thin_plate_spline")
        smoothing = kwargs.get("smoothing", 0.0)

        rbf = RBFInterpolator(points, values, kernel=kernel, smoothing=smoothing)
        interpolated_flat = rbf(pts_target)

    else:
        raise exceptions.GenericInternalError(
            f"Unsupported interpolation method: {method}"
        )

    return interpolated_flat.reshape(target_lon_mesh.shape)

import numpy as np
import pyvista as pv


from src import exceptions


def prepare_mesh_output(lons, lats, vals, variable, mask_cropped, step_row, step_col):
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

    tri_grid = thresholded.triangulate()
    tri_grid = tri_grid.clean()

    vertices = tri_grid.points.flatten()
    indices = tri_grid.cells.reshape((-1, 4))[:, 1:].flatten()
    values = tri_grid.point_data[variable]

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
    return out


def prepare_raw_output(vals, lons, lats, step_row, step_col):
    flat_vals = vals.flatten()
    flat_lons = lons.flatten()
    flat_lats = lats.flatten()

    valid_vals = flat_vals[~np.isnan(flat_vals)]
    if valid_vals.size == 0:
        raise exceptions.NoDataInSelection()
    return {
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


def prepare_geojson_output(vals, lons, lats, step_row, step_col):
    flat_vals = vals.flatten()
    flat_lons = lons.flatten()
    flat_lats = lats.flatten()

    valid_vals = flat_vals[~np.isnan(flat_vals)]
    if valid_vals.size == 0:
        raise exceptions.NoDataInSelection()

    features = []
    for i in range(flat_vals.shape[0]):
        if not np.isnan(flat_vals[i]):
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(flat_lons[i]), float(flat_lats[i])],
                    },
                    "properties": {"id": i, "value": float(flat_vals[i])},
                }
            )
    return {
        "type": "FeatureCollection",
        "bounds": {
            "min": np.min(valid_vals).item(),
            "max": np.max(valid_vals).item(),
        },
        "resolution_factor": {"row": step_row, "col": step_col},
        "features": features,
    }

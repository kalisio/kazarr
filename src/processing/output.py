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


def prepare_output(var_names, vals, lons, lats, global_props=None, var_props=None):
    if global_props is None:
        global_props = {}
    if var_props is None:
        var_props = {}

    if isinstance(var_names, str):
        var_names = [var_names]
    if len(var_names) != len(vals):
        raise exceptions.GenericInternalError(
            "Length of var_names must match length of vals"
        )

    flat_lons = lons.flatten().tolist()
    flat_lats = lats.flatten().tolist()
    vals_dict = {}
    one_point = lons.shape[0] == 1 and lats.shape[0] == 1

    no_data = True
    out_vars_props = {}
    for i, var_name in enumerate(var_names):
        var_vals = vals[i].flatten()
        valid_vals = var_vals[~np.isnan(var_vals)]
        if valid_vals.size == 0:
            continue
        no_data = False
        var_vals = np.where(np.isnan(var_vals), None, var_vals).tolist()
        vals_dict[var_name] = var_vals
        out_vars_props[var_name] = {
            "bounds": {
                "min": np.min(valid_vals).item(),
                "max": np.max(valid_vals).item(),
            },
            **var_props.get(var_name, {}),
        }
    if no_data:
        raise exceptions.NoDataInSelection()

    return flat_lons, flat_lats, vals_dict, global_props, out_vars_props, one_point


def prepare_raw_output(var_names, vals, lons, lats, global_props=None, var_props=None):
    flat_lons, flat_lats, vals_dict, collection_props, out_props, _ = (
        prepare_output(
            var_names,
            vals,
            lons,
            lats,
            global_props=global_props,
            var_props=var_props,
        )
    )

    return {
        "shape": vals[0].shape,
        **collection_props,
        "variables": out_props,
        "longitudes": flat_lons,
        "latitudes": flat_lats,
        "values": {
            **vals_dict
        },
    }


def prepare_geojson_output(
    var_names, vals, lons, lats, collection_props=None, var_props=None
):
    flat_lons, flat_lats, vals_dict, collection_props, out_props, has_one_point = (
        prepare_output(
            var_names,
            vals,
            lons,
            lats,
            global_props=collection_props,
            var_props=var_props,
        )
    )

    features = []
    for i in range(len(flat_lons)):
        if has_one_point:
            out_vals = vals_dict
        else:
            out_vals = {
                var_name: var_vals[i] for var_name, var_vals in vals_dict.items()
            }
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(flat_lons[i]), float(flat_lats[i])],
                },
                "properties": {"id": i, **out_vals},
            }
        )

    return {
        "type": "FeatureCollection",
        "properties": {**collection_props, "variables": out_props},
        "features": features,
    }

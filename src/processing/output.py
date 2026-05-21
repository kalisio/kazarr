import numpy as np
import pyvista as pv


from src import exceptions


def prepare_mesh_output(
    lons, lats, zs, vals, variable, mask_cropped, step_row, step_col
):
    if zs is None:
        zs = np.zeros_like(lons)

    lons_flat = np.ravel(lons, order="F")
    lats_flat = np.ravel(lats, order="F")
    zs_flat = np.ravel(zs, order="F")
    vals_flat = np.ravel(vals, order="F")

    grid = pv.StructuredGrid()
    grid.points = np.column_stack((lons_flat, lats_flat, zs_flat))

    if lons.ndim == 2:
        grid.dimensions = [lons.shape[0], lons.shape[1], 1]
    elif lons.ndim == 3:
        grid.dimensions = [lons.shape[0], lons.shape[1], lons.shape[2]]
    else:
        grid.dimensions = [lons.shape[0], 1, 1]

    grid.point_data[variable] = vals_flat

    valid_mask = np.ones_like(vals_flat)
    if mask_cropped is not None:
        valid_mask *= np.ravel(mask_cropped, order="F").astype(float)

    valid_mask[np.isnan(vals_flat)] = 0.0
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
    cells = tri_grid.cells

    if tri_grid.n_cells > 0:
        cell_size = cells[0]
        indices = cells.reshape((-1, cell_size + 1))[:, 1:].flatten()
    else:
        indices = np.array([], dtype=int)

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


def prepare_output(
    var_names, vals, lons, lats, zs=None, global_props=None, var_props=None, has_time_dimension=False
):
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
    flat_zs = zs.flatten().tolist() if zs is not None else None
    vals_dict = {}
    has_one_point = lons.size == 1 and lats.size == 1

    no_data = True
    out_vars_props = {}
    for i, var_name in enumerate(var_names):
        var_vals = vals[i].flatten()
        valid_vals = var_vals[~np.isnan(var_vals)]
        if valid_vals.size == 0:
            continue
        no_data = False
        var_vals = np.where(np.isnan(var_vals), None, var_vals)
        var_vals = (
            var_vals.reshape(vals[0].shape[0], -1) if has_time_dimension else var_vals
        )
        vals_dict[var_name] = var_vals.tolist()
        out_vars_props[var_name] = var_props.get(var_name, {})
        out_vars_props[var_name]["bounds"] = {
            "min": float(valid_vals.min()),
            "max": float(valid_vals.max()),
        }
    if no_data:
        raise exceptions.NoDataInSelection()

    return (
        flat_lons,
        flat_lats,
        flat_zs,
        vals_dict,
        global_props,
        out_vars_props,
        has_one_point,
    )


def prepare_raw_output(
    var_names, vals, lons, lats, zs=None, global_props=None, var_props=None, has_time_dimension=False
):
    flat_lons, flat_lats, flat_zs, vals_dict, collection_props, out_props, _ = (
        prepare_output(
            var_names,
            vals,
            lons,
            lats,
            zs=zs,
            global_props=global_props,
            var_props=var_props,
            has_time_dimension=has_time_dimension,
        )
    )

    data = {
        "longitudes": flat_lons,
        "latitudes": flat_lats,
        "values": {**vals_dict},
    }
    if flat_zs is not None:
        data["heights"] = flat_zs

    return {
        "shape": vals[0].shape,
        **collection_props,
        "variables": out_props,
        **data,
    }


def prepare_geojson_output(
    var_names, vals, lons, lats, zs=None, collection_props=None, var_props=None, has_time_dimension=False
):
    (
        flat_lons,
        flat_lats,
        flat_zs,
        vals_dict,
        collection_props,
        out_props,
        has_one_point,
    ) = prepare_output(
        var_names,
        vals,
        lons,
        lats,
        zs=zs,
        global_props=collection_props,
        var_props=var_props,
        has_time_dimension=has_time_dimension,
    )

    features = []
    for i in range(len(flat_lons)):
        out_vals = {}
        for var_name, var_vals in vals_dict.items():
            if has_one_point and len(var_vals) > 1:
                # Time series or multiple values for a single point
                out_vals[var_name] = var_vals
            elif has_time_dimension:
                # Time series for multiple points
                out_vals[var_name] = [var_vals[j][i] for j in range(len(var_vals))]
            else:
                # Spatial data (one value per point) or single scalar
                out_vals[var_name] = var_vals[i]

        coordinates = [float(flat_lons[i]), float(flat_lats[i])]
        if flat_zs is not None and flat_zs[i] is not None and not np.isnan(flat_zs[i]):
            coordinates.append(float(flat_zs[i]))
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": coordinates,
                },
                "properties": {"id": i, **out_vals},
            }
        )

    return {
        "type": "FeatureCollection",
        "properties": {**collection_props, "variables": out_props},
        "features": features,
    }

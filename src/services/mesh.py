import numpy as np
import pyvista as pv

from src import exceptions
from src.schemas.config import MeshExtractionConfig
from src.utils.data import dget, dgets, get_dataset_height_vars, get_height_var
from src.utils.file import load_dataset
from src.utils.logging import StepLoggerAndAborter
from src.processing.interpolation import extrapolate_edges_from_cell_data
from src.processing.contexts import BBoxContext

from typing import Any, Dict, Union, Optional
import threading


def get_mesh(
    dataset_id: str,
    format: str = "mesh",
    config: Union[Dict[str, Any], MeshExtractionConfig, None] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:

    if not isinstance(config, MeshExtractionConfig):
        config = MeshExtractionConfig.model_validate(config or {})
    step_logger = StepLoggerAndAborter(
        "mesh", parameters=(dataset_id, format, config), cancel_event=cancel_event
    )
    force_data_mapping = config.mesh.data_mapping

    bounding_box = BBoxContext.from_tuple(config.bbox)

    step_logger.step_start("Load dataset and config")
    dataset, dataset_config = load_dataset(dataset_id)

    lon_var, lat_var= dgets(
        dataset_config, ["variables.lon", "variables.lat"]
    )
    missing_vars = []
    if lon_var not in dataset:
        missing_vars.append(f"lon ({lon_var})")
    if lat_var not in dataset:
        missing_vars.append(f"lat ({lat_var})")

    height_vars = get_dataset_height_vars(dataset, dataset_config)
    height_var = None
    if isinstance(height_vars, str) or height_vars is None:
        height_var = height_vars
    elif config.is_3d and config.variable is not None:
        height_var = get_height_var(dataset, dataset_config, config.variable)
    elif config.is_3d and config.height_variable is not None:
        if config.height_variable not in dataset:
            missing_vars.append(f"height ({config.height_variable})")
        else:
            height_var = config.height_variable
    if len(missing_vars) > 0:
        raise exceptions.BadConfigurationVariable(missing_vars)
    
    if height_var is None and config.is_3d:
        raise exceptions.CantFindHeightVariable()

    lons = dataset[lon_var]
    lats = dataset[lat_var]

    heights_da = (
        dataset[height_var]
        if height_var is not None and height_var in dataset
        else None
    )
    is_3d_dataset = heights_da is not None and heights_da.ndim == 1

    is_regular_grid = lons.ndim == 1 and lats.ndim == 1 and lons.dims != lats.dims
    is_point_list = lons.ndim == 1 and lats.ndim == 1 and lons.dims == lats.dims

    levels_1d = None
    if is_3d_dataset and config.is_3d and is_regular_grid:
        levels_1d = heights_da.values
        if bounding_box.has_bb_z:
            bb_z_min = bounding_box.z_min if bounding_box.z_min is not None else -np.inf
            bb_z_max = bounding_box.z_max if bounding_box.z_max is not None else np.inf
            z_mask = (levels_1d >= bb_z_min) & (levels_1d <= bb_z_max)
            if not np.any(z_mask):
                raise exceptions.NoDataInSelection()
            levels_1d = levels_1d[z_mask]

    if is_regular_grid:
        lons_vals = lons.values
        lats_vals = lats.values
        if config.is_3d and levels_1d is not None:
            lons_points, lats_points, height_points = np.meshgrid(
                lons_vals, lats_vals, levels_1d, indexing="ij"
            )
        else:
            lons_points, lats_points = np.meshgrid(lons_vals, lats_vals, indexing="ij")
            height_points = np.zeros_like(lons_points)
    else:
        lons_points = lons.values
        lats_points = lats.values
        if is_3d_dataset and heights_da is not None:
            height_points = heights_da.values
        else:
            height_points = np.zeros_like(lons_points)

    cell_data = force_data_mapping != "vertices" and (
        force_data_mapping == "cells"
        or dget(dataset_config, "mesh_data_on_cells", False)
    )
    if cell_data and not is_point_list:
        step_logger.step_start("Cell to point geometry conversion")
        mesh_type = dget(dataset_config, "mesh_type", "auto")
        lons_points, lats_points, height_points = extrapolate_edges_from_cell_data(
            lons_points,
            lats_points,
            height_points if (is_3d_dataset and config.is_3d) else None,
            "radial" if mesh_type == "radial" else "rectilinear",
        )

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
        step_logger.step_start("Prepare output (mesh)")
        if is_point_list:
            points = np.column_stack((lons_points, lats_points, height_points))
            cloud = pv.PolyData(points)
            tri_grid = cloud.delaunay_2d()
            cells = tri_grid.faces
            vertices = tri_grid.points.flatten()
            indices = cells.reshape((-1, 4))[:, 1:].flatten()
        elif config.is_3d and lons_points.ndim == 3:
            grid = pv.StructuredGrid(lons_points, lats_points, height_points)
            tri_grid = grid.triangulate()
            tri_grid = tri_grid.clean()
            vertices = tri_grid.points.flatten()
            cells = tri_grid.cells
            if tri_grid.n_cells > 0:
                cell_size = cells[0]
                indices = cells.reshape((-1, cell_size + 1))[:, 1:].flatten()
            else:
                indices = np.array([], dtype=int)
        else:
            lons_2d = (
                lons_points.squeeze()
                if lons_points.ndim == 3 and 1 in lons_points.shape
                else lons_points
            )
            lats_2d = (
                lats_points.squeeze()
                if lats_points.ndim == 3 and 1 in lats_points.shape
                else lats_points
            )
            height_2d = (
                height_points.squeeze()
                if height_points.ndim == 3 and 1 in height_points.shape
                else height_points
            )
            grid = pv.StructuredGrid(lons_2d, lats_2d, height_2d)
            tri_grid = grid.triangulate()
            cells = tri_grid.cells
            vertices = tri_grid.points.flatten()
            indices = cells.reshape((-1, 4))[:, 1:].flatten()

        out = {
            "vertices": vertices.tolist(),
            "indices": indices.tolist(),
            "is_3d": config.is_3d,
        }

    step_logger.end()
    return out

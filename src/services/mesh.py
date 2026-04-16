import numpy as np
import pyvista as pv

from src import exceptions
from src.schemas.config import ExtractionConfig
from src.utils.data import dget, dgets
from src.utils.file import load_dataset
from src.utils.logging import StepDurationLogger
from src.processing.interpolation import extrapolate_edges_from_cell_data

from typing import Any, Dict, Union


def get_mesh(
    dataset_id: str,
    format: str = "mesh",
    config: Union[Dict[str, Any], ExtractionConfig, None] = None,
) -> Dict[str, Any]:
    if not isinstance(config, ExtractionConfig):
        config = ExtractionConfig.model_validate(config or {})
    step_logger = StepDurationLogger("mesh", parameters=(dataset_id, format, config))
    force_data_mapping = config.mesh.data_mapping

    step_logger.step_start("Load dataset and config")
    dataset, dataset_config = load_dataset(dataset_id)

    lon_var, lat_var = dgets(dataset_config, ["variables.lon", "variables.lat"])
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
        force_data_mapping == "cells"
        or dget(dataset_config, "mesh_data_on_cells", False)
    )
    if cell_data and not is_point_list:
        step_logger.step_start("Cell to point data conversion")
        mesh_type = dget(dataset_config, "mesh_type", "auto")
        lons_points, lats_points, height_points = extrapolate_edges_from_cell_data(
            lons, lats, None, "radial" if mesh_type == "radial" else "rectilinear"
        )
    else:
        lons_points, lats_points = lons.values, lats.values
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
        indices = cells.reshape((-1, 4))[:, 1:].flatten()

        out = {"vertices": vertices.tolist(), "indices": indices.tolist()}

    step_logger.end()
    return out

from fastapi import APIRouter, Depends, Query, Request
from starlette.concurrency import run_in_threadpool
import threading
import asyncio

import src.schemas.requests as models
from src.services import extraction
from src.utils.data import parse_query_dict
from src.utils.requests import watch_disconnection
import src.exceptions as exceptions

router = APIRouter(tags=["Extraction"])


@router.get("/datasets/{dataset:path}/extract", summary="Get data at a specific time")
async def extract_data(
    request: Request,
    base: models.MeshFormatParams = Depends(),
    bbox: models.BBoxParams = Depends(),
    time: models.TimeParams = Depends(),
    mesh: models.MeshParams = Depends(),
    spatial_interp: models.SpatialInterpolationParams = Depends(),
    resolution_limit: float | None = Query(
        None, description="The resolution limit for data extraction"
    ),
    is_3d: bool = Query(
        False,
        description="If True, performs a full 3D volume extraction. If False (default) and the dataset is 3D, a vertical coordinate must be provided.",
    ),
):
    interp_vars_params = base.interp_vars_params
    if interp_vars_params is not None and ":" in interp_vars_params:
        interp_vars_params = parse_query_dict(interp_vars_params)

    interp_spatial_params = spatial_interp.interp_spatial_params
    if interp_spatial_params is not None and ":" in interp_spatial_params:
        interp_spatial_params = parse_query_dict(interp_spatial_params)

    if base.variable is None:
        raise exceptions.MissingQueryParameter("variable")

    config = {
        "bbox": (
            bbox.lon_min,
            bbox.lat_min,
            bbox.lon_max,
            bbox.lat_max,
            bbox.z_min,
            bbox.z_max,
        ),
        "is_3d": is_3d,
        "as_dims": base.as_dims,
        "resolution_limit": resolution_limit,
        "mesh": {
            "tile_shape": (mesh.mesh_tile_size, mesh.mesh_tile_size)
            if mesh.mesh_tile_size is not None
            else None,
            "data_mapping": mesh.mesh_data_mapping,
        },
        "interpolation": {
            "vars": {
                "items": base.interp_vars,
                "time": time.interp_time,
                "method": base.interp_vars_method,
                "params": interp_vars_params,
            },
            "spatial": {
                "method": spatial_interp.interp_spatial_method,
                "params": interp_spatial_params,
            },
        },
    }

    cancel_event = threading.Event()
    watcher_task = asyncio.create_task(watch_disconnection(request, cancel_event))
    try:
        return await run_in_threadpool(
            extraction.extract,
            request,
            base.dataset,
            base.variable,
            time_range=time.time,
            format=base.format,
            config=config,
            cancel_event=cancel_event,
        )
    finally:
        watcher_task.cancel()

from fastapi import APIRouter, Query, Path, Request
from starlette.concurrency import run_in_threadpool
import asyncio
import threading

from src.utils.requests import watch_disconnection
from src.services import mesh as mesh_service

router = APIRouter(tags=["Mesh"])


@router.get(
    "/datasets/{dataset:path}/mesh", summary="Get mesh representation of the dataset"
)
async def mesh(
    request: Request,
    dataset: str = Path(..., description="The path to the dataset"),
    format: str = Query(
        "mesh",
        description="The format of the extracted data (Currently supported: 'mesh', 'geojson')",
    ),
    mesh_data_mapping: str | None = Query(
        None,
        description="Whether the data of the mesh is on cells or on vertices. This will override the dataset configuration. (Supported values: 'vertices', 'cells')",
    ),
    is_3d: bool = Query(
        False,
        description="If True, generates a 3D volumetric mesh using the vertical coordinate defined in the dataset configuration if the dataset use a unique one, otherwise, see 'variable' and 'height_variable' parameters.",
    ),
    variable: str | None = Query(
        None,
        description="The variable to base the mesh geometry on. Not mandatory if the dataset use a unique vertical coordinate.",
    ),
    height_variable: str | None = Query(
        None,
        description="The variable to use as height coordinate for the mesh geometry. This will override the dataset configuration and the 'variable' parameter.",
    ),
):
    config = {
        "is_3d": is_3d,
        "variable": variable,
        "height_variable": height_variable,
    }
    if mesh_data_mapping is not None:
        config["mesh"] = {"data_mapping": mesh_data_mapping}
    cancel_event = threading.Event()
    watcher_task = asyncio.create_task(watch_disconnection(request, cancel_event))
    try:
        return await run_in_threadpool(
            mesh_service.get_mesh,
            dataset,
            format=format,
            config=config,
            cancel_event=cancel_event,
        )
    finally:
        watcher_task.cancel()

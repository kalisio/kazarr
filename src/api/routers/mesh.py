import asyncio
import threading
from typing import Annotated

from fastapi import APIRouter, Path, Query, Request
from starlette.concurrency import run_in_threadpool

from src.services import mesh as mesh_service
from src.utils.requests import watch_disconnection

router = APIRouter(tags=["Mesh"])


@router.get(
    "/datasets/{dataset:path}/mesh", summary="Get mesh representation of the dataset"
)
async def mesh(
    request: Request,
    dataset: Annotated[str, Path(description="The path to the dataset")],
    format: Annotated[
        str,
        Query(
            description="The format of the extracted data (Currently supported: 'mesh', 'geojson')"
        ),
    ] = "mesh",
    mesh_data_mapping: Annotated[
        str | None,
        Query(
            description="Whether the data of the mesh is on cells or on vertices. This will override the dataset configuration. (Supported values: 'vertices', 'cells')",
        ),
    ] = None,
    is_3d: Annotated[
        bool,
        Query(
            description="If True, generates a 3D volumetric mesh using the vertical coordinate defined in the dataset configuration if the dataset use a unique one, otherwise, see 'variable' and 'level_variable' parameters.",
        ),
    ] = False,
    variable: Annotated[
        str | None,
        Query(
            description="The variable to base the mesh geometry on. Not mandatory if the dataset use a unique vertical coordinate."
        ),
    ] = None,
    level_variable: Annotated[
        str | None,
        Query(
            description="The variable to use as level coordinate for the mesh geometry. This will override the dataset configuration and the 'variable' parameter.",
        ),
    ] = None,
):
    config = {
        "is_3d": is_3d,
        "variable": variable,
        "level_variable": level_variable,
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

from fastapi import APIRouter, Query, Path
from starlette.concurrency import run_in_threadpool

from src.services import mesh as mesh_service

router = APIRouter(tags=["Mesh"])


@router.get(
    "/datasets/{dataset:path}/mesh", summary="Get mesh representation of the dataset"
)
async def mesh(
    dataset: str = Path(..., description="The path to the dataset"),
    format: str = Query(
        "mesh",
        description="The format of the extracted data (Currently supported: 'mesh', 'geojson')",
    ),
    mesh_data_mapping: str | None = Query(
        None,
        description="Whether the data of the mesh is on cells or on vertices. This will override the dataset configuration. (Supported values: 'vertices', 'cells')",
    ),
):
    config = {}
    if mesh_data_mapping is not None:
        config["mesh"] = {"data_mapping": mesh_data_mapping}
    return await run_in_threadpool(
        mesh_service.get_mesh, dataset, format=format, config=config
    )

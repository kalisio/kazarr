from fastapi import APIRouter, Query, Request
from fastapi.responses import RedirectResponse
from starlette.concurrency import run_in_threadpool

import src.services.dataset as dataset_service

router = APIRouter(tags=["Dataset"])


@router.get("/datasets", summary="List available datasets")
async def list_datasets(
    search_path: str = Query(None, description="The path to search for datasets"),
):
    return await run_in_threadpool(dataset_service.list_datasets, search_path)


@router.get("/datasets/", include_in_schema=False)
async def redirect_datasets(request: Request):
    url = request.url
    new_url = url.replace(path="/datasets")
    return RedirectResponse(url=new_url, status_code=301)


@router.get("/datasets/{dataset:path}/metadata", summary="Get dataset information")
async def dataset_metadata(dataset: str):
    return await run_in_threadpool(dataset_service.dataset_metadata, dataset)

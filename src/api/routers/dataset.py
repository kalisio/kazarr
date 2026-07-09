from email.utils import format_datetime, parsedate_to_datetime
import datetime as dt

from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import RedirectResponse
from starlette.concurrency import run_in_threadpool

import src.services.dataset as dataset_service
import src.utils.file as file_utils

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
async def dataset_metadata(dataset: str, request: Request, response: Response):
    # 304 Not Modified handling so that clients can know when a dataset has been updated
    # For now, only works for datasets stored on S3
    last_modified = await run_in_threadpool(
        file_utils.get_dataset_last_modified, dataset
    )

    if last_modified:
        # If last_modified is a string (can happen with some s3fs versions), parse it
        if isinstance(last_modified, str):
            try:
                # Attempt to parse ISO string
                last_modified = dt.datetime.fromisoformat(
                    last_modified.replace("Z", "+00:00")
                )
            except ValueError:
                pass

        # Ensure it's a timezone-aware UTC datetime for format_datetime(usegmt=True)
        if isinstance(last_modified, dt.datetime):
            if last_modified.tzinfo is not None:
                last_modified = last_modified.astimezone(dt.timezone.utc)
            else:
                last_modified = last_modified.replace(tzinfo=dt.timezone.utc)

            # Check the If-Modified-Since header
            if_modified_since = request.headers.get("if-modified-since")
            if if_modified_since:
                try:
                    ims_dt = parsedate_to_datetime(if_modified_since)
                    if last_modified <= ims_dt:
                        return Response(status_code=304)
                except (TypeError, ValueError):
                    pass

            # Add Last-Modified header to the response
            response.headers["Last-Modified"] = format_datetime(
                last_modified, usegmt=True
            )
            response.headers["Cache-Control"] = "no-cache"

    return await run_in_threadpool(dataset_service.dataset_metadata, dataset)

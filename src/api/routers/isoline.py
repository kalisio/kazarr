from fastapi import APIRouter, Depends, Query, Request
from starlette.concurrency import run_in_threadpool

import src.schemas.requests as models
from src.services import isoline as isoline_service
from src.utils.data import parse_query_dict
import src.exceptions as exceptions


router = APIRouter(tags=["Isoline"])


@router.get(
    "/datasets/{dataset:path}/isoline",
    summary="Get isolines for a specific variable at a specific time",
)
async def isoline_data(
    request: Request,
    base: models.BaseParams = Depends(),
    time: models.TimeParams = Depends(),
    levels: list[float] = Query(
        ..., description="List of levels for isoline generation"
    ),
):
    interp_vars_params = base.interp_vars_params
    if interp_vars_params is not None and ":" in interp_vars_params:
        interp_vars_params = parse_query_dict(interp_vars_params)

    if base.variable is None:
        raise exceptions.MissingQueryParameter("variable")

    config = {
        "as_dims": base.as_dims,
        "interpolation": {
            "vars": {
                "items": base.interp_vars,
                "time": time.interp_time,
                "method": base.interp_vars_method,
                "params": interp_vars_params,
            }
        },
    }

    return await run_in_threadpool(
        isoline_service.isoline,
        request,
        base.dataset,
        base.variable,
        levels,
        time=time.time,
        format=base.format,
        config=config,
    )

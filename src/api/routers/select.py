from fastapi import APIRouter, Depends, Request
from starlette.concurrency import run_in_threadpool

import src.schemas.requests as models
from src.services import extraction
from src.utils.data import parse_query_dict
import src.exceptions as exceptions


router = APIRouter(tags=["Select"])


@router.get(
    "/datasets/{dataset:path}/select",
    summary="Get data for free selection of dimensions and coordinates",
)
async def free_selection_data(
    request: Request,
    base: models.BaseParams = Depends(),
    time: models.TimeParams = Depends(),
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
        extraction.free_selection,
        request,
        base.dataset,
        base.variable,
        config=config,
    )

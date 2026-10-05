from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request, Response
from starlette.concurrency import run_in_threadpool

import src.schemas.requests as models
from src import exceptions
from src.processing.isoline import parse_thresholds
from src.services import isoline as isoline_service
from src.utils.data import parse_query_dict

router = APIRouter(tags=["Isoline"])


@router.get(
    "/datasets/{dataset:path}/isoline",
    summary="Get isolines for a specific variable at a specific time",
)
async def isoline_data(
    request: Request,
    base: Annotated[models.BaseParams, Depends()],
    time: Annotated[models.TimeParams, Depends()],
    bbox: Annotated[models.BBoxParams, Depends()],
    thresholds: Annotated[
        list[str],
        Query(
            description="Thresholds for isoline generation: either a list of values "
            "(thresholds=0&thresholds=5&thresholds=10), or a range 'min:max:step' where min "
            "and max are optional and taken from the data when omitted (e.g. '::5', '0::5', ':30:5')."
        ),
    ],
):
    interp_vars_params = base.interp_vars_params
    if interp_vars_params is not None and ":" in interp_vars_params:
        interp_vars_params = parse_query_dict(interp_vars_params)

    if base.variable is None:
        raise exceptions.MissingQueryParameter("variable")

    config = {
        "bbox": (bbox.lon_min, bbox.lat_min, bbox.lon_max, bbox.lat_max),
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

    content = await run_in_threadpool(
        isoline_service.isoline,
        request,
        base.dataset,
        base.variable,
        parse_thresholds(thresholds),
        time=time.time,
        format=base.format,
        config=config,
    )
    return Response(content=content, media_type="application/json")

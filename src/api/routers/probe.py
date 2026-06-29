from fastapi import APIRouter, Body, Depends, Query, Request
from starlette.concurrency import run_in_threadpool
import threading
import asyncio

import src.schemas.requests as models
from src.services import extraction
from src.utils.data import parse_query_dict
from src.utils.requests import watch_disconnection
import src.exceptions as exceptions


router = APIRouter(tags=["Probe"])


@router.get(
    "/datasets/{dataset:path}/probe",
    summary="Get data at specific coordinates over time",
)
async def probe_data(
    request: Request,
    base: models.MultipleVariablesParams = Depends(),
    lon: float = Query(..., description="The longitude coordinate to probe"),
    lat: float = Query(..., description="The latitude coordinate to probe"),
    level: float | None = Query(None, description="The level coordinate to probe"),
    time: models.MultiTimeParams = Depends(),
    spatial_interp: models.SpatialInterpolationParams = Depends(),
):
    interp_vars_params = base.interp_vars_params
    if interp_vars_params is not None and ":" in interp_vars_params:
        interp_vars_params = parse_query_dict(interp_vars_params)

    interp_spatial_params = spatial_interp.interp_spatial_params
    if interp_spatial_params is not None and ":" in interp_spatial_params:
        interp_spatial_params = parse_query_dict(interp_spatial_params)

    if base.variables is None and base.variable is None:
        raise exceptions.MissingQueryParameter("variables")
    if base.variables is not None and base.variable is not None:
        raise exceptions.UserInputBasedException(
            "INVALID_QUERY_PARAMETERS",
            "Cannot specify both 'variable' and 'variables' parameters. Please use 'variables' for probe endpoint.",
        )

    variables = [base.variable] if base.variable is not None else base.variables

    config = {
        "as_dims": base.as_dims,
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
            extraction.probe,
            request,
            base.dataset,
            variables,
            lon,
            lat,
            level=level,
            time_range=time.times if time.times else time.time,
            format=base.format,
            config=config,
            cancel_event=cancel_event,
        )
    finally:
        watcher_task.cancel()


@router.post(
    "/datasets/{dataset:path}/probes",
    summary="Get data at multiple coordinates (batch probe)",
)
async def probe_data_multi(
    request: Request,
    base: models.MultipleVariablesParams = Depends(),
    body: models.MultiProbeBody = Body(...),
    time: models.MultiTimeParams = Depends(),
    spatial_interp: models.SpatialInterpolationParams = Depends(),
):
    interp_vars_params = base.interp_vars_params
    if interp_vars_params is not None and ":" in interp_vars_params:
        interp_vars_params = parse_query_dict(interp_vars_params)

    interp_spatial_params = spatial_interp.interp_spatial_params
    if interp_spatial_params is not None and ":" in interp_spatial_params:
        interp_spatial_params = parse_query_dict(interp_spatial_params)

    if base.variables is None and base.variable is None:
        raise exceptions.MissingQueryParameter("variables")
    if base.variables is not None and base.variable is not None:
        raise exceptions.UserInputBasedException(
            "INVALID_QUERY_PARAMETERS",
            "Cannot specify both 'variable' and 'variables' parameters. Please use 'variables' for probe endpoint.",
        )

    variables = [base.variable] if base.variable is not None else base.variables

    config = {
        "as_dims": base.as_dims,
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

    if body.times:
        times = body.times
    elif time.times:
        times = time.times
    else:
        times = time.time

    cancel_event = threading.Event()
    watcher_task = asyncio.create_task(watch_disconnection(request, cancel_event))
    try:
        return await run_in_threadpool(
            extraction.multi_probe,
            request,
            base.dataset,
            variables,
            body.points,
            time_range=times,
            format=base.format,
            config=config,
            cancel_event=cancel_event,
        )
    finally:
        watcher_task.cancel()

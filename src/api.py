import sys
import os
# import logging
# import time

from fastapi import FastAPI, Path, Query, Request, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Literal

import src.handlers as handlers
import src.exceptions as exceptions
from src.utils import parse_query_dict

# logging.basicConfig(
#     level=logging.INFO,
#     # %(process)d insère automatiquement le PID
#     format="%(asctime)s - [Worker %(process)d] - %(levelname)s - %(message)s",
#     datefmt="%H:%M:%S"
# )

# logger = logging.getLogger(__name__)

app = FastAPI(
    title="kazarr API",
    version=os.getenv("APP_VERSION", "0.1.0"),
    description="A lightweight FastAPI service that exposes endpoints to interact with Zarr datasets stored in a Simple Storage Service (S3)",
    contact={
        "name": "Kalisio",
        "url": "https://kalisio.xyz",
        "email": "contact@kalisio.xyz",
    },
    docs_url=None,
)

# @app.middleware("http")
# async def log_requests(request: Request, call_next):
#     start_time = time.perf_counter()
#     response = await call_next(request)
#     process_time = (time.perf_counter() - start_time) * 1000
#     logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.2f}ms")
#     return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get(
    "/",
    summary="API Root",
    description="Provides basic information about the kazarr API.",
)
def read_root():
    return {
        "name": "kazarr API",
        "version": os.getenv("APP_VERSION", "0.1.0"),
        "description": "A lightweight FastAPI service that exposes endpoints to interact with Zarr datasets stored in a Simple Storage Service (S3)",
        "endpoints": [
            "/health",
            "/datasets",
            "/datasets/{dataset}/infos",
            "/datasets/{dataset}/extract",
            "/datasets/{dataset}/probe",
            "/datasets/{dataset}/isoline",
            "/datasets/{dataset}/select",
        ],
    }


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html(request: Request):
    """
    Handle passing JWT token from query params to Swagger UI for authenticated requests
    Handle passing JWT token from query params to "Try it out" requests in Swagger UI
    """
    token = request.query_params.get("jwt")

    openapi_url = app.openapi_url.lstrip("/")
    if token:
        openapi_url += f"?jwt={token}"

    response = get_swagger_ui_html(
        openapi_url=openapi_url,
        title=app.title + " - Swagger UI",
        swagger_ui_parameters={
            "defaultModelsExpandDepth": -1,
            "persistAuthorization": True,
        },
    )

    js_intercept_token = """
    const urlParams = new URLSearchParams(window.location.search);
    const jwt = urlParams.get('jwt');
    const path = window.location.pathname;
    const basePath = path.substring(0, path.lastIndexOf('/docs'));
    if (jwt) {
      ui.initOAuth({"persistAuthorization": true}); 
    }

    const originalFetch = window.fetch;
    window.fetch = function() {
      let url = arguments[0];
      if (typeof url === 'string') {
        if (basePath) {
          // Absolute URL
          if (url.startsWith(origin) && !url.startsWith(origin + basePath)) {
            url = url.replace(origin, origin + basePath);
          }
          // Relative URL
          else if (url.startsWith('/') && !url.startsWith(basePath)) {
            url = basePath + url;
          }
        }
        // Add token if not already present in URL
        if (jwt && !url.includes('jwt=')) {
          const separator = url.includes('?') ? '&' : '?';
          arguments[0] = url + separator + 'jwt=' + jwt;
        } else {
          arguments[0] = url;
        }
      }
      return originalFetch.apply(this, arguments);
    };
  """

    new_content = response.body.decode().replace(
        "</body>", f"<script>{js_intercept_token}</script></body>"
    )
    return HTMLResponse(content=new_content)


@app.get(
    "/health",
    summary="Health Check",
    description="Check the health status of the kazarr API.",
)
def health_check():
    return {"status": "ok"}


@app.get(
    "/datasets",
    summary="List available datasets",
    description="Retrieve a list of all available datasets.",
)
def list_datasets(
    search_path: str = Query(None, description="The path to search for datasets"),
):
    try:
        return handlers.list_datasets(search_path)
    except exceptions.GenericInternalError as e:
        sys.stderr.write(f"[ERR {e.error_code}]: {e.message}\n")
        raise HTTPException(status_code=500, detail="An internal error occured")
    except exceptions.ConfigurationBasedException as e:
        raise HTTPException(status_code=500, detail=e.get())
    except exceptions.UserInputBasedException as e:
        raise HTTPException(status_code=404, detail=e.get())


# As FastAPI redirect_slashes only works for static paths, we need to create a custom endpoint to handle redirection for the /datasets path with optional search_path query parameter
@app.get("/datasets/", include_in_schema=False)
async def redirect_datasets(request: Request):
    url = request.url
    new_url = url.replace(path="/datasets")
    return RedirectResponse(url=new_url, status_code=301)


# As FastAPI can't handle endpoints with a path parameter followed by a static segment,
# this endpoint will act as a proxy to route to the appropriate handler
# So, it need to handle every possible parameter that could be passed to the underlying handlers
# but they will be hidden to the OpenAPI documentation
# As every parameter will now be optional, we need to check by ourselves for required parameters
@app.get(
    "/datasets/{dataset:path}",
    summary="Get dataset information",
    description="Retrieve metadata and variables informations for a specified dataset.",
)
def dataset_infos(
    request: Request,
    dataset: str = Path(
        ..., description="The path to the dataset to retrieve information for"
    ),
    # == Extraction parameters == #
    variable: str = Query(
        None, include_in_schema=False
    ),  #! Must be defined for extraction, isoline and free selection
    lon_min: float | None = Query(None, include_in_schema=False),
    lat_min: float | None = Query(None, include_in_schema=False),
    lon_max: float | None = Query(None, include_in_schema=False),
    lat_max: float | None = Query(None, include_in_schema=False),
    time: str | None = Query(None, include_in_schema=False),
    resolution_limit: float | None = Query(None, include_in_schema=False),
    format: Literal["raw", "geojson", "mesh"] = Query("raw", include_in_schema=False),
    mesh_tile_size: int | None = Query(None, include_in_schema=False),
    mesh_interpolate: bool = Query(False, include_in_schema=False),
    mesh_data_mapping: str | None = Query(None, include_in_schema=False),
    time_interpolate: bool = Query(False, include_in_schema=False),
    interp_vars: list[str] = Query([], include_in_schema=False),
    as_dims: list[str] = Query([], include_in_schema=False),
    interpolation: str | None = Query(None, include_in_schema=False),
    # == Probe parameters == #
    variables: list[str] = Query(
        None, include_in_schema=False
    ),  #! Must be defined for probe
    lon: float = Query(None, include_in_schema=False),  #! Must be defined for probe
    lat: float = Query(None, include_in_schema=False),  #! Must be defined for probe
    height: float | None = Query(None, include_in_schema=False),
    interpolate: bool = Query(True, include_in_schema=False),
    # format => already defined in extraction parameters
    # as_dims => already defined in extraction parameters
    # == Isoline parameters == #
    # variable => already defined in extraction parameters
    # time => already defined in extraction parameters
    levels: list[float] = Query(
        None, include_in_schema=False
    ),  #! Must be defined for isoline
    # format => already defined in extraction parameters
    # time_interpolate => already defined in extraction parameters
    # as_dims => already defined in extraction parameters
    # == Free selection parameters == #
    # variable => already defined in extraction parameters
    # interp_vars => already defined in extraction parameters
    # as_dims => already defined in extraction parameters
    # == Mesh parameters == #
    # format => already defined in extraction parameters
    # mesh_data_mapping => already defined in extraction parameters
):
    try:
        if interpolation is not None and ":" in interpolation:
            interpolation = parse_query_dict(interpolation)

        if dataset.endswith("/extract"):
            if variable is None:
                raise exceptions.MissingQueryParameter("variable")

            format = {"type": format}
            format["force_data_mapping"] = mesh_data_mapping
            if format["type"] == "mesh":
                format["shape"] = (
                    (mesh_tile_size, mesh_tile_size)
                    if mesh_tile_size is not None
                    else None
                )
                format["interpolate"] = mesh_interpolate

            return handlers.extract(
                dataset[:-8],
                variable,
                request,
                time=time,
                bounding_box=(lon_min, lat_min, lon_max, lat_max),
                resolution_limit=resolution_limit,
                format=format,
                interp_vars=interp_vars,
                time_interpolate=time_interpolate,
                as_dims=as_dims,
                interp_config=interpolation,
            )
        elif dataset.endswith("/probe"):
            if variables is None:
                raise exceptions.MissingQueryParameter("variables")
            if lon is None or lat is None:
                raise exceptions.MissingQueryParameter(["lon", "lat"])
            return handlers.probe(
                dataset[:-6],
                variables,
                lon,
                lat,
                request,
                height=height,
                time=time,
                interpolate=interpolate,
                interp_config=interpolation,
                format=format,
                as_dims=as_dims,
            )
        elif dataset.endswith("/isoline"):
            if variable is None:
                raise exceptions.MissingQueryParameter("variable")
            if levels is None:
                raise exceptions.MissingQueryParameter("levels")
            return handlers.isoline(
                dataset[:-8],
                variable,
                levels,
                request,
                time=time,
                format=format,
                time_interpolate=time_interpolate,
                as_dims=as_dims,
            )
        elif dataset.endswith("/select"):
            if variable is None:
                raise exceptions.MissingQueryParameter("variable")
            return handlers.free_selection(
                dataset[:-7],
                variable,
                request,
                interp_vars=interp_vars,
                as_dims=as_dims,
            )
        elif dataset.endswith("/mesh"):
            return handlers.mesh(
                dataset[:-5],
                format=format,
                force_data_mapping=mesh_data_mapping,
            )
        return handlers.dataset_infos(dataset)
    except exceptions.GenericInternalError as e:
        sys.stderr.write(f"[ERR {e.error_code}]: {e.message}\n")
        raise HTTPException(status_code=500, detail="An internal error occured")
    except exceptions.ConfigurationBasedException as e:
        raise HTTPException(status_code=500, detail=e.get())
    except exceptions.UserInputBasedException as e:
        raise HTTPException(status_code=404, detail=e.get())


@app.get(
    "/datasets/{dataset:path}/extract",
    summary="Get data at a specific time",
    description="Retrieve data for a specified variable at a specific time and within an optional bounding box.",
)
def extract_data(
    dataset: str = Path(
        ..., description="The path to the dataset to extract data from"
    ),
    variable: str = Query(..., description="The variable to extract"),
    lon_min: float | None = Query(
        None, description="Minimum longitude of the bounding box"
    ),
    lat_min: float | None = Query(
        None, description="Minimum latitude of the bounding box"
    ),
    lon_max: float | None = Query(
        None, description="Maximum longitude of the bounding box"
    ),
    lat_max: float | None = Query(
        None, description="Maximum latitude of the bounding box"
    ),
    time: str | None = Query(None, description="The time value to extract"),
    resolution_limit: float | None = Query(
        None, description="The resolution limit for data extraction"
    ),
    format: str = Query(
        "raw",
        description="The format of the extracted data (Currently supported: 'raw', 'geojson', 'mesh' with additional parameters)",
    ),
    mesh_tile_size: int | None = Query(
        None,
        description="[format='mesh'] The size of the mesh tile to use when extracting data",
    ),
    mesh_interpolate: bool = Query(
        False, description="[format='mesh'] Whether to interpolate data on the mesh"
    ),
    mesh_data_mapping: str | None = Query(
        None,
        description="[format='mesh'] Whether the data of the mesh is on cells or on vertices. This will override the dataset configuration. (Supported values: 'vertices', 'cells')",
    ),
    interp_vars: list[str] = Query(
        [], description="Variables to interpolate during extraction"
    ),
    time_interpolate: bool = Query(
        False,
        description="Whether to interpolate values on time dimension or to get the closest time step",
    ),
    as_dims: list[str] = Query(
        [],
        description="If some variables have the same name as dimensions, will force them to be treated as dimensions",
    ),
    interpolation: str = Query(
        "method:linear,padding:1.0",
        description="Interpolation configuration. Must be defined with : \"interpolation=method:METHOD_NAME,padding:FLOAT_COEFF,optparam1:VALUE1,optparam2:VALUE2,...\" where METHOD_NAME is one of 'nearest', 'linear', 'cubic', 'idw' or 'rbf', FLOAT_COEFF is a coefficient of the bounding box size that will be use to add extra context to interpolation and optparam are optional parameters depending on the method (for example, for idw method, you can specify the radius and power parameters with interpolation=method:idw,radius:0.5,power:2). If not specified, it will default to linear interpolation. For more informations, see https://github.com/kalisio/kazarr",
    ),
):
    pass


@app.get(
    "/datasets/{dataset:path}/probe",
    summary="Get data at specific coordinates over time",
    description="Retrieve data for specified variables at specific coordinates over time.",
)
def probe_data(
    dataset: str = Path(..., description="The path to the dataset to probe"),
    variables: list[str] = Query(..., description="The variables to probe"),
    lon: float = Query(..., description="The longitude coordinate to probe"),
    lat: float = Query(..., description="The latitude coordinate to probe"),
    height: float | None = Query(None, description="The height coordinate to probe"),
    time: str | None = Query(None, description="The time value to probe"),
    interpolate: bool = Query(
        True,
        description="Whether to interpolate values on spatial dimensions or to get the closest grid point",
    ),
    interpolation: str = Query(
        "method:idw,radius:0.05,power:2",
        description='Interpolation configuration. Must be defined with : "interpolation=method:idw,radius:FLOAT,power:FLOAT". Only IDW method is avaible for now. If not specified, it will default to radius=0.05 and power=2.0. For more informations, see https://github.com/kalisio/kazarr',
    ),
    format: str = Query(
        "raw",
        description="The format of the probed data (Currently supported: 'raw', 'geojson')",
    ),
    as_dims: list[str] = Query(
        [],
        description="If some variables have the same name as dimensions, will force them to be treated as dimensions",
    ),
):
    pass


@app.get(
    "/datasets/{dataset:path}/isoline",
    summary="Get isolines for a specific variable at a specific time",
    description="Generate isolines for a specified variable at a specific time and for given levels.",
)
def isoline_data(
    dataset: str = Path(
        ..., description="The path to the dataset to generate isolines from"
    ),
    variable: str = Query(..., description="The variable to generate isolines for"),
    time: str | None = Query(
        None, description="The time value to use for isoline generation"
    ),
    levels: list[float] = Query(
        ..., description="List of levels for isoline generation"
    ),
    format: str = Query(
        "raw",
        description="The format of the extracted data (Currently supported: 'raw', 'geojson')",
    ),
    time_interpolate: bool = Query(
        False, description="Whether to interpolate values on time dimension"
    ),
    as_dims: list[str] = Query(
        [],
        description="If some variables have the same name as dimensions, will force them to be treated as dimensions",
    ),
):
    pass


@app.get(
    "/datasets/{dataset:path}/select",
    summary="Get data for free selection of dimensions and coordinates",
    description="Retrieve data for a specified variable based on free selection of dimensions and coordinates.",
)
def free_selection_data(
    dataset: str = Path(
        ..., description="The path to the dataset to perform selection on"
    ),
    variable: str = Query(..., description="The variable to perform selection on"),
    interp_vars: list[str] = Query(
        [], description="Variables to interpolate during selection"
    ),
    as_dims: list[str] = Query(
        [],
        description="If some variables have the same name as dimensions, will force them to be treated as dimensions",
    ),
):
    pass


@app.get(
    "/datasets/{dataset:path}/mesh",
    summary="Get mesh representation of the dataset",
    description="Retrieve a mesh representation of the dataset for visualization purposes.",
)
def mesh(
    dataset: str = Path(
        ..., description="The path to the dataset to get the mesh representation from"
    ),
    format: str = Query(
        "mesh",
        description="The format of the extracted data (Currently supported: 'mesh', 'geojson')",
    ),
    mesh_data_mapping: str | None = Query(
        None,
        description="Whether the data of the mesh is on cells or on vertices. This will override the dataset configuration. (Supported values: 'vertices', 'cells')",
    ),
):
    pass

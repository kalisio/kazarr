import sys, os

from fastapi import FastAPI, Path, Query, Request, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal

import src.handlers as handlers
import src.exceptions as exceptions

class RegisterDatasetRequest(BaseModel):
  name: str
  path: str
  description: str = ""
  config: dict = {}

app = FastAPI(
  title="kazarr API",
  version=os.getenv("APP_VERSION", "0.1.0"),
  description="A lightweight FastAPI service that exposes endpoints to interact with Zarr datasets stored in a Simple Storage Service (S3)",
  contact={
    "name": "Kalisio",
    "url": "https://kalisio.xyz",
    "email": "contact@kalisio.xyz"
  },
  docs_url=None
)

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
  description="Provides basic information about the kazarr API."
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
      "/datasets/{dataset}/select"
    ]
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
    }
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

  new_content = response.body.decode().replace("</body>", f"<script>{js_intercept_token}</script></body>")
  return HTMLResponse(content=new_content)

@app.get(
  "/health",
  summary="Health Check",
  description="Check the health status of the kazarr API."
)
def health_check():
  return {"status": "ok"}

@app.get(
  "/datasets",
  summary="List available datasets",
  description="Retrieve a list of all available datasets."
)
def list_datasets(
  search_path: str = Query(None, description="The path to search for datasets")
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


# As FastAPI can't handle endpoints with a path parameter followed by a static segment,
# this endpoint will act as a proxy to route to the appropriate handler
# So, it need to handle every possible parameter that could be passed to the underlying handlers
# but they will be hidden to the OpenAPI documentation
# As every parameter will now be optional, we need to check by ourselves for required parameters
@app.get(
  "/datasets/{dataset:path}",
  summary="Get dataset information",
  description="Retrieve metadata and variables informations for a specified dataset."
)
def dataset_infos(
  request: Request,
  dataset: str = Path(..., description="The path to the dataset to retrieve information for"),

  # == Extraction parameters == #
  variable: str = Query(None, include_in_schema=False), #! Must be defined for extraction, isoline and free selection
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
  as_dims: list[str] = Query([], include_in_schema=False),

  # == Probe parameters == #
  variables: list[str] = Query(None, include_in_schema=False), #! Must be defined for probe
  lon: float = Query(None, include_in_schema=False), #! Must be defined for probe
  lat: float = Query(None, include_in_schema=False), #! Must be defined for probe
  height: float | None = Query(None, include_in_schema=False),
  # as_dims => already defined in extraction parameters

  # == Isoline parameters == #
  # variable => already defined in extraction parameters
  # time => already defined in extraction parameters
  levels: list[float] = Query(None, include_in_schema=False), #! Must be defined for isoline
  # format => already defined in extraction parameters
  # time_interpolate => already defined in extraction parameters
  # as_dims => already defined in extraction parameters

  # == Free selection parameters == #
  # variable => already defined in extraction parameters
  interp_vars: list[float] = Query([], include_in_schema=False),
  # as_dims => already defined in extraction parameters
):
  try:
    if dataset.endswith("/extract"):
      if variable is None:
        raise exceptions.UserInputBasedException("The 'variable' parameter is required for data extraction")
      
      format = { "type": format }
      format["force_data_mapping"] = mesh_data_mapping
      if format["type"] == "mesh":
        format["shape"] = (mesh_tile_size, mesh_tile_size) if mesh_tile_size is not None else None
        format["interpolate"] = mesh_interpolate

      return handlers.extract(
        dataset[:-8],
        variable,
        request,
        time=time,
        bounding_box=(lon_min, lat_min, lon_max, lat_max),
        resolution_limit=resolution_limit,
        format=format,
        time_interpolate=time_interpolate,
        as_dims=as_dims
      )
    elif dataset.endswith("/probe"):
      if variables is None:
        raise exceptions.UserInputBasedException("The 'variables' parameter is required for probing data")
      if lon is None or lat is None:
        raise exceptions.UserInputBasedException("The 'lon' and 'lat' parameters are required for probing data")
      return handlers.probe(dataset[:-6], variables, lon, lat, request, height=height, as_dims=as_dims)
    elif dataset.endswith("/isoline"):
      if variable is None:
        raise exceptions.UserInputBasedException("The 'variable' parameter is required for isoline generation")
      if levels is None:
        raise exceptions.UserInputBasedException("The 'levels' parameter is required for isoline generation")
      return handlers.isoline(dataset[:-8], variable, levels, request, time=time, format=format, time_interpolate=time_interpolate, as_dims=as_dims)
    elif dataset.endswith("/select"):
      if variable is None:
        raise exceptions.UserInputBasedException("The 'variable' parameter is required for free selection")
      return handlers.free_selection(dataset[:-7], variable, request, interp_vars=interp_vars, as_dims=as_dims)
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
  request: Request,
  dataset: str = Path(..., description="The path to the dataset to extract data from"),
  variable: str = Query(..., description="The variable to extract"),
  lon_min: float | None = Query(None, description="Minimum longitude of the bounding box"),
  lat_min: float | None = Query(None, description="Minimum latitude of the bounding box"),
  lon_max: float | None = Query(None, description="Maximum longitude of the bounding box"),
  lat_max: float | None = Query(None, description="Maximum latitude of the bounding box"),
  time: str | None = Query(None, description="The time value to extract"),
  resolution_limit: float | None = Query(None, description="The resolution limit for data extraction"),
  format: str = Query("raw", description="The format of the extracted data (Currently supported: 'raw', 'geojson', 'mesh' with additional parameters)"),
  mesh_tile_size: int | None = Query(None, description="[format='mesh'] The size of the mesh tile to use when extracting data"),
  mesh_interpolate: bool = Query(False, description="[format='mesh'] Whether to interpolate data on the mesh"),
  mesh_data_mapping: str | None = Query(None, description="[format='mesh'] Whether the data of the mesh is on cells or on vertices. This will override the dataset configuration. (Supported values: 'vertices', 'cells')"),
  time_interpolate: bool = Query(False, description="Whether to interpolate values on time dimension"),
  as_dims: list[str] = Query([], description="If some variables have the same name as dimensions, will force them to be treated as dimensions")
):
  pass

@app.get(
  "/datasets/{dataset:path}/probe",
  summary="Get data at specific coordinates over time",
  description="Retrieve data for specified variables at specific coordinates over time."
)
def probe_data(
  request: Request,
  dataset: str = Path(..., description="The path to the dataset to probe"),
  variables: list[str] = Query(..., description="The variables to probe"),
  lon: float = Query(..., description="The longitude coordinate to probe"),
  lat: float = Query(..., description="The latitude coordinate to probe"),
  height: float | None = Query(None, description="The height coordinate to probe"),
  as_dims: list[str] = Query([], description="If some variables have the same name as dimensions, will force them to be treated as dimensions")
):
  pass

@app.get(
  "/datasets/{dataset:path}/isoline",
  summary="Get isolines for a specific variable at a specific time",
  description="Generate isolines for a specified variable at a specific time and for given levels."
)
def isoline_data(
  request: Request,
  dataset: str = Path(..., description="The path to the dataset to generate isolines from"),
  variable: str = Query(..., description="The variable to generate isolines for"),
  time: str | None = Query(None, description="The time value to use for isoline generation"),
  levels: list[float] = Query(..., description="List of levels for isoline generation"),
  format: str = Query("raw", description="The format of the extracted data (Currently supported: 'raw', 'geojson')"),
  time_interpolate: bool = Query(False, description="Whether to interpolate values on time dimension"),
  as_dims: list[str] = Query([], description="If some variables have the same name as dimensions, will force them to be treated as dimensions")
):
  pass

@app.get(
  "/datasets/{dataset:path}/select",
  summary="Get data for free selection of dimensions and coordinates",
  description="Retrieve data for a specified variable based on free selection of dimensions and coordinates."
)
def free_selection_data(
  request: Request,
  dataset: str = Path(..., description="The path to the dataset to perform selection on"),
  variable: str = Query(..., description="The variable to perform selection on"),
  interp_vars: list[float] = Query([], description="Variables to interpolate during selection"),
  as_dims: list[str] = Query([], description="If some variables have the same name as dimensions, will force them to be treated as dimensions")
):
  pass
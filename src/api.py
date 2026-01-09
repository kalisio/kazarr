import sys, os

from fastapi import FastAPI, Path, Query, Request, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

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

  js_intercept_token = f"""
    const urlParams = new URLSearchParams(window.location.search);
    const jwt = urlParams.get('jwt');
    const path = window.location.pathname;
    const basePath = path.substring(0, path.lastIndexOf('/docs'));
    if (jwt) {{
      ui.initOAuth({{"persistAuthorization": true}}); 
    }}

    const originalFetch = window.fetch;
    window.fetch = function() {{
      let url = arguments[0];
      if (typeof url === 'string') {{
        if (basePath && url.startsWith('/') && !url.startsWith(basePath)) {{
          url = basePath + url;
          arguments[0] = url;
        }}
        if (jwt && !url.includes('jwt=')) {{
          const separator = url.includes('?') ? '&' : '?';
          arguments[0] = url + separator + 'jwt=' + jwt;
        }}
      }}
      return originalFetch.apply(this, arguments);
    }};
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

@app.get("/datasets")
def list_datasets():
  try:
    return handlers.list_datasets()
  except exceptions.GenericInternalError as e:
    sys.stderr.write(f"[ERR {e.error_code}]: {e.message}\n")
    raise HTTPException(status_code=500, detail="An internal error occured")
  except exceptions.ConfigurationBasedException as e:
    raise HTTPException(status_code=500, detail=e.get())
  except exceptions.UserInputBasedException as e:
    raise HTTPException(status_code=404, detail=e.get())

@app.get(
  "/datasets/{dataset}",
  summary="Get dataset information",
  description="Retrieve metadata and variables informations for a specified dataset."
)
def dataset_infos(dataset: str = Path(..., description="The name of the dataset to retrieve information for")):
  try:
    return handlers.dataset_infos(dataset)
  except exceptions.GenericInternalError as e:
    sys.stderr.write(f"[ERR {e.error_code}]: {e.message}\n")
    raise HTTPException(status_code=500, detail="An internal error occured")
  except exceptions.ConfigurationBasedException as e:
    raise HTTPException(status_code=500, detail=e.get())
  except exceptions.UserInputBasedException as e:
    raise HTTPException(status_code=404, detail=e.get())

@app.get(
  "/datasets/{dataset}/extract",
  summary="Get data at a specific time",
  description="Retrieve data for a specified variable at a specific time and within an optional bounding box."
)
def extract_data(
  request: Request,
  dataset: str = Path(..., description="The name of the dataset to extract data from"),
  variable: str = Query(..., description="The variable to extract"),
  lon_min: float | None = Query(None, description="Minimum longitude of the bounding box"),
  lat_min: float | None = Query(None, description="Minimum latitude of the bounding box"),
  lon_max: float | None = Query(None, description="Maximum longitude of the bounding box"),
  lat_max: float | None = Query(None, description="Maximum latitude of the bounding box"),
  time: str | None = Query(None, description="The time value to extract"),
  resolution_limit: float | None = Query(None, description="The resolution limit for data extraction"),
  format: str = Query("raw", description="The format of the extracted data (Currently supported: 'raw', 'geojson')"),
  mesh_tile_size: int | None = Query(None, description="The size of the mesh tile to use when extracting data"),
  mesh_interpolate: bool = Query(False, description="Whether to interpolate the mesh"),
  as_dims: list[str] = Query([], description="If some variables have the same name as dimensions, will force them to be treated as dimensions")
):
  try:
    return handlers.extract(
      dataset,
      variable,
      request,
      time=time,
      bounding_box=(lon_min, lat_min, lon_max, lat_max),
      resolution_limit=resolution_limit,
      format=format,
      mesh_tile_shape=(mesh_tile_size, mesh_tile_size) if mesh_tile_size is not None else None,
      mesh_interpolate=mesh_interpolate,
      as_dims=as_dims
    )
  except exceptions.GenericInternalError as e:
    sys.stderr.write(f"[ERR {e.error_code}]: {e.message}\n")
    raise HTTPException(status_code=500, detail="An internal error occured")
  except exceptions.ConfigurationBasedException as e:
    raise HTTPException(status_code=500, detail=e.get())
  except exceptions.UserInputBasedException as e:
    raise HTTPException(status_code=404, detail=e.get())

@app.get(
  "/datasets/{dataset}/probe",
  summary="Get data at specific coordinates over time",
  description="Retrieve data for specified variables at specific coordinates over time."
)
def probe_data(
  request: Request,
  dataset: str = Path(..., description="The name of the dataset to probe"),
  variables: list[str] = Query(..., description="The variables to probe"),
  lon: float = Query(..., description="The longitude coordinate to probe"),
  lat: float = Query(..., description="The latitude coordinate to probe"),
  height: float | None = Query(None, description="The height coordinate to probe"),
  as_dims: list[str] = Query([], description="If some variables have the same name as dimensions, will force them to be treated as dimensions")
):
  try:
    return handlers.probe(dataset, variables, lon, lat, request, height=height, as_dims=as_dims)
  except exceptions.GenericInternalError as e:
    sys.stderr.write(f"[ERR {e.error_code}]: {e.message}\n")
    raise HTTPException(status_code=500, detail="An internal error occured")
  except exceptions.ConfigurationBasedException as e:
    raise HTTPException(status_code=500, detail=e.get())
  except exceptions.UserInputBasedException as e:
    raise HTTPException(status_code=404, detail=e.get())

@app.get(
  "/datasets/{dataset}/isoline",
  summary="Get isolines for a specific variable at a specific time",
  description="Generate isolines for a specified variable at a specific time and for given levels."
)
def isoline_data(
  request: Request,
  dataset: str = Path(..., description="The name of the dataset to generate isolines from"),
  variable: str = Query(..., description="The variable to generate isolines for"),
  time: str | None = Query(None, description="The time value to use for isoline generation"),
  levels: list[float] = Query(..., description="List of levels for isoline generation"),
  format: str = Query("raw", description="The format of the extracted data (Currently supported: 'raw', 'geojson')"),
  as_dims: list[str] = Query([], description="If some variables have the same name as dimensions, will force them to be treated as dimensions")
):
  try:
    return handlers.isoline(dataset, variable, levels, request, time=time, format=format, as_dims=as_dims)
  except exceptions.GenericInternalError as e:
    sys.stderr.write(f"[ERR {e.error_code}]: {e.message}\n")
    raise HTTPException(status_code=500, detail="An internal error occured")
  except exceptions.ConfigurationBasedException as e:
    raise HTTPException(status_code=500, detail=e.get())
  except exceptions.UserInputBasedException as e:
    raise HTTPException(status_code=404, detail=e.get())

@app.get(
  "/datasets/{dataset}/select",
  summary="Get data for free selection of dimensions and coordinates",
  description="Retrieve data for a specified variable based on free selection of dimensions and coordinates."
)
def free_selection_data(
  request: Request,
  dataset: str = Path(..., description="The name of the dataset to perform selection on"),
  variable: str = Query(..., description="The variable to perform selection on"),
  as_dims: list[str] = Query([], description="If some variables have the same name as dimensions, will force them to be treated as dimensions")
):
  try:
    return handlers.free_selection(dataset, variable, request, as_dims=as_dims)
  except exceptions.GenericInternalError as e:
    sys.stderr.write(f"[ERR {e.error_code}]: {e.message}\n")
    raise HTTPException(status_code=500, detail="An internal error occured")
  except exceptions.ConfigurationBasedException as e:
    raise HTTPException(status_code=500, detail=e.get())
  except exceptions.UserInputBasedException as e:
    raise HTTPException(status_code=404, detail=e.get())

@app.post(
  "/datasets",
  summary="Register a new dataset",
  description="Register a new Zarr dataset."
)
def register_dataset(
  request: RegisterDatasetRequest
):
  try:
    return handlers.register_dataset(request.name, request.path, request.description, request.config)
  except exceptions.GenericInternalError as e:
    sys.stderr.write(f"[ERR {e.error_code}]: {e.message}\n")
    raise HTTPException(status_code=500, detail="An internal error occured")
  except exceptions.ConfigurationBasedException as e:
    raise HTTPException(status_code=500, detail=e.get())
  except exceptions.UserInputBasedException as e:
    raise HTTPException(status_code=404, detail=e.get())

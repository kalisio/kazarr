import sys, os

from fastapi import FastAPI, Path, Query, Request, HTTPException

import src.utils as utils
import src.handlers as handlers
import src.exceptions as exceptions

app = FastAPI(
  title="KaZarr API",
  version=os.getenv("APP_VERSION", "0.1.0"),
  description="A lightweight FastAPI service that exposes endpoints to interact with Zarr datasets stored in a Simple Storage Service (S3)",
  contact={
    "name": "Kalisio",
    "url": "https://kalisio.xyz",
    "email": "contact@kalisio.xyz"
  }
)

@app.get(
  "/",
  summary="API Root",
  description="Provides basic information about the KaZarr API."
)
def read_root():
  return {
    "name": "KaZarr API",
    "version": os.getenv("APP_VERSION", "0.1.0"),
    "description": "A lightweight FastAPI service that exposes endpoints to interact with Zarr datasets stored in a Simple Storage Service (S3)",
    "endpoints": [
      "/health",
      "/datasets",
      "/datasets/{dataset}/infos",
      "/datasets/{dataset}/extract",
      "/datasets/{dataset}/probe",
      "/datasets/{dataset}/isoline",
    ]
  }

@app.get(
  "/health",
  summary="Health Check",
  description="Check the health status of the KaZarr API."
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
  time: str | None = Query(None, description="The time value to extract")
):
  try:
    return handlers.extract(dataset, variable, request, time=time, bounding_box=(lon_min, lat_min, lon_max, lat_max))
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
  height: float | None = Query(None, description="The height coordinate to probe")
):
  try:
    return handlers.probe(dataset, variables, lon, lat, request, height=height)
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
  levels: list[float] = Query(..., description="Comma-separated list of levels for isoline generation")
):
  try:
    return handlers.isoline(dataset, variable, levels, request, time=time)
  except exceptions.GenericInternalError as e:
    sys.stderr.write(f"[ERR {e.error_code}]: {e.message}\n")
    raise HTTPException(status_code=500, detail="An internal error occured")
  except exceptions.ConfigurationBasedException as e:
    raise HTTPException(status_code=500, detail=e.get())
  except exceptions.UserInputBasedException as e:
    raise HTTPException(status_code=404, detail=e.get())

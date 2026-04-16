import os
import sys
import logging
import time

from fastapi import FastAPI, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import src.exceptions as exceptions
from src.api.routers import dataset, extract, probe, isoline, mesh, select

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - [Worker %(process)d] - %(levelname)s - %(message)s",
#     datefmt="%H:%M:%S",
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False, # Cannot be True when allow_origins is ["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)


# @app.middleware("http")
# async def log_requests(request: Request, call_next):
#     start_time = time.perf_counter()
#     response = await call_next(request)
#     process_time = (time.perf_counter() - start_time) * 1000
#     try:
#         logger.info(
#             f"{request.method} {request.url.path} - {response.status_code} - {process_time:.2f}ms"
#         )
#     except Exception:
#         pass
#     return response


# Include all modular routers
app.include_router(dataset.router)
app.include_router(extract.router)
app.include_router(probe.router)
app.include_router(isoline.router)
app.include_router(select.router)
app.include_router(mesh.router)


@app.exception_handler(exceptions.GenericInternalError)
async def generic_internal_exception_handler(
    request: Request, exc: exceptions.GenericInternalError
):
    sys.stderr.write(f"[ERR {exc.error_code}]: {exc.message}\n")
    return JSONResponse(
        status_code=500, content={"detail": "An internal error occurred"}
    )


@app.exception_handler(exceptions.ConfigurationBasedException)
async def configuration_exception_handler(
    request: Request, exc: exceptions.ConfigurationBasedException
):
    return JSONResponse(status_code=500, content={"detail": exc.get()})


@app.exception_handler(exceptions.UserInputBasedException)
async def user_input_exception_handler(
    request: Request, exc: exceptions.UserInputBasedException
):
    return JSONResponse(status_code=400, content={"detail": exc.get()})


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
            "/datasets/{dataset}/metadata",
            "/datasets/{dataset}/extract",
            "/datasets/{dataset}/probe",
            "/datasets/{dataset}/isoline",
            "/datasets/{dataset}/select",
            "/datasets/{dataset}/mesh",
        ],
    }


@app.get(
    "/health",
    summary="Health Check",
    description="Check the health status of the kazarr API.",
)
def health_check():
    return {"status": "ok"}


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html(request: Request):
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
          if (url.startsWith(origin) && !url.startsWith(origin + basePath)) {
            url = url.replace(origin, origin + basePath);
          }
          else if (url.startsWith('/') && !url.startsWith(basePath)) {
            url = basePath + url;
          }
        }
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

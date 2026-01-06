import os, argparse

import src.api as api
from src.utils import enable_s3fs_debug_logging

import uvicorn

# Get host and port from environment variables or use defaults
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 8000))

def start_api(host, port, datasets_path, enable_debug=False):
  if enable_debug:
    os.environ["DEBUG"] = "1"
  if os.getenv("DATASETS_PATH") is None:
    os.environ["DATASETS_PATH"] = datasets_path

  uvicorn.run(api.app, host=host, port=port)

def main():
  parser = argparse.ArgumentParser(description="A lightweight FastAPI service that exposes endpoints to interact with Zarr datasets stored in a Simple Storage Service (S3)")
  parser.add_argument("-p", "--port", type=int, default=PORT, help="Port to run the API server on")
  parser.add_argument("-H", "--host", type=str, default=HOST, help="Host to run the API server on")
  parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging")
  parser.add_argument("--datasets-path", type=str, default="datasets.json", help="Path to the datasets configuration file")

  args = parser.parse_args()

  if args.debug:
    enable_s3fs_debug_logging()

  start_api(args.host, args.port, args.datasets_path, enable_debug=args.debug)

if __name__ == "__main__":
  main()
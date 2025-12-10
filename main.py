import os, argparse

import src.api as api

import uvicorn

# Get host and port from environment variables or use defaults
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 8000))

def start_api(host, port):
  uvicorn.run(api.app, host=host, port=port)

def main():
  parser = argparse.ArgumentParser(description="A lightweight FastAPI service that exposes endpoints to interact with Zarr datasets stored in a Simple Storage Service (S3)")
  parser.add_argument("-p", "--port", type=int, default=PORT, help="Port to run the API server on")
  parser.add_argument("-H", "--host", type=str, default=HOST, help="Host to run the API server on")

  args = parser.parse_args()

  start_api(args.host, args.port)

if __name__ == "__main__":
  main()
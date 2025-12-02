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
  subparsers = parser.add_subparsers(dest="command")

  parser_start_api = subparsers.add_parser("start-api", help="Start the Flask API server with Uvicorn")
  parser_start_api.add_argument("--host", type=str, default=HOST)
  parser_start_api.add_argument("--port", type=int, default=PORT)

  args = parser.parse_args()

  if args.command == "start-api":
    start_api(args.host, args.port)
  else:
    parser.print_help()

if __name__ == "__main__":
  main()
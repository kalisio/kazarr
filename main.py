import os
import argparse

from src.utils.logging import configure_logging

import uvicorn

# Get host and port from environment variables or use defaults
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 8000))
WORKERS_COUNT = int(os.getenv("WORKERS_COUNT", 1))


def start_api(host, port, workers, datasets_path, enable_debug=False, log_level="INFO"):
    if log_level.upper() == "DEBUG":
        enable_debug = True
    if os.getenv("DATASETS_PATH") is None:
        os.environ["DATASETS_PATH"] = datasets_path

    configure_logging(enable_debug, log_level)
    uvicorn_log_level = log_level.lower() if log_level.lower() in ["critical", "error", "warning", "info", "debug", "trace"] else "info"
    uvicorn.run("src.api.api:app", host=host, port=port, workers=workers, log_level=uvicorn_log_level)


def main():
    parser = argparse.ArgumentParser(
        description="A lightweight FastAPI service that exposes endpoints to interact with Zarr datasets stored in a Simple Storage Service (S3)"
    )
    parser.add_argument(
        "-p", "--port", type=int, default=PORT, help="Port to run the API server on"
    )
    parser.add_argument(
        "-H", "--host", type=str, default=HOST, help="Host to run the API server on"
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=WORKERS_COUNT,
        help="Number of worker processes for handling requests",
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the log level for the application",
    )
    parser.add_argument(
        "--datasets-path",
        type=str,
        default="/",
        help="Path to the datasets configuration file",
    )

    args = parser.parse_args()

    start_api(
        args.host, args.port, args.workers, args.datasets_path, enable_debug=args.debug, log_level=args.log_level
    )


if __name__ == "__main__":
    main()

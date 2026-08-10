FROM python:3.11-bookworm
LABEL maintainer="<contact@kalisio.xyz>"

# Install uv and uvx from the Astral SH container registry
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY . .

# /tmp is always world-writable regardless of the runtime UID.
# uv initializes its cache dir even with --no-cache, so we must point it
# somewhere accessible to avoid "Permission denied" on /.cache/uv
ENV UV_CACHE_DIR=/tmp/uv-cache

RUN uv sync --no-cache

EXPOSE 8000

CMD ["uv", "run", "python", "main.py", "-H", "0.0.0.0", "-p", "8000", "-d"]

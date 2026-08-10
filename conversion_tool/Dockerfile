FROM python:3.11-bookworm
LABEL maintainer="<contact@kalisio.xyz>"

# Install uv and uvx from the Astral SH container registry
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
  gcc \
  libc6-dev \
  libeccodes0 \
  libeccodes-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# /tmp is always world-writable regardless of the runtime UID.
# uv initializes its cache dir even with --no-cache, so we must point it
# somewhere accessible to avoid "Permission denied" on /.cache/uv
ENV UV_CACHE_DIR=/tmp/uv-cache

# Install the package along with its runtime dependencies and pyinstaller
RUN uv sync --group build --no-cache && uv run pyinstaller -F -n kazarr main.py

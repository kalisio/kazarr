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

ENV HOME=/app
COPY . ${HOME}
WORKDIR ${HOME}

# Install the package along with its runtime dependencies and pyinstaller
RUN chmod -R g=u /app && uv sync --group build --all-extras --no-cache && uv run pyinstaller -F -n kazarr \
  --add-data "/usr/share/eccodes/definitions:eccodes/definitions" \
  --collect-all eccodes \
  main.py

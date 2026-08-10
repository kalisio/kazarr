FROM python:3.11-bookworm
LABEL maintainer="<contact@kalisio.xyz>"

# Install uv and uvx from the Astral SH container registry
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY . .

# Force uv cache into a writable directory (avoids /.cache/uv permission error when $HOME is not set)
ENV UV_CACHE_DIR=/app/.cache/uv

RUN uv sync

EXPOSE 8000

CMD ["uv", "run", "python", "main.py", "-H", "0.0.0.0", "-p", "8000", "-d"]

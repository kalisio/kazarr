FROM python:3.11-bookworm
LABEL maintainer="<contact@kalisio.xyz>"

# Install uv and uvx from the Astral SH container registry
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY . .

RUN uv sync --no-cache

EXPOSE 8000

CMD ["uv", "run", "python", "main.py", "-H", "0.0.0.0", "-p", "8000", "-d"]

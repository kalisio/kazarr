FROM python:3.11-bookworm
LABEL maintainer="<contact@kalisio.xyz>"

# Install uv and uvx from the Astral SH container registry
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV HOME=/app
COPY . ${HOME}
WORKDIR ${HOME}

RUN chmod -R g=u /app && uv sync --no-cache

EXPOSE 8000

CMD ["uv", "run", "python", "main.py", "-H", "0.0.0.0", "-p", "8000", "-d"]

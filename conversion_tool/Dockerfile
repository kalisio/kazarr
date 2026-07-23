FROM python:3.11-bookworm
LABEL maintainer="<contact@kalisio.xyz>"

RUN apt-get update && apt-get install -y --no-install-recommends \
  gcc \
  libc6-dev \
  && rm -rf /var/lib/apt/lists/*

ENV HOME=/app
COPY . ${HOME}
WORKDIR ${HOME}

# Install the package along with its runtime dependencies and pyinstaller
RUN pip install ".[build]" && pyinstaller -F -n kazarr main.py

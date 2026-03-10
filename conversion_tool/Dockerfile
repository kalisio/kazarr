FROM python:3.11-bookworm
LABEL maintainer="<contact@kalisio.xyz>"

RUN apt-get update && apt-get install -y --no-install-recommends \
  gcc \
  libc6-dev \
  && rm -rf /var/lib/apt/lists/*

ENV HOME=/app
COPY . ${HOME}
WORKDIR ${HOME}

RUN pip install \
  python-dotenv \
  xarray \
  zarr \
  cfgrib \
  h5netcdf \
  numpy \
  pyproj \
  dask \
  s3fs \
  matplotlib \
  pyvista \
  pyinstaller

RUN pyinstaller -F -n conversion_tool main.py

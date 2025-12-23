FROM python:3.11-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
  gcc \
  libc6-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

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

COPY . .

RUN pyinstaller -F main.py

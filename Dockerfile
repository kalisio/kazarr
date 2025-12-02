FROM mambaorg/micromamba
LABEL maintainer="<contact@kalisio.xyz>"

ENV HOME=/kazarr
COPY . ${HOME}
WORKDIR ${HOME}

RUN micromamba install -y -n base -c conda-forge \
  python=3.11 \
  fastapi \
  uvicorn \
  xarray \
  zarr \
  cfgrib \
  numpy \
  pyproj \
  dask \
  s3fs \
  matplotlib \
  && micromamba clean --all --yes

EXPOSE 8000

CMD ["micromamba", "run", "-n", "base", "python", "main.py", "start-api", "--host", "0.0.0.0", "--port", "8000"]
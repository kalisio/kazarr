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
  pyvista \
  && micromamba clean --all --yes

EXPOSE 8000

CMD ["micromamba", "run", "-n", "base", "python", "main.py", "-H", "0.0.0.0", "-p", "8000"]
FROM python:3.11-bookworm
LABEL maintainer="<contact@kalisio.xyz>"

# Install uv and uvx from the Astral SH container registry
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
  gcc \
  libc6-dev \
  curl \
  bzip2 \
  && rm -rf /var/lib/apt/lists/*

# A clean, standalone ecCodes native library from conda-forge, installed into
# its own prefix with micromamba (not a full conda install). We deliberately
# do NOT use the PyPI "eccodeslib" package here: it vendors ECMWF's eckit
# toolkit, which in turn vendors its own copy of PROJ - and that copy coexists
# in the same PyInstaller-bundled process as pyproj's own bundled PROJ. The
# two are C++ libraries; loading two different builds of the same C++ classes
# in one process is an ODR violation and crashes with glibc heap corruption
# ("corrupted size vs. prev_size") on interpreter shutdown. conda-forge's
# eccodes package has no proj/eckit dependency at all (just libaec, OpenJPEG/
# Jasper, libnetcdf, HDF5 - see https://github.com/conda-forge/eccodes-feedstock),
# so it sidesteps the conflict entirely instead of trying to patch around it.
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local bin/micromamba && \
  /usr/local/bin/micromamba create -y -p /opt/eccodes -c conda-forge "eccodes>=2.42" && \
  rm -rf /opt/eccodes/pkgs /root/.cache

ENV ECCODES_DIR=/opt/eccodes
ENV ECCODES_DEFINITION_PATH=/opt/eccodes/share/eccodes/definitions

ENV HOME=/app
COPY . ${HOME}
WORKDIR ${HOME}

# Install the package along with its runtime dependencies and pyinstaller.
#
# `eccodes` (pinned >=2.45 in pyproject.toml - see the comment there) resolves
# to the pure-Python "none-any" wheel with no "eccodeslib" dependency, so it
# relies on findlibs to locate a native library at runtime - which is exactly
# the conda-forge one installed above (found via ECCODES_DIR). PyInstaller
# doesn't know about that external library on its own, so we bundle it
# explicitly: the .so at the top level of the bundle (where gribapi's loader
# and findlibs both expect a directly-loadable file), and the eccodes
# definitions data alongside the "eccodes" package tree.
#
# LD_LIBRARY_PATH is set only for this RUN step (not as a persistent ENV
# above): conda-forge's /opt/eccodes/lib ships its own copies of very common
# system libraries (e.g. libtinfo/libncursesw), with an ABI that can differ
# from Debian's. Exporting it image-wide would make unrelated system binaries
# (apt-installed tools, nano, etc.) pick up those conda builds instead of
# Debian's and crash - the same kind of ABI conflict that caused the PROJ
# issue this whole native-library setup exists to avoid. The frozen kazarr
# executable itself doesn't need it: PyInstaller bundles its own copy of
# libeccodes.so and resolves it via its own bundle path at runtime.
RUN chmod -R g=u /app && uv sync --group build --all-extras --no-cache && \
  LD_LIBRARY_PATH="/opt/eccodes/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    uv run pyinstaller -F -n kazarr \
      --collect-all eccodes \
      --add-data "/opt/eccodes/share/eccodes/definitions:eccodes/definitions" \
      --add-binary "/opt/eccodes/lib/libeccodes.so*:." \
      main.py

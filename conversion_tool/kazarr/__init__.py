"""
kazarr
======

A library to process various data formats (NetCDF, GRIB, ...) to Zarr datasets
compatible with the kazarr service.

Usage as a library::

    from kazarr import process, list_templates

    process(
        "path/to/data.nc",
        template="my_template",
        output_path="s3://my-bucket/output.zarr"
    )
"""

from kazarr.api import list_templates, process

__all__ = ["list_templates", "process"]

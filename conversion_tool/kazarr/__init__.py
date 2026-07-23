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

from kazarr.api import process, list_templates

__all__ = ["process", "list_templates"]

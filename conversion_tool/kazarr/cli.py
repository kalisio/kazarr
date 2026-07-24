import os
import sys
import json
import logging
import argparse

from kazarr.api import (
    process,
    list_templates,
    TEMPLATE_DEFAULT_PATH,
)


def _run():
    """Core CLI logic. Raises exceptions on error."""
    # Handle list-templates subcommand
    if len(sys.argv) > 1 and sys.argv[1] == "list-templates":
        parser_list = argparse.ArgumentParser(
            prog="kazarr list-templates",
            description="List available templates for processing.",
        )
        parser_list.add_argument(
            "--templates-path",
            type=str,
            default=TEMPLATE_DEFAULT_PATH,
            help="Path to templates configuration file (local or s3://) [default: templates.json]",
        )
        args_list = parser_list.parse_args(sys.argv[2:])
        templates = list_templates(args_list.templates_path)
        print("Available templates:")
        for template in templates:
            print(f"- {template}")
        return

    epilog_text = """
other commands:
  list-templates        List available templates for processing.
                        Usage: kazarr list-templates [--templates-path PATH]
"""

    parser = argparse.ArgumentParser(
        prog="kazarr",
        description="Process datasets (NetCDF, GRIB, etc.) to Zarr format for Kazarr.",
        epilog=epilog_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input_path",
        type=str,
        nargs="?",
        default=None,
        help="Path of the input file or folder used for generating the Zarr dataset (local or s3://)",
    )
    parser.add_argument(
        "-l",
        "--list-templates",
        action="store_true",
        help="Shortcut to list available templates for conversion and exit",
    )
    parser.add_argument(
        "-t",
        "--template",
        type=str,
        help="Template to use for the configuration of the new dataset",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=json.loads,
        default={},
        help="Additional configuration as JSON string",
    )
    parser.add_argument(
        "-a",
        "--args",
        action="append",
        default=[],
        help="Additional arguments that can be accessed in templates in the form key=value (can be used multiple times). In the template, these can be accessed as ARGS.key",
    )
    parser.add_argument(
        "-f",
        "--config-file",
        type=str,
        help="Path to a JSON file containing additional configuration",
    )
    parser.add_argument(
        "-d", "--description", type=str, help="Description of the new dataset"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output path for the processed dataset (local or s3://)",
    )
    parser.add_argument(
        "-p",
        "--pipeline",
        type=str,
        default="preprocess",
        help="Pipeline to use for processing the dataset [default: preprocess]",
    )
    parser.add_argument(
        "--templates-path",
        type=str,
        default=TEMPLATE_DEFAULT_PATH,
        help="Path to templates configuration file (local or s3://) [default: templates.json]",
    )
    parser.add_argument(
        "--data-mapping",
        type=str,
        choices=["vertices", "cells"],
        default="vertices",
        help="Whether to map data on mesh vertices or cells (default: vertices)",
    )
    parser.add_argument(
        "--mesh-type",
        type=str,
        choices=["auto", "regular", "rectilinear", "radial"],
        default="auto",
        help="Type of mesh to generate (default: auto, which infers from data between regular and rectilinear but not able to handle radial meshes)",
    )
    parser.add_argument(
        "--custom-eccodes-path",
        type=str,
        help="Path to a directory containing custom ecCodes (local) to be used for processing",
    )
    parser.add_argument(
        "--dask-dashboard",
        action="store_true",
        help="Whether to start a Dask dashboard for monitoring the processing (default: False)",
    )
    parser.add_argument(
        "--s3-storage-class",
        type=str,
        choices=[
            "STANDARD",
            "REDUCED_REDUNDANCY",
            "STANDARD_IA",
            "ONEZONE_IA",
            "INTELLIGENT_TIERING",
            "GLACIER",
            "DEEP_ARCHIVE",
            "OUTPOSTS",
            "GLACIER_IR",
            "SNOW",
            "EXPRESS_ONEZONE",
            "FSX_OPENZFS",
        ],
        default="STANDARD",
        help="S3 storage class for the output dataset (default: STANDARD)",
    )

    args = parser.parse_args()

    if not args.input_path:
        parser.print_help()
        sys.exit(1)

    process(
        args.input_path,
        template=args.template,
        config=args.config,
        config_file=args.config_file,
        template_args=args.args,
        description=args.description,
        output_path=args.output,
        pipeline_name=args.pipeline,
        templates_path=args.templates_path,
        data_mapping=args.data_mapping,
        mesh_type=args.mesh_type,
        custom_eccodes_path=args.custom_eccodes_path,
        dask_dashboard=args.dask_dashboard,
        s3_storage_class=args.s3_storage_class,
    )


def main():
    """CLI entry point with formatted error display."""
    logging.basicConfig(
        level=logging.INFO, format="[KAZARR] {%(levelname)s} %(message)s"
    )

    logger = logging.getLogger(__name__)
    try:
        _run()
    except Exception as e:
        error_message_length = len(str(e))
        try:
            terminal_width = os.get_terminal_size().columns
        except Exception:
            terminal_width = 16
        separator_length = min(error_message_length, terminal_width)
        logger.exception(
            "\n%s\n%s\n%s", "=" * separator_length, e, "=" * separator_length
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

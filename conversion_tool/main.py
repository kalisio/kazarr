import os
import json
import argparse

import src.pipelines as pipelines
from src.utils import load_json, merge, get_valid_template_args

TEMPLATE_DEFAULT_PATH = "templates.json"


def new_dataset(
    input_path,
    template=None,
    config={},
    config_file=None,
    template_args=[],
    description="",
    output_path=None,
    pipeline_name="preprocess",
    templates_path=TEMPLATE_DEFAULT_PATH,
    data_mapping="vertices",
    mesh_type="auto",
    dask_dashboard=False
):
    pipeline_config = {}
    if template is not None:
        templates = load_json(templates_path, config)
        if template in templates:
            pipeline_config = merge(templates[template], config)
    if config_file is not None:
        config_file_content = load_json(config_file, config)
        pipeline_config = merge(pipeline_config, config_file_content)
    if config is not None:
        pipeline_config = merge(pipeline_config, config)
    pipeline_config = merge(
        pipeline_config,
        {
            "description": description,
            "path": input_path,
            "ARGS": get_valid_template_args(template_args),
        },
    )
    if output_path is not None:
        pipeline_config["save_path"] = output_path
    if data_mapping is not None:
        pipeline_config["mesh_data_on_cells"] = data_mapping == "cells"
    if mesh_type is not None:
        pipeline_config["mesh_type"] = mesh_type
    if dask_dashboard:
        pipeline_config["enable_dask_dashboard"] = True
    _, config = pipelines.pipeline(pipeline_config, pipeline_name)


def list_templates(templates_path=TEMPLATE_DEFAULT_PATH):
    templates = load_json(templates_path)

    print("Available templates:")
    for template in templates.keys():
        print(f"- {template}")


def main():
    parser = argparse.ArgumentParser(description="Data Processing Application")
    subparsers = parser.add_subparsers(dest="command")

    parser_list_templates = subparsers.add_parser(
        "list-templates", help="List available templates for new datasets"
    )
    parser_list_templates.add_argument(
        "--templates-path",
        type=str,
        default=TEMPLATE_DEFAULT_PATH,
        help="Path to templates configuration file (local or s3://) [default: templates.json]",
    )

    parser_create_dataset = subparsers.add_parser(
        "new-dataset", help="Create a new zarr dataset"
    )
    parser_create_dataset.add_argument(
        "input_path",
        type=str,
        help="Path of the input file or folder used for generating this new dataset (local or s3://)",
    )
    parser_create_dataset.add_argument(
        "-t",
        "--template",
        type=str,
        help="Template to use for the configuration of the new dataset",
    )
    parser_create_dataset.add_argument(
        "-c",
        "--config",
        type=json.loads,
        default={},
        help="Additional configuration as JSON string",
    )
    parser_create_dataset.add_argument(
        "-a",
        "--args",
        action="append",
        default=[],
        help="Additional arguments that can be accessed in templates in the form key=value (can be used multiple times). In the template, these can be accessed as ARGS.key",
    )
    parser_create_dataset.add_argument(
        "-f",
        "--config-file",
        type=str,
        help="Path to a JSON file containing additional configuration",
    )
    parser_create_dataset.add_argument(
        "-d", "--description", type=str, help="Description of the new dataset"
    )
    parser_create_dataset.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output path for the processed dataset (local or s3://)",
    )
    parser_create_dataset.add_argument(
        "-p",
        "--pipeline",
        type=str,
        default="preprocess",
        help="Pipeline to use for processing the new dataset [default: preprocess]",
    )
    parser_create_dataset.add_argument(
        "--templates-path",
        type=str,
        default=TEMPLATE_DEFAULT_PATH,
        help="Path to templates configuration file (local or s3://) [default: templates.json]",
    )
    parser_create_dataset.add_argument(
        "--data-mapping",
        type=str,
        choices=["vertices", "cells"],
        default="vertices",
        help="Whether to map data on mesh vertices or cells (default: vertices)",
    )
    parser_create_dataset.add_argument(
        "--mesh-type",
        type=str,
        choices=["auto", "regular", "rectilinear", "radial"],
        default="auto",
        help="Type of mesh to generate (default: auto, which infers from data between regular and rectilinear but not able to handle radial meshes)",
    )
    parser_create_dataset.add_argument(
        "--dask-dashboard",
        action="store_true",
        help="Whether to start a Dask dashboard for monitoring the processing (default: False)",
    )

    args = parser.parse_args()

    if args.command == "new-dataset":
        new_dataset(
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
            dask_dashboard=args.dask_dashboard
        )
    elif args.command == "list-templates":
        list_templates(args.templates_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_message_prefix = "[KAZARR] ERROR: "
        error_message_length = len(str(e)) + len(error_message_prefix)
        try:
            terminal_width = os.get_terminal_size().columns
        except Exception:
            terminal_width = 16
        separator_length = min(error_message_length, terminal_width)
        print("=" * separator_length)
        print(f"{error_message_prefix}{e}")
        print("=" * separator_length)
        raise e

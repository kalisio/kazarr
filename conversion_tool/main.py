import json
import argparse

import src.pipelines as pipelines
from src.utils import load_json, merge


def new_dataset(
    dataset_name,
    input_path,
    template=None,
    config={},
    config_file=None,
    description="",
    output_path=None,
    pipeline_name="preprocess",
    templates_path="templates.json",
    data_mapping="vertices",
    mesh_type="auto",
):
    pipeline_config = {}
    if template is not None:
        templates = load_json(templates_path)
        if template in templates:
            pipeline_config = merge(templates[template], config)
    if config_file is not None:
        config_file_content = load_json(config_file)
        pipeline_config = merge(pipeline_config, config_file_content)
    if config is not None:
        pipeline_config = merge(pipeline_config, config)
    pipeline_config = merge(
        pipeline_config,
        {
            "name": dataset_name,
            "description": description,
            "path": input_path,
        },
    )
    if output_path is not None:
        pipeline_config["save_path"] = output_path
    if data_mapping is not None:
        pipeline_config["mesh_data_on_cells"] = data_mapping == "cells"
    if mesh_type is not None:
        pipeline_config["mesh_type"] = mesh_type
    _, config = pipelines.pipeline(pipeline_config, pipeline_name)


def list_templates(templates_path="templates.json"):
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
        default="templates.json",
        help="Path to templates configuration file (local or s3://) [default: templates.json]",
    )

    parser_create_dataset = subparsers.add_parser(
        "new-dataset", help="Create a new zarr dataset"
    )
    parser_create_dataset.add_argument(
        "dataset_name", type=str, help="Name of the new dataset"
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
        default="templates.json",
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

    args = parser.parse_args()

    if args.command == "new-dataset":
        new_dataset(
            args.dataset_name,
            args.input_path,
            template=args.template,
            config=args.config,
            config_file=args.config_file,
            description=args.description,
            output_path=args.output,
            pipeline_name=args.pipeline,
            templates_path=args.templates_path,
            data_mapping=args.data_mapping,
            mesh_type=args.mesh_type,
        )
    elif args.command == "list-templates":
        list_templates(args.templates_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

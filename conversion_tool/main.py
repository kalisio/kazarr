import json, argparse

import src.pipelines as pipelines
from src.utils import load_datasets, load_dataset_config, merge

def list_datasets(datasets_path="datasets.json"):
  datasets = load_datasets(datasets_path)

  print("Available datasets:")
  for dataset in datasets.keys():
    print(f"- {dataset}")

def list_pipelines(dataset_name, datasets_path="datasets.json"):
  dataset_config = load_dataset_config(dataset_name, datasets_path)

  pipelines_config = dataset_config.get("pipelines", {})
  print(f"Available pipelines for dataset {dataset_name}:")
  for pipeline in pipelines_config.keys():
    print(f"- {pipeline}")

def run_pipeline(dataset_name, pipeline_name, override_config={}):
  dataset, config = pipelines.run_pipeline(dataset_name, pipeline_name, override_config)
  print(f"[KAZARR] Pipeline {pipeline_name} executed successfully for dataset {dataset_name}.")
  return dataset, config

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
  registration_endpoint=None
):
  pipeline_config = {}
  if template is not None:
    templates = load_datasets(templates_path)
    if template in templates:
      pipeline_config = merge(templates[template], config)
  if config_file is not None:
    config_file_content = load_datasets(config_file)
    pipeline_config = merge(pipeline_config, config_file_content)
  pipeline_config = merge(pipeline_config, {
    "name": dataset_name,
    "description": description,
    "path": input_path,
  })
  if output_path is not None:
    pipeline_config["save_path"] = output_path
  if registration_endpoint is not None:
    pipeline_config["registration_endpoint_url"] = registration_endpoint
  dataset, config = pipelines.pipeline(pipeline_config, pipeline_name)

def list_templates(templates_path="templates.json"):
  templates = load_datasets(templates_path)

  print("Available templates:")
  for template in templates.keys():
    print(f"- {template}")

def main():
  parser = argparse.ArgumentParser(description="Data Processing Application")
  subparsers = parser.add_subparsers(dest="command")

  parser_list_templates = subparsers.add_parser("list-templates", help="List available templates for new datasets")
  parser_list_templates.add_argument("--templates-path", type=str, default="templates.json", help="Path to templates configuration file (local or s3://) [default: templates.json]")

  parser_create_dataset = subparsers.add_parser("new-dataset", help="Create a new zarr dataset")
  parser_create_dataset.add_argument("dataset_name", type=str, help="Name of the new dataset")
  parser_create_dataset.add_argument("input_path", type=str, help="Path of the input file or folder used for generating this new dataset (local or s3://)")
  parser_create_dataset.add_argument("-t", "--template", type=str, help="Template to use for the configuration of the new dataset")
  parser_create_dataset.add_argument("-c", "--config", type=json.loads, default={}, help="Additional configuration as JSON string")
  parser_create_dataset.add_argument("-f", "--config-file", type=str, help="Path to a JSON file containing additional configuration")
  parser_create_dataset.add_argument("-d", "--description", type=str, help="Description of the new dataset")
  parser_create_dataset.add_argument("-o", "--output", type=str, help="Output path for the processed dataset (local or s3://)")
  parser_create_dataset.add_argument("-p", "--pipeline", type=str, default="preprocess", help="Pipeline to use for processing the new dataset [default: preprocess]")
  parser_create_dataset.add_argument("--rgstr-endpoint", type=str, help="Endpoint URL for dataset registration service")
  parser_create_dataset.add_argument("--templates-path", type=str, default="templates.json", help="Path to templates configuration file (local or s3://) [default: templates.json]")

  # parser_list_datasets = subparsers.add_parser("list-datasets", help="List available datasets")

  # parser_list_pipelines = subparsers.add_parser("list-pipelines", help="List available pipelines for a dataset")
  # parser_list_pipelines.add_argument("dataset", type=str, help="Name of the dataset")

  # parser_run_pipeline = subparsers.add_parser("run-pipeline", help="Run a pipeline for a dataset")
  # parser_run_pipeline.add_argument("dataset", type=str, help="Name of the dataset")
  # parser_run_pipeline.add_argument("pipeline", type=str, help="Name of the pipeline")
  # parser_run_pipeline.add_argument("--config", type=json.loads, default={}, help="Override configuration as JSON string")
  # parser_run_pipeline.add_argument("--datasets-path", type=str, default="datasets.json", help="Path to datasets configuration file")
  # parser_run_pipeline.add_argument("--templates-path", type=str, default="templates.json", help="Path to templates configuration file")

  args = parser.parse_args()

  if args.command == "list-datasets":
    list_datasets()
  elif args.command == "list-pipelines":
    list_pipelines(args.dataset)
  elif args.command == "run-pipeline":
    run_pipeline(args.dataset, args.pipeline, args.config)
  elif args.command == "new-dataset":
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
      registration_endpoint=args.rgstr_endpoint
    )
  elif args.command == "list-templates":
    list_templates(args.templates_path)
  else:
    parser.print_help()

if __name__ == "__main__":
  main()
import kazarr.pipelines as pipelines
from kazarr.utils import load_json, merge, get_valid_template_args

TEMPLATE_DEFAULT_PATH = "templates.json"


def process(
    input_path,
    template=None,
    config=None,
    config_file=None,
    template_args=None,
    description="",
    output_path=None,
    pipeline_name="preprocess",
    templates_path=TEMPLATE_DEFAULT_PATH,
    data_mapping="vertices",
    mesh_type="auto",
    dask_dashboard=False,
    s3_storage_class="STANDARD",
):
    """Create a new Zarr dataset from the given input path.

    Args:
        input_path (str): Path of the input file or folder (local or s3://).
        template (str, optional): Template name to use for the pipeline configuration.
        config (dict, optional): Additional configuration as a dictionary.
        config_file (str, optional): Path to a JSON file with additional configuration.
        template_args (list, optional): List of "key=value" strings accessible as ARGS.key in templates.
        description (str, optional): Description of the new dataset.
        output_path (str, optional): Output path for the processed dataset (local or s3://).
        pipeline_name (str, optional): Pipeline to use for processing. Defaults to "preprocess".
        templates_path (str, optional): Path to templates configuration file. Defaults to "templates.json".
        data_mapping (str, optional): Whether to map data on mesh "vertices" or "cells". Defaults to "vertices".
        mesh_type (str, optional): Type of mesh to generate ("auto", "regular", "rectilinear", "radial"). Defaults to "auto".
        dask_dashboard (bool, optional): Whether to start a Dask dashboard. Defaults to False.
        s3_storage_class (str, optional): S3 storage class for the output dataset. Defaults to "STANDARD".

    Returns:
        tuple: The (dataset, config) result from the pipeline.
    """
    config = config or {}
    template_args = template_args or []
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
    if s3_storage_class is not None:
        pipeline_config["s3_storage_class"] = s3_storage_class
    return pipelines.pipeline(pipeline_config, pipeline_name)


def list_templates(templates_path=TEMPLATE_DEFAULT_PATH):
    """Return available templates from the given templates file.

    Args:
        templates_path (str, optional): Path to templates configuration file. Defaults to "templates.json".

    Returns:
        list[str]: List of available template names.
    """
    return list(load_json(templates_path).keys())

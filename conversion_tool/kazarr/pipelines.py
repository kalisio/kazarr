import time
import copy
import logging

from kazarr.utils import (
    load_dataset_config,
    dget,
    log_duration,
    camel_to_snake,
    merge,
    init_store_as_secondary,
)
from kazarr import processes as proc

logger = logging.getLogger(__name__)


def pipeline(config, name, dataset=None):
    """Run a defined pipeline of processes on the dataset according to the provided configuration."""
    start_time = time.time()
    pipelines = dget(config, "pipelines", {})

    if name not in pipelines:
        raise ValueError(f"Pipeline not found: {name}")

    logger.info('Starting pipeline "%s"', name)

    target_pipeline = pipelines[name]

    if config.get("enable_dask_dashboard", False):
        target_pipeline = ["init_dask_dashboard"] + target_pipeline

    for process in target_pipeline:
        process_type = "process"
        process_name = process
        process_params = {}
        if isinstance(process, dict):
            process_type = dget(process, "type")
            process_name = dget(process, "name")
            process_params = dget(process, "params", default={})
            if process_type is None or process_name is None:
                raise ValueError(
                    f'Invalid process definition in pipeline "{name}": {process}'
                )

        if process_type == "pipeline":
            pipeline_config = merge(
                copy.deepcopy(process_params), copy.deepcopy(config)
            )

            # store_as_secondary is provided on pipeline level
            # so we need to set it to False when running the sub-pipeline to avoid saving it when loading
            # as we want the final output of the sub-pipeline
            pipeline_store_as_secondary = dget(
                pipeline_config, "store_as_secondary", default=False
            )
            if pipeline_store_as_secondary:
                pipeline_config["store_as_secondary"] = False

            # Update config updated in sub-pipeline that need to be propagated to the main config
            out_pipeline_dataset, out_pipeline_config = pipeline(
                pipeline_config, process_name, dataset
            )
            if "global_config_update" in out_pipeline_config:
                for key in out_pipeline_config["global_config_update"]:
                    config[key] = out_pipeline_config[key]

            # When the sub-pipeline is finished, now we can store the output dataset as secondary
            if pipeline_store_as_secondary:
                _, config_with_secondary = init_store_as_secondary(
                    None,
                    out_pipeline_dataset,
                    merge(
                        {
                            "store_as_secondary": True,
                            "secondary_tag": dget(pipeline_config, "secondary_tag"),
                        },
                        copy.deepcopy(config),
                    ),
                )
                config["secondary_datasets"] = config_with_secondary.get(
                    "secondary_datasets", []
                )
            else:
                dataset = out_pipeline_dataset

            # Special case when delta_time_to_datetime process is used in a sub-pipeline
            # and changes the name of the time variable: we need to update the time variable
            # name in the global config so that following processes in the main pipeline can use it
            time_var_key = "variables.time"
            original_time_var = dget(config, time_var_key)
            new_time_var = dget(out_pipeline_config, time_var_key)
            if (
                original_time_var is not None
                and new_time_var is not None
                and original_time_var != new_time_var
            ):
                config["variables"]["time"] = new_time_var
            continue

        process_start_time = time.time()
        try:
            target_process = getattr(proc, process_name)
            dataset, config = target_process(dataset, {**config, **process_params})
            log_duration(process_start_time, f'Completed process "{process_name}"')
        except AttributeError as e:
            try:
                target_process = getattr(proc, camel_to_snake(process_name))
                dataset, config = target_process(dataset, {**config, **process_params})
                log_duration(process_start_time, f'Completed process "{process_name}"')
            except Exception:
                raise e
    log_duration(start_time, f'Completed pipeline "{name}"')
    return dataset, config


def run_pipeline(
    dataset_name, pipeline_name, override_config={}, datasets_path="datasets.json"
):
    dataset_config = load_dataset_config(dataset_name, datasets_path)
    merged_config = {**dataset_config, **override_config}
    return pipeline(merged_config, pipeline_name)

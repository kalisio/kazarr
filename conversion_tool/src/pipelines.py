import time
import copy

from src.utils import load_dataset_config, dget, print_duration, camel_to_snake, merge, update_store_as_secondary
from src import processes as proc


def pipeline(config, name, dataset=None):
    """Run a defined pipeline of processes on the dataset according to the provided configuration."""
    start_time = time.time()
    pipelines = dget(config, "pipelines", {})

    if name not in pipelines:
        raise ValueError(f"Pipeline not found: {name}")

    print(f'[KAZARR] Starting pipeline "{name}"')

    target_pipeline = pipelines[name]

    if config.get("enable_dask_dashboard", False):
        target_pipeline = ["init_dask_dashboard"] + target_pipeline

    time_var_has_changed = []

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
            pipeline_config = merge(copy.deepcopy(process_params), copy.deepcopy(config))
            dataset, out_pipeline_config = pipeline(pipeline_config, process_name, dataset)
            if "global_config_update" in out_pipeline_config:
                for key in out_pipeline_config["global_config_update"]:
                    config[key] = out_pipeline_config[key]

            # Special case when delta_time_to_datetime process is used in a sub-pipeline 
            # and changes the name of the time variable: we need to update the time variable 
            # name in the global config so that following processes in the main pipeline can use it
            original_time_var = dget(config, "variables.time")
            new_time_var = dget(out_pipeline_config, "variables.time")
            if original_time_var is not None and new_time_var is not None and original_time_var != new_time_var:
                time_var_has_changed.append(new_time_var)
            continue

        if time_var_has_changed and all(val == time_var_has_changed[0] for val in time_var_has_changed):
            config["variables"]["time"] = merge(time_var_has_changed[0], dget(config, "variables.time")) if isinstance(dget(config, "variables.time"), dict) else time_var_has_changed[0]

        process_start_time = time.time()
        try:
            target_process = getattr(proc, process_name)
            dataset, config = target_process(dataset, {**config, **process_params})
            dataset, config = update_store_as_secondary(dataset, config)
            print_duration(process_start_time, f'Completed process "{process_name}"')
        except AttributeError as e:
            try:
                target_process = getattr(proc, camel_to_snake(process_name))
                dataset, config = target_process(dataset, {**config, **process_params})
                print_duration(
                    process_start_time, f'Completed process "{process_name}"'
                )
            except Exception:
                raise e
    print_duration(start_time, f'Completed pipeline "{name}"')
    return dataset, config


def run_pipeline(
    dataset_name, pipeline_name, override_config={}, datasets_path="datasets.json"
):
    dataset_config = load_dataset_config(dataset_name, datasets_path)
    merged_config = {**dataset_config, **override_config}
    return pipeline(merged_config, pipeline_name)

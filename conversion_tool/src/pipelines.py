import time

from src.utils import load_dataset_config, dget, print_duration, camel_to_snake
from src import processes as proc

def pipeline(config, name, dataset=None):
  start_time = time.time()
  pipelines = dget(config, "pipelines", {})

  if name not in pipelines:
    raise ValueError(f"Pipeline not found: {name}")

  print(f"[KAZARR] Starting pipeline \"{name}\"")

  target_pipeline = pipelines[name]
  for process in target_pipeline:
    process_type = "process"
    process_name = process
    process_params = {}
    if isinstance(process, dict):
      process_type = dget(process, "type")
      process_name = dget(process, "name")
      process_params = dget(process, "params", default={})
      if process_type is None or process_name is None:
        raise ValueError(f"Invalid process definition in pipeline \"{name}\": {process}")

    if process_type == "pipeline":
      dataset, config = pipeline(pipelines, config, process_name, dataset)
      continue
    
    process_start_time = time.time()
    try:
      target_process = getattr(proc, process_name)
      dataset, config = target_process(dataset, {**config, **process_params})
      print_duration(process_start_time, f"Completed process \"{process_name}\"")
    except AttributeError as e:
      try:
        target_process = getattr(proc, camel_to_snake(process_name))
        dataset, config = target_process(dataset, {**config, **process_params})
        print_duration(process_start_time, f"")
      except Exception as e2:
        raise e
  print_duration(start_time, f"Completed pipeline \"{name}\"")
  return dataset, config
  
def run_pipeline(dataset_name, pipeline_name, override_config={}, datasets_path = "datasets.json"):
  dataset_config = load_dataset_config(dataset_name, datasets_path)
  merged_config = {**dataset_config, **override_config}
  return pipeline(merged_config, pipeline_name)

import os
import json
import logging
import time

from loguru import logger as log

log.add("logs/app_fastapi.log", rotation="10 MB", compression="zip", level="INFO")


class KazarrLoggerHandler(logging.Handler):
    def __init__(self):
        super().__init__()

    def emit(self, record):
        formatted_record = self.format(record)
        if formatted_record.startswith("CALL: get_object"):
            try:
                data = json.loads(formatted_record.split(" - ")[-1].replace("'", '"'))
                print(
                    f"[Kazarr - S3FS] Try downloading {data.get('Bucket')}/{data.get('Key')}"
                )
            except Exception:
                pass


def enable_s3fs_debug_logging():
    handler = KazarrLoggerHandler()
    handler.setLevel(logging.DEBUG)
    logger = logging.getLogger("s3fs")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)


class StepDurationLogger:
    def __init__(self, method_name, parameters=None):
        self.method_name = method_name
        self.parameters = parameters if parameters is not None else ()
        self.start_time = time.perf_counter()
        self.steps = []

    def step_start(self, step_name, auto_end_previous=True):
        if len(self.steps) > 0 and auto_end_previous:
            self.step_end()

        self.steps.append(
            {"name": step_name, "start_time": time.perf_counter(), "end_time": None}
        )

    def step_end(self):
        if len(self.steps) == 0:
            return

        current_step = self.steps[-1]
        if current_step["end_time"] is None:
            current_step["end_time"] = time.perf_counter()

    def log(self):
        if os.getenv("DEBUG") != "1":
            return

        total_end_time = time.perf_counter()
        total_duration = total_end_time - self.start_time

        log_message = f"[Kazarr - Performance] | Method:{self.method_name} | Parameters: {self.parameters} | Total duration: {total_duration:.4f} seconds"
        for step in self.steps:
            if step["end_time"] is not None:
                step_duration = step["end_time"] - step["start_time"]
                log_message += f" | Step {step['name']}: {step_duration:.4f}s"
            else:
                log_message += f" | Step '{step['name']}': not ended"

        log.info(log_message.strip())

    def end(self):
        self.step_end()
        self.log()

import os
import sys
import json
import logging
import time

from src import exceptions

from loguru import logger as log


def configure_logging(debug=False, log_level="INFO"):
    log.remove()
    log.add(
        "logs/app_fastapi.log",
        rotation="10 MB",
        compression="zip",
    )
    log.add(
        sys.stderr,
        level=log_level.upper(),
    )
    if debug:
        os.environ["DEBUG"] = "1"
        enable_s3fs_debug_logging()
    else:
        os.environ.pop("DEBUG", None)


class KazarrLoggerHandler(logging.Handler):
    def __init__(self):
        super().__init__()

    def emit(self, record):
        formatted_record = self.format(record)
        if formatted_record.startswith("CALL: get_object"):
            try:
                data = json.loads(formatted_record.split(" - ")[-1].replace("'", '"'))
                log.debug(
                    "[Kazarr - S3FS] Try downloading {bucket}/{key}",
                    bucket=data.get("Bucket"),
                    key=data.get("Key"),
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


class StepLoggerAndAborter(StepDurationLogger):
    def __init__(self, method_name, parameters=None, cancel_event=None):
        super().__init__(method_name, parameters)
        self.cancel_event = cancel_event

    def check_cancelled(self, step_name=None):
        if self.cancel_event and self.cancel_event.is_set():
            previousStep = (
                f"Previous step: {self.steps[-1]['name']}, " if self.steps else ""
            )
            nextStep = f"Next step: {step_name}" if step_name else ""
            raise exceptions.RequestCancelled(
                f"{self.method_name} process was cancelled by the client. ({previousStep}{nextStep})"
            )

    def step_start(self, step_name, auto_end_previous=True):
        self.check_cancelled(step_name=step_name)
        super().step_start(step_name, auto_end_previous)

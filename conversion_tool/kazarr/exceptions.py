class KazarrException(Exception):
    def __init__(self, error_code, message, payload=None):
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.payload = payload

    def get(self):
        out = {"error_code": self.error_code, "message": self.message}
        if self.payload is not None:
            out["payload"] = self.payload
        return out


class ConfigurationError(KazarrException):
    def __init__(self, message):
        super().__init__("CONFIGURATION_ERROR", message, None)


class MissingEnvironmentVariableError(ConfigurationError):
    def __init__(self, variable_name):
        super().__init__(f"Missing required environment variable: '{variable_name}'.")


class PipelineError(KazarrException):
    def __init__(self, message):
        super().__init__("PIPELINE_ERROR", message, None)


class LastOperationNotCompletedError(PipelineError):
    def __init__(self, history=None):
        message = "Pipeline stopped because last operation was not completed successfully, and data may be incomplete or corrupted."
        if history is not None and len(history) > 0:
            message = message + f" Last operation: {history[-1]}"
        super().__init__(message)


class DatasetError(KazarrException):
    def __init__(self, message):
        super().__init__("DATASET_ERROR", message, None)


class DatasetConfigurationError(DatasetError):
    def __init__(self, message):
        super().__init__(message=f"Dataset configuration error: {message}")


class DatasetLoadError(DatasetError):
    def __init__(self, path, message):
        super().__init__(f"Failed to load dataset from path '{path}': {message}")

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


class GenericInternalError(KazarrException):
    def __init__(self, message):
        super().__init__("GENERIC_INTERNAL_ERROR", message, None)


class ConfigurationBasedException(KazarrException):
    pass


class UserInputBasedException(KazarrException):
    pass


class MissingConfigurationElement(ConfigurationBasedException):
    def __init__(self, element_name):
        super().__init__(
            "MISSING_CONFIGURATION_ELEMENT",
            f"Missing required configuration element: '{element_name}'.",
            element_name,
        )


class MissingQueryParameter(UserInputBasedException):
    def __init__(self, parameter_names):
        if not isinstance(parameter_names, list):
            parameter_names = [parameter_names]
        if len(parameter_names) == 0:
            parameter_names = ["UNKNOWN"]

        if len(parameter_names) == 1:
            message = f"Missing required query parameter: '{parameter_names[0]}'."
        else:
            message = (
                "Missing required query parameters: '"
                + "', '".join(parameter_names[:-1])
                + "' and '"
                + parameter_names[-1]
                + "'."
            )
        super().__init__("MISSING_QUERY_PARAMETER", message, parameter_names)


class BadConfigurationVariable(ConfigurationBasedException):
    def __init__(self, variable_name):
        if not isinstance(variable_name, list):
            variable_name = [variable_name]
        if len(variable_name) == 0:
            variable_name = ["UNKNOWN"]

        if len(variable_name) == 1:
            message = f"Variable '{variable_name[0]}' not found in configuration or in dataset."
        if isinstance(variable_name, list) and len(variable_name) > 1:
            message = (
                "Variables '"
                + "', '".join(variable_name[:-1])
                + "' and '"
                + variable_name[-1]
                + "' not found in configuration or in dataset."
            )
        super().__init__("CONFIG_VARIABLE_NOT_FOUND", message, variable_name)


class TooManyDimensions(ConfigurationBasedException):
    def __init__(self, number_of_dimensions):
        super().__init__("TOO_MANY_DIMENSIONS", f"Latitude and longitude are {number_of_dimensions}D, but this dataset is configured as 2D.", number_of_dimensions)


class VariableCannotBeUsedForSelection(ConfigurationBasedException):
    def __init__(self, variable_name):
        super().__init__(
            "VARIABLE_CANNOT_BE_USED_FOR_SELECTION",
            f"Variable '{variable_name}' cannot be used for selection because it does have more than 1 dimension.",
            {"variable_name": variable_name},
        )

class DatasetNotFound(UserInputBasedException):
    def __init__(self, dataset_id):
        super().__init__(
            "DATASET_NOT_FOUND", f"Dataset '{dataset_id}' not found.", dataset_id
        )


class VariableNotFound(UserInputBasedException):
    def __init__(self, variable_name):
        if not isinstance(variable_name, list):
            variable_name = [variable_name]
        if len(variable_name) == 0:
            variable_name = ["UNKNOWN"]

        if len(variable_name) == 1:
            message = f"Variable '{variable_name[0]}' not found in dataset."
        if isinstance(variable_name, list) and len(variable_name) > 1:
            message = (
                "Variables '"
                + "', '".join(variable_name[:-1])
                + "' and '"
                + variable_name[-1]
                + "' not found in dataset."
            )
        super().__init__("VARIABLE_NOT_FOUND", message, variable_name)


class MissingDimensionsOrCoordinates(UserInputBasedException):
    def __init__(self, dimensions):
        message = "Missing required dimensions or coordinates: "
        elems = []
        payload = []
        for dim, coords in dimensions.items():
            elems.append(
                f"dimension '{dim}' or coordinate{'s' if len(coords) > 1 else ''} "
                + "', '".join(coords)
            )
            payload.append({"dimension": dim, "coordinates": coords})
        message += ", ".join(elems) + "."
        super().__init__("MISSING_DIMENSIONS_OR_COORDINATES", message, payload)


class TooFewPoints(UserInputBasedException):
    def __init__(self):
        super().__init__(
            "TOO_FEW_POINTS",
            "Not enough points with this bounding box to generate mesh data.",
        )


class NoDataInSelection(UserInputBasedException):
    def __init__(self, detail=""):
        if not detail or not isinstance(detail, str):
            detail = ""
        super().__init__(
            "NO_DATA_IN_SELECTION",
            f"No data available in the selected area and time. {detail}",
        )


class BadSelection(UserInputBasedException):
    def __init__(self, message):
        super().__init__("BAD_SELECTION", message)


class RequestCancelled(Exception):
    def __init__(self, message="Request was cancelled by the client"):
        super().__init__(message)


class DifferentTypesOfLevel(UserInputBasedException):
    def __init__(self):
        message = "Not all variables uses the same type of level."
        super().__init__("DIFFERENT_TYPES_OF_LEVEL", message)


class CantFindLevelVariable(UserInputBasedException):
    def __init__(self):
        message = "Unable to determine 'level' variable for 3D mesh generation."
        super().__init__("CANT_FIND_LEVEL_VARIABLE", message)


class InvalidTimeRange(UserInputBasedException):
    def __init__(self, message):
        super().__init__("INVALID_TIME_RANGE", message)


class InvalidDatetimeFormat(UserInputBasedException):
    def __init__(self, datetime_str):
        message = f"Invalid datetime format: '{datetime_str}'. Expected ISO 8601 format."
        super().__init__("INVALID_DATETIME_FORMAT", message, datetime_str)


class PathMissingTimes(UserInputBasedException):
    def __init__(self):
        message = (
            "Path mode ('path' or GeoJSON LineString) requires 'times' to be defined."
        )
        super().__init__("PATH_MISSING_TIMES", message)


class PathInvalidTimesLength(UserInputBasedException):
    def __init__(self, times, path_length):
        message = (
            f"Path mode requires exactly one time per point: "
            f"got {len(times)} times for {path_length} points."
        )
        super().__init__("PATH_INVALID_TIMES_LENGTH", message, {"times": times, "path_length": path_length})


class PathDoesNotSupportTimeRanges(UserInputBasedException):
    def __init__(self, invalid_times):
        message = (
            "Trajectory mode requires single timestamps, not time ranges. "
            f"Invalid values: {invalid_times}"
        )
        super().__init__("PATH_DOES_NOT_SUPPORT_TIME_RANGES", message, {"invalid_times": invalid_times})


class MultiProbeBodyMissingPoint(UserInputBasedException):
    def __init__(self):
        message = (
            "Body must contain 'points', 'path', or a GeoJSON FeatureCollection of Points/LineString."
        )
        super().__init__("MULTI_PROBE_BODY_MISSING_POINT", message)
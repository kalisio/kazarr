from fastapi import HTTPException

class KazarrException(Exception):
  def __init__(self, error_code, message, payload = None):
    super().__init__(message)
    self.error_code = error_code
    self.message = message
    self.payload = payload
  def get(self):
    out = { "error_code": self.error_code, "message": self.message }
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
    super().__init__("MISSING_CONFIGURATION_ELEMENT", f"Missing required configuration element: '{element_name}'.", element_name)

class BadConfigurationVariable(ConfigurationBasedException):
  def __init__(self, variable_name):
    if not isinstance(variable_name, list):
      variable_name = [variable_name]
    if len(variable_name) == 1:
      message = f"Variable '{variable_name[0]}' not found in configuration or in dataset."
    if isinstance(variable_name, list) and len(variable_name) > 1:
      message = "Variables '" + "', '".join(variable_name[:-1]) + "' and '" + variable_name[-1] + "' not found in configuration or in dataset."
    super().__init__("CONFIG_VARIABLE_NOT_FOUND", message, variable_name)

class DatasetNotFound(UserInputBasedException):
  def __init__(self, dataset_id):
    super().__init__("DATASET_NOT_FOUND", f"Dataset '{dataset_id}' not found.", dataset_id)

class VariableNotFound(UserInputBasedException):
  def __init__(self, variable_name):
    if not isinstance(variable_name, list):
      variable_name = [variable_name]
    if len(variable_name) == 1:
      message = f"Variable '{variable_name[0]}' not found in dataset."
    if isinstance(variable_name, list) and len(variable_name) > 1:
      message = "Variables '" + "', '".join(variable_name[:-1]) + "' and '" + variable_name[-1] + "' not found in dataset."
    super().__init__("VARIABLE_NOT_FOUND", message, variable_name)

class MissingDimensionsOrCoordinates(UserInputBasedException):
  def __init__(self, dimensions):
    message = "Missing required dimensions or coordinates: "
    elems = []
    payload = []
    for dim, coords in dimensions.items():
      elems.append(f"dimension '{dim}' or coordinate{'s' if len(coords) > 1 else ''} " + "', '".join(coords))
      payload.append({ "dimension": dim, "coordinates": coords })
    message += ", ".join(elems) + "."
    super().__init__("MISSING_DIMENSIONS_OR_COORDINATES", message, payload)

class TooFewPoints(UserInputBasedException):
  def __init__(self):
    super().__init__("TOO_FEW_POINTS", "Not enough points with this bounding box to generate mesh data.")

class NoDataInSelection(UserInputBasedException):
  def __init__(self):
    super().__init__("NO_DATA_IN_SELECTION", "No data available in the selected area and time.")

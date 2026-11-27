"""Show how the messages of a GRIB file map to the Xarray variables built by kazarr.

Templates select GRIB messages with GRIB keys (``filter_by_keys`` uses ``shortName``,
``stepType``, ``typeOfLevel``...), while the following processes (``keep_variables``,
``rename_variables``...) use the names of the Xarray variables. Those names come from
the ecCodes ``cfVarName`` (which depends on the ecCodes version and definitions) and may
be suffixed by kazarr to tell apart duplicates. This module lists both side by side.
"""

import json
import os

from kazarr.grib import (
    GRIB_TIME_RANGE_ATTRS,
    GRIB_TIME_RANGE_KEYS,
    grib_time_range_suffix,
)
from kazarr.processes import load_from_grib
from kazarr.utils import S3_PREFIX, load_custom_eccodes

try:
    import eccodes
except ImportError:
    eccodes = None

# Keys identifying a group of GRIB messages, also used in filter_by_keys
GROUP_KEYS = ("shortName", "stepType", "typeOfLevel")
UNKNOWN_NAMES = ("unknown", "undef")

NOTE_RENAMED = "renamed by kazarr"
NOTE_UNKNOWN = "unknown parameter (missing ecCodes definition?)"
NOTE_NOT_LOADED = "not loaded"


def _normalize(value):
    # ecCodes returns "unknown" for keys it cannot resolve, while cfgrib omits the attribute
    return "unknown" if value is None or value in UNKNOWN_NAMES else value


def _get_key(handle, key):
    try:
        return eccodes.codes_get(handle, key)
    except Exception:
        return None


def _time_range(values):
    """Time range suffix (e.g. '1h') from (lengthOfTimeRange, indicatorOfUnitForTimeRange)."""
    if all(isinstance(v, int) for v in values):
        return grib_time_range_suffix(*values)
    return None


def list_grib_message_groups(path):
    """Group the messages of a local GRIB file by shortName, stepType and typeOfLevel.

    Returns a dict {(shortName, stepType, typeOfLevel): {"paramId", "name", "messages",
    "time_ranges"}} in order of first appearance in the file, where "time_ranges"
    counts the messages of each time range (e.g. {"1h": 24, "3h": 8}).
    """
    if eccodes is None:
        raise ImportError(
            "eccodes is required to inspect GRIB files. Please install the tool with GRIB support: 'pip install .[grib]'."
        )
    groups = {}
    with open(path, "rb") as f:
        while (handle := eccodes.codes_grib_new_from_file(f)) is not None:
            try:
                key = tuple(_normalize(_get_key(handle, k)) for k in GROUP_KEYS)
                group = groups.setdefault(
                    key,
                    {
                        "paramId": _get_key(handle, "paramId"),
                        "name": _get_key(handle, "name"),
                        "messages": 0,
                        "time_ranges": {},
                    },
                )
                group["messages"] += 1
                time_range = _time_range(
                    [_get_key(handle, k) for k in GRIB_TIME_RANGE_KEYS]
                )
                group["time_ranges"][time_range] = (
                    group["time_ranges"].get(time_range, 0) + 1
                )
            finally:
                eccodes.codes_release(handle)
    return groups


def _variable_group_key(data_var):
    return tuple(_normalize(data_var.attrs.get(f"GRIB_{k}")) for k in GROUP_KEYS)


def _variable_time_range(data_var):
    return _time_range([data_var.attrs.get(a) for a in GRIB_TIME_RANGE_ATTRS])


def _base_name(data_var):
    """Name cfgrib gives to a variable, before any renaming by kazarr."""
    cf_var_name = data_var.attrs.get("GRIB_cfVarName")
    if cf_var_name and cf_var_name not in UNKNOWN_NAMES:
        return cf_var_name
    return data_var.attrs.get("GRIB_shortName")


def inspect_grib(path, backend_kwargs=None, custom_eccodes_path=None):
    """List the GRIB message groups of a file and the Xarray variable each one becomes.

    The file is loaded with load_from_grib, so variable names are the ones the
    pipelines will see. Data is not read, only metadata.

    Args:
        path (str): Path of a single GRIB file (local or s3://).
        backend_kwargs (dict, optional): cfgrib backend_kwargs, as in a template (e.g. filter_by_keys).
        custom_eccodes_path (str, optional): Path to custom ecCodes definitions.

    Returns:
        list[dict]: One row per message group or variable, with keys shortName,
        stepType, typeOfLevel, timeRange, paramId, name, messages, variable, dims and
        notes. timeRange is only set for variables split by time range or computed
        over a single one.
    """
    load_custom_eccodes(custom_eccodes_path)

    is_s3 = path.startswith(S3_PREFIX)
    if not is_s3 and not os.path.isfile(path):
        raise ValueError(f"'{path}' is not a file. inspect expects a single GRIB file.")

    config = {"load_path": path}
    if backend_kwargs:
        config["backend_kwargs"] = dict(backend_kwargs)
    dataset, _ = load_from_grib(None, config)

    # Raw messages are only listed for local files: S3 files are only opened through cfgrib
    groups = {} if is_s3 else list_grib_message_groups(path)

    variables_by_group = {}
    for var_name, data_var in dataset.data_vars.items():
        variables_by_group.setdefault(_variable_group_key(data_var), []).append(
            var_name
        )

    rows = []
    for key, group in groups.items():
        var_names = variables_by_group.pop(key, [None])
        split = len(var_names) > 1
        for var_name in var_names:
            row = _make_row(key, group, dataset, var_name)
            if split and row["timeRange"] in group["time_ranges"]:
                # Variable split by time range: only count the messages of its range
                row["messages"] = group["time_ranges"][row["timeRange"]]
            rows.append(row)

    # Variables that could not be matched to a group of messages (e.g. S3 files)
    for key, var_names in variables_by_group.items():
        for var_name in var_names:
            data_var = dataset[var_name]
            group = {
                "paramId": data_var.attrs.get("GRIB_paramId"),
                "name": data_var.attrs.get("GRIB_name"),
                "messages": None,
            }
            rows.append(_make_row(key, group, dataset, var_name))
    return rows


def _make_row(key, group, dataset, var_name):
    row = dict(zip(GROUP_KEYS, key, strict=True))
    row.update(
        timeRange=_variable_time_range(dataset[var_name]) if var_name else None,
        paramId=group["paramId"],
        name=group["name"],
        messages=group["messages"],
        variable=var_name,
        dims=list(dataset[var_name].dims) if var_name else [],
        notes=[],
    )
    if var_name is None:
        row["notes"].append(NOTE_NOT_LOADED)
    elif var_name != _base_name(dataset[var_name]):
        row["notes"].append(NOTE_RENAMED)
    if row["paramId"] == 0 or row["shortName"] in UNKNOWN_NAMES:
        row["notes"].append(NOTE_UNKNOWN)
    return row


def format_inspection(rows):
    """Format the result of inspect_grib as a text table."""
    headers = (
        "shortName",
        "stepType",
        "typeOfLevel",
        "timeRange",
        "paramId",
        "messages",
        "",
        "variable",
        "dims",
        "notes",
    )
    lines = [
        (
            str(row["shortName"]),
            str(row["stepType"]),
            str(row["typeOfLevel"]),
            row["timeRange"] or "-",
            str(row["paramId"]),
            "-" if row["messages"] is None else str(row["messages"]),
            "→",
            row["variable"] or "-",
            ", ".join(row["dims"]),
            "; ".join(row["notes"]),
        )
        for row in rows
    ]
    widths = [
        max([len(h)] + [len(line[i]) for line in lines]) for i, h in enumerate(headers)
    ]
    output = []
    for line in [headers, *lines]:
        output.append(
            "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(line)).rstrip()
        )
    return "\n".join(output)


def format_inspection_json(rows):
    return json.dumps(rows, indent=2, default=str)

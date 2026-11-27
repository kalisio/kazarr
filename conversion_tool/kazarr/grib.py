"""Opening of GRIB files with cfgrib, used by the GRIB loading processes.

cfgrib builds one Xarray dataset per group of compatible GRIB messages: the
variables sharing the same name across them are told apart here, and the
variables holding several statistical processing durations (time ranges) are
split into one variable per duration.
"""

import logging
import re

import xarray as xr

try:
    import cfgrib
except ImportError:
    cfgrib = None

logger = logging.getLogger(__name__)

# GRIB attributes used, in this order, to tell apart variables sharing the same name
# across the datasets returned by cfgrib.open_datasets (e.g. ptype avg / ptype max).
GRIB_DISAMBIGUATION_ATTRS = ("GRIB_stepType", "GRIB_typeOfLevel")
# Only short identifier-like values (instant, avg, max, surface, ...) are used as suffixes
_GRIB_SUFFIX_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9]{0,24}$")

# GRIB keys identifying a variable built by cfgrib
_GRIB_VARIABLE_KEYS = ("paramId", "stepType", "typeOfLevel")
# GRIB keys giving the duration of a statistical processing (average, maximum, accumulation...)
GRIB_TIME_RANGE_KEYS = ("lengthOfTimeRange", "indicatorOfUnitForTimeRange")
GRIB_TIME_RANGE_ATTRS = tuple(f"GRIB_{k}" for k in GRIB_TIME_RANGE_KEYS)
# Code table 4.4 (indicator of unit of time range): (suffix, multiplier)
_GRIB_TIME_RANGE_UNITS = {
    0: ("min", 1),
    1: ("h", 1),
    2: ("d", 1),
    10: ("h", 3),
    11: ("h", 6),
    12: ("h", 12),
    13: ("s", 1),
}


def disambiguate_grib_variables(datasets):
    """Rename variables found in several datasets so they can be merged.

    Every occurrence of a duplicated variable is suffixed with the value of the first
    attribute of GRIB_DISAMBIGUATION_ATTRS that differs between all occurrences
    (ptype -> ptype_avg / ptype_max), so names do not depend on the order of the
    messages in the GRIB file. When no attribute fits, falls back to a counter
    (ptype, ptype_2, ...), which does depend on that order.

    Returns a new list of datasets, input datasets are not modified.
    """
    occurrences = {}
    for i, ds in enumerate(datasets):
        for var in ds.data_vars:
            occurrences.setdefault(var, []).append(i)

    taken = set()
    for ds in datasets:
        taken.update(ds.data_vars)
        taken.update(ds.coords)

    renames = [{} for _ in datasets]
    for var, indexes in occurrences.items():
        if len(indexes) < 2:
            continue

        new_names = None
        for attr in GRIB_DISAMBIGUATION_ATTRS:
            values = [datasets[i][var].attrs.get(attr) for i in indexes]
            if not all(
                isinstance(v, str) and _GRIB_SUFFIX_PATTERN.match(v) for v in values
            ):
                continue
            if len(set(values)) != len(values):
                continue
            candidates = [f"{var}_{v}" for v in values]
            if taken.isdisjoint(candidates):
                new_names = candidates
                break

        if new_names is None:
            logger.warning(
                "         ! No GRIB attribute among %s tells apart the %d occurrences of %s, "
                "falling back to numbered names which depend on the order of the GRIB messages.",
                GRIB_DISAMBIGUATION_ATTRS,
                len(indexes),
                var,
            )
            # Keep the original name for the first occurrence, as before
            new_names = [var]
            counter = 2
            for _ in indexes[1:]:
                while f"{var}_{counter}" in taken:
                    counter += 1
                new_names.append(f"{var}_{counter}")
                counter += 1

        taken.update(new_names)
        for i, new_name in zip(indexes, new_names, strict=True):
            if new_name != var:
                renames[i][var] = new_name
                logger.info(
                    "         ! Renaming %s to %s in dataset %d to avoid conflicts.",
                    var,
                    new_name,
                    i,
                )

    return [
        ds.rename(rename) if rename else ds
        for ds, rename in zip(datasets, renames, strict=True)
    ]


def _open_grib_file_as_dataset(path, backend_kwargs):
    """Open a local GRIB file with cfgrib, merging the datasets it splits it into if needed."""
    backend_kwargs = {**backend_kwargs, "errors": "raise"}
    try:
        return xr.open_dataset(
            path, engine="cfgrib", chunks="auto", backend_kwargs=backend_kwargs
        )
    except cfgrib.dataset.DatasetBuildError:
        logger.warning(
            "Failed to load GRIB file with Xarray, trying to open as multiple datasets... This may cause higher memory usage and slower performance."
        )
    datasets = cfgrib.open_datasets(path, backend_kwargs=backend_kwargs)
    if len(datasets) == 1:
        return datasets[0].chunk("auto")

    try:
        logger.info("Multiple datasets found in GRIB file, attempting to merge them...")
        datasets = disambiguate_grib_variables(datasets)
        full_variables = set()
        for ds in datasets:
            full_variables.update(ds.data_vars)
            full_variables.update(ds.coords)
        # Use chunk("auto") on each dataset to enable Dask parallelism during merge, which can help reduce memory usage and speed up the process
        datasets = [ds.chunk("auto") for ds in datasets]
        new_dataset = xr.merge(datasets, compat="minimal", join="outer")
        removed_variables = (
            full_variables - set(new_dataset.data_vars) - set(new_dataset.coords)
        )
        if removed_variables:
            logger.info(
                "         ! The following variables were removed during merge due to conflicts: %s",
                removed_variables,
            )
    except Exception:
        logger.exception("Error occurred while merging datasets.")
        return None
    return new_dataset


def grib_time_range_suffix(length, unit):
    """Suffix describing a GRIB time range, e.g. (1, 1) -> '1h', (15, 0) -> '15min'."""
    name, factor = _GRIB_TIME_RANGE_UNITS.get(unit, (None, None))
    if name is None:
        return f"{length}unit{unit}"
    return f"{length * factor}{name}"


def _grib_variable_key(data_var):
    return tuple(data_var.attrs.get(f"GRIB_{k}") for k in _GRIB_VARIABLE_KEYS)


def list_grib_time_ranges(path, backend_kwargs):
    """List the time ranges of the statistically processed fields of a local GRIB file.

    Returns a dict {(paramId, stepType, typeOfLevel): {"ranges": [(lengthOfTimeRange,
    indicatorOfUnitForTimeRange), ...], "overlapping": bool}}, restricted to the
    messages selected by the filter_by_keys of backend_kwargs. "overlapping" is True
    when messages with different time ranges end at the same step, so they cannot be
    stored in a single variable indexed by step. Accumulations since the start of the
    forecast (0-1, 0-2...) have a different time range at each step but do not overlap.
    """
    index_kwargs = {}
    if "indexpath" in backend_kwargs:
        index_kwargs["indexpath"] = backend_kwargs["indexpath"]
    index = cfgrib.dataset.open_fileindex(
        cfgrib.messages.FileStream(path),
        index_keys=[*_GRIB_VARIABLE_KEYS, *GRIB_TIME_RANGE_KEYS, "time", "endStep"],
        filter_by_keys=backend_kwargs.get("filter_by_keys", {}),
        **index_kwargs,
    )
    time_ranges = {}
    for header_values, _ in index.iter_index():
        values = dict(zip(index.index_keys, header_values, strict=True))
        time_range = tuple(values[k] for k in GRIB_TIME_RANGE_KEYS)
        # Fields which are not statistically processed have no time range ("undef")
        if not all(isinstance(v, int) for v in time_range):
            continue
        key = tuple(values[k] for k in _GRIB_VARIABLE_KEYS)
        info = time_ranges.setdefault(key, {"ranges": {}, "ends": {}})
        info["ranges"][time_range] = None
        info["ends"].setdefault((values["time"], values["endStep"]), set()).add(
            time_range
        )
    return {
        key: {
            "ranges": list(info["ranges"]),
            "overlapping": any(len(r) > 1 for r in info["ends"].values()),
        }
        for key, info in time_ranges.items()
    }


def _set_time_range_attrs(data_var, time_range):
    data_var.attrs.update(dict(zip(GRIB_TIME_RANGE_ATTRS, time_range, strict=True)))


def open_grib_file(path, backend_kwargs=None):
    """Open a local GRIB file, splitting the variables mixing overlapping time ranges.

    cfgrib does not tell apart the messages of a parameter computed over different
    durations (e.g. ptype averaged over 1h and over 3h): they end up in a single
    variable indexed by the end of their time range, and messages ending at the same
    step are silently dropped. Such variables are loaded once per time range and
    suffixed with it (ptype_avg -> ptype_avg_1h / ptype_avg_3h). Accumulations since
    the start of the forecast (0-1, 0-2...) never end at the same step, so they are
    left untouched.

    Variables computed over a single time range get GRIB_lengthOfTimeRange and
    GRIB_indicatorOfUnitForTimeRange attributes.
    """
    backend_kwargs = backend_kwargs or {}
    time_ranges = list_grib_time_ranges(path, backend_kwargs)
    new_dataset = _open_grib_file_as_dataset(path, backend_kwargs)
    if new_dataset is None or not time_ranges:
        return new_dataset

    filter_by_keys = backend_kwargs.get("filter_by_keys", {})
    split_datasets = []
    for key, info in time_ranges.items():
        ranges = info["ranges"]
        var_names = [
            name
            for name, data_var in new_dataset.data_vars.items()
            if _grib_variable_key(data_var) == key
        ]
        if len(var_names) != 1:
            continue
        var = var_names[0]
        if len(ranges) == 1:
            _set_time_range_attrs(new_dataset[var], ranges[0])
        if not info["overlapping"]:
            continue

        new_names = [f"{var}_{grib_time_range_suffix(*r)}" for r in ranges]
        taken = set(new_dataset.data_vars) | set(new_dataset.coords)
        if len(set(new_names)) != len(new_names) or not taken.isdisjoint(new_names):
            logger.warning(
                "         ! %s mixes overlapping time ranges but cannot be split into %s, "
                "messages ending at the same step may be missing.",
                var,
                new_names,
            )
            continue

        logger.info(
            "         ! %s mixes overlapping time ranges, splitting it into %s.",
            var,
            ", ".join(new_names),
        )
        new_dataset = new_dataset.drop_vars(var)
        for time_range, new_name in zip(ranges, new_names, strict=True):
            split_filter = {
                **filter_by_keys,
                **dict(zip(_GRIB_VARIABLE_KEYS, key, strict=True)),
                **dict(zip(GRIB_TIME_RANGE_KEYS, time_range, strict=True)),
            }
            split_dataset = _open_grib_file_as_dataset(
                path, {**backend_kwargs, "filter_by_keys": split_filter}
            )
            (split_var,) = split_dataset.data_vars
            split_dataset = split_dataset.rename({split_var: new_name})
            _set_time_range_attrs(split_dataset[new_name], time_range)
            split_datasets.append(split_dataset.chunk("auto"))

    if not split_datasets:
        return new_dataset
    return xr.merge(
        [new_dataset, *split_datasets],
        compat="minimal",
        join="outer",
        combine_attrs="override",
    )

import math
from dataclasses import dataclass

import contourpy
import numpy as np

from src import exceptions
from src.utils import serialization

# Maximum number of thresholds a request can produce (protects against tiny steps)
MAX_THRESHOLDS = 1000
# Decimals kept for thresholds generated from a range (avoids 0.30000000000000004)
THRESHOLD_DECIMALS = 10


@dataclass
class ThresholdRange:
    """Thresholds defined as `min:max:step`. A missing bound is taken from the data."""

    min: float | None
    max: float | None
    step: float


def _parse_number(value, raw):
    try:
        number = float(value)
    except ValueError as e:
        raise exceptions.InvalidThresholds(raw, f"'{value}' is not a number.") from e
    if not math.isfinite(number):
        raise exceptions.InvalidThresholds(raw, f"'{value}' is not a finite number.")
    return number


def parse_thresholds(raw_thresholds: list[str]) -> list[float] | ThresholdRange:
    """Parse the `thresholds` query parameter.

    Either a list of values (`thresholds=0&thresholds=5`), or a single range
    `min:max:step` where min and/or max may be omitted (`:30:5`, `10::5`, `::5`).
    """
    raw_thresholds = [item.strip() for item in raw_thresholds if item.strip() != ""]
    if not raw_thresholds:
        raise exceptions.MissingQueryParameter("thresholds")

    if not any(":" in item for item in raw_thresholds):
        thresholds = []
        for item in raw_thresholds:
            threshold = _parse_number(item, item)
            if threshold not in thresholds:
                thresholds.append(threshold)
        return thresholds

    if len(raw_thresholds) > 1:
        raise exceptions.InvalidThresholds(
            raw_thresholds,
            "A range 'min:max:step' can't be combined with other thresholds.",
        )
    raw = raw_thresholds[0]
    parts = [part.strip() for part in raw.split(":")]
    if len(parts) != 3:
        raise exceptions.InvalidThresholds(raw, "Expected format: 'min:max:step'.")
    threshold_min = _parse_number(parts[0], raw) if parts[0] else None
    threshold_max = _parse_number(parts[1], raw) if parts[1] else None
    if not parts[2]:
        raise exceptions.InvalidThresholds(raw, "The step is required.")
    step = _parse_number(parts[2], raw)
    if step <= 0:
        raise exceptions.InvalidThresholds(raw, "The step must be strictly positive.")
    if (
        threshold_min is not None
        and threshold_max is not None
        and threshold_min > threshold_max
    ):
        raise exceptions.InvalidThresholds(raw, "min must be lower than or equal to max.")
    return ThresholdRange(threshold_min, threshold_max, step)


def resolve_thresholds(thresholds: list[float] | ThresholdRange, val) -> list[float]:
    """Compute the thresholds of a ThresholdRange (explicit lists are returned as is).

    - Explicit min: thresholds are min, min + step, min + 2*step, ...
    - Missing min: the first threshold is the first multiple of step greater than
      or equal to the minimum of the data, so that thresholds are round values and
      stay the same from one time step (or bbox) to another.
    - max (explicit, or the maximum of the data when missing) is inclusive.
    """
    if not isinstance(thresholds, ThresholdRange):
        return thresholds

    step = thresholds.step
    data_min, data_max = None, None
    if thresholds.min is None or thresholds.max is None:
        finite = np.asarray(val, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            raise exceptions.NoDataInSelection(
                "Thresholds can't be computed from the data: no valid value in the selection."
            )
        data_min, data_max = float(finite.min()), float(finite.max())

    threshold_max = thresholds.max if thresholds.max is not None else data_max
    if thresholds.min is not None:
        start = thresholds.min
    else:
        k = math.ceil(data_min / step)
        # Floating point noise may push the quotient just above an integer
        # (e.g. 0.3 / 0.1): step back if the previous threshold is still >= min.
        if round((k - 1) * step, THRESHOLD_DECIMALS) >= data_min:
            k -= 1
        start = k * step

    count = math.floor((threshold_max - start) / step) + 1
    # Same floating point issue for the max (inclusive), e.g. (0.8 - 0.2) / 0.2
    if round(start + count * step, THRESHOLD_DECIMALS) <= threshold_max:
        count += 1
    if count <= 0:
        return []
    if count > MAX_THRESHOLDS:
        raise exceptions.InvalidThresholds(
            f"{thresholds.min if thresholds.min is not None else ''}:"
            f"{thresholds.max if thresholds.max is not None else ''}:{step}",
            f"This range produces {count} thresholds, the maximum is {MAX_THRESHOLDS}. Use a larger step.",
        )
    return [round(start + i * step, THRESHOLD_DECIMALS) for i in range(count)]


def generate_isolines(lon, lat, val, thresholds):
    """Compute isolines of a 2D field with contourpy.

    lon/lat are either 1D (len(lon) == val.shape[1], len(lat) == val.shape[0])
    or 2D with the same shape as val. NaN values are masked.

    Returns, for each threshold (in the same order), the list of its lines, each
    line being a (n, 2) numpy array of [lon, lat] points. Disconnected lines are
    kept separated. Lines are kept as numpy arrays (no conversion to Python
    lists), to be serialized with serialize_isolines.
    """
    z = np.asarray(val, dtype=np.float64)
    invalid = ~np.isfinite(z)
    if invalid.all():
        return [[] for _ in thresholds]
    if invalid.any():
        z = np.ma.array(z, mask=invalid)

    generator = contourpy.contour_generator(
        np.asarray(lon, dtype=np.float64),
        np.asarray(lat, dtype=np.float64),
        z,
        name="serial",
        line_type=contourpy.LineType.ChunkCombinedOffset,
    )

    isolines = []
    for threshold in thresholds:
        points_chunks, offsets_chunks = generator.lines(threshold)
        lines = []
        for points, offsets in zip(points_chunks, offsets_chunks, strict=True):
            if points is None:
                continue
            lines.extend(np.split(points, offsets[1:-1]))
        isolines.append(lines)
    return isolines


def format_isoline_raw(isolines, thresholds):
    out = {}
    for i, threshold in enumerate(thresholds):
        out[threshold] = isolines[i]
    return out


def isoline_geometry(lines):
    """GeoJSON geometry of the lines of a threshold: a LineString when there is a
    single line, a MultiLineString otherwise (including when there is no line,
    as an empty MultiLineString)."""
    if len(lines) == 1:
        return {"type": "LineString", "coordinates": lines[0]}
    return {"type": "MultiLineString", "coordinates": lines}


def format_isoline_geojson(isolines, thresholds):
    features = []
    for i, threshold in enumerate(thresholds):
        features.append(
            {
                "type": "Feature",
                "geometry": isoline_geometry(isolines[i]),
                "properties": {"id": i, "threshold": threshold},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def serialize_isolines(out) -> bytes:
    """Serialize a formatted output (raw or geojson) to JSON bytes."""
    return serialization.dumps(out)

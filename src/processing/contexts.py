from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.exceptions import InvalidDatetimeFormat, InvalidTimeRange


@dataclass
class BBoxContext:
    lon_min: float | None = None
    lat_min: float | None = None
    lon_max: float | None = None
    lat_max: float | None = None
    level_min: float | None = None
    level_max: float | None = None
    has_bb_lon: bool = False
    has_bb_lat: bool = False
    has_bb_level: bool = False
    has_bb: bool = False

    @classmethod
    def from_tuple(cls, bbox: tuple | None):
        if bbox is None:
            return cls()
        if len(bbox) >= 6:
            lon_min, lat_min, lon_max, lat_max, level_min, level_max = bbox[:6]
        else:
            lon_min, lat_min, lon_max, lat_max = bbox[:4]
            level_min, level_max = None, None
        has_bb_lon = lon_min is not None or lon_max is not None
        has_bb_lat = lat_min is not None or lat_max is not None
        has_bb_level = level_min is not None or level_max is not None
        has_bb = has_bb_lon or has_bb_lat
        return cls(
            lon_min,
            lat_min,
            lon_max,
            lat_max,
            level_min,
            level_max,
            has_bb_lon,
            has_bb_lat,
            has_bb_level,
            has_bb,
        )


@dataclass
class GridIndices:
    col_min: int = 0
    col_max: int = 0
    row_min: int = 0
    row_max: int = 0
    level_min: int = 0
    level_max: int = 0
    width_raw: int = 1
    height_raw: int = 1
    depth_raw: int = 1
    point_indices: np.ndarray | None = None
    n_points: int = 0
    step_row: int = 1
    step_col: int = 1
    step_level: int = 1


@dataclass
class TimeRange:
    start: str | None = None
    end: str | None = None
    has_time_range: bool = False
    # Set by validate(): True when the requested time / range
    # does not overlap the dataset time extent at all.
    is_out_of_bounds: bool = False

    @classmethod
    def from_string(cls, time_range: str | None):
        if not time_range:
            return cls()

        if "/" in time_range:
            splitted_range = time_range.split("/")
            if len(splitted_range) != 2:
                raise InvalidTimeRange(
                    "Time range should be in the format 'start/end' or a single time value."
                )
            start_raw, end_raw = splitted_range
            has_time_range = True
        else:
            start_raw, end_raw = time_range, None
            has_time_range = False

        def parse_and_strip_tz(time_str):
            if time_str in ("..", "", None):
                return None
            try:
                dt = pd.to_datetime(time_str)
                if dt.tz is not None:
                    dt = dt.tz_localize(None)

                return dt.isoformat()
            except (ValueError, TypeError):
                raise InvalidDatetimeFormat(time_str)

        start = parse_and_strip_tz(start_raw)
        end = parse_and_strip_tz(end_raw)

        if start is None and end is None:
            has_time_range = False

        return cls(start, end, has_time_range)

    def get_indexer(self):
        if self.has_time_range:
            return slice(self.start, self.end)
        else:
            return self.start
        
    def has_time(self):
        return self.start is not None or (self.end is not None and self.has_time_range)

    def validate(self, min_time, max_time, dtype=None) -> bool:
        """
        Compute and store whether this time / range falls entirely outside
        [min_time, max_time]. Returns the new value of is_out_of_bounds.

        - Single time value: out of bounds if before min_time or after max_time.
        - Time range: out of bounds if it does not overlap [min_time, max_time]
          (a None start/end is treated as unbounded on that side).
        - No time at all: never out of bounds (whole dataset).
        """

        def to_time(t_val):
            if t_val is None:
                return None
            return np.array(t_val, dtype=dtype) if dtype is not None else t_val

        start = to_time(self.start)
        end = to_time(self.end)

        if not self.has_time_range:
            self.is_out_of_bounds = start is not None and bool(
                start < min_time or start > max_time
            )
        else:
            self.is_out_of_bounds = bool(
                (end is not None and end < min_time)
                or (start is not None and start > max_time)
            )
        return self.is_out_of_bounds


@dataclass
class MultiTimeRange:
    ranges: list[TimeRange] | None

    def __post_init__(self):
        if self.ranges is None:
            self.ranges = []

    @classmethod
    def from_strings(cls, times: list[str] | str | None):
        if not times:
            return cls([TimeRange.from_string(None)])
        if isinstance(times, str):
            times = [times]

        ranges = []
        for t in times:
            ranges.append(TimeRange.from_string(t))

        if not ranges:
            ranges = [TimeRange.from_string(None)]

        return cls(ranges)

    def has_time(self):
        return any(r.has_time() for r in self.ranges)

    def validate(self, min_time, max_time, dtype=None) -> bool:
        """
        Compute and store is_out_of_bounds on every sub-range.
        Returns True if all sub-ranges are out of bounds.
        """
        for r in self.ranges:
            r.validate(min_time, max_time, dtype)
        return self.all_out_of_bounds()

    def all_out_of_bounds(self) -> bool:
        return bool(self.ranges) and all(r.is_out_of_bounds for r in self.ranges)

    def any_out_of_bounds(self) -> bool:
        return any(r.is_out_of_bounds for r in self.ranges)

    def get_out_of_bounds_ranges(self) -> list[TimeRange]:
        return [r for r in self.ranges if r.is_out_of_bounds]

    def get_in_bounds_ranges(self) -> list[TimeRange]:
        return [r for r in self.ranges if not r.is_out_of_bounds]

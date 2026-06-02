from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.exceptions import InvalidTimeRange, InvalidDatetimeFormat


@dataclass
class BBoxContext:
    lon_min: float | None = None
    lat_min: float | None = None
    lon_max: float | None = None
    lat_max: float | None = None
    z_min: float | None = None
    z_max: float | None = None
    has_bb_lon: bool = False
    has_bb_lat: bool = False
    has_bb_z: bool = False
    has_bb: bool = False

    @classmethod
    def from_tuple(cls, bbox: tuple | None):
        if bbox is None:
            return cls()
        if len(bbox) >= 6:
            lon_min, lat_min, lon_max, lat_max, z_min, z_max = bbox[:6]
        else:
            lon_min, lat_min, lon_max, lat_max = bbox[:4]
            z_min, z_max = None, None
        has_bb_lon = lon_min is not None or lon_max is not None
        has_bb_lat = lat_min is not None or lat_max is not None
        has_bb_z = z_min is not None or z_max is not None
        has_bb = has_bb_lon or has_bb_lat
        return cls(
            lon_min,
            lat_min,
            lon_max,
            lat_max,
            z_min,
            z_max,
            has_bb_lon,
            has_bb_lat,
            has_bb_z,
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

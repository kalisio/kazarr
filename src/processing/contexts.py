from dataclasses import dataclass
import numpy as np


@dataclass
class BBoxContext:
    lon_min: float | None = None
    lat_min: float | None = None
    lon_max: float | None = None
    lat_max: float | None = None
    has_bb_lon: bool = False
    has_bb_lat: bool = False
    has_bb: bool = False

    @classmethod
    def from_tuple(cls, bbox: tuple | None):
        lon_min, lat_min, lon_max, lat_max = (
            (None, None, None, None) if bbox is None else bbox
        )
        has_bb_lon = lon_min is not None or lon_max is not None
        has_bb_lat = lat_min is not None or lat_max is not None
        has_bb = has_bb_lon or has_bb_lat
        return cls(lon_min, lat_min, lon_max, lat_max, has_bb_lon, has_bb_lat, has_bb)


@dataclass
class GridIndices:
    col_min: int = 0
    col_max: int = 0
    row_min: int = 0
    row_max: int = 0
    width_raw: int = 1
    height_raw: int = 1
    point_indices: np.ndarray | None = None
    n_points: int = 0
    step_row: int = 1
    step_col: int = 1

from dataclasses import dataclass
from typing import List, Literal, Optional
from fastapi import Query, Path
from pydantic import BaseModel, model_validator

from src.exceptions import (
    PathMissingTimes,
    PathInvalidTimesLength,
    PathDoesNotSupportTimeRanges,
    MultiProbeBodyMissingPoint,
)


@dataclass
class BaseParams:
    dataset: str = Path(..., description="The path to the dataset to extract data from")
    variable: str = Query(..., description="Name of the variable to query")
    format: Literal["raw", "geojson"] = Query(
        "raw",
        description="The format of the extracted data (Currently supported: 'raw', 'geojson')",
    )
    interp_vars: list[str] = Query(
        [],
        description="List of variables to interpolate",
    )
    interp_vars_method: Literal[
        "linear",
        "nearest",
        "zero",
        "slinear",
        "quadratic",
        "cubic",
        "quintic",
        "polynomial",
        "pchip",
        "barycentric",
        "krogh",
        "akima",
        "makima",
    ] = Query(
        "nearest",
        description="The method to use for interpolation of variables.",
    )
    interp_vars_params: str | None = Query(
        None,
        description='Interpolation configuration. Must be defined with : "interpolation=optparam1:VALUE1,optparam2:VALUE2,..." where optparam are optional parameters depending on the method.',
    )
    as_dims: list[str] = Query(
        [],
        description="If some variables have the same name as dimensions, will force them to be treated as dimensions",
    )


@dataclass
class MultipleVariablesParams(BaseParams):
    variable: str = Query(
        None,
        description="Name of the variable to query. One of 'variable' or 'variables' must be provided",
    )
    variables: list[str] = Query(
        None,
        description="List of variables to query. One of 'variable' or 'variables' must be provided",
    )


@dataclass
class MeshFormatParams(BaseParams):
    format: Literal["raw", "geojson", "mesh"] = Query(
        "raw",
        description="The format of the extracted data (Currently supported: 'raw', 'geojson', 'mesh' with additional parameters)",
    )


@dataclass
class MultipleVariablesMeshFormatParams(MultipleVariablesParams):
    format: Literal["raw", "geojson", "mesh"] = Query(
        "raw",
        description="The format of the extracted data (Currently supported: 'raw', 'geojson', 'mesh' with additional parameters)",
    )


@dataclass
class BBoxParams:
    lon_min: float | None = Query(
        None, description="Minimum longitude of the bounding box"
    )
    lat_min: float | None = Query(
        None, description="Minimum latitude of the bounding box"
    )
    lon_max: float | None = Query(
        None, description="Maximum longitude of the bounding box"
    )
    lat_max: float | None = Query(
        None, description="Maximum latitude of the bounding box"
    )
    level_min: float | None = Query(
        None,
        description="Minimum vertical coordinate (altitude/depth) of the bounding box",
    )
    level_max: float | None = Query(
        None,
        description="Maximum vertical coordinate (altitude/depth) of the bounding box",
    )


@dataclass
class TimeParams:
    time: str | None = Query(
        None,
        description="Time for which to retrieve data.",
    )
    interp_time: bool = Query(
        False,
        description="Whether to interpolate values on time dimension or to get the closest time step. Shortcut to interp_vars=TIME_VARIABLE_NAME.",
    )

@dataclass
class MultiTimeParams(TimeParams):
    times: list[str] | None = Query(
        None,
        description="List of times for which to retrieve data.",
    )


@dataclass
class MeshParams:
    mesh_tile_size: int | None = Query(
        None,
        description="[format='mesh'] The size of the mesh tile to use when extracting data",
    )
    mesh_data_mapping: str | None = Query(
        "vertices",
        description="[format='mesh'] Whether the data of the mesh is on cells or on vertices. This will override the dataset configuration.",
    )


@dataclass
class SpatialInterpolationParams:
    interp_spatial_method: Literal["nearest", "linear", "cubic", "idw", "rbf"] = Query(
        "nearest",
        description="The method to use for spatial interpolation.",
    )
    interp_spatial_params: str = Query(
        "padding:1.0",
        description='Interpolation configuration. Must be defined with : "interpolation=padding:FLOAT_COEFF,optparam1:VALUE1,optparam2:VALUE2,..."',
    )


class ProbePoint(BaseModel):
    lon: float
    lat: float
    level: Optional[float] = None


class GeoJSONGeometry(BaseModel):
    type: str
    coordinates: list

    def as_probe_point(self) -> "ProbePoint":
        """Convert a Point geometry to a ProbePoint."""
        return ProbePoint(
            lon=self.coordinates[0],
            lat=self.coordinates[1],
            level=self.coordinates[2] if len(self.coordinates) > 2 else None,
        )

    def as_probe_points(self) -> "List[ProbePoint]":
        """Convert a LineString geometry to a list of ProbePoints."""
        return [
            ProbePoint(
                lon=coord[0],
                lat=coord[1],
                level=coord[2] if len(coord) > 2 else None,
            )
            for coord in self.coordinates
        ]


class GeoJSONFeature(BaseModel):
    type: str
    geometry: GeoJSONGeometry
    properties: Optional[dict] = None


class GeoJSONFeatureCollection(BaseModel):
    type: str
    features: list[GeoJSONFeature]


class MultiProbeBody(BaseModel):
    # ad hoc format — normal mode (all times for each point)
    points: List[ProbePoint] | None = None
    # ad hoc format — trajectory mode (time i for point i)
    path: List[ProbePoint] | None = None
    times: list[str] | None = None
    # GeoJSON FeatureCollection format
    type: str | None = None
    features: list[GeoJSONFeature] | None = None
    # Resolved after validation
    is_path: bool = False

    @model_validator(mode="after")
    def resolve_points(self) -> "MultiProbeBody":
        if self.type == "FeatureCollection" and self.features is not None:
            line_string_points: List[ProbePoint] = []
            point_points: List[ProbePoint] = []
            for f in self.features:
                if f.geometry.type == "LineString":
                    line_string_points.extend(f.geometry.as_probe_points())
                    self.times = f.properties.get("times") if f.properties else None
                    break  # Only one LineString is allowed, so we can stop after the first one
                elif f.geometry.type == "Point":
                    point_points.append(f.geometry.as_probe_point())

            if line_string_points:
                self.path = line_string_points
            elif point_points:
                self.points = point_points

        if self.path is not None:
            self.is_path = True
            if not self.times:
                raise PathMissingTimes()
            if len(self.times) != len(self.path):
                raise PathInvalidTimesLength(self.times, len(self.path))
            invalid_times = [t for t in self.times if "/" in t]
            if invalid_times:
                raise PathDoesNotSupportTimeRanges(invalid_times)
        elif not self.points:
            raise MultiProbeBodyMissingPoint()
        return self

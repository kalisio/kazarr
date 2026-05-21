from dataclasses import dataclass
from typing import List, Literal, Optional
from fastapi import Query, Path
from pydantic import BaseModel, model_validator


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
    interp_vars_method: Literal["nearest", "linear", "cubic", "idw", "rbf"] = Query(
        "nearest",
        description="The method to use for interpolation of variables. Supported values are 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'quintic', 'polynomial', 'pchip', 'barycentric', 'krogh', 'akima', 'makima'. Default is 'nearest'.",
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
    z_min: float | None = Query(
        None,
        description="Minimum vertical coordinate (altitude/depth) of the bounding box",
    )
    z_max: float | None = Query(
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
        description="The method to use for spatial interpolation. Supported values are 'nearest', 'linear', 'cubic', 'idw' and 'rbf'. Default is 'linear'.",
    )
    interp_spatial_params: str = Query(
        "padding:1.0",
        description='Interpolation configuration. Must be defined with : "interpolation=padding:FLOAT_COEFF,optparam1:VALUE1,optparam2:VALUE2,..."',
    )


class ProbePoint(BaseModel):
    lon: float
    lat: float
    height: Optional[float] = None


class GeoJSONPoint(BaseModel):
    type: str
    coordinates: list[float]


class GeoJSONFeature(BaseModel):
    type: str
    geometry: GeoJSONPoint


class GeoJSONFeatureCollection(BaseModel):
    type: str
    features: list[GeoJSONFeature]


class MultiProbeBody(BaseModel):
    # ad hoc format
    points: List[ProbePoint] | None = None
    # GeoJSON FeatureCollection format
    type: str | None = None
    features: list[GeoJSONFeature] | None = None

    @model_validator(mode="after")
    def resolve_points(self) -> "MultiProbeBody":
        if self.type == "FeatureCollection" and self.features is not None:
            self.points = [
                ProbePoint(
                    lon=f.geometry.coordinates[0],
                    lat=f.geometry.coordinates[1],
                    height=f.geometry.coordinates[2]
                    if len(f.geometry.coordinates) > 2
                    else None,
                )
                for f in self.features
                if f.geometry.type == "Point"
            ]
        if not self.points:
            raise ValueError(
                "Body must contain 'points' or a GeoJSON FeatureCollection of Points"
            )
        return self

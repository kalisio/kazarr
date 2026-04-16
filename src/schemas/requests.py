from dataclasses import dataclass, fields, make_dataclass
from typing import Literal, Type, TypeVar
from fastapi import Query, Path
from pydantic_core import PydanticUndefined


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
    variable: str = Query(None, description="Name of the variable to query. One of 'variable' or 'variables' must be provided")
    variables: list[str] = Query(None, description="List of variables to query. One of 'variable' or 'variables' must be provided")


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


T = TypeVar("T")


def clone_with_hidden_fields(
    cls: Type[T], new_name: str = None, except_fields: list[str] = None
) -> Type[T]:
    if new_name is None:
        new_name = f"{cls.__name__}Hidden"

    new_fields = []

    for f in fields(cls):
        old_param = f.default

        if hasattr(old_param, "default"):
            param_cls = type(old_param)
            default_val = (
                ... if old_param.default is PydanticUndefined else old_param.default
            )
            description = getattr(old_param, "description", None)
        else:
            param_cls = Query
            default_val = old_param
            description = None

        show_in_schema = (
            False if except_fields is None or f.name not in except_fields else True
        )

        new_param = param_cls(
            default=default_val,
            description=description,
            include_in_schema=show_in_schema,
        )

        new_fields.append((f.name, f.type, new_param))

    return make_dataclass(new_name, new_fields)

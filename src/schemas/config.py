from typing import Literal, Any
from pydantic import BaseModel, Field


class MeshConfig(BaseModel):
    tile_shape: tuple[int, int] | None = None
    data_mapping: Literal["vertices", "cells"] = "vertices"


class InterpolationVarsConfig(BaseModel):
    items: list[str] = Field(default_factory=list)
    time: bool = False
    method: str = "nearest"
    params: dict[str, Any] | None = Field(default_factory=dict)


class InterpolationSpatialConfig(BaseModel):
    method: str = "nearest"
    # Setting padding to 1.0 as default in spatial interpolation parameters
    params: dict[str, Any] | None = Field(default_factory=lambda: {"padding": 1.0})


class InterpolationConfig(BaseModel):
    vars: InterpolationVarsConfig = Field(default_factory=InterpolationVarsConfig)
    spatial: InterpolationSpatialConfig = Field(
        default_factory=InterpolationSpatialConfig
    )


class ExtractionConfig(BaseModel):
    bbox: tuple[float | None, float | None, float | None, float | None] | None = None
    as_dims: list[str] | None = Field(default_factory=list)
    resolution_limit: float | None = None
    mesh: MeshConfig = Field(default_factory=MeshConfig)
    interpolation: InterpolationConfig = Field(default_factory=InterpolationConfig)

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class MeshConfig(BaseModel):
    tile_shape: tuple[int, int] | None = None
    data_mapping: Literal["vertices", "cells"] = "vertices"


# Default method used when interpolation is requested through a boolean
# (`interp_time`, `interp_vars`, `interp_spatial=true`) without an explicit method
DEFAULT_INTERPOLATION_METHOD = "linear"


class InterpolationVarsConfig(BaseModel):
    items: list[str] = Field(default_factory=list)
    time: bool = False
    # None = not given: resolved by InterpolationConfig (see below)
    method: str | None = None
    params: dict[str, Any] | None = Field(default_factory=dict)


class InterpolationSpatialConfig(BaseModel):
    # `interp_spatial`: None = not given, True = interpolate, False = strict
    # (no interpolation, and for probes on point lists, the probe must match
    # one of the points)
    enabled: bool | None = None
    # None = not given: resolved by InterpolationConfig (see below)
    method: str | None = None
    # Setting padding to 1.0 as default in spatial interpolation parameters
    params: dict[str, Any] | None = Field(default_factory=lambda: {"padding": 1.0})

    @property
    def strict(self) -> bool:
        """Strict mode: `interp_spatial=false` with no method other than
        nearest (an explicit method wins over `interp_spatial`)."""
        return self.enabled is False and self.method == "nearest"


class InterpolationConfig(BaseModel):
    vars: InterpolationVarsConfig = Field(default_factory=InterpolationVarsConfig)
    spatial: InterpolationSpatialConfig = Field(
        default_factory=InterpolationSpatialConfig
    )

    @model_validator(mode="after")
    def resolve_methods(self):
        """Resolve the effective interpolation methods, with this priority:
        explicit method > interpolation booleans > nearest.

        - vars: interpolating the time (`interp_time`) or some variables
          (`interp_vars`) without a method means linear interpolation.
        - spatial: `interp_spatial=true` without a method means linear,
          `interp_spatial=false` or no `interp_spatial` means nearest.
        """
        if self.vars.method is None:
            self.vars.method = (
                DEFAULT_INTERPOLATION_METHOD
                if self.vars.time or self.vars.items
                else "nearest"
            )
        if self.spatial.method is None:
            self.spatial.method = (
                DEFAULT_INTERPOLATION_METHOD if self.spatial.enabled else "nearest"
            )
        return self


TimeReduceMethod = Literal["delta", "sum", "mean", "min", "max"]


class ExtractionConfig(BaseModel):
    bbox: tuple[float | None, ...] | None = None
    is_3d: bool = False
    time_reduce: TimeReduceMethod | None = None
    as_dims: list[str] | None = Field(default_factory=list)
    resolution_limit: float | None = None
    mesh: MeshConfig = Field(default_factory=MeshConfig)
    interpolation: InterpolationConfig = Field(default_factory=InterpolationConfig)


class MeshExtractionConfig(ExtractionConfig):
    variable: str | None = None
    level_variable: str | None = None

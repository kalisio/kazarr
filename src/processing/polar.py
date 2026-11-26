"""Bilinear interpolation in polar coordinates (radar like grids).

The structure of the polar grid (center, azimuth of every beam, range of every
gate) is detected and stored in the dataset metadata by the conversion tool,
under the "polar_grid" key (see conversion_tool/kazarr/polar.py). Azimuths and
ranges are geodesic (WGS84) azimuths and distances from the center: they are
computed here exactly the same way for the probes.

The interpolation is bilinear in (range, azimuth): every probe gets the 4
nodes of the polar cell it falls in and their weights, which are then used
like the neighbors and weights of IDW.
"""

from dataclasses import dataclass

import numpy as np
from pyproj import Geod

from src import exceptions

POLAR_GRID_KEY = "polar_grid"
POLAR_BILINEAR_METHOD = "polar_bilinear"
GEOD = Geod(ellps="WGS84")

# Numerical tolerances on the grid boundaries
RANGE_EPSILON_M = 1e-3
AZIMUTH_EPSILON_DEG = 1e-9


@dataclass
class PolarStencil:
    # Storage indices of the 4 nodes of each probe's cell, shape (N, 4)
    azimuth_indices: np.ndarray
    range_indices: np.ndarray
    # Bilinear weights of these nodes, shape (N, 4), summing to 1
    weights: np.ndarray
    # Probes outside the grid (beyond the last gate, or outside the sector)
    outside: np.ndarray


def get_polar_grid(dataset_id: str, dataset_config: dict, lon_dims: tuple) -> dict:
    """Return the polar grid metadata of a dataset, or raise an explicit error."""
    polar_grid = dataset_config.get(POLAR_GRID_KEY)
    if not polar_grid:
        raise exceptions.BadSelection(
            f"Spatial interpolation '{POLAR_BILINEAR_METHOD}' is not available for "
            f"dataset '{dataset_id}': it has no polar grid metadata. Either it is not "
            "a polar grid, or it was converted before polar grid detection was "
            "added to the conversion tool and must be converted again."
        )
    expected_dims = (
        *polar_grid.get("leading_dims", {}).keys(),
        polar_grid["azimuth_dim"],
        polar_grid["range_dim"],
    )
    if tuple(lon_dims) != expected_dims:
        raise exceptions.GenericInternalError(
            f"Polar grid metadata of dataset '{dataset_id}' does not match its "
            f"coordinates dimensions ({expected_dims} != {tuple(lon_dims)})."
        )
    return polar_grid


def _azimuth_axis(polar_grid: dict):
    """Beam offsets from the first beam (degrees, increasing from 0), the
    storage index of each of them, and whether the axis is periodic."""
    azimuths = np.asarray(polar_grid["azimuths"], dtype=float)
    direction = float(polar_grid.get("azimuth_direction", 1))
    closure = polar_grid["closure"]
    if closure == "duplicated":
        # The last beam is the first one stored again: interpolate between the
        # distinct beams only, the seam is handled as for a closed grid.
        azimuths = azimuths[:-1]
    offsets = np.cumsum(
        np.concatenate([[0.0], (direction * np.diff(azimuths)) % 360.0])
    )
    indices = np.arange(len(azimuths))
    periodic = closure in ("closed", "duplicated")
    if periodic:
        offsets = np.append(offsets, 360.0)
        indices = np.append(indices, 0)
    return azimuths[0], direction, offsets, indices, periodic


def polar_bilinear_stencil(
    polar_grid: dict, lons: np.ndarray, lats: np.ndarray
) -> PolarStencil:
    """Compute the polar cells and bilinear weights of the given probes.

    Probes closer to the center than the first gate are interpolated between
    the two surrounding beams, at the first gate. Probes beyond the last gate,
    or outside the sector of an open grid, are flagged as outside.
    """
    lons = np.atleast_1d(np.asarray(lons, dtype=float))
    lats = np.atleast_1d(np.asarray(lats, dtype=float))
    n = lons.size
    center = polar_grid["center"]
    if n == 1:
        # pyproj handles size 1 arrays as scalars (through a deprecated
        # array to scalar conversion): give it scalars directly.
        az, _, dist = GEOD.inv(
            float(center["lon"]), float(center["lat"]), float(lons[0]), float(lats[0])
        )
    else:
        az, _, dist = GEOD.inv(
            np.full(n, center["lon"]), np.full(n, center["lat"]), lons, lats
        )
    az = np.atleast_1d(np.asarray(az, dtype=float))
    dist = np.atleast_1d(np.asarray(dist, dtype=float))

    # --- Range axis ---
    ranges = np.asarray(polar_grid["ranges"], dtype=float)
    range_order = np.argsort(ranges)  # storage index of each sorted gate
    sorted_ranges = ranges[range_order]
    outside = dist > sorted_ranges[-1] + RANGE_EPSILON_M
    clamped = np.clip(dist, sorted_ranges[0], sorted_ranges[-1])
    i0 = np.clip(
        np.searchsorted(sorted_ranges, clamped, side="right") - 1,
        0,
        len(sorted_ranges) - 2,
    )
    t_range = (clamped - sorted_ranges[i0]) / (
        sorted_ranges[i0 + 1] - sorted_ranges[i0]
    )
    t_range = np.clip(t_range, 0.0, 1.0)
    range_lo, range_hi = range_order[i0], range_order[i0 + 1]

    # --- Azimuth axis ---
    first_azimuth, direction, offsets, beam_indices, periodic = _azimuth_axis(
        polar_grid
    )
    delta = (direction * (az - first_azimuth)) % 360.0
    if not periodic:
        span = offsets[-1]
        # Just before the first beam (numerical noise around 0)
        delta = np.where(delta > 360.0 - AZIMUTH_EPSILON_DEG, 0.0, delta)
        outside |= delta > span + AZIMUTH_EPSILON_DEG
        delta = np.clip(delta, 0.0, span)
    j0 = np.clip(
        np.searchsorted(offsets, delta, side="right") - 1, 0, len(offsets) - 2
    )
    t_azimuth = (delta - offsets[j0]) / (offsets[j0 + 1] - offsets[j0])
    t_azimuth = np.clip(t_azimuth, 0.0, 1.0)
    beam_lo, beam_hi = beam_indices[j0], beam_indices[j0 + 1]

    azimuth_indices = np.stack([beam_lo, beam_lo, beam_hi, beam_hi], axis=1)
    range_indices = np.stack([range_lo, range_hi, range_lo, range_hi], axis=1)
    weights = np.stack(
        [
            (1 - t_azimuth) * (1 - t_range),
            (1 - t_azimuth) * t_range,
            t_azimuth * (1 - t_range),
            t_azimuth * t_range,
        ],
        axis=1,
    )
    return PolarStencil(azimuth_indices, range_indices, weights, outside)

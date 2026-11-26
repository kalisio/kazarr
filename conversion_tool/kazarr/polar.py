"""Detection of polar grids (radar like grids) from their node coordinates.

A polar grid is a structured grid whose nodes lie on beams going out of a
center: along the azimuth axis the nodes turn around the center at a constant
range, along the range axis they move away from the center on a constant
azimuth. Its structure (center, azimuth of every beam, range of every gate) is
not stored in the source files, only the longitude / latitude of the nodes, so
it is recovered here and stored in the dataset metadata. The API uses it for
the "polar_bilinear" spatial interpolation of probes.

For now, the azimuth axis is the second to last dimension of the lon/lat
arrays and the range axis is the last one. A grid stored the other way around
fails the consistency checks below and is simply not detected as polar.

Azimuths and ranges are geodesic (WGS84) azimuths and distances from the
center, computed with pyproj.Geod. The API must compute them the same way.
"""

import logging

import numpy as np
from pyproj import Geod

logger = logging.getLogger(__name__)

POLAR_GRID_KEY = "polar_grid"
# Result of the last detection and fingerprint of the coordinates it was run
# on, kept in the conversion config only (not stored in the dataset)
POLAR_GRID_DETECTION_KEY = "polar_grid_detection"
GEOD = Geod(ellps="WGS84")

# Tolerances, relative to the smallest step of the grid along each axis
RANGE_TOLERANCE = 0.05
AZIMUTH_TOLERANCE = 0.05
# A full circle grid closes on its first beam with a step close to the others.
# Beyond this factor, the gap is considered as the hole of a sector grid.
CLOSING_STEP_FACTOR = 1.5
# Maximum ratio between the range of the first gate and the depth of the grid
MAX_INNER_RADIUS_RATIO = 1.0


def _wrap_deg(angle):
    """Wrap angles (degrees) to ]-180, 180]."""
    return -((-np.asarray(angle) + 180.0) % 360.0 - 180.0)


def _to_unit_vectors(lons, lats):
    lon_r, lat_r = np.radians(lons), np.radians(lats)
    return np.stack(
        [np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)],
        axis=-1,
    )


def _find_center(lons, lats):
    """Least squares intersection of the beams, as (lon, lat, max offset in m).

    The nodes are projected with a gnomonic projection, where great circles
    (geodesics on the sphere) are straight lines: every beam is then fitted
    with a line, and the center is the point closest to all of them.
    """
    xyz = _to_unit_vectors(lons, lats)
    tangent = xyz.reshape(-1, 3).mean(axis=0)
    tangent /= np.linalg.norm(tangent)
    ref = (
        np.array([0.0, 0.0, 1.0])
        if abs(tangent[2]) < 0.9
        else np.array([1.0, 0.0, 0.0])
    )
    e1 = np.cross(ref, tangent)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(tangent, e1)
    dot = xyz @ tangent
    if np.any(dot <= 0):
        return None
    u = (xyz @ e1) / dot
    v = (xyz @ e2) / dot

    normals, offsets = [], []
    for j in range(u.shape[0]):
        pts = np.column_stack([u[j], v[j]])
        mean = pts.mean(axis=0)
        centered = pts - mean
        if np.allclose(centered, 0):
            continue
        _, _, vt = np.linalg.svd(centered)
        direction = vt[0]
        normal = np.array([-direction[1], direction[0]])
        normals.append(normal)
        offsets.append(normal @ mean)
    if len(normals) < 2:
        return None
    normals = np.array(normals)
    offsets = np.array(offsets)
    center_uv, _, rank, _ = np.linalg.lstsq(normals, offsets, rcond=None)
    if rank < 2:
        # All beams are parallel
        return None
    # Distance of the center to each beam line, in gnomonic units ~ radians
    # close to the tangent point: converted to meters for diagnostics.
    max_offset = float(np.max(np.abs(normals @ center_uv - offsets))) * 6371008.8

    center_xyz = tangent + center_uv[0] * e1 + center_uv[1] * e2
    center_xyz /= np.linalg.norm(center_xyz)
    center_lon = float(np.degrees(np.arctan2(center_xyz[1], center_xyz[0])))
    center_lat = float(np.degrees(np.arcsin(np.clip(center_xyz[2], -1.0, 1.0))))
    return center_lon, center_lat, max_offset


def _circular_mean_deg(angles, axis):
    rad = np.radians(angles)
    return np.degrees(
        np.arctan2(
            np.nanmean(np.sin(rad), axis=axis), np.nanmean(np.cos(rad), axis=axis)
        )
    )


def detect_polar_grid(lons, lats, dims):
    """Detect whether the given node coordinates form a polar grid.

    lons, lats: arrays of node coordinates (degrees), sharing the dimensions
    `dims`. The azimuth axis is dims[-2], the range axis is dims[-1]. Leading
    dimensions (if any) must hold the same coordinates for every index.

    Returns a tuple (metadata, reason): the polar grid metadata to store in the
    dataset configuration, or None and the reason why the grid is not polar.
    """
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    if lons.shape != lats.shape or lons.ndim < 2 or len(dims) != lons.ndim:
        return None, "longitudes and latitudes are not structured 2D arrays"
    if not (np.all(np.isfinite(lons)) and np.all(np.isfinite(lats))):
        return None, "coordinates contain non finite values"
    if np.nanmax(np.abs(lats)) > 90 or np.nanmax(np.abs(lons)) > 360:
        return None, "coordinates are not longitudes / latitudes in degrees"

    # Leading dimensions: the horizontal grid must be the same for all of them
    lead_shape = lons.shape[:-2]
    flat_lons = lons.reshape((-1, *lons.shape[-2:]))
    flat_lats = lats.reshape((-1, *lats.shape[-2:]))
    if not (
        np.allclose(flat_lons, flat_lons[0]) and np.allclose(flat_lats, flat_lats[0])
    ):
        return None, "the horizontal grid changes along the leading dimensions"
    lons2d, lats2d = flat_lons[0], flat_lats[0]
    n_az, n_rng = lons2d.shape
    if n_az < 2 or n_rng < 2:
        return None, "less than 2 beams or 2 gates"

    found = _find_center(lons2d, lats2d)
    if found is None:
        return None, "beams do not intersect"
    center_lon, center_lat, max_center_offset = found

    az, _, dist = GEOD.inv(
        np.full(lons2d.size, center_lon),
        np.full(lons2d.size, center_lat),
        lons2d.ravel(),
        lats2d.ravel(),
    )
    az = np.asarray(az).reshape(lons2d.shape)
    dist = np.asarray(dist).reshape(lons2d.shape)

    # Ranges: one per gate, shared by all beams
    ranges = np.median(dist, axis=0)
    range_steps = np.diff(ranges)
    if not (np.all(range_steps > 0) or np.all(range_steps < 0)):
        return None, "ranges are not monotonic along the range axis"
    min_range_step = float(np.min(np.abs(range_steps)))
    # Nearly parallel lines (e.g. a rectilinear grid) always "intersect" very
    # far away, where they look like beams. The center of a polar grid is
    # expected to be close to it: the empty disk around the center must not be
    # wider than the grid itself.
    range_min, range_max = float(np.min(ranges)), float(np.max(ranges))
    if range_min > MAX_INNER_RADIUS_RATIO * (range_max - range_min):
        return None, (
            f"the center is too far from the grid ({range_min:.0f} m for a "
            f"{range_max - range_min:.0f} m deep grid)"
        )
    max_range_deviation = float(np.max(np.abs(dist - ranges[None, :])))
    if max_range_deviation > RANGE_TOLERANCE * min_range_step:
        return None, (
            f"ranges vary across beams ({max_range_deviation:.3f} m, "
            f"gate step {min_range_step:.3f} m)"
        )

    # Azimuths: one per beam, shared by all gates. Nodes at the center itself
    # (a gate at range 0) have no azimuth.
    at_center = dist < RANGE_TOLERANCE * min_range_step
    az_valid = np.where(at_center, np.nan, az)
    if np.all(at_center.all(axis=0)):
        return None, "all nodes are at the center"
    azimuths = _circular_mean_deg(az_valid, axis=1)
    if not np.all(np.isfinite(azimuths)):
        return None, "some beams have no azimuth"
    az_steps = _wrap_deg(np.diff(azimuths))
    direction = 1.0 if np.sum(az_steps) >= 0 else -1.0
    signed_steps = direction * az_steps
    if np.min(signed_steps) <= 0:
        return None, "azimuths are not monotonic along the azimuth axis"

    # Closure of the azimuth axis
    span = float(np.sum(signed_steps))
    tol = AZIMUTH_TOLERANCE * float(np.median(signed_steps))
    if abs(span - 360.0) <= tol:
        closure = "duplicated"  # last beam is the first one stored again
        steps_to_check = signed_steps
    else:
        closing_step = float((direction * (azimuths[0] - azimuths[-1])) % 360.0)
        if span < 360.0 and 0 < closing_step <= CLOSING_STEP_FACTOR * float(
            np.max(signed_steps)
        ):
            closure = "closed"  # full circle, without duplicated beam
            steps_to_check = np.append(signed_steps, closing_step)
        elif span < 360.0:
            closure = "open"  # sector
            steps_to_check = signed_steps
        else:
            return None, "beams overlap (azimuth span over 360 degrees)"
    min_az_step = float(np.min(steps_to_check))

    max_azimuth_deviation = float(
        np.nanmax(np.abs(_wrap_deg(az_valid - azimuths[:, None])))
    )
    if max_azimuth_deviation > AZIMUTH_TOLERANCE * min_az_step:
        return None, (
            f"azimuths vary along beams ({max_azimuth_deviation:.5f} deg, "
            f"beam step {min_az_step:.5f} deg)"
        )

    metadata = {
        "center": {"lon": center_lon, "lat": center_lat},
        "azimuth_dim": dims[-2],
        "range_dim": dims[-1],
        "leading_dims": {d: int(s) for d, s in zip(dims[:-2], lead_shape, strict=True)},
        # Azimuths (degrees, clockwise from north, in ]-180, 180]) of every
        # beam, in storage order, and ranges (meters) of every gate
        "azimuths": [float(a) for a in azimuths],
        "ranges": [float(r) for r in ranges],
        # "closed": full circle, "duplicated": full circle whose last beam is
        # the first one stored again, "open": sector
        "closure": closure,
        # +1 when azimuths increase (clockwise) with the beam index, -1 otherwise
        "azimuth_direction": int(direction),
        "quality": {
            "max_center_offset_m": max_center_offset,
            "max_range_deviation_m": max_range_deviation,
            "max_azimuth_deviation_deg": max_azimuth_deviation,
        },
    }
    return metadata, None

import hashlib
import os

import numpy as np
from loguru import logger as log
from scipy.spatial import cKDTree

_spatial_index_cache: dict[tuple, cKDTree] = {}
MAX_CACHE_SIZE = os.getenv("KDTREE_MAX_CACHE_SIZE", "10")


def _get_array_hash(arr: np.ndarray) -> str:
    """Compute a fast hash for a numpy array.
    We use the shape, dtype, and a sample of values to keep it fast.
    """
    sample_indices = np.linspace(0, arr.size - 1, min(100, arr.size), dtype=int)
    sample_data = arr.ravel()[sample_indices]

    hasher = hashlib.md5(usedforsecurity=False)
    hasher.update(str(arr.shape).encode())
    hasher.update(str(arr.dtype).encode())
    hasher.update(sample_data.tobytes())
    return hasher.hexdigest()


def lonlat_to_xyz(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project geographic coordinates (lon, lat in degrees) onto the unit
    sphere as 3D Cartesian coordinates.
    """
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)
    cos_lat = np.cos(lat_rad)
    x = cos_lat * np.cos(lon_rad)
    y = cos_lat * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return x, y, z


def lonlat_to_sphere_points(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Convenience wrapper: (lon, lat) arrays -> an (N, 3) array of unit-sphere
    XYZ points, ready to feed into cKDTree or to column_stack with extra
    (e.g. level) columns.
    """
    x, y, z = lonlat_to_xyz(lon, lat)
    return np.column_stack((x, y, z))


def deg_to_chord_distance(deg: np.ndarray | float) -> np.ndarray | float:
    """Convert an angular great-circle distance in degrees to the equivalent
    unit-sphere chord (straight-line) distance, so it can be used as a
    radius threshold for queries against unit-sphere-projected points.
    """
    rad = np.radians(deg)
    return 2.0 * np.sin(rad / 2.0)


def chord_to_deg_distance(chord: np.ndarray | float) -> np.ndarray | float:
    """Convert a unit-sphere chord distance back to the true angular
    great-circle distance in degrees. Inverse of `deg_to_chord_distance`.
    """
    # Clip for numerical safety: chord should be in [0, 2], but floating
    # point round-trips can push it a hair outside due to rounding.
    chord = np.clip(chord, 0.0, 2.0)
    return np.degrees(2.0 * np.arcsin(chord / 2.0))


def get_cached_ckdtree(
    points: np.ndarray,
    dataset_id: str | None = None,
    coord_vars: tuple[str, ...] | None = None,
) -> cKDTree:
    """Retrieve or build a cKDTree for the given points.

    If dataset_id and coord_vars are provided, they are used for caching.
    Otherwise, a hash of the points is used.
    """
    # Create cache key
    if dataset_id and coord_vars:
        # We still add a data hash to be safe if the file changed or subsetting happened
        data_hash = _get_array_hash(points)
        cache_key = (dataset_id, coord_vars, data_hash)
    else:
        data_hash = _get_array_hash(points)
        cache_key = (data_hash,)

    if cache_key in _spatial_index_cache:
        log.info(
            "[KAZARR] Using cached spatial index (cKDTree) for points with shape {shape}",
            shape=points.shape,
        )
        return _spatial_index_cache[cache_key]

    log.info(
        "[KAZARR] Building new spatial index (cKDTree) for points with shape {shape}",
        shape=points.shape,
    )
    tree = cKDTree(points)

    # Manage cache size
    if len(_spatial_index_cache) >= int(MAX_CACHE_SIZE):
        log.warning("[KAZARR] Cache size limit reached, evicting oldest entry")
        # Simple FIFO-ish eviction: remove a random entry or the first one
        _spatial_index_cache.pop(next(iter(_spatial_index_cache)))

    _spatial_index_cache[cache_key] = tree
    return tree


# ---------------------------------------------------------------------------
# Point-in-grid test for structured irregular grids (curvilinear, radial)
# ---------------------------------------------------------------------------
#
# On a structured grid (2D lon/lat arrays indexed [..., j, i]), the cells are
# implicitly known from the array indices: cell (j, i) has corners
# (j, i), (j, i+1), (j+1, i+1), (j+1, i). No mesh needs to be built: the
# nearest grid node (already given by the cKDTree) tells us which few cells
# can contain a probe, and a point-in-quadrilateral test on those cells tells
# whether the probe lies inside the grid -- holes (NaN coordinates) included.
#
# All geometry is done on the unit sphere and each candidate cell is projected
# onto the plane tangent to the probe with a gnomonic projection (great
# circles -> straight lines), so the test is exact for cells with
# great-circle edges and is unaffected by the antimeridian or the poles.

_grid_geometry_cache: dict[tuple, "StructuredGridGeometry"] = {}


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return v / norm


def _median_spacing(xyz: np.ndarray, axis: int) -> float:
    """Median chord distance between consecutive nodes along `axis`."""
    d = np.linalg.norm(np.diff(xyz, axis=axis), axis=-1)
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(d)) if d.size else 0.0


def _axis_closure(xyz: np.ndarray, axis: int) -> str:
    """How the last line of nodes along `axis` relates to the first one:

    - "duplicate": it repeats the first line (a closed ring stored with its
      closing line duplicated, e.g. a 360 deg radial grid whose last branch is
      the first one again),
    - "periodic": it is adjacent to the first line without repeating it (a
      360 deg radial grid whose last branch is one step before the first one),
    - "open": anything else (regional grids, radial sectors).
    """
    n = xyz.shape[axis]
    if n < 3:
        return "open"
    spacing = _median_spacing(xyz, axis)
    if spacing == 0.0:
        return "open"
    first = np.take(xyz, 0, axis=axis)
    last = np.take(xyz, n - 1, axis=axis)
    gap = np.linalg.norm(last - first, axis=-1)
    gap = gap[np.isfinite(gap)]
    if gap.size == 0:
        return "open"
    gap = float(np.median(gap))
    if gap <= 1e-3 * spacing:
        return "duplicate"
    if gap <= 1.5 * spacing:
        return "periodic"
    return "open"


def _cell_edges_along_axis(xyz: np.ndarray, axis: int, periodic: bool) -> np.ndarray:
    """Turn cell-center coordinates into cell-corner coordinates along `axis`
    (on the unit sphere). Edge k is the boundary on the lower side of cell k,
    so that cell k of the data stays cell k of the corner grid.
    """
    n = xyz.shape[axis]
    if n == 1:
        return xyz
    c = np.moveaxis(xyz, axis, 0)
    mids = _normalize(0.5 * (c[:-1] + c[1:]))
    if periodic:
        wrap = _normalize(0.5 * (c[-1] + c[0]))[np.newaxis]
        edges = np.concatenate([wrap, mids], axis=0)  # n edges, cell k = [e_k, e_k+1 mod n]
    else:
        first = _normalize(2.0 * c[0] - mids[0])[np.newaxis]
        last = _normalize(2.0 * c[-1] - mids[-1])[np.newaxis]
        edges = np.concatenate([first, mids, last], axis=0)  # n + 1 edges
    return np.moveaxis(edges, 0, axis)


class StructuredGridGeometry:
    """Cell geometry of a structured irregular grid, used to tell whether a
    probe falls inside the grid.

    `lons` / `lats` have shape (..., J, I): the two last axes are the
    horizontal structured axes, any leading axes (e.g. a level axis) are kept.
    When `data_on_cells` is True the coordinates are cell centers and the grid
    extends half a cell beyond them; otherwise they are the cell corners.
    """

    def __init__(self, lons: np.ndarray, lats: np.ndarray, data_on_cells: bool = False):
        lons = np.asarray(lons, dtype=float)
        lats = np.asarray(lats, dtype=float)
        if lons.ndim < 2 or lons.shape != lats.shape:
            raise ValueError("Structured grid needs lon/lat arrays of identical shape with ndim >= 2")
        self.node_shape = lons.shape
        self.data_on_cells = data_on_cells

        x, y, z = lonlat_to_xyz(lons, lats)
        nodes = np.stack([x, y, z], axis=-1)  # (..., J, I, 3)
        j_axis, i_axis = nodes.ndim - 3, nodes.ndim - 2

        # A duplicated closing line is dropped and the axis treated as
        # periodic: otherwise a probe whose nearest node is the duplicate
        # would not see the cells on the other side of the seam.
        closure_j = _axis_closure(nodes, j_axis)
        if closure_j == "duplicate":
            nodes = np.take(nodes, np.arange(nodes.shape[j_axis] - 1), axis=j_axis)
        closure_i = _axis_closure(nodes, i_axis)
        if closure_i == "duplicate":
            nodes = np.take(nodes, np.arange(nodes.shape[i_axis] - 1), axis=i_axis)
        self.periodic_j = closure_j != "open"
        self.periodic_i = closure_i != "open"
        # Node index -> index in the (possibly de-duplicated) node grid.
        self.nodes_j, self.nodes_i = nodes.shape[j_axis], nodes.shape[i_axis]

        if data_on_cells:
            corners = _cell_edges_along_axis(nodes, j_axis, self.periodic_j)
            corners = _cell_edges_along_axis(corners, i_axis, self.periodic_i)
        else:
            corners = nodes
        self.corners = corners
        self.n_j, self.n_i = corners.shape[-3], corners.shape[-2]
        # Number of cells along each axis (a periodic axis has one more: the
        # cell joining the last line of corners back to the first one).
        self.cells_j = self.n_j if self.periodic_j else self.n_j - 1
        self.cells_i = self.n_i if self.periodic_i else self.n_i - 1
        spacing = max(_median_spacing(corners, j_axis), _median_spacing(corners, i_axis))
        # Tolerance (tangent-plane units ~ radians) so that probes lying
        # exactly on a cell edge or on the outer boundary count as inside.
        self.edge_tolerance = 1e-6 * spacing if spacing > 0 else 1e-12
        # Largest cell diagonal (chord): a probe inside a cell is at most this
        # far from that cell's corners, hence from its nearest grid node. Any
        # probe farther than that from its nearest node is outside for sure.
        self.max_cell_diameter = self._max_cell_diameter()

    def _max_cell_diameter(self) -> float:
        c = self.corners
        if self.periodic_j:
            c = np.concatenate([c, c[..., :1, :, :]], axis=-3)
        if self.periodic_i:
            c = np.concatenate([c, c[..., :, :1, :]], axis=-2)
        if c.shape[-3] < 2 or c.shape[-2] < 2:
            return 0.0
        d1 = np.linalg.norm(c[..., 1:, 1:, :] - c[..., :-1, :-1, :], axis=-1)
        d2 = np.linalg.norm(c[..., 1:, :-1, :] - c[..., :-1, 1:, :], axis=-1)
        d = np.fmax(d1, d2)
        return float(np.nanmax(d)) if np.isfinite(d).any() else 0.0

    def _candidate_offsets(self, window: int) -> list[int]:
        # Node mode: the cells touching node (j0, i0) have their lower corner
        # at j0-1 or j0. Cell mode: the probe's nearest cell center is (j0, i0)
        # itself, so look at the cells around it.
        upper = window + 1 if self.data_on_cells else window
        return list(range(-window, upper))

    def contains(
        self,
        query_xyz: np.ndarray,
        nearest_flat_idx: np.ndarray,
        nearest_dist: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return a boolean mask telling which probes lie inside the grid.

        query_xyz: (N, 3) unit-sphere coordinates of the probes.
        nearest_flat_idx: (N,) flat index (into the lon/lat arrays) of each
        probe's nearest grid node, as returned by the cKDTree.
        nearest_dist: optional (N,) chord distance to that node, as returned by
        the cKDTree. Used to reject far-away probes without testing any cell.
        """
        query_xyz = np.asarray(query_xyz, dtype=float).reshape(-1, 3)
        nearest = np.unravel_index(np.asarray(nearest_flat_idx).ravel(), self.node_shape)
        prefix = nearest[:-2]
        j0 = np.mod(nearest[-2], self.nodes_j)  # duplicated closing line -> first line
        i0 = np.mod(nearest[-1], self.nodes_i)

        inside = np.zeros(len(query_xyz), dtype=bool)
        if nearest_dist is not None:
            # A probe inside a cell is at most one cell diagonal away from the
            # cell's corners / center, hence from its nearest grid node.
            reach = self.max_cell_diameter * (1.0 + 1e-6)
            todo = np.flatnonzero(~(np.asarray(nearest_dist).ravel() > reach))
        else:
            todo = np.arange(len(query_xyz))
        # A first pass on the cells right around the nearest node is enough
        # for any reasonable grid; a wider second pass covers strongly sheared
        # cells whose nearest node is not one of their own corners.
        for window in (1, 2):
            if todo.size == 0:
                break
            hit = self._contains_in_window(
                query_xyz[todo], tuple(p[todo] for p in prefix), j0[todo], i0[todo], window
            )
            inside[todo[hit]] = True
            todo = todo[~hit]
        return inside

    def _contains_in_window(self, q, prefix, j0, i0, window):
        offsets = self._candidate_offsets(window)
        dj, di = np.meshgrid(offsets, offsets, indexing="ij")
        dj, di = dj.ravel(), di.ravel()  # (C,)

        cj = j0[:, None] + dj[None, :]  # (N, C) lower corner of candidate cells
        ci = i0[:, None] + di[None, :]
        valid = np.ones(cj.shape, dtype=bool)
        if self.periodic_j:
            cj = np.mod(cj, self.n_j)
        else:
            valid &= (cj >= 0) & (cj < self.cells_j)
        if self.periodic_i:
            ci = np.mod(ci, self.n_i)
        else:
            valid &= (ci >= 0) & (ci < self.cells_i)
        cj = np.clip(cj, 0, self.n_j - 1)
        ci = np.clip(ci, 0, self.n_i - 1)
        cj1 = np.mod(cj + 1, self.n_j) if self.periodic_j else np.minimum(cj + 1, self.n_j - 1)
        ci1 = np.mod(ci + 1, self.n_i) if self.periodic_i else np.minimum(ci + 1, self.n_i - 1)

        pre = tuple(np.broadcast_to(p[:, None], cj.shape) for p in prefix)
        quad = np.stack(
            [
                self.corners[pre + (cj, ci)],
                self.corners[pre + (cj, ci1)],
                self.corners[pre + (cj1, ci1)],
                self.corners[pre + (cj1, ci)],
            ],
            axis=2,
        )  # (N, C, 4, 3)

        # Tangent-plane basis at each probe (east, north), robust at the poles.
        ref = np.where(np.abs(q[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
        e1 = _normalize(np.cross(ref, q))
        e2 = np.cross(q, e1)

        # Gnomonic projection of the corners: the probe is the origin.
        dot_p = np.einsum("ncks,ns->nck", quad, q)
        with np.errstate(invalid="ignore", divide="ignore"):
            u = np.einsum("ncks,ns->nck", quad, e1) / dot_p
            v = np.einsum("ncks,ns->nck", quad, e2) / dot_p
        valid &= np.all(dot_p > 0, axis=2)  # cell on the probe's hemisphere

        u_a, v_a = u, v
        u_b, v_b = np.roll(u, -1, axis=2), np.roll(v, -1, axis=2)

        # Crossing-number test of the origin against each quadrilateral: count
        # the edges crossed by the ray going from the origin along +u.
        straddles = (v_a > 0) != (v_b > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            u_cross = u_a + (0.0 - v_a) * (u_b - u_a) / (v_b - v_a)
        crossings = np.sum(straddles & (u_cross > 0), axis=2)
        in_quad = (crossings % 2) == 1

        # Origin lying on an edge (shared edge or outer boundary) -> inside.
        eu, ev = u_b - u_a, v_b - v_a
        seg_len2 = eu * eu + ev * ev
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.clip(-(u_a * eu + v_a * ev) / seg_len2, 0.0, 1.0)
        t = np.where(seg_len2 > 0, t, 0.0)
        dist2 = (u_a + t * eu) ** 2 + (v_a + t * ev) ** 2
        on_edge = np.any(dist2 <= self.edge_tolerance**2, axis=2)

        return np.any(valid & (in_quad | on_edge), axis=1)


def get_cached_structured_grid_geometry(
    lons: np.ndarray,
    lats: np.ndarray,
    dataset_id: str | None = None,
    coord_vars: tuple[str, ...] | None = None,
    data_on_cells: bool = False,
) -> StructuredGridGeometry:
    """Retrieve or build the cell geometry of a structured irregular grid."""
    cache_key = (
        dataset_id,
        coord_vars,
        data_on_cells,
        _get_array_hash(np.asarray(lons)),
        _get_array_hash(np.asarray(lats)),
    )
    geometry = _grid_geometry_cache.get(cache_key)
    if geometry is not None:
        return geometry

    log.info(
        "[KAZARR] Building structured grid geometry for grid with shape {shape}",
        shape=np.shape(lons),
    )
    geometry = StructuredGridGeometry(lons, lats, data_on_cells=data_on_cells)
    if len(_grid_geometry_cache) >= int(MAX_CACHE_SIZE):
        _grid_geometry_cache.pop(next(iter(_grid_geometry_cache)), None)
    _grid_geometry_cache[cache_key] = geometry
    return geometry

"""
test_isoline.py — Tests for the isoline endpoint.

Datasets are written directly as Zarr (no conversion tool needed) with analytic
fields, so that generated isolines can be checked against the exact solution.
"""

import os

import numpy as np
import pytest
import xarray as xr
from fastapi.testclient import TestClient

import utils
from src import exceptions
from src.processing.isoline import ThresholdRange, parse_thresholds, resolve_thresholds

TMP_FOLDER = os.environ.get("TEST_TMP_FOLDER", "test/tests_tmp")

REGULAR = "isoline_regular"
REGULAR_0_360 = "isoline_regular_0_360"
CURVILINEAR = "isoline_curvilinear"
WITH_NAN = "isoline_nan"
POINT_LIST = "isoline_points"

BUMPS = [(-4.0, 45.0), (4.0, 45.0)]  # (lon, lat) centers of two gaussian bumps


def two_bumps(lon, lat):
    return sum(np.exp(-((lon - x) ** 2 + (lat - y) ** 2)) for x, y in BUMPS)


def ring(lon, lat, lon0=0.0, lat0=0.0):
    return np.sqrt((lon - lon0) ** 2 + (lat - lat0) ** 2)


def save(ds, name, lon="lon", lat="lat"):
    ds.attrs["kazarr"] = {"variables": {"lon": lon, "lat": lat}}
    ds.to_zarr(os.path.join(TMP_FOLDER, f"{name}.zarr"), mode="w", zarr_format=2, consolidated=True)


def all_coords(lines):
    if not lines:
        return np.empty((0, 2))
    return np.concatenate([np.asarray(line) for line in lines])


@pytest.fixture(scope="module", autouse=True)
def datasets():
    os.makedirs(TMP_FOLDER, exist_ok=True)

    # Regular grid (0.1°) with two separate bumps
    lon = np.round(np.arange(-10, 10.001, 0.1), 6)
    lat = np.round(np.arange(40, 50.001, 0.1), 6)
    ds = xr.Dataset(
        {"value": (("lat", "lon"), two_bumps(*np.meshgrid(lon, lat)))},
        coords={"lat": lat, "lon": lon},
    ).chunk({"lat": 50, "lon": 50})
    save(ds, REGULAR)

    # Global regular grid in [0, 360[ with a disc centered on lon=0 (dataset seam)
    lon = np.arange(0.0, 360.0, 1.0)
    lat = np.arange(-60.0, 60.001, 1.0)
    LON, LAT = np.meshgrid(lon, lat)
    ds = xr.Dataset(
        {"value": (("lat", "lon"), ring((LON + 180.0) % 360.0 - 180.0, LAT))},
        coords={"lat": lat, "lon": lon},
    )
    save(ds, REGULAR_0_360)

    # Curvilinear (rotated) grid with a leading vertical dimension of size 1
    j, i = np.meshgrid(np.arange(60), np.arange(80), indexing="ij")
    angle = np.radians(30)
    lon2d = -3 + 0.1 * (i * np.cos(angle) - j * np.sin(angle))
    lat2d = 43 + 0.1 * (i * np.sin(angle) + j * np.cos(angle))
    ds = xr.Dataset(
        {"value": (("k", "y", "x"), ring(lon2d, lat2d, 1.0, 46.5)[None])},
        coords={
            "longitude": (("k", "y", "x"), lon2d[None]),
            "latitude": (("k", "y", "x"), lat2d[None]),
        },
    )
    save(ds, CURVILINEAR, lon="longitude", lat="latitude")

    # Regular grid with a NaN area on the east half
    lon = np.arange(-5, 5.001, 0.1)
    lat = np.arange(-5, 5.001, 0.1)
    LON, LAT = np.meshgrid(lon, lat)
    values = ring(LON, LAT)
    values[LON > 0] = np.nan
    ds = xr.Dataset({"value": (("lat", "lon"), values)}, coords={"lat": lat, "lon": lon})
    save(ds, WITH_NAN)

    # Point list: not supported
    ds = xr.Dataset(
        {"value": (("point",), np.arange(10.0))},
        coords={
            "lon": (("point",), np.arange(10.0)),
            "lat": (("point",), np.arange(10.0)),
        },
    )
    save(ds, POINT_LIST)

    yield
    utils.cleanup_test_files()


@pytest.mark.Isoline
class TestIsoline:
    def get(self, client, dataset, query):
        return client.get(f"/datasets/{dataset}/isoline?variable=value&{query}")

    # ------------------------------------------------------------------
    # Isolines generation (contourpy)
    # ------------------------------------------------------------------

    def test_disjoint_lines_are_not_merged(self, client: TestClient):
        """Each bump produces its own closed line (matplotlib's paths used to
        merge all the lines of a threshold into a single list of points)."""
        response = self.get(client, REGULAR, "thresholds=0.5")
        assert response.status_code == 200
        lines = response.json()["0.5"]
        assert len(lines) == 2
        for line in lines:
            line = np.asarray(line)
            assert line.shape[1] == 2
            np.testing.assert_allclose(line[0], line[-1])  # closed
        centers = sorted(np.asarray(line).mean(axis=0)[0] for line in lines)
        np.testing.assert_allclose(centers, [-4, 4], atol=0.05)

    def test_vertices_are_on_the_isoline(self, client: TestClient):
        response = self.get(client, REGULAR, "thresholds=0.2&thresholds=0.5&thresholds=0.9")
        data = response.json()
        for threshold in (0.2, 0.5, 0.9):
            coords = all_coords(data[str(threshold)])
            assert len(coords) > 0
            np.testing.assert_allclose(
                two_bumps(coords[:, 0], coords[:, 1]), threshold, atol=0.02
            )

    def test_unsorted_thresholds_and_threshold_out_of_range(self, client: TestClient):
        response = self.get(client, REGULAR, "thresholds=100&thresholds=0.5")
        assert response.status_code == 200
        data = response.json()
        assert list(data.keys()) == ["100.0", "0.5"]
        assert data["100.0"] == []
        assert len(data["0.5"]) == 2

    def test_geojson(self, client: TestClient):
        response = self.get(client, REGULAR, "thresholds=0.5&thresholds=100&format=geojson")
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "FeatureCollection"
        first, empty = data["features"]
        assert first["type"] == "Feature"
        assert first["properties"] == {"id": 0, "threshold": 0.5}
        assert first["geometry"]["type"] == "MultiLineString"
        assert len(first["geometry"]["coordinates"]) == 2
        assert empty["properties"] == {"id": 1, "threshold": 100.0}
        assert empty["geometry"] == {"type": "MultiLineString", "coordinates": []}

    def test_geojson_matches_raw(self, client: TestClient):
        raw = self.get(client, REGULAR, "thresholds=0.2&thresholds=0.5").json()
        geojson = self.get(client, REGULAR, "thresholds=0.2&thresholds=0.5&format=geojson").json()
        for feature in geojson["features"]:
            key = str(feature["properties"]["threshold"])
            geometry = feature["geometry"]
            lines = (
                [geometry["coordinates"]]
                if geometry["type"] == "LineString"
                else geometry["coordinates"]
            )
            assert lines == raw[key]

    def test_geojson_single_line_is_a_linestring(self, client: TestClient):
        """A threshold with a single line gives a LineString, several lines a
        MultiLineString."""
        response = self.get(
            client, REGULAR, "thresholds=0.5&format=geojson&lon_min=-8&lon_max=0"
        )
        assert response.status_code == 200
        geometry = response.json()["features"][0]["geometry"]
        assert geometry["type"] == "LineString"
        coords = np.asarray(geometry["coordinates"])
        assert coords.ndim == 2
        assert coords.shape[1] == 2
        np.testing.assert_allclose(two_bumps(coords[:, 0], coords[:, 1]), 0.5, atol=0.02)

        raw = self.get(client, REGULAR, "thresholds=0.5&lon_min=-8&lon_max=0").json()
        assert geometry["coordinates"] == raw["0.5"][0]

    def test_curvilinear_grid(self, client: TestClient):
        response = self.get(client, CURVILINEAR, "thresholds=1&thresholds=2")
        assert response.status_code == 200
        for threshold, lines in response.json().items():
            assert len(lines) == 1
            coords = all_coords(lines)
            np.testing.assert_allclose(
                ring(coords[:, 0], coords[:, 1], 1.0, 46.5), float(threshold), atol=0.01
            )

    def test_nan_values(self, client: TestClient):
        response = self.get(client, WITH_NAN, "thresholds=2")
        assert response.status_code == 200
        coords = all_coords(response.json()["2.0"])
        assert len(coords) > 0
        assert coords[:, 0].max() <= 1e-9
        np.testing.assert_allclose(ring(coords[:, 0], coords[:, 1]), 2, atol=0.01)

    def test_gzip_response(self, client: TestClient):
        """Large responses are gzip compressed when the client accepts it, and
        decompress to the same content."""
        compressed = self.get(client, REGULAR, "thresholds=::0.1")
        assert compressed.headers.get("content-encoding") == "gzip"
        assert compressed.headers["content-type"] == "application/json"
        identity = client.get(
            f"/datasets/{REGULAR}/isoline?variable=value&thresholds=::0.1",
            headers={"Accept-Encoding": "identity"},
        )
        assert "content-encoding" not in identity.headers
        assert compressed.json() == identity.json()
        assert int(compressed.headers["content-length"]) < len(identity.content) / 2

    def test_point_list_is_rejected(self, client: TestClient):
        response = self.get(client, POINT_LIST, "thresholds=5")
        assert response.status_code == 400
        assert "BAD_SELECTION" in response.text

    # ------------------------------------------------------------------
    # Bounding box
    # ------------------------------------------------------------------

    def test_bbox(self, client: TestClient):
        response = self.get(
            client, REGULAR, "thresholds=0.5&lon_min=-8&lon_max=0&lat_min=42&lat_max=48"
        )
        assert response.status_code == 200
        lines = response.json()["0.5"]
        assert len(lines) == 1  # only the western bump
        coords = all_coords(lines)
        assert coords[:, 0].min() >= -8.1
        assert coords[:, 0].max() <= 0.1

    def test_bbox_cutting_isolines(self, client: TestClient):
        """Isolines crossing the bbox are cut at its edges (+ 1 cell of padding)."""
        response = self.get(client, REGULAR, "thresholds=0.5&lon_min=-4&lon_max=4")
        assert response.status_code == 200
        lines = response.json()["0.5"]
        assert len(lines) == 2
        coords = all_coords(lines)
        assert coords[:, 0].min() >= -4.1 - 1e-9
        assert coords[:, 0].max() <= 4.1 + 1e-9
        for line in lines:  # half circles: open lines
            assert line[0] != line[-1]

    def test_bbox_single_side(self, client: TestClient):
        response = self.get(client, REGULAR, "thresholds=0.5&lon_min=0")
        lines = response.json()["0.5"]
        assert len(lines) == 1
        assert all_coords(lines)[:, 0].min() >= -0.1 - 1e-9

    def test_bbox_outside_data(self, client: TestClient):
        response = self.get(client, REGULAR, "thresholds=0.5&lon_min=100&lon_max=110")
        assert response.status_code == 400
        assert "NO_DATA_IN_SELECTION" in response.text

    def test_bbox_across_antimeridian(self, client: TestClient):
        """On a [0, 360[ dataset, a disc centered on lon=0 is split in two
        lines without bbox, and returned as a single continuous line with a
        bbox using negative longitudes."""
        full = self.get(client, REGULAR_0_360, "thresholds=10").json()["10.0"]
        assert len(full) == 2

        response = self.get(client, REGULAR_0_360, "thresholds=10&lon_min=-30&lon_max=30")
        assert response.status_code == 200
        lines = response.json()["10.0"]
        assert len(lines) == 1
        coords = all_coords(lines)
        np.testing.assert_allclose(ring(coords[:, 0], coords[:, 1]), 10, atol=0.1)
        assert coords[:, 0].min() < -9
        assert coords[:, 0].max() > 9

    def test_curvilinear_grid_bbox(self, client: TestClient):
        """On curvilinear grids, the bbox selects the smallest block of grid
        cells covering it: lines may exceed the bbox, but only a part of the
        circle is computed."""
        full = all_coords(self.get(client, CURVILINEAR, "thresholds=1").json()["1.0"])
        response = self.get(
            client, CURVILINEAR, "thresholds=1&lon_min=1.5&lat_min=46.5&lon_max=3&lat_max=48"
        )
        assert response.status_code == 200
        coords = all_coords(response.json()["1.0"])
        assert 0 < len(coords) < len(full)
        # The north-east quarter of the circle is fully covered
        north_east = full[(full[:, 0] >= 1.5) & (full[:, 1] >= 46.5)]
        assert {tuple(p) for p in north_east} <= {tuple(p) for p in coords}


class TestThresholdsParsing:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            (["5"], [5.0]),
            (["10", "0", "10", "5"], [10.0, 0.0, 5.0]),  # order kept, duplicates removed
            (["0:10:2"], ThresholdRange(0.0, 10.0, 2.0)),
            (["::0.5"], ThresholdRange(None, None, 0.5)),
            ([":30:5"], ThresholdRange(None, 30.0, 5.0)),
            (["-10::5"], ThresholdRange(-10.0, None, 5.0)),
            ([" 1 : 2 : 0.5 "], ThresholdRange(1.0, 2.0, 0.5)),
        ],
    )
    def test_parse(self, raw, expected):
        assert parse_thresholds(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            ["abc"],
            ["nan"],
            ["1:2"],  # step missing
            ["1:2:"],  # step missing
            ["::"],
            ["::0"],
            ["::-1"],
            ["5:1:1"],  # min > max
            ["a::1"],
            ["1:2:3:4"],
            ["0:10:2", "15"],  # range combined with values
        ],
    )
    def test_parse_invalid(self, raw):
        with pytest.raises(exceptions.InvalidThresholds):
            parse_thresholds(raw)

    def test_parse_empty(self):
        with pytest.raises(exceptions.MissingQueryParameter):
            parse_thresholds([""])

    def test_resolve_explicit_list(self):
        assert resolve_thresholds([3.0, 1.0], np.zeros((2, 2))) == [3.0, 1.0]

    def test_resolve_explicit_bounds(self):
        # max is inclusive, no floating point artifacts
        assert resolve_thresholds(ThresholdRange(0.2, 0.8, 0.2), None) == [0.2, 0.4, 0.6, 0.8]
        assert resolve_thresholds(ThresholdRange(0, 10, 3), None) == [0, 3, 6, 9]

    def test_resolve_bounds_from_data(self):
        val = np.array([[273.4, 281.0], [np.nan, 296.2]])
        # First threshold aligned on a multiple of step, max taken from the data
        assert resolve_thresholds(ThresholdRange(None, None, 5), val) == [275, 280, 285, 290, 295]
        assert resolve_thresholds(ThresholdRange(None, 285, 5), val) == [275, 280, 285]
        # Explicit min is used as is
        assert resolve_thresholds(ThresholdRange(271, None, 10), val) == [271, 281, 291]
        # Data minimum exactly on a multiple of step is included
        assert resolve_thresholds(ThresholdRange(None, None, 0.1), np.array([0.3, 0.5])) == [0.3, 0.4, 0.5]

    def test_resolve_empty_range(self):
        val = np.array([10.0, 20.0])
        assert resolve_thresholds(ThresholdRange(30, None, 5), val) == []

    def test_resolve_only_nan(self):
        with pytest.raises(exceptions.NoDataInSelection):
            resolve_thresholds(ThresholdRange(None, None, 1), np.full((2, 2), np.nan))

    def test_resolve_too_many_thresholds(self):
        with pytest.raises(exceptions.InvalidThresholds):
            resolve_thresholds(ThresholdRange(0, 1, 1e-6), None)


@pytest.mark.Isoline
class TestIsolineThresholdsRange:
    def get(self, client, dataset, query):
        return client.get(f"/datasets/{dataset}/isoline?variable=value&{query}")

    def test_range_from_data(self, client: TestClient):
        """two_bumps values are in ]0, 1]: '::0.25' gives 0.25, 0.5, 0.75, 1.0."""
        response = self.get(client, REGULAR, "thresholds=::0.25")
        assert response.status_code == 200
        data = response.json()
        assert list(data.keys()) == ["0.25", "0.5", "0.75", "1.0"]
        explicit = self.get(client, REGULAR, "thresholds=0.25&thresholds=0.5&thresholds=0.75").json()
        for key in ("0.25", "0.5", "0.75"):
            assert data[key] == explicit[key]

    def test_range_explicit_bounds(self, client: TestClient):
        response = self.get(client, REGULAR, "thresholds=0.2:0.8:0.2&format=geojson")
        assert response.status_code == 200
        thresholds = [f["properties"]["threshold"] for f in response.json()["features"]]
        assert thresholds == [0.2, 0.4, 0.6, 0.8]

    def test_range_partial_bounds(self, client: TestClient):
        assert list(self.get(client, REGULAR, "thresholds=:0.5:0.25").json()) == ["0.25", "0.5"]
        assert list(self.get(client, REGULAR, "thresholds=0.6::0.2").json()) == ["0.6", "0.8", "1.0"]

    def test_range_uses_bbox_data(self, client: TestClient):
        """Bounds are taken from the data inside the bbox: on the NaN dataset
        (distance to the center), values in [2, 3]x[-1, 1] are in [2, ~3.2]."""
        response = self.get(
            client, WITH_NAN, "thresholds=::1&lon_min=-3&lon_max=-2&lat_min=-1&lat_max=1"
        )
        assert response.status_code == 200
        assert list(response.json().keys()) == ["2.0", "3.0"]

    @pytest.mark.parametrize("thresholds", ["1:2", "::0", "::-1", "5:1:1", "a::1", "::1e-9"])
    def test_invalid_range(self, client: TestClient, thresholds):
        response = self.get(client, REGULAR, f"thresholds={thresholds}")
        assert response.status_code == 400
        assert "INVALID_THRESHOLDS" in response.text

    def test_invalid_value(self, client: TestClient):
        response = self.get(client, REGULAR, "thresholds=0.5&thresholds=abc")
        assert response.status_code == 400
        assert "INVALID_THRESHOLDS" in response.text

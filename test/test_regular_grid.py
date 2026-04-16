"""
test_regular_grid.py — Tests for regular grid datasets.

A "regular grid" has 1D lat and lon coordinates on independent dimensions,
e.g. dataset[lat].dims == ("lat",) and dataset[lon].dims == ("lon",).
This is the simplest and most common case (e.g. ERA5-style data).
"""

import os
import pytest
from fastapi.testclient import TestClient

import utils


# ---------------------------------------------------------------------------
# Dataset parameters — used both for generation and for assertions
# ---------------------------------------------------------------------------
TIMES = 5
LATS = 10
LONS = 15
LAT_START = 45.1
LAT_STEP = 0.1
LON_START = 10.1
LON_STEP = 0.1

DATASET_NAME = "regular_grid"
DATASET_NAME_COMPLEX = "regular_grid_complex"
TMP_FOLDER = os.environ.get("TEST_TMP_FOLDER", "test/tests_tmp")
LEVELS = 3


@pytest.mark.RegularGrid
class TestRegularGrid:
    @pytest.fixture(scope="class", autouse=True)
    def cleanup(self):
        yield
        utils.cleanup_test_files()

    # ------------------------------------------------------------------
    # Setup: generate + convert dataset
    # ------------------------------------------------------------------

    def test_generate_dataset(self):
        """Generate a synthetic regular-grid dataset and save as NetCDF."""
        description = {
            "Value": {
                "type": "float",
                "method": "linear",
                "step": 1,
                "dimensions": [
                    {
                        "name": "time",
                        "type": "date",
                        "size": TIMES,
                        "start": "2026-01-01",
                        "freq": "D",
                    },
                    {
                        "name": "lat",
                        "type": "latitude",
                        "size": LATS,
                        "start": LAT_START,
                        "step": LAT_STEP,
                    },
                    {
                        "name": "lon",
                        "type": "longitude",
                        "size": LONS,
                        "start": LON_START,
                        "step": LON_STEP,
                    },
                ],
            }
        }
        dataset = utils.DatasetGenerator(description=description).generate()
        dataset.save(DATASET_NAME, to_netcdf=True)

        assert os.path.exists(os.path.join(TMP_FOLDER, f"{DATASET_NAME}.nc"))

    def test_convert_dataset(self, convert):
        """Convert the NetCDF to Zarr format via conversion_tool."""
        output_path = os.path.join(TMP_FOLDER, f"{DATASET_NAME}.zarr")
        convert(
            dataset_name=DATASET_NAME,
            input_path=os.path.join(TMP_FOLDER, f"{DATASET_NAME}.nc"),
            output_path=output_path,
            config={
                "variables": {"lon": "lon", "lat": "lat", "time": "time"},
                "pipelines": {
                    "preprocess": ["load_from_netcdf", "unify_chunks", "save"]
                },
                "version": 2,
            },
        )

        assert os.path.exists(output_path)

    # ------------------------------------------------------------------
    # Extract — raw format
    # ------------------------------------------------------------------

    def test_extract(self, client: TestClient):
        """Basic extraction at a specific time step returns expected bounds."""
        response = client.get(
            f"/datasets/{DATASET_NAME}/extract?variable=Value&time=2026-01-03"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["bounds"] == {"min": 300, "max": 449}
        assert "data" in data
        assert "longitudes" in data["data"]
        assert "latitudes" in data["data"]
        assert "values" in data["data"]
        assert len(data["data"]["values"]) == LATS * LONS

    def test_extract_time_interpolation(self, client: TestClient):
        """Time interpolation between two steps returns midpoint values."""
        response = client.get(
            f"/datasets/{DATASET_NAME}/extract"
            "?variable=Value&time=2026-01-03T12:00:00&interp_time=true&interp_vars_method=linear"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["bounds"] == {"min": 375, "max": 524}

    def test_extract_geojson(self, client: TestClient):
        """GeoJSON format returns a valid FeatureCollection."""
        response = client.get(
            f"/datasets/{DATASET_NAME}/extract?variable=Value&time=2026-01-01&format=geojson"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "FeatureCollection"
        assert "features" in data
        assert len(data["features"]) > 0
        # Each feature should have a point geometry and a value property
        feature = data["features"][0]
        assert feature["geometry"]["type"] == "Point"
        assert "value" in feature["properties"]

    def test_extract_tile(self, client: TestClient):
        """Bounding box extraction returns only points inside the box."""
        lat_min = LAT_START + 0.25
        lat_max = LAT_START + LAT_STEP * LATS - 0.25
        lon_min = LON_START + 0.25
        lon_max = LON_START + LON_STEP * LONS - 0.25

        response = client.get(
            f"/datasets/{DATASET_NAME}/extract?variable=Value&time=2026-01-01"
            f"&lat_min={lat_min}&lat_max={lat_max}&lon_min={lon_min}&lon_max={lon_max}"
        )

        assert response.status_code == 200
        data = response.json()

        min_lat = min_lon = float("inf")
        max_lat = max_lon = float("-inf")
        for idx, value in enumerate(data["data"]["values"]):
            if value is not None:
                current_lat = data["data"]["latitudes"][idx]
                current_lon = data["data"]["longitudes"][idx]
                min_lat = min(min_lat, current_lat)
                max_lat = max(max_lat, current_lat)
                min_lon = min(min_lon, current_lon)
                max_lon = max(max_lon, current_lon)

        assert min_lat >= lat_min
        assert max_lat <= lat_max
        assert min_lon >= lon_min
        assert max_lon <= lon_max

    def test_extract_bounding_box_out_of_range(self, client: TestClient):
        """Bounding box entirely outside dataset extent should return an error."""
        response = client.get(
            f"/datasets/{DATASET_NAME}/extract?variable=Value&time=2026-01-01"
            "&lat_min=-90&lat_max=-80&lon_min=-180&lon_max=-170"
        )

        assert response.status_code == 400

    # ------------------------------------------------------------------
    # Extract — mesh format
    # ------------------------------------------------------------------

    def test_extract_mesh(self, client: TestClient):
        """Mesh extraction with tile size returns correct vertex and index counts."""
        mesh_tile_size = 5
        response = client.get(
            f"/datasets/{DATASET_NAME}/extract?variable=Value&time=2026-01-01"
            f"&format=mesh&mesh_tile_size={mesh_tile_size}"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["bounds"] == {"min": 0, "max": 149}
        assert len(data["values"]) == mesh_tile_size * mesh_tile_size
        assert "vertices" in data
        assert "indices" in data
        # Expected triangles: (N-1)*(N-1)*2, each with 3 indices
        expected_indices = (mesh_tile_size - 1) ** 2 * 6
        assert len(data["indices"]) == expected_indices

    def test_extract_mesh_interpolation(self, client: TestClient):
        """Mesh extraction with spatial interpolation returns expected edge values."""
        mesh_tile_size = 5
        response = client.get(
            f"/datasets/{DATASET_NAME}/extract?variable=Value&time=2026-01-01"
            f"&format=mesh&mesh_tile_size={mesh_tile_size}&interp_spatial_method=linear"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["bounds"] == {"min": 0, "max": 149}
        assert len(data["values"]) == mesh_tile_size * mesh_tile_size
        # First row should be linearly interpolated from min to max
        for i in range(mesh_tile_size):
            assert data["values"][i] == pytest.approx((LONS - 1) / 4 * i)

    # ------------------------------------------------------------------
    # Probe
    # ------------------------------------------------------------------

    def test_probe_nearest(self, client: TestClient):
        """Probe at a known grid point returns a single scalar value."""
        lon = LON_START
        lat = LAT_START
        response = client.get(
            f"/datasets/{DATASET_NAME}/probe?variables=Value&lon={lon}&lat={lat}"
        )

        assert response.status_code == 200
        data = response.json()
        assert "variables" in data
        assert "Value" in data["variables"]
        values = data["variables"]["Value"]["values"]
        assert isinstance(values, list)
        assert len(values) > 0

    def test_probe_time_interpolation(self, client: TestClient):
        """Probe with time interpolation returns midpoint value."""
        lon = LON_START
        lat = LAT_START
        # Get value at t=day2 and t=day3
        r1 = client.get(
            f"/datasets/{DATASET_NAME}/probe?variables=Value&lon={lon}&lat={lat}&time=2026-01-02"
        )
        r2 = client.get(
            f"/datasets/{DATASET_NAME}/probe?variables=Value&lon={lon}&lat={lat}&time=2026-01-03"
        )
        r_interp = client.get(
            f"/datasets/{DATASET_NAME}/probe?variables=Value&lon={lon}&lat={lat}"
            "&time=2026-01-02T12:00:00&interp_time=true&interp_vars_method=linear"
        )

        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r_interp.status_code == 200

        v1 = r1.json()["variables"]["Value"]["values"]
        v2 = r2.json()["variables"]["Value"]["values"]
        v_interp = r_interp.json()["variables"]["Value"]["values"]

        v1 = v1[0] if isinstance(v1, list) else v1
        v2 = v2[0] if isinstance(v2, list) else v2
        v_interp = v_interp[0] if isinstance(v_interp, list) else v_interp

        assert v_interp == pytest.approx((v1 + v2) / 2)

    # ------------------------------------------------------------------
    # Select
    # ------------------------------------------------------------------

    def test_select(self, client: TestClient):
        """Free selection endpoint returns data array for given dimensions."""
        response = client.get(
            f"/datasets/{DATASET_NAME}/select?variable=Value&time=2026-01-01"
        )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert isinstance(data["data"], list)

    def test_select_unknown_variable(self, client: TestClient):
        """Selecting an unknown variable returns 400."""
        response = client.get(
            f"/datasets/{DATASET_NAME}/select?variable=NonExistentVar&time=2026-01-01"
        )

        assert response.status_code == 400

    # ------------------------------------------------------------------
    # Dataset infos
    # ------------------------------------------------------------------

    def test_dataset_infos(self, client: TestClient):
        """Dataset infos endpoint returns expected structure."""
        response = client.get(f"/datasets/{DATASET_NAME}/infos")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == DATASET_NAME
        assert "variables" in data
        assert "coordinates" in data
        assert "Value" in data["variables"]
        assert data["bounding_box"] is not None
        assert data["time_bounds"] is not None

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_extract_height_interpolation(self):
        """Height interpolation is not applicable to regular 2D grids."""
        pytest.skip("Height interpolation not applicable to regular 2D grids")


@pytest.mark.RegularGrid
class TestRegularGridComplex:
    @pytest.fixture(scope="class", autouse=True)
    def cleanup(self):
        yield
        utils.cleanup_test_files()

    # ------------------------------------------------------------------
    # Setup: generate + convert dataset
    # ------------------------------------------------------------------

    def test_generate_dataset(self):
        """Generate a complex regular-grid dataset with heterogeneous variables."""
        description = {
            "Value1": {
                "type": "float",
                "method": "linear",
                "bounds": {"min": 0, "max": 100},
                "dimensions": [
                    {
                        "name": "time",
                        "type": "date",
                        "size": TIMES,
                        "start": "2026-01-01",
                        "freq": "D",
                    },
                    {
                        "name": "lat",
                        "type": "latitude",
                        "size": LATS,
                        "start": LAT_START,
                        "step": LAT_STEP,
                    },
                    {
                        "name": "lon",
                        "type": "longitude",
                        "size": LONS,
                        "start": LON_START,
                        "step": LON_STEP,
                    },
                ],
            },
            "Value2": {
                "type": "float",
                "method": "linear",
                "bounds": {"min": 200, "max": 300},
                "dimensions": [
                    {
                        "name": "time",
                        "type": "date",
                        "size": TIMES,
                        "start": "2026-01-01",
                        "freq": "D",
                    },
                    {
                        "name": "lat",
                        "type": "latitude",
                        "size": LATS,
                        "start": LAT_START,
                        "step": LAT_STEP,
                    },
                    {
                        "name": "lon",
                        "type": "longitude",
                        "size": LONS,
                        "start": LON_START,
                        "step": LON_STEP,
                    },
                ],
            },
            "Value3": {
                "type": "float",
                "method": "linear",
                "bounds": {"min": 500, "max": 600},
                "dimensions": [
                    {
                        "name": "time",
                        "type": "date",
                        "size": TIMES,
                        "start": "2026-01-01",
                        "freq": "D",
                    },
                    {
                        "name": "level",
                        "type": "float",
                        "method": "linear",
                        "size": LEVELS,
                        "start": 0,
                        "step": 1,
                    },
                    {
                        "name": "lat",
                        "type": "latitude",
                        "size": LATS,
                        "start": LAT_START,
                        "step": LAT_STEP,
                    },
                    {
                        "name": "lon",
                        "type": "longitude",
                        "size": LONS,
                        "start": LON_START,
                        "step": LON_STEP,
                    },
                ],
            },
        }
        dataset = utils.DatasetGenerator(description=description).generate()
        dataset.save(DATASET_NAME_COMPLEX, to_netcdf=True)

        assert os.path.exists(os.path.join(TMP_FOLDER, f"{DATASET_NAME_COMPLEX}.nc"))

    def test_convert_dataset(self, convert):
        """Convert the complex NetCDF to Zarr format."""
        output_path = os.path.join(TMP_FOLDER, f"{DATASET_NAME_COMPLEX}.zarr")
        convert(
            dataset_name=DATASET_NAME_COMPLEX,
            input_path=os.path.join(TMP_FOLDER, f"{DATASET_NAME_COMPLEX}.nc"),
            output_path=output_path,
            config={
                "variables": {"lon": "lon", "lat": "lat", "time": "time"},
                "pipelines": {
                    "preprocess": ["load_from_netcdf", "unify_chunks", "save"]
                },
                "version": 2,
            },
        )

        assert os.path.exists(output_path)

    # ------------------------------------------------------------------
    # Probe
    # ------------------------------------------------------------------

    def test_probe_multi_variable_with_extra_dim(self, client: TestClient):
        """
        Probe multiple variables where one has an extra dimension.
        Verify that specifying the extra dimension doesn't impact other variables.
        """
        lon = LON_START
        lat = LAT_START
        level = 1
        time = "2026-01-01"

        response = client.get(
            f"/datasets/{DATASET_NAME_COMPLEX}/probe"
            f"?variables=Value1&variables=Value2&variables=Value3&lon={lon}&lat={lat}&time={time}&level={level}"
        )

        assert response.status_code == 200
        data = response.json()
        assert "variables" in data

        assert "Value1" in data["variables"]
        assert "Value2" in data["variables"]
        assert "Value3" in data["variables"]

        value1 = data["variables"]["Value1"]["values"]
        value2 = data["variables"]["Value2"]["values"]
        value3 = data["variables"]["Value3"]["values"]
        assert isinstance(value1, list) and len(data["variables"]["Value1"]["values"]) == 1 or isinstance(value1, (int, float))
        assert isinstance(value2, list) and len(data["variables"]["Value2"]["values"]) == 1 or isinstance(value2, (int, float))
        assert isinstance(value3, list) and len(data["variables"]["Value3"]["values"]) == 1 or isinstance(value3, (int, float))

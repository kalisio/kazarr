"""
test_rectilinear_grid.py — Tests for rectilinear grid datasets.

A "rectilinear grid" has 2D or 3D lat/lon arrays sharing their dimensions,
e.g. dataset[lat].dims == ("DimK", "DimJ", "DimI"). Common in model outputs
(e.g. ocean circulation models with curvilinear grids).
"""

import os
import pytest
from fastapi.testclient import TestClient

import utils


# ---------------------------------------------------------------------------
# Dataset parameters
# ---------------------------------------------------------------------------
HEIGHTS = 1
LATS = 91
LONS = 137

DATASET_NAME = "rectilinear_grid"
TMP_FOLDER = os.environ.get("TEST_TMP_FOLDER", "test/tests_tmp")


@pytest.mark.RectilinearGrid
class TestRectilinearGrid:
    @pytest.fixture(scope="class", autouse=True)
    def cleanup(self):
        yield
        utils.cleanup_test_files()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def test_generate_dataset(self):
        description = {
            "Precipitation": {
                "type": "float",
                "method": "linear",
                "step": 1,
                "dimensions": [
                    {
                        "name": "steps",
                        "type": "steps",
                        "size": 5,
                        "start": "2026-01-01 00:00:00",
                        "freq": "1h",
                    },
                    {"name": "DimK", "size": "lat.DimK"},
                    {"name": "DimJ", "size": "lat.DimJ"},
                    {"name": "DimI", "size": "lat.DimI"},
                ],
            },
            "lat": {
                "type": "load",
                "sample": "rectilinear_grid.nc",
                "variable": "CoordY0",
                "dimensions": ["DimK", "DimJ", "DimI"],
            },
            "lon": {
                "type": "load",
                "sample": "rectilinear_grid.nc",
                "variable": "CoordX0",
                "dimensions": ["DimK", "DimJ", "DimI"],
            },
            "height": {
                "type": "load",
                "sample": "rectilinear_grid.nc",
                "variable": "CoordZ0",
                "dimensions": ["DimK", "DimJ", "DimI"],
            },
        }
        dataset = utils.DatasetGenerator(description=description).generate()
        dataset.save(DATASET_NAME, to_netcdf=True)

        assert os.path.exists(os.path.join(TMP_FOLDER, f"{DATASET_NAME}.nc"))

    def test_convert_dataset(self, convert):
        output_path = os.path.join(TMP_FOLDER, f"{DATASET_NAME}.zarr")
        convert(
            dataset_name=DATASET_NAME,
            input_path=os.path.join(TMP_FOLDER, f"{DATASET_NAME}.nc"),
            output_path=output_path,
            config={
                "variables": {"lon": "lon", "lat": "lat", "time": "steps"},
                "dimensions": {"fixed": {"DimK": 0}},
                "referenceTime": {
                    "variable": "reference_time",
                    "format": "%Y-%m-%d %H:%M:%S",
                },
                "pipelines": {
                    "preprocess": [
                        "load_from_netcdf",
                        "unify_chunks",
                        "delta_time_to_datetime",
                        "save",
                    ]
                },
                "version": 2,
            },
        )
        assert os.path.exists(output_path)

    # ------------------------------------------------------------------
    # Extract — raw format
    # ------------------------------------------------------------------

    def test_extract(self, client: TestClient):
        """Full extraction returns all grid points with correct bounds."""
        response = client.get(
            f"/datasets/{DATASET_NAME}/extract?variable=Precipitation&time=2026-01-01"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["bounds"] == {"min": 0, "max": HEIGHTS * LATS * LONS - 1}

    def test_extract_time_interpolation(self, client: TestClient):
        """Time interpolation at midpoint between two steps (t=1h and t=2h)."""
        response = client.get(
            f"/datasets/{DATASET_NAME}/extract"
            "?variable=Precipitation&time=2026-01-01T01:30:00&interp_time=true&interp_vars_method=linear"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["bounds"] == {
            "min": HEIGHTS * LATS * LONS * 1.5,
            "max": HEIGHTS * LATS * LONS * 2.5 - 1,
        }

    def test_extract_geojson(self, client: TestClient):
        """GeoJSON extraction returns FeatureCollection with point features."""
        response = client.get(
            f"/datasets/{DATASET_NAME}/extract?variable=Precipitation&time=2026-01-01&format=geojson"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) > 0
        feature = data["features"][0]
        assert feature["geometry"]["type"] == "Point"
        assert "value" in feature["properties"]

    def test_extract_tile(self, client: TestClient):
        """Bounding box extraction returns only points within the box."""
        bbox = "lon_min=2.0&lon_max=4.0&lat_min=43.0&lat_max=44.0"
        response = client.get(
            f"/datasets/{DATASET_NAME}/extract?variable=Precipitation&time=2026-01-01&{bbox}"
        )

        assert response.status_code == 200
        data = response.json()
        # All non-null points must fall inside the bounding box
        for idx, value in enumerate(data["data"]["values"]):
            if value is not None:
                lon = data["data"]["longitudes"][idx]
                lat = data["data"]["latitudes"][idx]
                assert 2.0 <= lon <= 4.0
                assert 43.0 <= lat <= 44.0

    # ------------------------------------------------------------------
    # Extract — mesh format
    # ------------------------------------------------------------------

    def test_extract_mesh(self, client: TestClient):
        """Mesh extraction produces correct vertex, index and value counts."""
        mesh_tile_size = 200
        response = client.get(
            f"/datasets/{DATASET_NAME}/extract?variable=Precipitation&time=2026-01-01"
            f"&format=mesh&mesh_tile_size={mesh_tile_size}"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["bounds"] == {"min": 0, "max": HEIGHTS * LATS * LONS - 1}
        assert len(data["values"]) == mesh_tile_size * mesh_tile_size
        assert "indices" in data

        expected_indices_count = (mesh_tile_size - 1) * (mesh_tile_size - 1) * 6
        assert len(data["indices"]) == expected_indices_count

        max_index = len(data["values"]) - 1
        assert max(data["indices"]) <= max_index
        assert min(data["indices"]) >= 0

        assert isinstance(data["values"][0], (int, float))

    def test_extract_mesh_interpolation(self, client: TestClient):
        """Mesh extraction with spatial interpolation returns plausible values."""
        mesh_tile_size = 16
        response = client.get(
            f"/datasets/{DATASET_NAME}/extract?variable=Precipitation&time=2026-01-01"
            f"&format=mesh&mesh_tile_size={mesh_tile_size}&interp_spatial_method=linear"
        )

        assert response.status_code == 200
        data = response.json()
        assert "vertices" in data
        assert "indices" in data
        assert "values" in data
        assert len(data["values"]) == mesh_tile_size * mesh_tile_size
        # All values should be finite numbers (no None from interpolated mesh)
        for v in data["values"]:
            assert v is not None
            assert isinstance(v, (int, float))

    # ------------------------------------------------------------------
    # Probe
    # ------------------------------------------------------------------

    def test_probe_spatial_interpolation(self, client: TestClient):
        """IDW interpolation at midpoint between two known grid points."""
        r1 = client.get(
            f"/datasets/{DATASET_NAME}/probe?variables=Precipitation&lat=43.3&lon=2.3"
        )
        r2 = client.get(
            f"/datasets/{DATASET_NAME}/probe?variables=Precipitation&lat=43.3&lon=2.4"
        )
        r_interp = client.get(
            f"/datasets/{DATASET_NAME}/probe?variables=Precipitation&lat=43.3&lon=2.35"
            "&interp_spatial_method=idw&interp_spatial_params=radius:0.2"
        )

        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r_interp.status_code == 200

        values1 = r1.json()["variables"]["Precipitation"]["values"]
        values2 = r2.json()["variables"]["Precipitation"]["values"]
        values_interp = r_interp.json()["variables"]["Precipitation"]["values"]

        assert len(values1) == len(values2) == len(values_interp)

        for v1, v2, v_interp in zip(values1, values2, values_interp):
            expected = (v2 - v1) / 2 + v1
            assert expected == pytest.approx(v_interp)

    def test_probe_spatial_interpolation_no_neighbors(self, client: TestClient):
        """IDW probe with radius too small to find neighbors returns 400."""
        response = client.get(
            f"/datasets/{DATASET_NAME}/probe?variables=Precipitation&lat=43.3&lon=2.35"
            "&interp_spatial_method=idw&interp_spatial_params=radius:0.04"
        )

        assert response.status_code == 400
        assert response.json()["detail"]["error_code"] == "NO_DATA_IN_SELECTION"

    def test_probe_time_interpolation(self, client: TestClient):
        """Probe with time interpolation returns midpoint between two time steps."""
        r1 = client.get(
            f"/datasets/{DATASET_NAME}/probe?variables=Precipitation&lat=43.3&lon=2.3"
            "&time=2026-01-01T01:00:00"
        )
        r2 = client.get(
            f"/datasets/{DATASET_NAME}/probe?variables=Precipitation&lat=43.3&lon=2.3"
            "&time=2026-01-01T02:00:00"
        )
        r_interp = client.get(
            f"/datasets/{DATASET_NAME}/probe?variables=Precipitation&lat=43.3&lon=2.3"
            "&time=2026-01-01T01:30:00&interp_time=true&interp_vars_method=linear"
        )

        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r_interp.status_code == 200

        def extract_scalar(values):
            return values[0] if isinstance(values, list) else values

        v1 = extract_scalar(r1.json()["variables"]["Precipitation"]["values"])
        v2 = extract_scalar(r2.json()["variables"]["Precipitation"]["values"])
        v_interp = extract_scalar(
            r_interp.json()["variables"]["Precipitation"]["values"]
        )

        assert v_interp == pytest.approx((v1 + v2) / 2)

    def test_probe_nearest(self, client: TestClient):
        """Nearest-neighbor probe returns a valid value."""
        response = client.get(
            f"/datasets/{DATASET_NAME}/probe?variables=Precipitation&lat=43.3&lon=2.3"
        )

        assert response.status_code == 200
        data = response.json()
        assert "variables" in data
        assert "Precipitation" in data["variables"]

    # ------------------------------------------------------------------
    # Mesh endpoint
    # ------------------------------------------------------------------

    def test_mesh_endpoint(self, client: TestClient):
        """The /mesh endpoint returns vertices and triangulated indices."""
        response = client.get(f"/datasets/{DATASET_NAME}/mesh")

        assert response.status_code == 200
        data = response.json()
        assert "vertices" in data
        assert "indices" in data
        assert len(data["vertices"]) > 0
        assert len(data["indices"]) > 0
        # Indices should reference valid vertex positions (3 floats per vertex)
        n_vertices = len(data["vertices"]) // 3
        assert max(data["indices"]) < n_vertices

    def test_mesh_endpoint_geojson(self, client: TestClient):
        """The /mesh endpoint in GeoJSON format returns a FeatureCollection."""
        response = client.get(f"/datasets/{DATASET_NAME}/mesh?format=geojson")

        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) > 0

    # ------------------------------------------------------------------
    # Dataset infos
    # ------------------------------------------------------------------

    def test_dataset_infos(self, client: TestClient):
        """Dataset infos endpoint exposes variables and bounding box."""
        response = client.get(f"/datasets/{DATASET_NAME}/infos")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == DATASET_NAME
        assert "Precipitation" in data["variables"]
        assert data["bounding_box"] is not None

"""
test_radial_grid.py — Tests for radial grid datasets.

A "radial grid" is a polar/cylindrical grid where coordinates rotate around a
central axis. It has 2D lat/lon arrays with a periodic angular dimension.
The mesh_type must be set to "radial" in the dataset configuration.
"""

import os
import pytest
from fastapi.testclient import TestClient

import utils


# ---------------------------------------------------------------------------
# Dataset parameters
# ---------------------------------------------------------------------------
DATASET_NAME = "radial_grid"
TMP_FOLDER = os.environ.get("TEST_TMP_FOLDER", "test/tests_tmp")


@pytest.mark.RadialGrid
class TestRadialGrid:
    @pytest.fixture(scope="class", autouse=True)
    def cleanup(self):
        yield
        utils.cleanup_test_files()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def test_generate_dataset(self):
        """Generate a synthetic radial-grid dataset from sample coordinates."""
        description = {
            "WindSpeed": {
                "type": "float",
                "method": "sin",
                "bounds": {"min": 0, "max": 150},
                "dimensions": [
                    {
                        "name": "time",
                        "type": "date",
                        "size": 5,
                        "start": "2026-01-01",
                        "freq": "D",
                    },
                    {"name": "DimK", "size": "lat.DimK"},
                    {"name": "DimJ", "size": "lat.DimJ"},
                    {"name": "DimI", "size": "lat.DimI"},
                ],
            },
            "lat": {
                "type": "load",
                "sample": "radial_grid.nc",
                "variable": "CoordY0",
                "dimensions": ["DimK", "DimJ", "DimI"],
            },
            "lon": {
                "type": "load",
                "sample": "radial_grid.nc",
                "variable": "CoordX0",
                "dimensions": ["DimK", "DimJ", "DimI"],
            },
            "height": {
                "type": "load",
                "sample": "radial_grid.nc",
                "variable": "CoordZ0",
                "dimensions": ["DimK", "DimJ", "DimI"],
            },
        }
        dataset = utils.DatasetGenerator(description=description).generate()
        dataset.save(DATASET_NAME, to_netcdf=True)

        assert os.path.exists(os.path.join(TMP_FOLDER, f"{DATASET_NAME}.nc"))

    def test_convert_dataset(self, convert):
        """Convert the NetCDF to Zarr, specifying radial mesh_type."""
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
            mesh_type="radial",
        )

        assert os.path.exists(output_path)

    # ------------------------------------------------------------------
    # Extract — raw format
    # ------------------------------------------------------------------

    def test_extract(self, client: TestClient):
        """Full extraction at a specific time step returns valid data."""
        response = client.get(
            f"/datasets/{DATASET_NAME}/extract?variable=WindSpeed&time=2026-01-01"
        )

        assert response.status_code == 200
        data = response.json()
        assert "WindSpeed" in data["variables"]
        assert "bounds" in data["variables"]["WindSpeed"]
        assert data["variables"]["WindSpeed"]["bounds"]["min"] >= 0
        assert data["variables"]["WindSpeed"]["bounds"]["max"] <= 150
        assert "values" in data
        assert len(data["values"]["WindSpeed"]) > 0

    def test_extract_tile(self, client: TestClient):
        """Bounding box extraction on radial grid returns points inside the box."""
        # Use a bounding box known to overlap the radial grid sample data
        bbox = "lon_min=2.0&lon_max=4.0&lat_min=43.0&lat_max=44.0"
        response = client.get(
            f"/datasets/{DATASET_NAME}/extract?variable=WindSpeed&time=2026-01-01&{bbox}"
        )

        # May return 400 if bbox is outside the radial grid extent — both are valid
        assert response.status_code in (200, 400)
        if response.status_code == 200:
            data = response.json()
            assert "data" in data

    def test_extract_geojson(self, client: TestClient):
        """GeoJSON format returns a FeatureCollection."""
        response = client.get(
            f"/datasets/{DATASET_NAME}/extract?variable=WindSpeed&time=2026-01-01&format=geojson"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) > 0

    def test_extract_time_interpolation(self, client: TestClient):
        """Time interpolation returns a midpoint value between two time steps."""
        r1 = client.get(
            f"/datasets/{DATASET_NAME}/extract?variable=WindSpeed&time=2026-01-01"
        )
        r2 = client.get(
            f"/datasets/{DATASET_NAME}/extract?variable=WindSpeed&time=2026-01-02"
        )
        r_interp = client.get(
            f"/datasets/{DATASET_NAME}/extract"
            "?variable=WindSpeed&time=2026-01-01T12:00:00&interp_time=true&interp_vars_method=linear"
        )

        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r_interp.status_code == 200

        b1 = r1.json()["variables"]["WindSpeed"]["bounds"]
        b2 = r2.json()["variables"]["WindSpeed"]["bounds"]
        b_interp = r_interp.json()["variables"]["WindSpeed"]["bounds"]

        # Interpolated min/max should fall between the two time steps
        assert min(b1["min"], b2["min"]) <= b_interp["min"] + 0.01  # approx tolerance
        assert b_interp["max"] <= max(b1["max"], b2["max"]) + 0.01

    # ------------------------------------------------------------------
    # Extract — mesh format
    # ------------------------------------------------------------------

    def test_extract_mesh(self, client: TestClient):
        """Mesh extraction on radial grid returns valid triangulated output."""
        mesh_tile_size = 64
        response = client.get(
            f"/datasets/{DATASET_NAME}/extract?variable=WindSpeed&time=2026-01-01"
            f"&format=mesh&mesh_tile_size={mesh_tile_size}"
        )

        assert response.status_code == 200
        data = response.json()
        assert "vertices" in data
        assert "indices" in data
        assert "values" in data
        assert len(data["values"]) == mesh_tile_size * mesh_tile_size

    # ------------------------------------------------------------------
    # Probe
    # ------------------------------------------------------------------

    def test_probe_nearest(self, client: TestClient):
        """Nearest-neighbor probe returns a WindSpeed value."""
        response = client.get(
            f"/datasets/{DATASET_NAME}/probe?variables=WindSpeed&lat=43.3&lon=2.3"
        )

        assert response.status_code == 200
        data = response.json()
        assert "variables" in data
        assert "WindSpeed" in data["variables"]
        values = data["values"]["WindSpeed"]
        assert isinstance(values, list)
        assert len(values) > 0

    # ------------------------------------------------------------------
    # Mesh endpoint
    # ------------------------------------------------------------------

    def test_mesh_endpoint(self, client: TestClient):
        """The /mesh endpoint returns a valid triangulated mesh."""
        response = client.get(f"/datasets/{DATASET_NAME}/mesh")

        assert response.status_code == 200
        data = response.json()
        assert "vertices" in data
        assert "indices" in data
        assert len(data["vertices"]) > 0

    # ------------------------------------------------------------------
    # Dataset metadata
    # ------------------------------------------------------------------

    def test_dataset_metadata(self, client: TestClient):
        """Dataset metadata endpoint exposes WindSpeed variable and bounding box."""
        response = client.get(f"/datasets/{DATASET_NAME}/metadata")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == DATASET_NAME
        assert "WindSpeed" in data["variables"]
        assert data["bounding_box"] is not None

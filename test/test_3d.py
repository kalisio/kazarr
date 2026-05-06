"""
test_3d.py — Tests for 3D (volumetric) dataset extraction and mesh.

A "3D dataset" is a regular grid with an additional vertical dimension
(e.g. pressure levels, altitude layers). The dataset has shape (time, level, lat, lon).

Tests cover:
- 2D slice extraction from a 3D dataset (providing a level coordinate)
- Full 3D volume extraction (is_3d=true)
- 3D extraction with a vertical bounding box (z_min / z_max)
- 3D GeoJSON output including height coordinates
- 2D mesh from a 3D dataset (slice at a given level)
- 3D volumetric mesh generation
- Probe on a 3D dataset with height
- Error cases: missing level on 2D request, z bbox out of range
"""

import os
import pytest
from fastapi.testclient import TestClient

import utils

# ---------------------------------------------------------------------------
# Dataset parameters
# ---------------------------------------------------------------------------
TIMES = 3
LEVELS = 4
LATS = 8
LONS = 10

LAT_START = 43.0
LAT_STEP = 0.5
LON_START = 2.0
LON_STEP = 0.5
LEVEL_START = 100.0  # e.g. pressure levels in hPa
LEVEL_STEP = 100.0  # 100, 200, 300, 400

DATASET_NAME = "dataset_3d"
TMP_FOLDER = os.environ.get("TEST_TMP_FOLDER", "test/tests_tmp")


@pytest.mark.Dataset3D
class TestDataset3D:
    @pytest.fixture(scope="class", autouse=True)
    def cleanup(self):
        yield
        utils.cleanup_test_files()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def test_generate_dataset(self):
        """Generate a synthetic 3D dataset (time × level × lat × lon) and save as NetCDF."""
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
                        "name": "level",
                        "type": "float",
                        "size": LEVELS,
                        "start": LEVEL_START,
                        "step": LEVEL_STEP,
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
        """Convert the 3D NetCDF to Zarr with lon/lat/level/time variable mappings."""
        output_path = os.path.join(TMP_FOLDER, f"{DATASET_NAME}.zarr")
        convert(
            dataset_name=DATASET_NAME,
            input_path=os.path.join(TMP_FOLDER, f"{DATASET_NAME}.nc"),
            output_path=output_path,
            config={
                "variables": {
                    "lon": "lon",
                    "lat": "lat",
                    "time": "time",
                    "height": "level",
                },
                "pipelines": {
                    "preprocess": ["load_from_netcdf", "unify_chunks", "save"]
                },
                "version": 2,
            },
        )
        assert os.path.exists(output_path)

    # ------------------------------------------------------------------
    # Extract — 2D slice from 3D dataset
    # ------------------------------------------------------------------

    def test_extract_2d_slice(self, client: TestClient):
        """Extracting a 2D slice by providing a level coordinate returns a flat grid."""
        response = client.get(
            f"/datasets/{DATASET_NAME}/extract"
            f"?variable=Value&time=2026-01-01&level={LEVEL_START}"
        )
        assert response.status_code == 200
        data = response.json()
        assert "values" in data
        assert "longitudes" in data
        assert "latitudes" in data
        assert "heights" not in data  # 2D slice: no height in output
        assert len(data["values"]["Value"]) == LATS * LONS

    def test_extract_2d_missing_level_fails(self, client: TestClient):
        """Requesting 2D extraction (is_3d=false) without a level should fail."""
        response = client.get(
            f"/datasets/{DATASET_NAME}/extract?variable=Value&time=2026-01-01"
        )
        # Should return 400 or 422 because the level dimension is unsatisfied
        assert response.status_code in (400, 422)

    # ------------------------------------------------------------------
    # Extract — full 3D volume
    # ------------------------------------------------------------------

    def test_extract_3d_volume(self, client: TestClient):
        """3D extraction returns all levels × lat × lon points."""
        response = client.get(
            f"/datasets/{DATASET_NAME}/extract"
            f"?variable=Value&time=2026-01-01&is_3d=true"
        )
        assert response.status_code == 200
        data = response.json()
        assert "values" in data
        assert "longitudes" in data
        assert "latitudes" in data
        assert "heights" in data
        assert len(data["values"]["Value"]) == LONS * LATS * LEVELS

    def test_extract_3d_heights_present(self, client: TestClient):
        """Heights array in 3D output must have the same length as values."""
        response = client.get(
            f"/datasets/{DATASET_NAME}/extract"
            f"?variable=Value&time=2026-01-01&is_3d=true"
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["heights"]) == len(data["values"]["Value"])

    def test_extract_3d_z_bbox(self, client: TestClient):
        """3D extraction with z_min/z_max returns only levels inside the range."""
        z_min = LEVEL_START
        z_max = LEVEL_START + LEVEL_STEP  # only first 2 levels

        response = client.get(
            f"/datasets/{DATASET_NAME}/extract"
            f"?variable=Value&time=2026-01-01&is_3d=true"
            f"&z_min={z_min}&z_max={z_max}"
        )
        assert response.status_code == 200
        data = response.json()
        heights = data["heights"]
        assert all(z_min <= h <= z_max for h in heights)
        # 2 levels × LATS × LONS
        assert len(data["values"]["Value"]) == 2 * LATS * LONS

    def test_extract_3d_z_bbox_out_of_range(self, client: TestClient):
        """Z bounding box outside dataset extent should return an error."""
        response = client.get(
            f"/datasets/{DATASET_NAME}/extract"
            f"?variable=Value&time=2026-01-01&is_3d=true"
            f"&z_min=9999&z_max=99999"
        )
        assert response.status_code in (400, 404, 422)

    # ------------------------------------------------------------------
    # Extract — GeoJSON 3D
    # ------------------------------------------------------------------

    def test_extract_3d_geojson(self, client: TestClient):
        """GeoJSON 3D output embeds height as the third coordinate."""
        response = client.get(
            f"/datasets/{DATASET_NAME}/extract"
            f"?variable=Value&time=2026-01-01&is_3d=true&format=geojson"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "FeatureCollection"
        features = data["features"]
        assert len(features) > 0
        # Each feature must have a 3-element coordinate array [lon, lat, height]
        first_coords = features[0]["geometry"]["coordinates"]
        assert len(first_coords) == 3

    # ------------------------------------------------------------------
    # Mesh — 2D slice
    # ------------------------------------------------------------------

    def test_mesh_2d_slice(self, client: TestClient):
        """Mesh endpoint on a 3D dataset without is_3d returns a 2D triangulated surface."""
        response = client.get(f"/datasets/{DATASET_NAME}/mesh")
        assert response.status_code == 200
        data = response.json()
        assert "vertices" in data
        assert "indices" in data
        assert data["is_3d"] is False
        assert len(data["vertices"]) > 0
        assert len(data["indices"]) > 0

    # ------------------------------------------------------------------
    # Mesh — 3D volume
    # ------------------------------------------------------------------

    def test_mesh_3d_volume(self, client: TestClient):
        """Mesh endpoint with is_3d=true returns a volumetric tetrahedral mesh."""
        response = client.get(f"/datasets/{DATASET_NAME}/mesh?is_3d=true")
        assert response.status_code == 200
        data = response.json()
        assert "vertices" in data
        assert "indices" in data
        assert data["is_3d"] is True
        assert len(data["vertices"]) > 0
        assert len(data["indices"]) > 0

    def test_mesh_3d_volume_z_bbox(self, client: TestClient):
        """3D mesh with z_min/z_max uses only levels inside the range."""
        z_min = LEVEL_START
        z_max = LEVEL_START + LEVEL_STEP

        response = client.get(
            f"/datasets/{DATASET_NAME}/mesh?is_3d=true&z_min={z_min}&z_max={z_max}"
        )
        assert response.status_code == 200
        data = response.json()
        assert "vertices" in data
        assert data["is_3d"] is True
        # Vertices are flat [x, y, z, x, y, z, ...]
        # All z-components should lie within [z_min, z_max]
        vertices = data["vertices"]
        z_values = [vertices[i] for i in range(2, len(vertices), 3)]
        assert all(z_min <= z <= z_max for z in z_values)

    def test_mesh_3d_geojson(self, client: TestClient):
        """Mesh endpoint with is_3d=true and format=geojson returns a FeatureCollection with 3D coords."""
        response = client.get(
            f"/datasets/{DATASET_NAME}/mesh?is_3d=true&format=geojson"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "FeatureCollection"
        features = data["features"]
        assert len(features) > 0
        first_coords = features[0]["geometry"]["coordinates"]
        assert len(first_coords) == 3

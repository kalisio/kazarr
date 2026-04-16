"""
test_compare_grids.py — Performance benchmarks comparing regular vs rectilinear grids.
"""

import os
from time import perf_counter

import pytest
from fastapi.testclient import TestClient

import utils


LATS = 91
LONS = 137

REGULAR_DATASET = "regular_grid"
RECTILINEAR_DATASET = "rectilinear_grid"
TMP_FOLDER = os.environ.get("TEST_TMP_FOLDER", "test/tests_tmp")

# Performance thresholds (seconds) — generous to avoid CI flakiness while
# still detecting catastrophic regressions.
THRESHOLDS = {
    "extract_global": 10.0,
    "extract_mesh_no_tile": 15.0,
    "extract_mesh_tile": 20.0,
    "extract_mesh_tile_interp": 30.0,
    "extract_tile": 10.0,
    "extract_tile_mesh": 15.0,
    "extract_tile_mesh_tile": 20.0,
    "extract_tile_mesh_tile_interp": 30.0,
}

MESH_TILE_SIZE = 400
BOUNDING_BOX = "lon_min=2.8124999999999996&lon_max=4.21875&lat_min=43.068887774169625&lat_max=44.08758502824518"


def _timed_get(client: TestClient, url: str) -> tuple[float, dict]:
    """Run a GET request and return (duration_seconds, json_response)."""
    start = perf_counter()
    response = client.get(url)
    duration = perf_counter() - start
    assert response.status_code == 200, (
        f"Request failed ({response.status_code}): {url}"
    )
    return duration, response.json()


@pytest.mark.CompareRegularVsRectilinear
class TestRegularGridVsRectilinearGrid:
    @pytest.fixture(scope="class", autouse=True)
    def setup_datasets(self, convert):
        """Generate and convert both datasets with identical data (cosine distribution)."""
        common_precip_config = {
            "type": "float",
            "method": "cos",
            "periods": 12,
        }

        # --- Regular grid ---
        regular_description = {
            "Precipitation": {
                **common_precip_config,
                "dimensions": [
                    {
                        "name": "steps",
                        "type": "steps",
                        "size": 5,
                        "start": "2026-01-01 00:00:00",
                        "freq": "1h",
                    },
                    {
                        "name": "lat",
                        "type": "latitude",
                        "size": LATS,
                        "start": 42.1,
                        "step": 0.1,
                    },
                    {
                        "name": "lon",
                        "type": "longitude",
                        "size": LONS,
                        "start": -5.2,
                        "step": 0.1,
                    },
                ],
            }
        }
        utils.DatasetGenerator(description=regular_description).generate().save(
            REGULAR_DATASET, to_netcdf=True
        )
        convert(
            dataset_name=REGULAR_DATASET,
            input_path=os.path.join(TMP_FOLDER, f"{REGULAR_DATASET}.nc"),
            output_path=os.path.join(TMP_FOLDER, f"{REGULAR_DATASET}.zarr"),
            config={
                "variables": {"lon": "lon", "lat": "lat", "time": "steps"},
                "pipelines": {
                    "preprocess": [
                        "load_from_netcdf",
                        "unify_chunks",
                        "delta_time_to_datetime",
                        "save",
                    ]
                },
                "referenceTime": {
                    "variable": "reference_time",
                    "format": "%Y-%m-%d %H:%M:%S",
                },
                "version": 2,
            },
        )

        # --- Rectilinear grid (same data, curvilinear coordinates) ---
        rectilinear_description = {
            "Precipitation": {
                **common_precip_config,
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
        utils.DatasetGenerator(description=rectilinear_description).generate().save(
            RECTILINEAR_DATASET, to_netcdf=True
        )
        convert(
            dataset_name=RECTILINEAR_DATASET,
            input_path=os.path.join(TMP_FOLDER, f"{RECTILINEAR_DATASET}.nc"),
            output_path=os.path.join(TMP_FOLDER, f"{RECTILINEAR_DATASET}.zarr"),
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

        yield
        utils.cleanup_test_files()

    # ------------------------------------------------------------------
    # Global extract
    # ------------------------------------------------------------------

    def test_compare_global_extract(self, client: TestClient):
        """Both grids return valid results within the time threshold."""
        url = "?variable=Precipitation&time=2026-01-01"

        reg_duration, reg_data = _timed_get(
            client, f"/datasets/{REGULAR_DATASET}/extract{url}"
        )
        rec_duration, rec_data = _timed_get(
            client, f"/datasets/{RECTILINEAR_DATASET}/extract{url}"
        )

        print(
            f"\nGlobal extract — Regular: {reg_duration:.3f}s | Rectilinear: {rec_duration:.3f}s"
        )

        assert reg_duration < THRESHOLDS["extract_global"], (
            f"Regular grid extract too slow: {reg_duration:.2f}s"
        )
        assert rec_duration < THRESHOLDS["extract_global"], (
            f"Rectilinear grid extract too slow: {rec_duration:.2f}s"
        )

        # Both grids have the same data distribution — min should be ~0
        assert reg_data["bounds"]["min"] == pytest.approx(
            rec_data["bounds"]["min"], abs=1.0
        )

    def test_compare_global_extract_mesh(self, client: TestClient):
        """Mesh format: both grids respond in time and produce valid mesh output."""
        url = "?variable=Precipitation&time=2026-01-01&format=mesh"

        reg_duration, reg_data = _timed_get(
            client, f"/datasets/{REGULAR_DATASET}/extract{url}"
        )
        rec_duration, rec_data = _timed_get(
            client, f"/datasets/{RECTILINEAR_DATASET}/extract{url}"
        )

        print(
            f"\nGlobal mesh — Regular: {reg_duration:.3f}s | Rectilinear: {rec_duration:.3f}s"
        )

        assert reg_duration < THRESHOLDS["extract_mesh_no_tile"]
        assert rec_duration < THRESHOLDS["extract_mesh_no_tile"]

        for data in (reg_data, rec_data):
            assert "vertices" in data
            assert "indices" in data
            assert len(data["vertices"]) > 0

    def test_compare_global_extract_mesh_tiled(self, client: TestClient):
        """Mesh with tile size: both grids produce correct vertex/index counts."""
        url = f"?variable=Precipitation&time=2026-01-01&format=mesh&mesh_tile_size={MESH_TILE_SIZE}"

        reg_duration, reg_data = _timed_get(
            client, f"/datasets/{REGULAR_DATASET}/extract{url}"
        )
        rec_duration, rec_data = _timed_get(
            client, f"/datasets/{RECTILINEAR_DATASET}/extract{url}"
        )

        print(
            f"\nTiled mesh — Regular: {reg_duration:.3f}s | Rectilinear: {rec_duration:.3f}s"
        )

        assert reg_duration < THRESHOLDS["extract_mesh_tile"]
        assert rec_duration < THRESHOLDS["extract_mesh_tile"]

        expected_values = MESH_TILE_SIZE * MESH_TILE_SIZE
        for data in (reg_data, rec_data):
            assert len(data["values"]) == expected_values

    def test_compare_global_extract_mesh_interpolated_linear(self, client: TestClient):
        """Mesh with linear interpolation: both grids respond in time."""
        url = (
            f"?variable=Precipitation&time=2026-01-01&format=mesh"
            f"&mesh_tile_size={MESH_TILE_SIZE}&interp_spatial_method=linear"
        )

        reg_duration, _ = _timed_get(
            client, f"/datasets/{REGULAR_DATASET}/extract{url}"
        )
        rec_duration, _ = _timed_get(
            client, f"/datasets/{RECTILINEAR_DATASET}/extract{url}"
        )

        print(
            f"\nMesh (linear interp) — Regular: {reg_duration:.3f}s | Rectilinear: {rec_duration:.3f}s"
        )

        assert reg_duration < THRESHOLDS["extract_mesh_tile_interp"]
        assert rec_duration < THRESHOLDS["extract_mesh_tile_interp"]

    def test_compare_global_extract_mesh_interpolated_cubic(self, client: TestClient):
        """Mesh with cubic interpolation: both grids respond in time."""
        url = (
            f"?variable=Precipitation&time=2026-01-01&format=mesh"
            f"&mesh_tile_size={MESH_TILE_SIZE}&interp_spatial_method=cubic"
        )

        reg_duration, _ = _timed_get(
            client, f"/datasets/{REGULAR_DATASET}/extract{url}"
        )
        rec_duration, _ = _timed_get(
            client, f"/datasets/{RECTILINEAR_DATASET}/extract{url}"
        )

        print(
            f"\nMesh (cubic interp) — Regular: {reg_duration:.3f}s | Rectilinear: {rec_duration:.3f}s"
        )

        assert reg_duration < THRESHOLDS["extract_mesh_tile_interp"]
        assert rec_duration < THRESHOLDS["extract_mesh_tile_interp"]

    # ------------------------------------------------------------------
    # Tile (bounding box) extract
    # ------------------------------------------------------------------

    def test_compare_tile_extract(self, client: TestClient):
        """Tile extract: both grids return consistent bounds within the bbox."""
        url = f"?variable=Precipitation&time=2026-01-01&{BOUNDING_BOX}"

        reg_duration, reg_data = _timed_get(
            client, f"/datasets/{REGULAR_DATASET}/extract{url}"
        )
        rec_duration, rec_data = _timed_get(
            client, f"/datasets/{RECTILINEAR_DATASET}/extract{url}"
        )

        print(
            f"\nTile extract — Regular: {reg_duration:.3f}s | Rectilinear: {rec_duration:.3f}s"
        )

        assert reg_duration < THRESHOLDS["extract_tile"]
        assert rec_duration < THRESHOLDS["extract_tile"]

        # Both should return data in the same geographic area
        assert reg_data["bounds"]["min"] == pytest.approx(
            rec_data["bounds"]["min"], rel=0.1
        )
        assert reg_data["bounds"]["max"] == pytest.approx(
            rec_data["bounds"]["max"], rel=0.1
        )

    def test_compare_tile_extract_mesh(self, client: TestClient):
        """Tile mesh: both grids produce valid mesh structure."""
        url = f"?variable=Precipitation&time=2026-01-01&format=mesh&{BOUNDING_BOX}"

        reg_duration, reg_data = _timed_get(
            client, f"/datasets/{REGULAR_DATASET}/extract{url}"
        )
        rec_duration, rec_data = _timed_get(
            client, f"/datasets/{RECTILINEAR_DATASET}/extract{url}"
        )

        print(
            f"\nTile mesh — Regular: {reg_duration:.3f}s | Rectilinear: {rec_duration:.3f}s"
        )

        assert reg_duration < THRESHOLDS["extract_tile_mesh"]
        assert rec_duration < THRESHOLDS["extract_tile_mesh"]

        for data in (reg_data, rec_data):
            assert len(data["vertices"]) > 0
            assert len(data["indices"]) > 0

    def test_compare_tile_extract_mesh_tiled(self, client: TestClient):
        """Tile mesh with tile size: both grids return correct value counts."""
        url = (
            f"?variable=Precipitation&time=2026-01-01&format=mesh"
            f"&mesh_tile_size={MESH_TILE_SIZE}&{BOUNDING_BOX}"
        )

        reg_duration, reg_data = _timed_get(
            client, f"/datasets/{REGULAR_DATASET}/extract{url}"
        )
        rec_duration, rec_data = _timed_get(
            client, f"/datasets/{RECTILINEAR_DATASET}/extract{url}"
        )

        print(
            f"\nTile tiled mesh — Regular: {reg_duration:.3f}s | Rectilinear: {rec_duration:.3f}s"
        )

        assert reg_duration < THRESHOLDS["extract_tile_mesh_tile"]
        assert rec_duration < THRESHOLDS["extract_tile_mesh_tile"]

        expected = MESH_TILE_SIZE * MESH_TILE_SIZE
        assert len(reg_data["values"]) == expected
        assert len(rec_data["values"]) == expected

    def test_compare_tile_extract_mesh_interpolated_linear(self, client: TestClient):
        """Tile mesh with linear interpolation: both grids respond in time."""
        url = (
            f"?variable=Precipitation&time=2026-01-01&format=mesh"
            f"&mesh_tile_size={MESH_TILE_SIZE}&interp_spatial_method=linear&{BOUNDING_BOX}"
        )

        reg_duration, _ = _timed_get(
            client, f"/datasets/{REGULAR_DATASET}/extract{url}"
        )
        rec_duration, _ = _timed_get(
            client, f"/datasets/{RECTILINEAR_DATASET}/extract{url}"
        )

        print(
            f"\nTile mesh (linear) — Regular: {reg_duration:.3f}s | Rectilinear: {rec_duration:.3f}s"
        )

        assert reg_duration < THRESHOLDS["extract_tile_mesh_tile_interp"]
        assert rec_duration < THRESHOLDS["extract_tile_mesh_tile_interp"]

    def test_compare_tile_extract_mesh_interpolated_cubic(self, client: TestClient):
        """Tile mesh with cubic interpolation: both grids respond in time."""
        url = (
            f"?variable=Precipitation&time=2026-01-01&format=mesh"
            f"&mesh_tile_size={MESH_TILE_SIZE}&interp_spatial_method=cubic&{BOUNDING_BOX}"
        )

        reg_duration, _ = _timed_get(
            client, f"/datasets/{REGULAR_DATASET}/extract{url}"
        )
        rec_duration, _ = _timed_get(
            client, f"/datasets/{RECTILINEAR_DATASET}/extract{url}"
        )

        print(
            f"\nTile mesh (cubic) — Regular: {reg_duration:.3f}s | Rectilinear: {rec_duration:.3f}s"
        )

        assert reg_duration < THRESHOLDS["extract_tile_mesh_tile_interp"]
        assert rec_duration < THRESHOLDS["extract_tile_mesh_tile_interp"]

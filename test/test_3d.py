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

DATASET_REGULAR = "dataset_regular_3d"
TMP_FOLDER = os.environ.get("TEST_TMP_FOLDER", "test/tests_tmp")


@pytest.mark.Dataset3D
class TestRegularGrid3D:
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
        dataset.save(DATASET_REGULAR, to_netcdf=True)

        assert os.path.exists(os.path.join(TMP_FOLDER, f"{DATASET_REGULAR}.nc"))

    def test_convert_dataset(self, convert):
        """Convert the 3D NetCDF to Zarr with lon/lat/level/time variable mappings."""
        output_path = os.path.join(TMP_FOLDER, f"{DATASET_REGULAR}.zarr")
        convert(
            input_path=os.path.join(TMP_FOLDER, f"{DATASET_REGULAR}.nc"),
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
            f"/datasets/{DATASET_REGULAR}/extract"
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
            f"/datasets/{DATASET_REGULAR}/extract?variable=Value&time=2026-01-01"
        )
        # Should return 400 or 422 because the level dimension is unsatisfied
        assert response.status_code in (400, 422)

    # ------------------------------------------------------------------
    # Extract — full 3D volume
    # ------------------------------------------------------------------

    def test_extract_3d_volume(self, client: TestClient):
        """3D extraction returns all levels × lat × lon points."""
        response = client.get(
            f"/datasets/{DATASET_REGULAR}/extract"
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
            f"/datasets/{DATASET_REGULAR}/extract"
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
            f"/datasets/{DATASET_REGULAR}/extract"
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
            f"/datasets/{DATASET_REGULAR}/extract"
            f"?variable=Value&time=2026-01-01&is_3d=true"
            f"&z_min=9999&z_max=99999"
        )
        assert response.status_code in (400, 404, 422)

    def test_extract_cells_data_mapping(self, client: TestClient):
        """Extract endpoint with mesh_data_mapping=cells returns a list of cell values."""
        response = client.get(
            f"/datasets/{DATASET_REGULAR}/extract"
            f"?variable=Value&time=2026-01-01&is_3d=true&mesh_data_mapping=cells"
        )

        assert response.status_code == 200
        data = response.json()
        assert "values" in data
        assert "Value" in data["values"]
        assert isinstance(data["values"]["Value"], list)
        assert len(data["values"]["Value"]) > 0
        assert all(isinstance(v, (int, float)) for v in data["values"]["Value"])
        assert len(data["longitudes"]) == (LONS + 1) * (LATS + 1) * (LEVELS + 1)

    # ------------------------------------------------------------------
    # Extract — GeoJSON 3D
    # ------------------------------------------------------------------

    def test_extract_3d_geojson(self, client: TestClient):
        """GeoJSON 3D output embeds height as the third coordinate."""
        response = client.get(
            f"/datasets/{DATASET_REGULAR}/extract"
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
        response = client.get(f"/datasets/{DATASET_REGULAR}/mesh")
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
        response = client.get(f"/datasets/{DATASET_REGULAR}/mesh?is_3d=true")
        assert response.status_code == 200
        data = response.json()
        assert "vertices" in data
        assert "indices" in data
        assert data["is_3d"] is True
        assert len(data["vertices"]) > 0
        assert len(data["indices"]) > 0

    def test_mesh_3d_geojson(self, client: TestClient):
        """Mesh endpoint with is_3d=true and format=geojson returns a FeatureCollection with 3D coords."""
        response = client.get(
            f"/datasets/{DATASET_REGULAR}/mesh?is_3d=true&format=geojson"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "FeatureCollection"
        features = data["features"]
        assert len(features) > 0
        first_coords = features[0]["geometry"]["coordinates"]
        assert len(first_coords) == 3

    # ------------------------------------------------------------------
    # Dataset metadata
    # ------------------------------------------------------------------

    def test_dataset_metadata_vertical_axis(self, client: TestClient):
        """Dataset metadata exposes Humidity variable and bounding box."""
        response = client.get(f"/datasets/{DATASET_REGULAR}/metadata")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == DATASET_REGULAR
        assert "vertical_axis" in data
        assert "level" in data["vertical_axis"]
        assert data["vertical_axis"]["level"]["min"] == LEVEL_START
        assert (
            data["vertical_axis"]["level"]["max"]
            == LEVEL_START + (LEVELS - 1) * LEVEL_STEP
        )


DATASET_NON_REGULAR = "dataset_3d_non_regular"
NR_LATS = 11
NR_LONS = 39
NR_LEVELS = 3
NR_LEVEL_START = 0
NR_LEVEL_STEP = 100


@pytest.mark.Dataset3D
class TestNonRegularGrid3D:
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
                    {"name": "DimK", "size": "lat.DimK"},
                    {"name": "DimJ", "size": "lat.DimJ"},
                    {"name": "DimI", "size": "lat.DimI"},
                ],
            },
            "lat": {
                "type": "load",
                "sample": "3d_grid.nc",
                "variable": "CoordY0",
                "dimensions": ["DimK", "DimJ", "DimI"],
            },
            "lon": {
                "type": "load",
                "sample": "3d_grid.nc",
                "variable": "CoordX0",
                "dimensions": ["DimK", "DimJ", "DimI"],
            },
            "level": {
                "type": "load",
                "sample": "3d_grid.nc",
                "variable": "CoordZ0",
                "dimensions": ["DimK", "DimJ", "DimI"],
            },
        }
        dataset = utils.DatasetGenerator(description=description).generate()
        dataset.save(DATASET_NON_REGULAR, to_netcdf=True)

        assert os.path.exists(os.path.join(TMP_FOLDER, f"{DATASET_NON_REGULAR}.nc"))

    def test_convert_dataset(self, convert):
        """Convert the 3D NetCDF to Zarr with lon/lat/height/time variable mappings."""
        output_path = os.path.join(TMP_FOLDER, f"{DATASET_NON_REGULAR}.zarr")
        convert(
            input_path=os.path.join(TMP_FOLDER, f"{DATASET_NON_REGULAR}.nc"),
            output_path=output_path,
            config={
                "variables": {
                    "lon": "lon",
                    "lat": "lat",
                    "time": "time",
                    "height": "level",
                },
                "reprojection": {"fromCrs": "EPSG:32631", "toCrs": "EPSG:4326"},
                "pipelines": {
                    "preprocess": [
                        "load_from_netcdf",
                        "reproject_coordinates",
                        "unify_chunks",
                        "save",
                    ]
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
            f"/datasets/{DATASET_NON_REGULAR}/extract"
            f"?variable=Value&time=2026-01-01&DimK=1"
        )
        assert response.status_code == 200
        data = response.json()
        assert "values" in data
        assert "longitudes" in data
        assert "latitudes" in data
        assert "heights" not in data  # 2D slice: no height in output
        assert len(data["values"]["Value"]) == NR_LONS * NR_LATS

    def test_extract_2d_missing_level_fails(self, client: TestClient):
        """Requesting 2D extraction (is_3d=false) without a level should fail."""
        client.get(
            f"/datasets/{DATASET_NON_REGULAR}/extract?variable=Value&time=2026-01-01"
        )
        pytest.skip(
            "With irregular grids, extract allow latitude and longitude coordinates not to be fixed, and so their dimensions are not fixed either. But, as height share the same dimensions as lat/lon, it is currently not possible to check if the level dimension is fixed or not. This test should be re-enabled once we have a better way to check if the level dimension is satisfied or not."
        )

    # ------------------------------------------------------------------
    # Extract — full 3D volume
    # ------------------------------------------------------------------

    def test_extract_3d_volume(self, client: TestClient):
        """3D extraction returns all levels × lat × lon points."""
        response = client.get(
            f"/datasets/{DATASET_NON_REGULAR}/extract"
            f"?variable=Value&time=2026-01-01&is_3d=true"
        )
        assert response.status_code == 200
        data = response.json()
        assert "values" in data
        assert "longitudes" in data
        assert "latitudes" in data
        assert "heights" in data
        assert len(data["values"]["Value"]) == NR_LEVELS * NR_LONS * NR_LATS

    def test_extract_3d_heights_present(self, client: TestClient):
        """Heights array in 3D output must have the same length as values."""
        response = client.get(
            f"/datasets/{DATASET_NON_REGULAR}/extract"
            f"?variable=Value&time=2026-01-01&is_3d=true"
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["heights"]) == len(data["values"]["Value"])

    def test_extract_3d_z_bbox(self, client: TestClient):
        """3D extraction with z_min/z_max returns only levels inside the range."""
        z = 100  # only one level

        response = client.get(
            f"/datasets/{DATASET_NON_REGULAR}/extract"
            f"?variable=Value&time=2026-01-01&is_3d=true"
            f"&z_min={z}&z_max={z}"
        )
        assert response.status_code == 200
        data = response.json()
        heights = data["heights"]
        assert all(z == h for h in heights)
        # 1 level × LATS × LONS
        assert len(data["values"]["Value"]) == 1 * NR_LATS * NR_LONS

    def test_extract_3d_z_bbox_out_of_range(self, client: TestClient):
        """Z bounding box outside dataset extent should return an error."""
        response = client.get(
            f"/datasets/{DATASET_NON_REGULAR}/extract"
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
            f"/datasets/{DATASET_NON_REGULAR}/extract"
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
        response = client.get(f"/datasets/{DATASET_NON_REGULAR}/mesh")
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
        response = client.get(f"/datasets/{DATASET_NON_REGULAR}/mesh?is_3d=true")
        assert response.status_code == 200
        data = response.json()
        assert "vertices" in data
        assert "indices" in data
        assert data["is_3d"] is True
        assert len(data["vertices"]) > 0
        assert len(data["indices"]) > 0

    def test_mesh_3d_geojson(self, client: TestClient):
        """Mesh endpoint with is_3d=true and format=geojson returns a FeatureCollection with 3D coords."""
        response = client.get(
            f"/datasets/{DATASET_NON_REGULAR}/mesh?is_3d=true&format=geojson"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "FeatureCollection"
        features = data["features"]
        assert len(features) > 0
        first_coords = features[0]["geometry"]["coordinates"]
        assert len(first_coords) == 3

    # ------------------------------------------------------------------
    # Dataset metadata
    # ------------------------------------------------------------------

    def test_dataset_metadata_vertical_axis(self, client: TestClient):
        """Dataset metadata exposes Humidity variable and bounding box."""
        response = client.get(f"/datasets/{DATASET_NON_REGULAR}/metadata")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == DATASET_NON_REGULAR
        assert "vertical_axis" in data
        assert "level" in data["vertical_axis"]
        assert data["vertical_axis"]["level"]["min"] == NR_LEVEL_START
        assert (
            data["vertical_axis"]["level"]["max"]
            == NR_LEVEL_START + (NR_LEVELS - 1) * NR_LEVEL_STEP
        )


DATASET_MULTILEVEL = "dataset_3d_multilevel"


@pytest.mark.Dataset3D
class TestDataset3DMultiLevel:
    @pytest.fixture(scope="class", autouse=True)
    def cleanup(self):
        yield
        utils.cleanup_test_files()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def test_generate_dataset(self):
        description = {
            "temperature": {
                "type": "float",
                "method": "linear",
                "attributes": {"typeOfLevel": "isobaricInhPa"},
                "dimensions": [
                    {
                        "name": "time",
                        "type": "date",
                        "size": TIMES,
                        "start": "2026-01-01",
                        "freq": "D",
                    },
                    {
                        "name": "isobaricInhPa",
                        "type": "float",
                        "size": LEVELS,
                        "start": 100.0,
                        "step": 100.0,
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
            "wind": {
                "type": "float",
                "method": "linear",
                "attributes": {"typeOfLevel": "heightAboveGround"},
                "dimensions": [
                    {
                        "name": "time",
                        "size": TIMES,
                    },
                    {
                        "name": "heightAboveGround",
                        "type": "float",
                        "size": 2,
                        "start": 2.0,
                        "step": 8.0,
                    },
                    {
                        "name": "lat",
                        "size": LATS,
                    },
                    {
                        "name": "lon",
                        "size": LONS,
                    },
                ],
            },
        }
        dataset = utils.DatasetGenerator(description=description).generate()
        dataset.save(DATASET_MULTILEVEL, to_netcdf=True)
        assert os.path.exists(os.path.join(TMP_FOLDER, f"{DATASET_MULTILEVEL}.nc"))

    def test_convert_dataset(self, convert):
        output_path = os.path.join(TMP_FOLDER, f"{DATASET_MULTILEVEL}.zarr")
        convert(
            input_path=os.path.join(TMP_FOLDER, f"{DATASET_MULTILEVEL}.nc"),
            output_path=output_path,
            config={
                "variables": {
                    "lon": "lon",
                    "lat": "lat",
                    "time": "time",
                    "height": "ATTRS.typeOfLevel",
                },
                "pipelines": {
                    "preprocess": ["load_from_netcdf", "unify_chunks", "save"]
                },
                "version": 2,
            },
        )
        assert os.path.exists(output_path)

    # ------------------------------------------------------------------
    # Mesh — 3D volume
    # ------------------------------------------------------------------

    def test_mesh_3d_variable_specific(self, client: TestClient):
        response = client.get(
            f"/datasets/{DATASET_MULTILEVEL}/mesh?is_3d=true&variable=wind"
        )
        assert response.status_code == 200
        data = response.json()
        assert "vertices" in data
        assert data["is_3d"] is True

    def test_mesh_3d_no_variable(self, client: TestClient):
        response = client.get(f"/datasets/{DATASET_MULTILEVEL}/mesh?is_3d=true")
        # Should fail with NO_HEIGHT_VARIABLE because the dataset has multiple height types
        assert response.status_code in (400, 422)

    def test_mesh_3d_height_variable_specific(self, client: TestClient):
        response = client.get(
            f"/datasets/{DATASET_MULTILEVEL}/mesh?is_3d=true&height_variable=heightAboveGround"
        )
        assert response.status_code == 200
        data = response.json()
        assert "vertices" in data
        assert data["is_3d"] is True

    # ------------------------------------------------------------------
    # Probe
    # ------------------------------------------------------------------

    def test_probe_multilevel(self, client: TestClient):
        response = client.post(
            f"/datasets/{DATASET_MULTILEVEL}/probes?time=2026-01-01&variables=temperature&variables=wind",
            json={"points": [{"lon": LON_START, "lat": LAT_START}]},
        )
        assert response.status_code == 400

    def test_probe_multilevel_geojson(self, client: TestClient):
        response = client.post(
            f"/datasets/{DATASET_MULTILEVEL}/probes?time=2026-01-01&variables=temperature&variables=wind",
            json={
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [LON_START, LAT_START],
                        },
                    }
                ],
            },
        )
        assert response.status_code == 400

    def test_probe(self, client: TestClient):
        response = client.post(
            f"/datasets/{DATASET_MULTILEVEL}/probes?time=2026-01-01&variables=temperature&isobaricInhPa=100",
            json={"points": [{"lon": LON_START, "lat": LAT_START}]},
        )
        assert response.status_code == 200
        data = response.json()
        assert "times" in data
        assert "variables" in data
        assert "temperature" in data["variables"]
        values = data["values"]["temperature"]
        assert isinstance(values, list)
        assert len(values) == 1  # 1 probe points
        assert all(
            isinstance(v, list) and len(v) == 1 for v in values
        )  # 1 time steps requested

    def test_probe_geojson(self, client: TestClient):
        response = client.post(
            f"/datasets/{DATASET_MULTILEVEL}/probes?time=2026-01-01&variables=temperature&isobaricInhPa=100",
            json={
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [LON_START, LAT_START],
                        },
                    }
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "times" in data
        assert "variables" in data
        assert "temperature" in data["variables"]
        values = data["values"]["temperature"]
        assert isinstance(values, list)
        assert len(values) == 1  # 1 probe points
        assert all(
            isinstance(v, list) and len(v) == 1 for v in values
        )  # 1 time steps requested

    # ------------------------------------------------------------------
    # Extract
    # ------------------------------------------------------------------

    def test_extract_with_spatial_dimension_fixed(self, client: TestClient):
        response = client.get(
            f"/datasets/{DATASET_MULTILEVEL}/extract?variable=wind&time=2026-01-01&is_3d=true&heightAboveGround=7.0"
        )
        assert response.status_code == 200
        data = response.json()
        assert "values" in data
        assert "longitudes" in data
        assert "latitudes" in data
        assert "heights" in data

    def test_extract(self, client: TestClient):
        response = client.get(
            f"/datasets/{DATASET_MULTILEVEL}/extract?variable=wind&time=2026-01-01&is_3d=true"
        )
        assert response.status_code == 200
        data = response.json()
        assert "values" in data
        assert "longitudes" in data
        assert "latitudes" in data
        assert "heights" in data

    # ------------------------------------------------------------------
    # Dataset metadata
    # ------------------------------------------------------------------

    def test_dataset_metadata_vertical_axis(self, client: TestClient):
        """Dataset metadata exposes Humidity variable and bounding box."""
        response = client.get(f"/datasets/{DATASET_MULTILEVEL}/metadata")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == DATASET_MULTILEVEL
        assert "vertical_axis" in data
        assert "isobaricInhPa" in data["vertical_axis"]
        assert data["vertical_axis"]["isobaricInhPa"]["min"] == 100.0
        assert (
            data["vertical_axis"]["isobaricInhPa"]["max"]
            == 100.0 + (LEVELS - 1) * 100.0
        )
        assert "heightAboveGround" in data["vertical_axis"]
        assert data["vertical_axis"]["heightAboveGround"]["min"] == 2.0
        assert data["vertical_axis"]["heightAboveGround"]["max"] == 2.0 + 8.0

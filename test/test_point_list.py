"""
test_point_list.py — Tests for point list datasets.

A "point list" dataset has 1D lat/lon arrays sharing the same single dimension,
e.g. dataset[lat].dims == dataset[lon].dims == ("DimN",). This represents a
collection of individual stations or observation points (no grid structure).
"""

import os
import pytest
from fastapi.testclient import TestClient

import numpy as np

import utils


# ---------------------------------------------------------------------------
# Dataset parameters
# ---------------------------------------------------------------------------
DATASET_NAME = "point_list"
TMP_FOLDER = os.environ.get("TEST_TMP_FOLDER", "test/tests_tmp")

# Station coordinates (10 French cities)
STATION_NAMES = [
    b"Paris",
    b"Marseille",
    b"Lyon",
    b"Toulouse",
    b"Nice",
    b"Nantes",
    b"Montpellier",
    b"Strasbourg",
    b"Bordeaux",
    b"Lille",
]
STATION_LATS = [
    48.856614,
    43.296482,
    45.764043,
    43.604652,
    43.710173,
    47.218371,
    43.610769,
    48.573405,
    44.837789,
    50.629250,
]
STATION_LONS = [
    2.352222,
    5.369780,
    4.835659,
    1.444209,
    7.261953,
    -1.553621,
    3.876716,
    7.752111,
    -0.579180,
    3.057256,
]
N_STATIONS = len(STATION_LATS)
N_TIMES = 5


@pytest.mark.PointList
class TestPointList:
    @pytest.fixture(scope="class", autouse=True)
    def cleanup(self):
        yield
        utils.cleanup_test_files()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def test_generate_dataset(self):
        """Generate a synthetic point-list dataset (stations across France)."""
        description = {
            "Humidity": {
                "type": "float",
                "method": "cos",
                "bounds": {"min": 0, "max": 100},
                "dimensions": [
                    {
                        "name": "time",
                        "type": "date",
                        "size": N_TIMES,
                        "start": "2026-01-01",
                        "freq": "1h",
                    },
                    {"name": "DimN", "size": "lat.DimN"},
                ],
            },
            "lat": {
                "type": "array",
                "values": STATION_LATS,
                "dimensions": ["DimN"],
            },
            "lon": {
                "type": "array",
                "values": STATION_LONS,
                "dimensions": ["DimN"],
            },
            "height": {
                "type": "array",
                "values": [0.0] * N_STATIONS,
                "dimensions": ["DimN"],
            },
            "name": {
                "type": "array",
                "values": STATION_NAMES,
                "dimensions": ["DimN"],
            }
        }
        dataset = utils.DatasetGenerator(description=description).generate()
        dataset.save(DATASET_NAME, to_netcdf=True)

        assert os.path.exists(os.path.join(TMP_FOLDER, f"{DATASET_NAME}.nc"))

    def test_convert_dataset(self, convert):
        """Convert the NetCDF to Zarr format."""
        output_path = os.path.join(TMP_FOLDER, f"{DATASET_NAME}.zarr")
        convert(
            dataset_name=DATASET_NAME,
            input_path=os.path.join(TMP_FOLDER, f"{DATASET_NAME}.nc"),
            output_path=output_path,
            config={
                "variables": {"lon": "lon", "lat": "lat", "height": "height", "time": "time"},
                "assignCoords": { "name": "DimN" },
                "pipelines": {
                    "preprocess": ["load_from_netcdf", "assign_coords", "unify_chunks", "save"]
                },
                "version": 2,
            },
        )

        assert os.path.exists(output_path)

    # ------------------------------------------------------------------
    # Extract — raw format
    # ------------------------------------------------------------------

    def test_extract(self, client: TestClient):
        """Full extraction returns all N_STATIONS points."""
        response = client.get(
            f"/datasets/{DATASET_NAME}/extract?variable=Humidity&time=2026-01-01"
        )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        non_null = [v for v in data["data"]["values"] if v is not None]
        assert len(non_null) == N_STATIONS
        assert data["bounds"]["min"] >= 0
        assert data["bounds"]["max"] <= 100

    def test_extract_geojson(self, client: TestClient):
        """GeoJSON extraction returns one Feature per station."""
        response = client.get(
            f"/datasets/{DATASET_NAME}/extract?variable=Humidity&time=2026-01-01&format=geojson"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) == N_STATIONS

        for feature in data["features"]:
            assert feature["geometry"]["type"] == "Point"
            coords = feature["geometry"]["coordinates"]
            assert len(coords) == 2
            assert -180 <= coords[0] <= 180
            assert -90 <= coords[1] <= 90

    def test_extract_with_bbox(self, client: TestClient):
        """Bounding box extraction returns only stations inside the box."""
        # Box covering roughly southern France (should include Marseille, Nice, etc.)
        bbox = "lon_min=0.0&lon_max=8.0&lat_min=43.0&lat_max=44.5"
        response = client.get(
            f"/datasets/{DATASET_NAME}/extract?variable=Humidity&time=2026-01-01&{bbox}"
        )

        assert response.status_code == 200
        data = response.json()
        non_null_count = sum(1 for v in data["data"]["values"] if v is not None)
        assert non_null_count > 0
        assert non_null_count < N_STATIONS  # Not all stations are in the box

    def test_extract_bbox_no_stations(self, client: TestClient):
        """Bounding box with no stations returns 400."""
        # Middle of the ocean — no stations there
        bbox = "lon_min=-100.0&lon_max=-90.0&lat_min=0.0&lat_max=10.0"
        response = client.get(
            f"/datasets/{DATASET_NAME}/extract?variable=Humidity&time=2026-01-01&{bbox}"
        )

        assert response.status_code == 400

    def test_extract_time_interpolation(self, client: TestClient):
        """Time interpolation at midpoint returns values between two time steps."""
        r1 = client.get(
            f"/datasets/{DATASET_NAME}/extract?variable=Humidity&time=2026-01-01T01:00:00"
        )
        r2 = client.get(
            f"/datasets/{DATASET_NAME}/extract?variable=Humidity&time=2026-01-01T02:00:00"
        )
        r_interp = client.get(
            f"/datasets/{DATASET_NAME}/extract"
            "?variable=Humidity&time=2026-01-01T01:30:00&interp_time=true&interp_vars_method=linear"
        )

        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r_interp.status_code == 200

        b1 = r1.json()["data"]["values"]
        b2 = r2.json()["data"]["values"]
        b_interp = r_interp.json()["data"]["values"]

        assert b_interp[0] == pytest.approx((b1[0] + b2[0]) / 2, rel=0.01)
        assert b_interp[-1] == pytest.approx((b1[-1] + b2[-1]) / 2, rel=0.01)

    def test_extract_point_name(self, client: TestClient):
        """Extracted GeoJSON features include station names as properties."""
        time_index = 1
        city_index = 9  # Lille
        response = client.get(
            f"/datasets/{DATASET_NAME}/extract?variable=Humidity&time=2026-01-01T0{time_index}:00:00&name={STATION_NAMES[city_index].decode()}"
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]["values"]) == 1
        flat_data = np.linspace(0, 2 * np.pi, len(STATION_NAMES) * N_TIMES)
        phi = flat_data[time_index * len(STATION_NAMES) + city_index]
        lille_value = ((np.cos(phi) + 1) / 2) * 100
        assert data["data"]["values"][0] == pytest.approx(lille_value, abs=0.1)


    # ------------------------------------------------------------------
    # Probe
    # ------------------------------------------------------------------

    def test_probe_nearest(self, client: TestClient):
        """Nearest-neighbor probe at a station location returns its Humidity value."""
        # Probe very close to Paris
        response = client.get(
            f"/datasets/{DATASET_NAME}/probe?variables=Humidity&lat=48.856&lon=2.352&DimN=0"
        )

        assert response.status_code == 200
        data = response.json()
        assert "variables" in data
        assert "Humidity" in data["variables"]
        values = data["variables"]["Humidity"]["values"]
        assert isinstance(values, list)
        assert len(values) > 0
        # All time steps should have valid humidity values
        for v in values:
            assert isinstance(v, (int, float))
            assert 0 <= v <= 100

    def test_probe_time_at_station(self, client: TestClient):
        """Probe at a specific time returns a single value."""
        response = client.get(
            f"/datasets/{DATASET_NAME}/probe?variables=Humidity&lat=48.856&lon=2.352"
            "&time=2026-01-01T01:00:00"
        )

        assert response.status_code == 200
        data = response.json()
        values = data["variables"]["Humidity"]["values"]
        assert (isinstance(values, list) and len(values) == 1) or isinstance(values, (int, float))

    def test_probe_with_height(self, client: TestClient):
        """Probe with height dimension falls back to nearest station."""
        response = client.get(
            f"/datasets/{DATASET_NAME}/probe?variables=Humidity&lat=48.856&lon=2.352&height=0.0"
        )

        assert response.status_code == 200
        data = response.json()
        assert "variables" in data
        assert "Humidity" in data["variables"]

    def test_probe_time_interpolation(self, client: TestClient):
        """Probe with time interpolation returns midpoint value."""
        r1 = client.get(
            f"/datasets/{DATASET_NAME}/probe?variables=Humidity&lat=48.856&lon=2.352"
            "&time=2026-01-01T01:00:00"
        )
        r2 = client.get(
            f"/datasets/{DATASET_NAME}/probe?variables=Humidity&lat=48.856&lon=2.352"
            "&time=2026-01-01T02:00:00"
        )
        r_interp = client.get(
            f"/datasets/{DATASET_NAME}/probe?variables=Humidity&lat=48.856&lon=2.352"
            "&time=2026-01-01T01:30:00&interp_time=true&interp_vars_method=linear"
        )

        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r_interp.status_code == 200

        def scalar(values):
            return values[0] if isinstance(values, list) else values

        v1 = scalar(r1.json()["variables"]["Humidity"]["values"])
        v2 = scalar(r2.json()["variables"]["Humidity"]["values"])
        v_interp = scalar(r_interp.json()["variables"]["Humidity"]["values"])

        assert v_interp == pytest.approx((v1 + v2) / 2, rel=0.01)

    # ------------------------------------------------------------------
    # Dataset metadata
    # ------------------------------------------------------------------

    def test_dataset_metadata(self, client: TestClient):
        """Dataset metadata exposes Humidity variable and bounding box."""
        response = client.get(f"/datasets/{DATASET_NAME}/metadata")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == DATASET_NAME
        assert "Humidity" in data["variables"]
        assert data["bounding_box"] is not None
        # Bounding box should span the station coordinates
        bb = data["bounding_box"]
        assert bb["lon"]["min"] == pytest.approx(min(STATION_LONS), abs=0.1)
        assert bb["lon"]["max"] == pytest.approx(max(STATION_LONS), abs=0.1)
        assert bb["lat"]["min"] == pytest.approx(min(STATION_LATS), abs=0.1)
        assert bb["lat"]["max"] == pytest.approx(max(STATION_LATS), abs=0.1)

    # ------------------------------------------------------------------
    # Select
    # ------------------------------------------------------------------

    def test_select(self, client: TestClient):
        """Free selection returns a list of time-series values."""
        response = client.get(
            f"/datasets/{DATASET_NAME}/select?variable=Humidity&time=2026-01-01"
        )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert isinstance(data["data"], (list, float, int))

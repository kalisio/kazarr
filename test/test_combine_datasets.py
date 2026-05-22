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
DATASET_NAME_RADIAL = "radial_grid_combine"
TMP_FOLDER = os.environ.get("TEST_TMP_FOLDER", "test/tests_tmp")


@pytest.mark.CombineDatasets
class TestCombineRadialGrid:
    @pytest.fixture(scope="class", autouse=True)
    def cleanup(self):
        yield
        utils.cleanup_test_files()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def test_generate_dataset_v1(self):
        """Generate a synthetic point-list dataset (stations across France)."""
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
            "Temperature": {
                "type": "float",
                "method": "cos",
                "bounds": {"min": -20, "max": 40},
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
        dataset.save(f"{DATASET_NAME_RADIAL}_v1", to_netcdf=True)

        assert os.path.exists(os.path.join(TMP_FOLDER, f"{DATASET_NAME_RADIAL}_v1.nc"))

    def test_convert_dataset_v1(self, convert):
        """Convert the NetCDF to Zarr format."""
        output_path = os.path.join(TMP_FOLDER, f"{DATASET_NAME_RADIAL}_v1.zarr")
        convert(
            input_path=os.path.join(TMP_FOLDER, f"{DATASET_NAME_RADIAL}_v1.nc"),
            output_path=output_path,
            config={
                "variables": {
                    "lon": "lon",
                    "lat": "lat",
                    "height": "height",
                    "time": "time",
                },
                "pipelines": {
                    "preprocess": [
                        "load_from_netcdf",
                        "unify_chunks",
                        "save",
                    ]
                },
                "version": 2,
            },
        )

        assert os.path.exists(output_path)

    def test_generate_dataset_v2(self):
        """Generate a synthetic point-list dataset (stations across France)."""
        description = {
            "WindSpeed": {
                "type": "float",
                "method": "sin",
                "bounds": {"min": 150, "max": 300},
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
            "Precipitation": {
                "type": "float",
                "method": "cos",
                "bounds": {"min": 50, "max": 80},
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
        dataset.save(f"{DATASET_NAME_RADIAL}_v2", to_netcdf=True)

        assert os.path.exists(os.path.join(TMP_FOLDER, f"{DATASET_NAME_RADIAL}_v2.nc"))

    def test_convert_dataset_v2(self, convert):
        """Convert the NetCDF to Zarr format."""
        output_path = os.path.join(TMP_FOLDER, f"{DATASET_NAME_RADIAL}_v2.zarr")
        convert(
            input_path=os.path.join(TMP_FOLDER, f"{DATASET_NAME_RADIAL}_v2.nc"),
            output_path=output_path,
            config={
                "variables": {
                    "lon": "lon",
                    "lat": "lat",
                    "height": "height",
                    "time": "time",
                },
                "pipelines": {
                    "preprocess": [
                        "load_from_netcdf",
                        "unify_chunks",
                        "save",
                    ]
                },
                "version": 2,
            },
        )

        assert os.path.exists(output_path)

    def test_combine_datasets(self, convert):
        """Combine the two Zarr datasets into one."""
        output_path = os.path.join(TMP_FOLDER, f"{DATASET_NAME_RADIAL}_combined.zarr")
        convert(
            input_path=os.path.join(TMP_FOLDER, f"{DATASET_NAME_RADIAL}_v1.zarr"),
            output_path=output_path,
            config={
                "variables": {
                    "lon": "lon",
                    "lat": "lat",
                    "height": "height",
                    "time": "time",
                },
                "pipelines": {
                    "preprocess": [
                        "load_from_zarr",
                        {
                            "type": "process",
                            "name": "load_from_zarr",
                            "params": {
                                "store_as_secondary": True,
                                "load_path": os.path.join(
                                    TMP_FOLDER, f"{DATASET_NAME_RADIAL}_v2.zarr"
                                ),
                            },
                        },
                        {
                            "type": "process",
                            "name": "combine_at_time",
                            "params": {"combine_time": "2026-01-03"},
                        },
                        "unify_chunks",
                        "save",
                    ]
                },
                "version": 2,
            },
        )

        assert os.path.exists(output_path)

    def test_combined_dataset_variables(self, client: TestClient):
        """Check that the combined dataset contains the expected variables and data."""
        response = client.get(
            f"/datasets/{DATASET_NAME_RADIAL}_combined/metadata"
        )

        assert response.status_code == 200
        metadata = response.json()
        expected_variables = {"lon", "lat", "height", "WindSpeed", "Precipitation", "Temperature"}
        assert set(metadata["variables"].keys()) == expected_variables

    def test_combined_dataset_common_variable(self, client: TestClient):

        response = client.get(
            f"/datasets/{DATASET_NAME_RADIAL}_combined/select?variable=WindSpeed"
        )

        assert response.status_code == 200
        data = response.json()
        np_arr = np.array(data["data"])
        first_part = np_arr[:3]  # First 2 time steps should be from dataset v1 (0-150)
        second_part = np_arr[3:]  # Last 3 time steps should be from
        assert first_part.max() <= 150
        assert second_part.min() >= 150

    def test_combined_dataset_removed_variable(self, client: TestClient):

        response = client.get(
            f"/datasets/{DATASET_NAME_RADIAL}_combined/select?variable=Temperature"
        )

        assert response.status_code == 200
        data = response.json()
        np_arr = np.array(data["data"])
        first_part = np_arr[:3]  # First 2 time steps should be from dataset v1 (0-150)
        second_part = np_arr[3:]  # Last 3 time steps should be from
        assert first_part.min() >= -20
        assert first_part.max() <= 40
        assert np.equal(second_part, None).all() # Should be filled with None since Temperature is missing in dataset v2

    def test_combined_dataset_added_variable(self, client: TestClient):

        response = client.get(
            f"/datasets/{DATASET_NAME_RADIAL}_combined/select?variable=Precipitation"
        )

        assert response.status_code == 200
        data = response.json()
        np_arr = np.array(data["data"])
        first_part = np_arr[:3]  # First 2 time steps should be from dataset v1 (0-150)
        second_part = np_arr[3:]  # Last 3 time steps should be from
        assert np.equal(
            first_part, None
        ).all()  # Should be filled with None since Precipitation is missing in dataset v1
        assert second_part.min() >= 50
        assert second_part.max() <= 80



DATASET_NAME_POINT_LIST = "point_list_combine"
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
N_REDUCED_AMOUNT = 3


@pytest.mark.CombineDatasets
class TestPointList:
    @pytest.fixture(scope="class", autouse=True)
    def cleanup(self):
        yield
        utils.cleanup_test_files()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def test_generate_dataset_v1(self):
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
                "values": STATION_LATS[:-N_REDUCED_AMOUNT],
                "dimensions": ["DimN"],
            },
            "lon": {
                "type": "array",
                "values": STATION_LONS[:-N_REDUCED_AMOUNT],
                "dimensions": ["DimN"],
            },
            "height": {
                "type": "array",
                "values": [0.0] * (N_STATIONS - N_REDUCED_AMOUNT),
                "dimensions": ["DimN"],
            },
            "name": {
                "type": "array",
                "values": STATION_NAMES[:-N_REDUCED_AMOUNT],
                "dimensions": ["DimN"],
            },
        }
        dataset = utils.DatasetGenerator(description=description).generate()
        dataset.save(f"{DATASET_NAME_POINT_LIST}_v1", to_netcdf=True)

        assert os.path.exists(os.path.join(TMP_FOLDER, f"{DATASET_NAME_POINT_LIST}_v1.nc"))

    def test_convert_dataset_v1(self, convert):
        """Convert the NetCDF to Zarr format."""
        output_path = os.path.join(TMP_FOLDER, f"{DATASET_NAME_POINT_LIST}_v1.zarr")
        convert(
            input_path=os.path.join(TMP_FOLDER, f"{DATASET_NAME_POINT_LIST}_v1.nc"),
            output_path=output_path,
            config={
                "variables": {
                    "lon": "lon",
                    "lat": "lat",
                    "height": "height",
                    "time": "time",
                },
                "assignCoords": {"name": "DimN"},
                "pipelines": {
                    "preprocess": [
                        "load_from_netcdf",
                        "assign_coords",
                        "unify_chunks",
                        "save",
                    ]
                },
                "version": 2,
            },
        )

        assert os.path.exists(output_path)

    def test_generate_dataset_v2(self):
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
                "values": STATION_LATS[N_REDUCED_AMOUNT:],
                "dimensions": ["DimN"],
            },
            "lon": {
                "type": "array",
                "values": STATION_LONS[N_REDUCED_AMOUNT:],
                "dimensions": ["DimN"],
            },
            "height": {
                "type": "array",
                "values": [0.0] * (N_STATIONS - N_REDUCED_AMOUNT),
                "dimensions": ["DimN"],
            },
            "name": {
                "type": "array",
                "values": STATION_NAMES[N_REDUCED_AMOUNT:],
                "dimensions": ["DimN"],
            },
        }
        dataset = utils.DatasetGenerator(description=description).generate()
        dataset.save(f"{DATASET_NAME_POINT_LIST}_v2", to_netcdf=True)

        assert os.path.exists(
            os.path.join(TMP_FOLDER, f"{DATASET_NAME_POINT_LIST}_v2.nc")
        )

    def test_convert_dataset_v2(self, convert):
        """Convert the NetCDF to Zarr format."""
        output_path = os.path.join(TMP_FOLDER, f"{DATASET_NAME_POINT_LIST}_v2.zarr")
        convert(
            input_path=os.path.join(TMP_FOLDER, f"{DATASET_NAME_POINT_LIST}_v2.nc"),
            output_path=output_path,
            config={
                "variables": {
                    "lon": "lon",
                    "lat": "lat",
                    "height": "height",
                    "time": "time",
                },
                "assignCoords": {"name": "DimN"},
                "pipelines": {
                    "preprocess": [
                        "load_from_netcdf",
                        "assign_coords",
                        "unify_chunks",
                        "save",
                    ]
                },
                "version": 2,
            },
        )

        assert os.path.exists(output_path)

    def test_combine_datasets(self, convert):
        """Combine the two Zarr datasets into one."""
        output_path = os.path.join(TMP_FOLDER, f"{DATASET_NAME_POINT_LIST}_combined.zarr")
        convert(
            input_path=os.path.join(TMP_FOLDER, f"{DATASET_NAME_POINT_LIST}_v1.zarr"),
            output_path=output_path,
            config={
                "variables": {
                    "lon": "lon",
                    "lat": "lat",
                    "height": "height",
                    "time": "time",
                },
                "assignCoords": {"name": "DimN"},
                "pipelines": {
                    "preprocess": [
                        "load_from_zarr",
                        {
                            "type": "process",
                            "name": "load_from_zarr",
                            "params": {
                                "store_as_secondary": True,
                                "load_path": os.path.join(
                                    TMP_FOLDER, f"{DATASET_NAME_POINT_LIST}_v2.zarr"
                                ),
                            },
                        },
                        {
                            "type": "process",
                            "name": "combine_at_time",
                            "params": {"combine_time": "2026-01-01T02:00:00"},
                        },
                        "unify_chunks",
                        "save",
                    ]
                },
                "version": 2,
            },
        )

        assert os.path.exists(output_path)

    def test_combined_dataset_points(self, client: TestClient):
        """Check that the combined dataset contains the expected points."""
        response = client.get(
            f"/datasets/{DATASET_NAME_POINT_LIST}_combined/select?variable=name"
        )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) == N_TIMES
        first_part = data["data"][:-N_REDUCED_AMOUNT]
        second_part = data["data"][N_REDUCED_AMOUNT:]
        first_part_stations = [name.decode() for name in STATION_NAMES[:-N_REDUCED_AMOUNT]]
        second_part_stations = [name.decode() for name in STATION_NAMES[N_REDUCED_AMOUNT:]]
        assert np.equal(first_part, first_part_stations).all()
        assert np.equal(second_part, second_part_stations).all()

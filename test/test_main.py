import sys
import os
from pathlib import Path
import utils

from fastapi.testclient import TestClient
import pytest
from matplotlib import pyplot as plt
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
tool_dir = root_dir / "conversion_tool"

# As Python does not support relative imports, we need to use workarounds to import modules from both src and conversion_tool/src for testing
# Import from base project
sys.path.append(str(root_dir))
import src.handlers as handlers  # noqa
from src.api import app  # noqa

# Clear modules to allow re-importing conversion_tool/src ones after
modules_to_clear = ["src", "src.utils", "src.handlers", "src.processes"]
for m in modules_to_clear:
    if m in sys.modules:
        del sys.modules[m]

# Update path to include conversion_tool/src for testing
sys.path.insert(0, str(tool_dir))

# Import conversion_tool modules (after clearing src ones)
import main as conversion_tool  # noqa


client = TestClient(app)

os.environ["TEST_TMP_FOLDER"] = "test/tests_tmp"


@pytest.mark.General
class TestGeneral:
    def test_health(self):
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_list_datasets(self):
        response = client.get("/datasets")

        assert response.status_code == 200


@pytest.mark.RegularGrid
class TestRegularGrid:
    @pytest.fixture(scope="class", autouse=True)
    def setup_and_cleanup(self):
        yield
        utils.cleanup_test_files()

    times = 5
    lats = 10
    lons = 15
    lat_start = 45.1
    lat_step = 0.1
    lon_start = 10.1
    lon_step = 0.1

    def test_generate_dataset(self):
        description = {
            "Value": {
                "type": "float",
                "method": "linear",
                "step": 1,
                "dimensions": [
                    {
                        "name": "time",
                        "type": "date",
                        "size": self.times,
                        "start": "2026-01-01",
                        "freq": "D",
                    },
                    {
                        "name": "lat",
                        "type": "latitude",
                        "size": self.lats,
                        "start": self.lat_start,
                        "step": self.lat_step,
                    },
                    {
                        "name": "lon",
                        "type": "longitude",
                        "size": self.lons,
                        "start": self.lon_start,
                        "step": self.lon_step,
                    },
                ],
            }
        }
        generator = utils.DatasetGenerator(description=description)
        dataset = generator.generate()
        dataset.save("regular_grid", to_netcdf=True)

        # Check if file exists
        saved_file = os.path.join(
            os.environ.get("TEST_TMP_FOLDER", "tmp"), "regular_grid.nc"
        )
        assert os.path.exists(saved_file)

    def test_convert_dataset(self):
        conversion_tool.new_dataset(
            dataset_name="regular_grid",
            input_path=os.path.join(
                os.environ.get("TEST_TMP_FOLDER", "tmp"), "regular_grid.nc"
            ),
            output_path=os.path.join(
                os.environ.get("TEST_TMP_FOLDER", "tmp"), "regular_grid.zarr"
            ),
            config={
                "variables": {
                    "lon": "lon",
                    "lat": "lat",
                    "time": "time",
                },
                "pipelines": {
                    "preprocess": ["load_from_netcdf", "unify_chunks", "save"]
                },
                "version": 2,
            },
        )

        # Check if folder exists
        saved_folder = os.path.join(
            os.environ.get("TEST_TMP_FOLDER", "tmp"), "regular_grid.zarr"
        )
        assert os.path.exists(saved_folder)

    def test_extract(self):
        response = client.get(
            "/datasets/regular_grid/extract?variable=Value&time=2026-01-03"
        )

        assert response.status_code == 200
        json_response = response.json()
        assert json_response["bounds"] == {"min": 300, "max": 449}

    def test_extract_time_interpolation(self):
        response = client.get(
            "/datasets/regular_grid/extract?variable=Value&time=2026-01-03T12:00:00&time_interpolate=true"
        )

        assert response.status_code == 200
        json_response = response.json()
        assert json_response["bounds"] == {"min": 375, "max": 524}

    def test_extract_mesh(self):
        mesh_tile_size = 5
        response = client.get(
            "/datasets/regular_grid/extract?variable=Value&time=2026-01-01&format=mesh&mesh_tile_size="
            + str(mesh_tile_size)
        )

        assert response.status_code == 200
        json_response = response.json()
        assert json_response["bounds"] == {"min": 0, "max": 149}
        assert len(json_response["values"]) == mesh_tile_size * mesh_tile_size
        assert json_response["values"][mesh_tile_size * 1 + 1] == 33

    def test_extract_mesh_interpolation(self):
        mesh_tile_size = 5
        response = client.get(
            f"/datasets/regular_grid/extract?variable=Value&time=2026-01-01&format=mesh&mesh_tile_size={str(mesh_tile_size)}&mesh_interpolate=true"
        )

        assert response.status_code == 200
        json_response = response.json()
        assert json_response["bounds"] == {"min": 0, "max": 149}
        assert len(json_response["values"]) == mesh_tile_size * mesh_tile_size
        for i in range(mesh_tile_size):
            assert json_response["values"][i] == pytest.approx((self.lons - 1) / 4 * i)

    def test_extract_tile(self):
        lat = (self.lat_start + 0.25, self.lat_start + self.lat_step * self.lats - 0.25)
        lon = (self.lon_start + 0.25, self.lon_start + self.lon_step * self.lons - 0.25)
        response = client.get(
            f"/datasets/regular_grid/extract?variable=Value&time=2026-01-01&lat_min={lat[0]}&lat_max={lat[1]}&lon_min={lon[0]}&lon_max={lon[1]}"
        )

        assert response.status_code == 200
        json_response = response.json()

        min_lat, max_lat, min_lon, max_lon = None, None, None, None
        for index, value in enumerate(json_response["data"]["values"]):
            if value is not None:
                current_lat = json_response["data"]["latitudes"][index]
                current_lon = json_response["data"]["longitudes"][index]

                if min_lat is None or current_lat < min_lat:
                    min_lat = current_lat
                if max_lat is None or current_lat > max_lat:
                    max_lat = current_lat
                if min_lon is None or current_lon < min_lon:
                    min_lon = current_lon
                if max_lon is None or current_lon > max_lon:
                    max_lon = current_lon

        assert min_lat >= lat[0] and max_lat <= lat[1]
        assert min_lon >= lon[0] and max_lon <= lon[1]

        # plt.scatter(longitudes, latitudes, c=values)
        # plt.colorbar(label='Value')
        # plt.xlabel('Longitude')
        # plt.ylabel('Latitude')
        # plt.title('Extracted Tile Values')
        # plt.show()


@pytest.mark.RectilinearGrid
class TestRectilinearGrid:
    @pytest.fixture(scope="class", autouse=True)
    def setup_and_cleanup(self):
        yield
        # utils.cleanup_test_files()

    heights = 1
    lats = 91
    lons = 137

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

        generator = utils.DatasetGenerator(description=description)
        dataset = generator.generate()
        dataset.save("rectilinear_grid", to_netcdf=True)

    def test_convert_dataset(self):
        output_path = os.path.join(
            os.environ.get("TEST_TMP_FOLDER", "tmp"), "rectilinear_grid.zarr"
        )
        conversion_tool.new_dataset(
            dataset_name="rectilinear_grid",
            input_path=os.path.join(
                os.environ.get("TEST_TMP_FOLDER", "tmp"), "rectilinear_grid.nc"
            ),
            output_path=output_path,
            config={
                "variables": {
                    "lon": "lon",
                    "lat": "lat",
                    "time": "steps",
                },
                "dimensions": {
                    "fixed": {
                        "DimK": 0,
                    }
                },
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

        # Check if folder exists
        assert os.path.exists(output_path)

    def test_extract(self):
        response = client.get(
            "/datasets/rectilinear_grid/extract?variable=Precipitation&time=2026-01-01"
        )

        assert response.status_code == 200
        assert response.json()["bounds"] == {
            "min": 0,
            "max": self.heights * self.lats * self.lons - 1,
        }


@pytest.mark.RadialGrid
class TestRadialGrid:
    @pytest.fixture(scope="class", autouse=True)
    def setup_and_cleanup(self):
        yield
        utils.cleanup_test_files()

    def test_generate_dataset(self):
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

        generator = utils.DatasetGenerator(description=description)
        dataset = generator.generate()
        dataset.save("radial_grid", to_netcdf=True)

    def test_convert_dataset(self):
        conversion_tool.new_dataset(
            dataset_name="radial_grid",
            input_path=os.path.join(
                os.environ.get("TEST_TMP_FOLDER", "tmp"), "radial_grid.nc"
            ),
            output_path=os.path.join(
                os.environ.get("TEST_TMP_FOLDER", "tmp"), "radial_grid.zarr"
            ),
            config={
                "variables": {
                    "lon": "lon",
                    "lat": "lat",
                    "time": "time",
                },
                "pipelines": {
                    "preprocess": ["load_from_netcdf", "unify_chunks", "save"]
                },
                "version": 2,
            },
        )

        # Check if folder exists
        saved_folder = os.path.join(
            os.environ.get("TEST_TMP_FOLDER", "tmp"), "radial_grid.zarr"
        )
        assert os.path.exists(saved_folder)


@pytest.mark.PointList
class TestPointList:
    @pytest.fixture(scope="class", autouse=True)
    def setup_and_cleanup(self):
        yield
        utils.cleanup_test_files()

    def test_generate_dataset(self):
        description = {
            "Humidity": {
                "type": "float",
                "method": "cos",
                "bounds": {"min": 0, "max": 100},
                "dimensions": [
                    {
                        "name": "time",
                        "type": "date",
                        "size": 5,
                        "start": "2026-01-01",
                        "freq": "1h",
                    },
                    {"name": "DimN", "size": "lat.DimN"},
                ],
            },
            "lat": {
                "type": "array",
                "values": [
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
                ],
                "dimensions": ["DimN"],
            },
            "lon": {
                "type": "array",
                "values": [
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
                ],
                "dimensions": ["DimN"],
            },
            "height": {
                "type": "array",
                "values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "dimensions": ["DimN"],
            },
        }
        generator = utils.DatasetGenerator(description=description)
        dataset = generator.generate()
        dataset.save("point_list", to_netcdf=True)

    def test_convert_dataset(self):
        output_path = os.path.join(
            os.environ.get("TEST_TMP_FOLDER", "tmp"), "point_list.zarr"
        )
        conversion_tool.new_dataset(
            dataset_name="point_list",
            input_path=os.path.join(
                os.environ.get("TEST_TMP_FOLDER", "tmp"), "point_list.nc"
            ),
            output_path=output_path,
            config={
                "variables": {
                    "lon": "lon",
                    "lat": "lat",
                    "time": "time",
                },
                "pipelines": {
                    "preprocess": ["load_from_netcdf", "unify_chunks", "save"]
                },
                "version": 2,
            },
        )

        # Check if folder exists
        assert os.path.exists(output_path)

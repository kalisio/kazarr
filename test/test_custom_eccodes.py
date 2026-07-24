import os

import pytest
import eccodes
import numpy as np
from fastapi.testclient import TestClient

import utils

TMP_FOLDER = os.environ.get("TEST_TMP_FOLDER", "test/tests_tmp")
CUSTOM_ECCODES_FOLDER = "./test/samples/custom_eccodes"
DATASET_NAME = "grib2_dataset"


@pytest.mark.CustomEccodes
class TestCustomEccodes:
    @pytest.fixture(scope="class", autouse=True)
    def cleanup(self):
        yield
        utils.cleanup_test_files()

    def generate_dummy_grib2(
        self, output_path, discipline=0, category=3, number=254, centre="lfpw"
    ):
        # Create a new GRIB2 message using a sample template
        gid = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")

        try:
            eccodes.codes_set(gid, "discipline", discipline)
            eccodes.codes_set(gid, "parameterCategory", category)
            eccodes.codes_set(gid, "parameterNumber", number)
            eccodes.codes_set(gid, "centre", centre)

            nx = eccodes.codes_get(gid, "Ni")
            ny = eccodes.codes_get(gid, "Nj")
            dummy_data = np.random.rand(nx * ny)

            eccodes.codes_set_values(gid, dummy_data)

            with open(output_path, "wb") as f:
                eccodes.codes_write(gid, f)

        finally:
            eccodes.codes_release(gid)

    def test_generate_dummy_grib2(self):
        output_path = os.path.join(TMP_FOLDER, f"{DATASET_NAME}.grib2")
        self.generate_dummy_grib2(output_path)
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

    def test_convert_grib2(self, convert):
        output_path = os.path.join(TMP_FOLDER, f"{DATASET_NAME}.zarr")
        convert(
            input_path=os.path.join(TMP_FOLDER, f"{DATASET_NAME}.grib2"),
            output_path=output_path,
            config={
                "variables": {
                    "lon": "longitude",
                    "lat": "latitude",
                    "time": "valid_time",
                },
                "pipelines": {
                    "preprocess": [
                        "load_from_grib",
                        "unify_chunks",
                        "save",
                    ]
                },
                "version": 2,
            },
            custom_eccodes_path=CUSTOM_ECCODES_FOLDER,
        )
        assert os.path.exists(output_path)

    def test_variable_name(self, client: TestClient):
        response = client.get(f"/datasets/{DATASET_NAME}/metadata")

        assert response.status_code == 200
        data = response.json()
        assert "variables" in data
        assert "coordinates" in data
        assert "longitude" in data["coordinates"]
        assert "latitude" in data["coordinates"]
        assert "CV" in data["variables"]

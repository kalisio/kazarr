"""
test_time_reduce.py — Tests for the `time_reduce` parameter of the extract endpoint.

`time_reduce` collapses a time range ('start/end') into a single field:
- 'delta': last step minus first step (e.g. accumulated precipitation)
- 'sum', 'mean', 'min', 'max': aggregation over every step of the range
"""

import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from fastapi.testclient import TestClient

import utils

TMP_FOLDER = os.environ.get("TEST_TMP_FOLDER", "test/tests_tmp")

# ---------------------------------------------------------------------------
# Regular grid, time as a dimension. Value = t * (LATS * LONS) + flat_index
# so the value at a given cell increases by LATS * LONS (150) per day.
# ---------------------------------------------------------------------------
TIMES = 5
LATS = 10
LONS = 15
CELLS = LATS * LONS
DATASET_NAME = "time_reduce_grid"

# ---------------------------------------------------------------------------
# Météo-France like: cumulative precipitation since the start of the run,
# time ('valid_time') is a coordinate on the 'step' dimension (cfgrib layout).
# tp[step, lat, lon] = step * rate[lat, lon]
# ---------------------------------------------------------------------------
MF_DATASET_NAME = "time_reduce_mf"
MF_STEPS = 7
MF_RUN = pd.Timestamp("2026-01-01T00:00:00")
MF_LATS = np.round(np.arange(43.0, 44.0, 0.1), 2)
MF_LONS = np.round(np.arange(1.0, 2.2, 0.1), 2)


def _mf_rate():
    lat_idx, lon_idx = np.meshgrid(
        np.arange(len(MF_LATS)), np.arange(len(MF_LONS)), indexing="ij"
    )
    return (1.0 + lat_idx + 0.5 * lon_idx).astype(np.float32)


@pytest.mark.TimeReduce
class TestTimeReduce:
    @pytest.fixture(scope="class", autouse=True)
    def cleanup(self):
        yield
        utils.cleanup_test_files()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

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
                        "size": TIMES,
                        "start": "2026-01-01",
                        "freq": "D",
                    },
                    {
                        "name": "lat",
                        "type": "latitude",
                        "size": LATS,
                        "start": 45.1,
                        "step": 0.1,
                    },
                    {
                        "name": "lon",
                        "type": "longitude",
                        "size": LONS,
                        "start": 10.1,
                        "step": 0.1,
                    },
                ],
            }
        }
        utils.DatasetGenerator(description=description).generate().save(
            DATASET_NAME, to_netcdf=True
        )

        steps = np.arange(MF_STEPS)
        tp = steps[:, None, None] * _mf_rate()[None, :, :]
        # As in GRIB files, no accumulation at the start of the run (step 0)
        tp[0] = np.nan
        # Last step is missing on the first latitude row
        tp[-1, 0, :] = np.nan
        ds = xr.Dataset(
            {
                "tp": (
                    ("step", "latitude", "longitude"),
                    tp.astype(np.float32),
                    {"units": "kg m**-2"},
                )
            },
            coords={
                "step": ("step", pd.to_timedelta(steps, unit="h")),
                "valid_time": ("step", MF_RUN + pd.to_timedelta(steps, unit="h")),
                "latitude": ("latitude", MF_LATS),
                "longitude": ("longitude", MF_LONS),
            },
        )
        ds.to_netcdf(os.path.join(TMP_FOLDER, f"{MF_DATASET_NAME}.nc"))

    def test_convert_dataset(self, convert):
        convert(
            input_path=os.path.join(TMP_FOLDER, f"{DATASET_NAME}.nc"),
            output_path=os.path.join(TMP_FOLDER, f"{DATASET_NAME}.zarr"),
            config={
                "variables": {"lon": "lon", "lat": "lat", "time": "time"},
                "pipelines": {
                    "preprocess": ["load_from_netcdf", "unify_chunks", "save"]
                },
                "version": 2,
            },
            mesh_type="regular",
        )
        convert(
            input_path=os.path.join(TMP_FOLDER, f"{MF_DATASET_NAME}.nc"),
            output_path=os.path.join(TMP_FOLDER, f"{MF_DATASET_NAME}.zarr"),
            config={
                "variables": {
                    "lon": "longitude",
                    "lat": "latitude",
                    "time": "valid_time",
                },
                "pipelines": {
                    "preprocess": ["load_from_netcdf", "unify_chunks", "save"]
                },
                "version": 2,
            },
            mesh_type="regular",
        )
        assert os.path.exists(os.path.join(TMP_FOLDER, f"{DATASET_NAME}.zarr"))
        assert os.path.exists(os.path.join(TMP_FOLDER, f"{MF_DATASET_NAME}.zarr"))

    # ------------------------------------------------------------------
    # Reduction methods
    # ------------------------------------------------------------------

    def _extract(self, client, query, dataset=DATASET_NAME):
        return client.get(f"/datasets/{dataset}/extract?{query}")

    def test_delta(self, client: TestClient):
        response = self._extract(
            client, "variable=Value&time=2026-01-02/2026-01-04&time_reduce=delta"
        )
        assert response.status_code == 200
        data = response.json()
        # Single field, no time dimension in the output
        assert "times" not in data
        assert len(data["values"]["Value"]) == CELLS
        assert np.allclose(data["values"]["Value"], 2 * CELLS)
        assert data["time_reduce"] == {
            "method": "delta",
            "start": "2026-01-02T00:00:00.000000000",
            "end": "2026-01-04T00:00:00.000000000",
            "count": 2,
        }

    @pytest.mark.parametrize(
        "method,expected_offset",
        [
            ("mean", 2 * CELLS),  # value of the middle day (2026-01-03)
            ("min", 1 * CELLS),  # first day of the range
            ("max", 3 * CELLS),  # last day of the range
        ],
    )
    def test_aggregations(self, client: TestClient, method, expected_offset):
        response = self._extract(
            client, f"variable=Value&time=2026-01-02/2026-01-04&time_reduce={method}"
        )
        assert response.status_code == 200
        data = response.json()
        assert "times" not in data
        np.testing.assert_allclose(
            data["values"]["Value"], np.arange(CELLS) + expected_offset
        )
        assert data["time_reduce"]["count"] == 3

    def test_sum(self, client: TestClient):
        response = self._extract(
            client, "variable=Value&time=2026-01-02/2026-01-04&time_reduce=sum"
        )
        assert response.status_code == 200
        np.testing.assert_allclose(
            response.json()["values"]["Value"], 3 * (np.arange(CELLS) + 2 * CELLS)
        )

    @pytest.mark.parametrize("method", ["delta", "sum", "mean", "min", "max"])
    def test_time_interpolation_is_ignored(self, client: TestClient, method):
        """Time interpolation is disabled with time_reduce: the result is the
        same as without it (dataset time steps only)."""
        query = "variable=Value&time=2026-01-01T12:00:00/2026-01-04T12:00:00"
        reference = self._extract(client, f"{query}&time_reduce={method}")
        response = self._extract(
            client,
            f"{query}&time_reduce={method}&interp_time=true&interp_vars_method=linear",
        )
        assert reference.status_code == 200
        assert response.status_code == 200
        assert response.json()["values"] == reference.json()["values"]
        assert response.json()["time_reduce"] == reference.json()["time_reduce"]
        assert response.json()["time_reduce"]["start"].startswith("2026-01-02T00")
        assert response.json()["time_reduce"]["end"].startswith("2026-01-04T00")

    def test_time_interpolation_is_ignored_with_interp_vars(self, client: TestClient):
        """Same when the time variable is passed explicitly in interp_vars."""
        response = self._extract(
            client,
            "variable=Value&time=2026-01-02/2026-01-04&time_reduce=mean"
            "&interp_vars=time&interp_vars_method=linear",
        )
        assert response.status_code == 200
        np.testing.assert_allclose(
            response.json()["values"]["Value"], np.arange(CELLS) + 2 * CELLS
        )

    def test_delta_with_bbox(self, client: TestClient):
        response = self._extract(
            client,
            "variable=Value&time=2026-01-01/2026-01-05&time_reduce=delta"
            "&lon_min=10.3&lon_max=10.6&lat_min=45.3&lat_max=45.6",
        )
        assert response.status_code == 200
        values = np.array(response.json()["values"]["Value"], dtype=float)
        values = values[~np.isnan(values)]
        assert values.size > 0
        assert np.allclose(values, 4 * CELLS)

    def test_delta_geojson(self, client: TestClient):
        response = self._extract(
            client,
            "variable=Value&time=2026-01-02/2026-01-04&time_reduce=delta&format=geojson",
        )
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "FeatureCollection"
        assert data["properties"]["time_reduce"]["method"] == "delta"
        assert "times" not in data["properties"]
        assert all(f["properties"]["Value"] == 2 * CELLS for f in data["features"])

    def test_delta_mesh(self, client: TestClient):
        response = self._extract(
            client,
            "variable=Value&time=2026-01-02/2026-01-04&time_reduce=delta&format=mesh",
        )
        assert response.status_code == 200

    # ------------------------------------------------------------------
    # Errors
    # ------------------------------------------------------------------

    def test_requires_time_range(self, client: TestClient):
        response = self._extract(
            client, "variable=Value&time=2026-01-02&time_reduce=delta"
        )
        assert response.status_code == 400
        assert "TIME_REDUCE_REQUIRES_TIME_RANGE" in response.text

    def test_requires_time(self, client: TestClient):
        response = self._extract(client, "variable=Value&time_reduce=mean")
        assert response.status_code == 400
        assert "TIME_REDUCE_REQUIRES_TIME_RANGE" in response.text

    def test_delta_single_step(self, client: TestClient):
        """A range covering a single time step cannot give a delta."""
        response = self._extract(
            client,
            "variable=Value&time=2026-01-02T06:00:00/2026-01-02T18:00:00&time_reduce=delta",
        )
        assert response.status_code == 400
        assert "TIME_REDUCE_NOT_ENOUGH_STEPS" in response.text

    def test_invalid_method(self, client: TestClient):
        response = self._extract(
            client, "variable=Value&time=2026-01-02/2026-01-04&time_reduce=median"
        )
        assert response.status_code == 422

    # ------------------------------------------------------------------
    # Météo-France like accumulation
    # ------------------------------------------------------------------

    def test_mf_accumulation(self, client: TestClient):
        """Precipitation accumulated between +2h and +5h = 3 * rate."""
        response = self._extract(
            client,
            "variable=tp&time=2026-01-01T02:00:00/2026-01-01T05:00:00&time_reduce=delta",
            dataset=MF_DATASET_NAME,
        )
        assert response.status_code == 200
        data = response.json()
        np.testing.assert_allclose(
            data["values"]["tp"], (3 * _mf_rate()).ravel(), rtol=1e-6
        )
        assert data["time_reduce"]["count"] == 2

    def test_mf_accumulation_snaps_to_steps(self, client: TestClient):
        """Without interpolation, the bounds are the first/last steps inside the range."""
        response = self._extract(
            client,
            "variable=tp&time=2026-01-01T01:30:00/2026-01-01T04:30:00&time_reduce=delta",
            dataset=MF_DATASET_NAME,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["time_reduce"]["start"].startswith("2026-01-01T02:00:00")
        assert data["time_reduce"]["end"].startswith("2026-01-01T04:00:00")
        np.testing.assert_allclose(
            data["values"]["tp"], (2 * _mf_rate()).ravel(), rtol=1e-6
        )

    # ------------------------------------------------------------------
    # Null values are ignored by 'delta'
    # ------------------------------------------------------------------

    def test_mf_delta_null_start(self, client: TestClient):
        """Start of the run is null everywhere: the end value is returned."""
        response = self._extract(
            client,
            "variable=tp&time=2026-01-01T00:00:00/2026-01-01T03:00:00&time_reduce=delta",
            dataset=MF_DATASET_NAME,
        )
        assert response.status_code == 200
        np.testing.assert_allclose(
            response.json()["values"]["tp"], (3 * _mf_rate()).ravel(), rtol=1e-6
        )

    def test_mf_delta_null_end(self, client: TestClient):
        """End is null on the first row: the start value is returned there."""
        response = self._extract(
            client,
            "variable=tp&time=2026-01-01T02:00:00/2026-01-01T06:00:00&time_reduce=delta",
            dataset=MF_DATASET_NAME,
        )
        assert response.status_code == 200
        expected = 4 * _mf_rate()
        expected[0, :] = 2 * _mf_rate()[0, :]
        np.testing.assert_allclose(
            response.json()["values"]["tp"], expected.ravel(), rtol=1e-6
        )

    def test_mf_delta_both_null(self, client: TestClient):
        """Both bounds null on the first row: the result stays null there."""
        response = self._extract(
            client,
            "variable=tp&time=2026-01-01T00:00:00/2026-01-01T06:00:00&time_reduce=delta",
            dataset=MF_DATASET_NAME,
        )
        assert response.status_code == 200
        values = np.array(response.json()["values"]["tp"], dtype=float).reshape(
            len(MF_LATS), len(MF_LONS)
        )
        assert np.isnan(values[0]).all()
        np.testing.assert_allclose(values[1:], 6 * _mf_rate()[1:], rtol=1e-6)

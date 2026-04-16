"""
test_general.py — General API tests (health, dataset listing, dataset infos).
"""

import pytest
from fastapi.testclient import TestClient


@pytest.mark.General
class TestGeneral:
    def test_health(self, client: TestClient):
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_list_datasets(self, client: TestClient):
        response = client.get("/datasets")

        assert response.status_code == 200
        data = response.json()
        assert "datasets" in data
        assert isinstance(data["datasets"], list)

    def test_list_datasets_with_search_path(self, client: TestClient):
        response = client.get("/datasets?search_path=/")

        assert response.status_code == 200

    def test_dataset_not_found(self, client: TestClient):
        response = client.get("/datasets/nonexistent_dataset_xyz/infos")

        assert response.status_code in (400, 404, 500)

    def test_dataset_infos_not_found(self, client: TestClient):
        response = client.get("/datasets/nonexistent_dataset_xyz/infos")

        assert response.status_code != 200

    def test_extract_missing_variable(self, client: TestClient):
        """Calling extract without a variable parameter should return 400."""
        response = client.get("/datasets/some_dataset/extract")

        # FastAPI returns 422 for missing required query params
        assert response.status_code in (400, 422)

    def test_probe_missing_coordinates(self, client: TestClient):
        """Calling probe without lat/lon should return 422."""
        response = client.get("/datasets/some_dataset/probe?variables=Value")

        assert response.status_code == 422

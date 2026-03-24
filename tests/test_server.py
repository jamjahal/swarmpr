"""Tests for the FastAPI server."""

import pytest
from fastapi.testclient import TestClient

from swarmpr.server.app import app


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app.

    Returns:
        A TestClient instance.
    """
    return TestClient(app)


class TestServer:
    """Tests for the FastAPI server endpoints."""

    def test_health_check(self, client: TestClient):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"

    def test_history_empty(self, client: TestClient):
        """Test that history is initially empty."""
        response = client.get("/history")
        assert response.status_code == 200
        assert response.json() == []

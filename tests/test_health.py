"""Tests for health endpoints."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from hybrid_rag.app import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_db_connected():
    with patch("hybrid_rag.api.health.check_db_connection", return_value=True):
        response = client.get("/health/db")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["database"] is True


def test_health_db_disconnected():
    with patch("hybrid_rag.api.health.check_db_connection", return_value=False):
        response = client.get("/health/db")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unavailable"
        assert data["database"] is False

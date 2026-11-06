"""Tests for /ready endpoint [DEPLOY-02, DEPLOY-05]."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from hybrid_rag.app import app

client = TestClient(app)


class TestReadinessEndpoint:
    @patch("hybrid_rag.ingest.embedder.get_embedder")
    @patch("hybrid_rag.api.health.check_db_connection", return_value=True)
    def test_all_healthy(self, mock_db, mock_model):
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is True
        assert data["checks"]["database"] is True
        assert data["checks"]["embedding_model"] is True

    @patch("hybrid_rag.api.health.check_db_connection", return_value=False)
    def test_db_down_returns_503(self, mock_db):
        with patch("hybrid_rag.ingest.embedder.get_embedder"):
            response = client.get("/ready")

        assert response.status_code == 503
        data = response.json()
        assert data["ready"] is False
        assert data["checks"]["database"] is False

    @patch("hybrid_rag.api.health.check_db_connection", return_value=True)
    def test_embedding_model_not_loaded_returns_503(self, mock_db):
        with patch(
            "hybrid_rag.ingest.embedder.get_embedder",
            side_effect=Exception("model not found"),
        ):
            response = client.get("/ready")

        assert response.status_code == 503
        data = response.json()
        assert data["ready"] is False
        assert data["checks"]["embedding_model"] is False
        assert data["checks"]["database"] is True

    @patch("hybrid_rag.api.health.check_db_connection", return_value=False)
    def test_all_down_returns_503(self, mock_db):
        with patch(
            "hybrid_rag.ingest.embedder.get_embedder",
            side_effect=Exception("down"),
        ):
            response = client.get("/ready")

        assert response.status_code == 503
        data = response.json()
        assert data["ready"] is False
        assert all(not v for v in data["checks"].values())

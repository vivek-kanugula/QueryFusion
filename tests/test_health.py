"""Tests for health endpoints."""

from unittest.mock import patch

import pytest
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


@pytest.mark.asyncio
async def test_lifespan_bm25_init_error_logged(caplog):
    import logging
    from fastapi import FastAPI
    from hybrid_rag.app import lifespan

    dummy_app = FastAPI()
    with patch(
        "hybrid_rag.retrieval.bm25_store.get_bm25_store",
        side_effect=RuntimeError("Store init failed"),
    ):
        with caplog.at_level(logging.ERROR):
            async with lifespan(dummy_app):
                pass
    assert "BM25 index initialization failed" in caplog.text

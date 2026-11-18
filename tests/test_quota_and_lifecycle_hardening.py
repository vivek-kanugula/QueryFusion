"""Tests for multi-key rotation, Groq auxiliary offloading, retention, and rate limits."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from hybrid_rag.app import app
from hybrid_rag.db.engine import get_db
from hybrid_rag.generation.llm_client import (
    _call_gemini,
    call_auxiliary_llm,
    get_active_gemini_key,
    get_gemini_api_keys,
    rotate_to_next_gemini_key,
)
from hybrid_rag.ingest.index_manager import cleanup_expired_documents
from hybrid_rag.models.document import Document
from hybrid_rag.settings import settings

client = TestClient(app)


def test_gemini_multi_key_pool():
    with patch.object(settings, "gemini_api_keys", ["key1", "key2", "key3"]):
        keys = get_gemini_api_keys()
        assert keys == ["key1", "key2", "key3"]

        k1 = get_active_gemini_key()
        k2 = rotate_to_next_gemini_key()
        assert k1 != k2
        assert k2 in keys


@patch("google.genai.Client")
def test_gemini_multi_key_rotation_on_429(mock_client_cls):
    # First key throws 429, second key succeeds
    mock_instance_1 = MagicMock()
    mock_instance_1.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED")

    mock_instance_2 = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = '{"answer": "recovered with key 2"}'
    mock_instance_2.models.generate_content.return_value = mock_resp

    mock_client_cls.side_effect = [mock_instance_1, mock_instance_2]

    with patch.object(settings, "gemini_api_keys", ["keyA", "keyB"]):
        resp = _call_gemini(
            model_name="gemini-3.5-flash-lite",
            prompt="Hello",
            system_instruction="Be concise",
            temperature=0.0,
        )
        assert resp == '{"answer": "recovered with key 2"}'
        assert mock_client_cls.call_count == 2


@patch("hybrid_rag.generation.llm_client._call_groq")
def test_call_auxiliary_llm_prefers_groq(mock_call_groq):
    mock_call_groq.return_value = '{"queries": ["q1", "q2"]}'
    with patch.object(settings, "groq_api_key", "gsk-valid-key"):
        result = call_auxiliary_llm(
            prompt="Test prompt",
            system_instruction="Extract queries",
            temperature=0.0,
        )
        assert result == '{"queries": ["q1", "q2"]}'
        assert mock_call_groq.called


@patch("hybrid_rag.generation.llm_client._call_groq")
@patch("hybrid_rag.generation.llm_client._call_gemini")
def test_call_auxiliary_llm_falls_back_to_gemini(mock_call_gemini, mock_call_groq):
    mock_call_groq.side_effect = Exception("Groq network timeout")
    mock_call_gemini.return_value = '{"queries": ["gemini_fallback"]}'

    with patch.object(settings, "groq_api_key", "gsk-valid-key"):
        result = call_auxiliary_llm(
            prompt="Test prompt",
            system_instruction="Extract queries",
            temperature=0.0,
            fallback_to_gemini=True,
        )
        assert result == '{"queries": ["gemini_fallback"]}'
        assert mock_call_groq.called
        assert mock_call_gemini.called


def test_cleanup_expired_documents_logic():
    mock_session = MagicMock()

    now = datetime.now(UTC)
    old_time = now - timedelta(hours=48)

    expired_doc = MagicMock(spec=Document)
    expired_doc.id = uuid.uuid4()
    expired_doc.is_evictable = True
    expired_doc.created_at = old_time
    expired_doc.chunking_strategy = "fixed"

    # Query returns 1 expired document
    mock_session.query.return_value.filter.return_value.filter.return_value.all.return_value = [
        expired_doc
    ]

    with patch("hybrid_rag.ingest.index_manager.delete_document") as mock_delete:
        evicted = cleanup_expired_documents(mock_session, max_age_hours=24)
        assert evicted == 1
        mock_delete.assert_called_once_with(mock_session, expired_doc.id, "fixed")


def test_trigger_cleanup_endpoint():
    mock_session = MagicMock()
    mock_session.query.return_value.filter.return_value.filter.return_value.all.return_value = []

    def _override():
        yield mock_session

    app.dependency_overrides[get_db] = _override
    try:
        response = client.post("/api/v1/documents/cleanup?max_age_hours=12")
        assert response.status_code == 200
        data = response.json()
        assert "evicted" in data
        assert data["max_age_hours"] == 12
    finally:
        app.dependency_overrides.clear()


def test_daily_query_rate_limit():
    from hybrid_rag.api.query import _check_rate_limit, _request_counts

    _request_counts.clear()
    ip = "198.51.100.1"

    with patch.object(settings, "rate_limit_per_day", 3), patch.object(
        settings, "rate_limit_per_minute", 100
    ):
        _check_rate_limit(ip)
        _check_rate_limit(ip)
        _check_rate_limit(ip)

        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc:
            _check_rate_limit(ip)
        assert exc.value.status_code == 429
        assert "Daily query quota exceeded" in exc.value.detail

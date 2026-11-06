"""Tests for golden document eviction protection [DATA-04]."""

import uuid
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from hybrid_rag.app import app
from hybrid_rag.db.engine import get_db
from hybrid_rag.models.document import Document

client = TestClient(app)


def _override_db(mock_session):
    def _get_db_override():
        yield mock_session

    return _get_db_override


def _make_doc(*, evictable: bool) -> MagicMock:
    doc = MagicMock(spec=Document)
    doc.id = uuid.uuid4()
    doc.filename = "test.pdf"
    doc.is_evictable = evictable
    doc.chunking_strategy = "fixed"
    return doc


class TestSingleDeleteProtection:
    def test_protected_document_returns_403(self):
        doc = _make_doc(evictable=False)
        mock_session = MagicMock()
        mock_session.get.return_value = doc

        app.dependency_overrides[get_db] = _override_db(mock_session)
        try:
            response = client.delete(f"/api/v1/documents/{doc.id}")
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 403
        assert "protected" in response.json()["detail"].lower()

    @patch("hybrid_rag.api.documents.delete_document")
    def test_evictable_document_deletes(self, mock_delete):
        doc = _make_doc(evictable=True)
        mock_session = MagicMock()
        mock_session.get.return_value = doc

        app.dependency_overrides[get_db] = _override_db(mock_session)
        try:
            response = client.delete(f"/api/v1/documents/{doc.id}")
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 200
        assert response.json()["deleted"] == str(doc.id)
        mock_delete.assert_called_once()

    def test_missing_document_returns_404(self):
        mock_session = MagicMock()
        mock_session.get.return_value = None

        app.dependency_overrides[get_db] = _override_db(mock_session)
        try:
            response = client.delete(f"/api/v1/documents/{uuid.uuid4()}")
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 404


class TestBatchDeleteProtection:
    def test_batch_with_protected_doc_returns_403(self):
        protected = _make_doc(evictable=False)
        evictable = _make_doc(evictable=True)
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.all.return_value = [
            protected,
            evictable,
        ]

        app.dependency_overrides[get_db] = _override_db(mock_session)
        try:
            response = client.post(
                "/api/v1/documents/batch-delete",
                json={"document_ids": [str(protected.id), str(evictable.id)]},
            )
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 403
        assert "protected" in response.json()["detail"].lower()

    @patch("hybrid_rag.api.documents.delete_document")
    def test_batch_all_evictable_deletes(self, mock_delete):
        docs = [_make_doc(evictable=True) for _ in range(3)]
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.all.return_value = docs

        app.dependency_overrides[get_db] = _override_db(mock_session)
        try:
            response = client.post(
                "/api/v1/documents/batch-delete",
                json={"document_ids": [str(d.id) for d in docs]},
            )
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 200
        assert len(response.json()["deleted"]) == 3
        assert mock_delete.call_count == 3

    def test_batch_empty_ids_returns_400(self):
        mock_session = MagicMock()
        app.dependency_overrides[get_db] = _override_db(mock_session)
        try:
            response = client.post(
                "/api/v1/documents/batch-delete",
                json={"document_ids": []},
            )
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 400

    def test_batch_all_protected_returns_403(self):
        docs = [_make_doc(evictable=False) for _ in range(2)]
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.all.return_value = docs

        app.dependency_overrides[get_db] = _override_db(mock_session)
        try:
            response = client.post(
                "/api/v1/documents/batch-delete",
                json={"document_ids": [str(d.id) for d in docs]},
            )
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 403

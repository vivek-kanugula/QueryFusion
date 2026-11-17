"""Unit and integration tests for Neon Object Storage and health check."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from hybrid_rag.app import app
from hybrid_rag.storage.s3 import (
    check_storage_connection,
    delete_file,
    download_file,
    generate_presigned_url,
    get_s3_client,
    is_storage_configured,
    reset_storage_client,
    upload_file,
)

client = TestClient(app)


def test_is_storage_configured():
    with patch("hybrid_rag.storage.s3.settings") as mock_settings:
        mock_settings.aws_endpoint_url_s3 = "https://s3.neon.tech"
        mock_settings.aws_access_key_id = "test_key"
        mock_settings.aws_secret_access_key = "test_secret"
        assert is_storage_configured() is True

        mock_settings.aws_endpoint_url_s3 = None
        assert is_storage_configured() is False


def test_get_s3_client_unconfigured():
    reset_storage_client()
    with patch("hybrid_rag.storage.s3.is_storage_configured", return_value=False):
        assert get_s3_client() is None


def test_storage_crud_mocked():
    reset_storage_client()
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {"Body": MagicMock(read=lambda: b"hello world")}
    mock_s3.generate_presigned_url.return_value = "https://signed.url/test"

    with patch("hybrid_rag.storage.s3.get_s3_client", return_value=mock_s3):
        # Check connection
        assert check_storage_connection() is True

        # Upload
        assert upload_file("doc1.txt", b"hello world") == "doc1.txt"
        mock_s3.put_object.assert_called_once()

        # Download
        assert download_file("doc1.txt") == b"hello world"

        # Presigned URL
        assert generate_presigned_url("doc1.txt") == "https://signed.url/test"

        # Delete
        assert delete_file("doc1.txt") is True
        mock_s3.delete_object.assert_called_once()


def test_health_storage_endpoint():
    with patch("hybrid_rag.storage.s3.is_storage_configured", return_value=False):
        response = client.get("/health/storage")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unconfigured"

    with (
        patch("hybrid_rag.storage.s3.is_storage_configured", return_value=True),
        patch("hybrid_rag.storage.s3.check_storage_connection", return_value=True),
    ):
        response = client.get("/health/storage")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["storage"] is True

    with (
        patch("hybrid_rag.storage.s3.is_storage_configured", return_value=True),
        patch("hybrid_rag.storage.s3.check_storage_connection", return_value=False),
    ):
        response = client.get("/health/storage")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unavailable"
        assert data["storage"] is False


def test_bm25_blob_persistence_roundtrip(tmp_path):
    """Verify BM25 index snapshot serializes and loads from object storage."""
    from hybrid_rag.retrieval.bm25_store import Bm25Store
    from hybrid_rag.settings import settings

    uploaded_blobs: dict[str, bytes] = {}
    deleted_blobs: list[str] = []

    def mock_upload(key: str, data: bytes, content_type: str = "application/octet-stream"):
        uploaded_blobs[key] = data
        return key

    def mock_download(key: str):
        return uploaded_blobs.get(key)

    def mock_delete(key: str):
        uploaded_blobs.pop(key, None)
        deleted_blobs.append(key)
        return True

    store = Bm25Store()
    texts = [
        "Hybrid RAG combines dense vector and sparse lexical search.",
        "PostgreSQL with pgvector provides robust semantic retrieval.",
    ]
    chunk_ids = ["c1", "c2"]

    with (
        patch.object(settings, "bm25_use_blob", True),
        patch("hybrid_rag.storage.s3.is_storage_configured", return_value=True),
        patch("hybrid_rag.storage.s3.upload_file", side_effect=mock_upload),
        patch("hybrid_rag.storage.s3.download_file", side_effect=mock_download),
        patch.dict("os.environ", {"BM25_CACHE_DIR": str(tmp_path / "cache1")}),
    ):
        count = store.rebuild(texts, chunk_ids, "fixed")
        assert count == 2
        assert "bm25-indices/fixed.tar.gz" in uploaded_blobs

    # Test loading in a completely fresh store and clean cache dir
    fresh_store = Bm25Store()
    with (
        patch.object(settings, "bm25_use_blob", True),
        patch("hybrid_rag.storage.s3.is_storage_configured", return_value=True),
        patch("hybrid_rag.storage.s3.download_file", side_effect=mock_download),
        patch("hybrid_rag.storage.s3.delete_file", side_effect=mock_delete),
        patch.dict("os.environ", {"BM25_CACHE_DIR": str(tmp_path / "cache2")}),
    ):
        loaded = fresh_store._try_load_persistent("fixed")
        assert loaded is True
        assert fresh_store.has_index("fixed")
        results = fresh_store.search("sparse lexical search", "fixed", top_k=2)
        assert len(results) >= 1
        assert results[0][0] == "c1"

        fresh_store.clear("fixed")
        assert "bm25-indices/fixed.tar.gz" in deleted_blobs
        assert not fresh_store.has_index("fixed")


"""Tests for deterministic UUIDv5 chunk ID generation."""

import uuid

from hybrid_rag.core.chunk_id import make_chunk_id


def test_deterministic():
    doc_id = uuid.UUID("12345678-1234-5678-1234-567812345678")
    a = make_chunk_id(doc_id, "fixed", 0)
    b = make_chunk_id(doc_id, "fixed", 0)
    assert a == b


def test_different_index():
    doc_id = uuid.UUID("12345678-1234-5678-1234-567812345678")
    a = make_chunk_id(doc_id, "fixed", 0)
    b = make_chunk_id(doc_id, "fixed", 1)
    assert a != b


def test_different_strategy():
    doc_id = uuid.UUID("12345678-1234-5678-1234-567812345678")
    a = make_chunk_id(doc_id, "fixed", 0)
    b = make_chunk_id(doc_id, "semantic", 0)
    assert a != b


def test_different_document():
    doc_a = uuid.UUID("12345678-1234-5678-1234-567812345678")
    doc_b = uuid.UUID("87654321-4321-8765-4321-876543218765")
    a = make_chunk_id(doc_a, "fixed", 0)
    b = make_chunk_id(doc_b, "fixed", 0)
    assert a != b


def test_returns_uuid5():
    doc_id = uuid.UUID("12345678-1234-5678-1234-567812345678")
    result = make_chunk_id(doc_id, "fixed", 0)
    assert isinstance(result, uuid.UUID)
    assert result.version == 5

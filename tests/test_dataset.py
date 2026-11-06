"""Tests for evaluation dataset ground-truth ID matching and resolution."""

import uuid
from unittest.mock import MagicMock

import pytest

from hybrid_rag.eval.dataset import (
    ChunkInfo,
    GoldenItem,
    SourceSpan,
    _matches_document,
    _normalize_doc_id,
    build_doc_id_map,
    map_spans_to_chunks,
)
from hybrid_rag.eval.runner import run_single_question
from hybrid_rag.schemas.pipeline import PipelineConfig


def test_normalize_doc_id():
    assert _normalize_doc_id("doc-troubleshooting") == "troubleshooting"
    assert _normalize_doc_id("doc-troubleshooting.md") == "troubleshooting"
    assert _normalize_doc_id("troubleshooting.md") == "troubleshooting"
    assert _normalize_doc_id("doc_api_reference") == "api-reference"
    assert _normalize_doc_id("deployment.html") == "deployment"
    assert _normalize_doc_id("") == ""
    assert _normalize_doc_id(None) == ""


def test_matches_document_direct():
    chunk = ChunkInfo(
        chunk_id="c1",
        document_id="doc-quickstart",
        char_offset_start=0,
        char_offset_end=100,
    )
    assert _matches_document("doc-quickstart", chunk) is True
    assert _matches_document("doc-security", chunk) is False


def test_matches_document_doc_id_map():
    doc_uuid = str(uuid.uuid4())
    chunk = ChunkInfo(
        chunk_id="c1",
        document_id=doc_uuid,
        char_offset_start=0,
        char_offset_end=100,
    )
    doc_id_map = {"doc-troubleshooting": doc_uuid}

    assert _matches_document("doc-troubleshooting", chunk, doc_id_map=doc_id_map) is True
    assert _matches_document("doc-security", chunk, doc_id_map=doc_id_map) is False


def test_matches_document_name():
    doc_uuid = str(uuid.uuid4())
    chunk = ChunkInfo(
        chunk_id="c1",
        document_id=doc_uuid,
        char_offset_start=0,
        char_offset_end=100,
        document_name="troubleshooting.md",
    )
    assert _matches_document("doc-troubleshooting", chunk) is True
    assert _matches_document("doc-security", chunk) is False


def test_matches_document_slug_normalization():
    chunk = ChunkInfo(
        chunk_id="c1",
        document_id="troubleshooting",
        char_offset_start=0,
        char_offset_end=100,
    )
    assert _matches_document("doc-troubleshooting", chunk) is True


def test_build_doc_id_map_from_db():
    from hybrid_rag.models.document import Document

    doc_uuid = uuid.uuid4()
    mock_doc = Document(
        id=doc_uuid,
        filename="troubleshooting.md",
        file_type="md",
        file_hash="hash1",
        size_bytes=100,
        chunking_strategy="fixed",
    )

    mock_db = MagicMock()
    mock_db.query.return_value.all.return_value = [mock_doc]

    id_map = build_doc_id_map(mock_db)
    assert id_map[str(doc_uuid)] == str(doc_uuid)
    assert id_map["troubleshooting.md"] == str(doc_uuid)
    assert id_map["troubleshooting"] == str(doc_uuid)
    assert id_map["doc-troubleshooting"] == str(doc_uuid)


def test_build_doc_id_map_from_list():
    doc_uuid = str(uuid.uuid4())
    docs = [{"id": doc_uuid, "filename": "architecture.md"}]

    id_map = build_doc_id_map(docs)
    assert id_map[doc_uuid] == doc_uuid
    assert id_map["architecture.md"] == doc_uuid
    assert id_map["architecture"] == doc_uuid
    assert id_map["doc-architecture"] == doc_uuid


def test_map_spans_to_chunks_with_uuid_and_doc_id_map():
    doc_uuid = str(uuid.uuid4())
    chunks = [
        ChunkInfo(
            chunk_id="c1",
            document_id=doc_uuid,
            char_offset_start=1000,
            char_offset_end=1300,
            document_name="troubleshooting.md",
        ),
        ChunkInfo(
            chunk_id="c2",
            document_id=doc_uuid,
            char_offset_start=2000,
            char_offset_end=2400,
            document_name="troubleshooting.md",
        ),
    ]

    spans = [SourceSpan(document_id="doc-troubleshooting", start_char=1024, end_char=1280)]

    # Without doc_id_map but with document_name resolution:
    matched = map_spans_to_chunks(spans, chunks)
    assert "c1" in matched
    assert "c2" not in matched

    # With doc_id_map explicitly:
    doc_id_map = {"doc-troubleshooting": doc_uuid}
    matched_mapped = map_spans_to_chunks(spans, chunks, doc_id_map=doc_id_map)
    assert "c1" in matched_mapped


def test_run_single_question_propagates_doc_id_map():
    doc_uuid = str(uuid.uuid4())
    chunks = [
        ChunkInfo(
            chunk_id="c1",
            document_id=doc_uuid,
            char_offset_start=1000,
            char_offset_end=1300,
        )
    ]
    item = GoldenItem(
        id="q1",
        question="How to troubleshoot?",
        expected_answer="Check logs.",
        category="troubleshooting",
        source_spans=[
            SourceSpan(document_id="doc-troubleshooting", start_char=1024, end_char=1280)
        ],
    )
    config = PipelineConfig()

    def retrieve_fn(q, c):
        return ["c1"]

    doc_id_map = {"doc-troubleshooting": doc_uuid}
    result = run_single_question(item, config, retrieve_fn, chunks, k=5, doc_id_map=doc_id_map)

    assert "c1" in result.relevant_ids
    assert result.metrics["precision_at_k"] == pytest.approx(0.2)
    assert result.metrics["recall_at_k"] == pytest.approx(1.0)

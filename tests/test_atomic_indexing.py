"""Atomic Indexing and Semantic Chunker Hardening Test Suite.

Verifies failure injection, rollback atomicity, and embedder session
reuse/memory stability under concurrent load.
"""

from __future__ import annotations

import ctypes
import threading
import tracemalloc
import uuid
from ctypes import wintypes
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hybrid_rag.ingest.chunkers import ChunkOutput, chunk_semantic
from hybrid_rag.ingest.embedder import get_embedder, reset_embedder
from hybrid_rag.ingest.index_manager import (
    delete_document,
    index_document,
)
from hybrid_rag.models.document import Document


# ---------------------------------------------------------------------------
# Windows Process Memory Measurement Helper
# ---------------------------------------------------------------------------
class PROCESS_MEMORY_COUNTERS(ctypes.Structure):  # noqa: N801
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def get_working_set_bytes() -> int:
    try:
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        h_proc = kernel32.GetCurrentProcess()
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        if psapi.GetProcessMemoryInfo(h_proc, ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize
    except Exception:
        pass
    return 0


# ---------------------------------------------------------------------------
# Shared Test Fixtures
# ---------------------------------------------------------------------------


def _make_doc(doc_id=None):
    return Document(
        id=doc_id or uuid.uuid4(),
        filename="adversarial_test.txt",
        file_type="txt",
        file_hash=f"advhash_{uuid.uuid4().hex[:12]}",
        size_bytes=500,
        chunking_strategy="fixed",
    )


def _make_chunks(n=4):
    return [
        ChunkOutput(
            text=(
                f"Adversarial chunk {i} content. "
                "Dense vector retrieval uses approximate nearest neighbor search. "
                "Cross-encoder reranking refines initial retrieval results."
            ),
            chunk_index=i,
            token_count=25,
            page_number=1,
            section_heading="Section Alpha",
            char_offset_start=i * 85,
            char_offset_end=(i + 1) * 85,
        )
        for i in range(n)
    ]


# ===========================================================================
# SECTION 1: Atomic PG+BM25 Writes and Rollback
# ===========================================================================


@patch("hybrid_rag.retrieval.bm25_store.get_bm25_store")
@patch("hybrid_rag.ingest.index_manager.embed_texts")
def test_bm25_failure_compensating_rollback(mock_embed, mock_bm25_getter):
    """Inject failure during BM25 update phase.

    Verify PostgreSQL transaction is rolled back and compensating BM25 rebuild is invoked.
    """
    mock_embed.return_value = [np.zeros(384) for _ in range(3)]
    mock_bm25 = MagicMock()
    mock_bm25.add_chunks.return_value = -1
    mock_bm25.rebuild_from_db.side_effect = [
        RuntimeError("BM25 index lock contention / disk failure"),
        0,
    ]
    mock_bm25_getter.return_value = mock_bm25

    db = MagicMock()

    doc = _make_doc()
    chunks = _make_chunks(3)

    with pytest.raises(RuntimeError, match="BM25 index lock contention"):
        index_document(db, doc, chunks, "fixed")

    db.commit.assert_not_called()
    db.rollback.assert_called_once()

    assert mock_bm25.rebuild_from_db.call_count == 2


@patch("hybrid_rag.retrieval.bm25_store.get_bm25_store")
@patch("hybrid_rag.ingest.index_manager.embed_texts")
def test_db_commit_failure_rollback(mock_embed, mock_bm25_getter):
    """Inject failure during final db.commit() phase.

    Verify rollback and compensating BM25 rebuild.
    """
    mock_embed.return_value = [np.zeros(384) for _ in range(4)]
    mock_bm25 = MagicMock()
    mock_bm25.add_chunks.return_value = 4
    mock_bm25_getter.return_value = mock_bm25

    db = MagicMock()
    db.commit.side_effect = RuntimeError("PostgreSQL transaction serialization failure")

    doc = _make_doc()
    chunks = _make_chunks(4)

    with pytest.raises(RuntimeError, match="PostgreSQL transaction serialization failure"):
        index_document(db, doc, chunks, "fixed")

    assert mock_bm25.rebuild_from_db.call_count >= 1
    db.rollback.assert_called_once()


def test_delete_document_db_error_triggers_rollback():
    """Inject DB error during delete_document; verify rollback is invoked."""
    db = MagicMock()
    doc_id = uuid.uuid4()
    mock_doc = _make_doc(doc_id)
    db.get.return_value = mock_doc
    db.commit.side_effect = RuntimeError("Foreign key constraint violation during delete")

    with pytest.raises(RuntimeError, match="Foreign key constraint violation"):
        delete_document(db, doc_id, "fixed")

    db.rollback.assert_called_once()


def test_delete_document_idempotent_when_doc_absent():
    """Verify delete_document executes safely when document not found in Postgres."""
    db = MagicMock()
    doc_id = uuid.uuid4()
    db.get.return_value = None

    delete_document(db, doc_id, "fixed")

    db.delete.assert_not_called()


# ===========================================================================
# SECTION 2: Semantic Chunker Embedder Session Reuse & Memory
# ===========================================================================


@pytest.fixture(autouse=True)
def clean_embedder():
    """Ensure clean embedder singleton state."""
    reset_embedder()
    yield
    reset_embedder()


def test_semantic_chunker_single_embedder_init_across_10_calls():
    """Call chunk_semantic 10 times consecutively with varying texts.

    Check that fastembed.TextEmbedding.__init__ is called only ONCE.
    """
    import fastembed

    init_calls = 0
    orig_init = fastembed.TextEmbedding.__init__

    def spy_init(self, *args, **kwargs):
        nonlocal init_calls
        init_calls += 1
        orig_init(self, *args, **kwargs)

    # Warm up singleton to observe first init
    embedder_ref = get_embedder()
    assert embedder_ref is not None

    sample_texts = [
        (
            f"This is document paragraph {i}. "
            "Machine learning optimizes complex vector representations across high dimensions. "
            "Semantic boundaries emerge where contextual similarity drops sharply. "
            "Deep neural networks enable dense information retrieval and robust semantic search."
        )
        for i in range(10)
    ]

    with patch.object(fastembed.TextEmbedding, "__init__", spy_init):
        for text in sample_texts:
            chunks = chunk_semantic(text, max_tokens=30)
            assert len(chunks) >= 1
            for c in chunks:
                assert text[c.char_offset_start : c.char_offset_end] == c.text

    assert init_calls == 0, f"Expected 0 new TextEmbedding initializations, observed {init_calls}"


def test_semantic_chunker_multithreaded_concurrency():
    """Concurrently call chunk_semantic across 5 worker threads.

    Assert no race condition, all threads complete, and get_embedder() returns the identical object.
    """
    thread_count = 5
    results: list[list[ChunkOutput]] = [None] * thread_count
    embedder_instances: list[int] = [None] * thread_count
    errors: list[Exception] = []

    test_passages = [
        (
            f"Thread passage {i} sentence alpha. Natural language processing requires embeddings. "
            f"Sentence beta discusses information retrieval and vector databases. "
            f"Sentence gamma covers retrieval-augmented generation architectures."
        )
        for i in range(thread_count)
    ]

    # Pre-warm embedder
    main_embedder = get_embedder()

    def worker(idx: int):
        try:
            emb = get_embedder()
            embedder_instances[idx] = id(emb)
            chunks = chunk_semantic(test_passages[idx], max_tokens=25)
            results[idx] = chunks
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(thread_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert not errors, f"Errors in concurrent semantic chunking: {errors}"
    assert all(r is not None and len(r) >= 1 for r in results)

    for idx, emb_id in enumerate(embedder_instances):
        assert emb_id == id(main_embedder), (
            f"Thread {idx} got different embedder instance {emb_id} != {id(main_embedder)}"
        )


def test_semantic_chunker_memory_stability():
    """Assert process memory remains stable over 15 consecutive semantic chunking runs.

    Ensure no memory leak occurs from accumulating duplicate ONNX sessions or models.
    """
    get_embedder()
    import gc

    gc.collect()

    tracemalloc.start()

    initial_working_set = get_working_set_bytes()
    initial_traced, _ = tracemalloc.get_traced_memory()

    test_text = (
        "Artificial intelligence models process natural language by transforming words. "
        "Dense vectors represent linguistic semantic content in continuous geometric space. "
        "When two concepts have similar meanings, their vector embeddings point closely. "
        "Cosine similarity measures the angle between two embedding vectors. "
        "Chunking strategies partition long documents into smaller segments suitable for context. "
        "Fixed-length chunking splits text strictly by character or token limits. "
        "Structure-aware chunking respects paragraph headers, sections, and markdown boundaries. "
        "Semantic chunking calculates sentence similarities and cuts at transition valleys. "
        "Hybrid search combines dense vector retrieval with sparse lexical BM25 matching. "
        "Reciprocal Rank Fusion merges ranked lists without requiring score normalization."
    )

    for _ in range(15):
        chunks = chunk_semantic(test_text, max_tokens=40)
        assert len(chunks) >= 2

    gc.collect()
    final_traced, peak_traced = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    final_working_set = get_working_set_bytes()

    traced_delta_mb = (final_traced - initial_traced) / (1024 * 1024)
    peak_traced / (1024 * 1024)

    if final_working_set > 0 and initial_working_set > 0:
        working_set_delta_mb = (final_working_set - initial_working_set) / (1024 * 1024)
    else:
        working_set_delta_mb = 0.0

    assert traced_delta_mb < 15.0, f"Excessive heap memory growth: {traced_delta_mb:.2f} MB"
    assert working_set_delta_mb < 50.0, (
        f"Excessive working set growth: {working_set_delta_mb:.2f} MB"
    )


def test_semantic_chunker_custom_embedder_injection():
    """Verify chunk_semantic accepts mock embedder without get_embedder()."""
    mock_emb = MagicMock()
    mock_emb.embed.return_value = [np.ones(384) * 0.1 for _ in range(3)]

    text = "First sentence. Second sentence. Third sentence."
    with patch("hybrid_rag.ingest.embedder.get_embedder") as mock_get_emb:
        chunks = chunk_semantic(text, max_tokens=10, embedder=mock_emb)
        assert len(chunks) >= 1
        mock_get_emb.assert_not_called()
        mock_emb.embed.assert_called_once()


def test_reset_embedder_lifecycle():
    """Verify reset_embedder() resets cached singleton and allows reconfiguration."""
    emb1 = get_embedder()
    assert emb1 is not None

    reset_embedder()

    mock_replacement = MagicMock()
    reset_embedder(mock_replacement)
    assert get_embedder() is mock_replacement

    reset_embedder(None)
    emb2 = get_embedder()
    assert emb2 is not mock_replacement

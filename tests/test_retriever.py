"""Integration tests for retrieval pipeline with pgvector and BM25."""

import uuid
from unittest.mock import MagicMock, patch

from hybrid_rag.models.document import DocumentChunk
from hybrid_rag.retrieval.bm25_store import Bm25Store
from hybrid_rag.retrieval.retriever import (
    dense_retrieve,
    hybrid_retrieve,
    sparse_retrieve,
)
from hybrid_rag.schemas.pipeline import PipelineConfig


def test_dense_retrieve_returns_candidates():
    config = PipelineConfig(chunking_strategy="fixed", dense_top_k=3)

    doc_id = uuid.uuid4()
    chunk1 = DocumentChunk(
        id=uuid.uuid4(),
        document_id=doc_id,
        chunk_index=0,
        content="First chunk",
        token_count=5,
    )
    chunk2 = DocumentChunk(
        id=uuid.uuid4(),
        document_id=doc_id,
        chunk_index=1,
        content="Second chunk",
        token_count=5,
    )

    mock_db = MagicMock(spec=["query"])
    mock_query = mock_db.query.return_value
    mock_filtered = mock_query.join.return_value.filter.return_value
    mock_filtered.order_by.return_value.limit.return_value.all.return_value = [
        (chunk1, 0.05),
        (chunk2, 0.20),
    ]

    with patch("hybrid_rag.ingest.embedder.embed_query") as mock_embed:
        import numpy as np

        mock_embed.return_value = np.zeros(384)
        stage = dense_retrieve("test query", config, mock_db)

    assert stage.stage == "dense"
    assert len(stage.candidates) == 2
    assert stage.candidates[0].chunk_id == str(chunk1.id)
    assert round(stage.candidates[0].score, 2) == 0.95
    assert stage.candidates[0].rank == 1
    assert stage.duration_ms > 0


def test_sparse_retrieve_returns_candidates():
    config = PipelineConfig(chunking_strategy="fixed", sparse_top_k=3)
    store = Bm25Store()
    store.rebuild(
        texts=["machine learning algorithms", "database indexing", "neural networks"],
        chunk_ids=["c1", "c2", "c3"],
        strategy="fixed",
    )

    stage = sparse_retrieve("machine learning", config, store)

    assert stage.stage == "sparse"
    assert len(stage.candidates) > 0
    assert stage.candidates[0].rank == 1
    assert stage.duration_ms > 0
    assert all(c.metadata.get("source") == "sparse" for c in stage.candidates)


def test_sparse_retrieve_no_index():
    config = PipelineConfig(chunking_strategy="fixed", sparse_top_k=3)
    store = Bm25Store()
    stage = sparse_retrieve("anything", config, store)
    assert stage.stage == "sparse"
    assert len(stage.candidates) == 0


def test_hybrid_retrieve_populates_trace():
    config = PipelineConfig(
        chunking_strategy="fixed", dense_top_k=3, sparse_top_k=3, fusion_top_k=5
    )

    doc_id = uuid.uuid4()
    chunk1 = DocumentChunk(
        id=uuid.uuid4(),
        document_id=doc_id,
        chunk_index=0,
        content="Dense match content",
        token_count=5,
    )
    mock_db = MagicMock(spec=["query"])
    mock_query = mock_db.query.return_value
    mock_filtered = mock_query.join.return_value.filter.return_value
    mock_filtered.order_by.return_value.limit.return_value.all.return_value = [
        (chunk1, 0.1),
    ]

    store = Bm25Store()
    store.rebuild(
        texts=["some text content"],
        chunk_ids=["c2"],
        strategy="fixed",
    )

    with patch("hybrid_rag.ingest.embedder.embed_query") as mock_embed:
        import numpy as np

        mock_embed.return_value = np.zeros(384)
        trace = hybrid_retrieve("test query", config, mock_db, store)

    assert trace.query == "test query"
    assert len(trace.stages) == 3
    assert trace.stages[0].stage == "dense"
    assert trace.stages[1].stage == "sparse"
    assert trace.stages[2].stage == "rrf"
    assert trace.total_duration_ms > 0

    dense_latency = trace.stages[0].metadata.get("dense_latency_ms")
    sparse_latency = trace.stages[1].metadata.get("sparse_latency_ms")
    rrf_latency = trace.stages[2].metadata.get("rrf_latency_ms")
    assert dense_latency is not None and dense_latency > 0
    assert sparse_latency is not None and sparse_latency > 0
    assert rrf_latency is not None and rrf_latency >= 0


def test_bm25_store_rebuild_and_search():
    store = Bm25Store()
    count = store.rebuild(
        texts=[
            "PostgreSQL database configuration",
            "Python machine learning tutorial",
            "Docker container deployment",
        ],
        chunk_ids=["c1", "c2", "c3"],
        strategy="fixed",
    )

    assert count == 3
    assert store.has_index("fixed")
    assert not store.has_index("semantic")

    results = store.search("PostgreSQL database", "fixed", top_k=3)
    assert len(results) > 0
    assert results[0][0] == "c1"
    assert results[0][1] > 0


def test_bm25_store_clear():
    store = Bm25Store()
    store.rebuild(["text"], ["c1"], "fixed")
    assert store.has_index("fixed")
    store.clear("fixed")
    assert not store.has_index("fixed")


def test_bm25_store_empty_rebuild():
    store = Bm25Store()
    store.rebuild(["text"], ["c1"], "fixed")
    assert store.has_index("fixed")
    count = store.rebuild([], [], "fixed")
    assert count == 0
    assert not store.has_index("fixed")


def test_bm25_store_technical_tokens():
    store = Bm25Store()
    store.rebuild(
        texts=["Fix DB-204 error in config.json"],
        chunk_ids=["c1"],
        strategy="fixed",
    )
    results = store.search("DB-204", "fixed", top_k=1)
    assert len(results) == 1
    assert results[0][0] == "c1"


def test_dense_retrieve_pgvector_session():
    import uuid

    from hybrid_rag.models.document import DocumentChunk

    mock_db = MagicMock(spec=["query"])
    doc_id = uuid.uuid4()
    chunk1 = DocumentChunk(
        id=uuid.uuid4(),
        document_id=doc_id,
        chunk_index=0,
        content="Postgres pgvector test content",
        token_count=10,
    )
    mock_query = mock_db.query.return_value
    mock_filtered = mock_query.join.return_value.filter.return_value
    mock_filtered.order_by.return_value.limit.return_value.all.return_value = [
        (chunk1, 0.15),
    ]

    config = PipelineConfig(chunking_strategy="fixed", dense_top_k=5)
    with patch("hybrid_rag.ingest.embedder.embed_query") as mock_embed:
        import numpy as np

        mock_embed.return_value = np.zeros(384)
        stage = dense_retrieve("pgvector search", config, mock_db)

    assert stage.stage == "dense"
    assert len(stage.candidates) == 1
    assert stage.candidates[0].chunk_id == str(chunk1.id)
    assert round(stage.candidates[0].score, 2) == 0.85
    assert stage.candidates[0].content == "Postgres pgvector test content"
    assert stage.candidates[0].metadata["document_id"] == str(doc_id)


def test_bm25_store_rebuild_from_db():
    import uuid

    from hybrid_rag.models.document import DocumentChunk

    mock_db = MagicMock()
    c1 = DocumentChunk(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        chunk_index=0,
        content="BM25 test chunk content",
        token_count=5,
    )
    mock_chain = mock_db.query.return_value.join.return_value.filter.return_value
    mock_chain.order_by.return_value.all.return_value = [c1]

    store = Bm25Store()
    count = store.rebuild_from_db(mock_db, "fixed")
    assert count == 1
    assert store.has_index("fixed")
    results = store.search("BM25 test", "fixed", top_k=1)
    assert len(results) == 1
    assert results[0][0] == str(c1.id)


def test_hydrate_candidates_populates_content_and_document_id():
    import uuid

    from hybrid_rag.models.document import DocumentChunk
    from hybrid_rag.retrieval.retriever import hydrate_candidates
    from hybrid_rag.schemas.pipeline import CandidateResult

    doc_id = uuid.uuid4()
    chunk_id = uuid.uuid4()
    chunk = DocumentChunk(
        id=chunk_id,
        document_id=doc_id,
        chunk_index=0,
        content="Hydrated passage content text.",
    )

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.all.return_value = [chunk]

    candidates = [
        CandidateResult(chunk_id=str(chunk_id), score=0.8, rank=1, content="", metadata={})
    ]

    hydrate_candidates(candidates, mock_db)

    assert candidates[0].content == "Hydrated passage content text."
    assert candidates[0].metadata["document_id"] == str(doc_id)


def test_full_retrieve_hydrates_without_rerank():
    import uuid

    from hybrid_rag.models.document import DocumentChunk
    from hybrid_rag.retrieval.retriever import full_retrieve
    from hybrid_rag.schemas.pipeline import CandidateResult, PipelineConfig, QueryTrace, StageTrace

    chunk_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    chunk = DocumentChunk(
        id=chunk_id,
        document_id=doc_id,
        content="Passage without rerank.",
    )

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.all.return_value = [chunk]

    config = PipelineConfig(rerank_enabled=False)
    fake_trace = QueryTrace(query="test", config=config)
    fake_trace.add_stage(
        StageTrace(
            stage="rrf",
            candidates=[
                CandidateResult(
                    chunk_id=str(chunk_id), score=0.016, rank=1, content="", metadata={}
                )
            ],
            duration_ms=1.0,
        )
    )

    with patch("hybrid_rag.retrieval.retriever.hybrid_retrieve", return_value=fake_trace):
        result_trace = full_retrieve("test", config, db=mock_db)

    candidates = result_trace.stages[-1].candidates
    assert len(candidates) == 1
    assert candidates[0].content == "Passage without rerank."
    assert candidates[0].metadata["document_id"] == str(doc_id)


def test_full_retrieve_fallback_session():
    import uuid

    from hybrid_rag.models.document import DocumentChunk
    from hybrid_rag.retrieval.retriever import full_retrieve
    from hybrid_rag.schemas.pipeline import CandidateResult, PipelineConfig, QueryTrace, StageTrace

    chunk_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    chunk = DocumentChunk(
        id=chunk_id,
        document_id=doc_id,
        content="Fallback session passage text.",
    )

    mock_fallback_session = MagicMock()
    mock_fallback_session.query.return_value.filter.return_value.all.return_value = [chunk]

    mock_session_maker = MagicMock()
    mock_session_maker.return_value.__enter__.return_value = mock_fallback_session

    config = PipelineConfig(rerank_enabled=False)
    fake_trace = QueryTrace(query="test", config=config)
    fake_trace.add_stage(
        StageTrace(
            stage="rrf",
            candidates=[
                CandidateResult(
                    chunk_id=str(chunk_id), score=0.016, rank=1, content="", metadata={}
                )
            ],
            duration_ms=1.0,
        )
    )

    with (
        patch("hybrid_rag.retrieval.retriever.hybrid_retrieve", return_value=fake_trace),
        patch("hybrid_rag.db.engine.SessionLocal", mock_session_maker),
    ):
        result_trace = full_retrieve("test", config)

    candidates = result_trace.stages[-1].candidates
    assert len(candidates) == 1
    assert candidates[0].content == "Fallback session passage text."
    assert candidates[0].metadata["document_id"] == str(doc_id)

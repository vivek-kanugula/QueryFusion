"""Tests for scoped document filtering across dense, sparse, and hybrid retrieval."""

import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hybrid_rag.api.query import router
from hybrid_rag.models.document import DocumentChunk
from hybrid_rag.retrieval.bm25_store import Bm25Store
from hybrid_rag.retrieval.retriever import dense_retrieve, full_retrieve, sparse_retrieve
from hybrid_rag.schemas.pipeline import CandidateResult, PipelineConfig, QueryTrace, StageTrace


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_dense_retrieve_filters_by_document_ids():
    doc1_id = uuid.uuid4()

    config = PipelineConfig(
        chunking_strategy="fixed",
        dense_top_k=3,
        document_ids=(str(doc1_id),),
    )

    chunk1 = DocumentChunk(
        id=uuid.uuid4(),
        document_id=doc1_id,
        chunk_index=0,
        content="Document 1 content",
        token_count=5,
    )

    mock_db = MagicMock(spec=["query"])
    mock_query = mock_db.query.return_value
    mock_filtered = mock_query.join.return_value.filter.return_value
    mock_filtered.order_by.return_value.limit.return_value.all.return_value = [
        (chunk1, 0.10),
    ]

    with patch("hybrid_rag.ingest.embedder.embed_query") as mock_embed:
        import numpy as np

        mock_embed.return_value = np.zeros(384)
        stage = dense_retrieve("test query", config, mock_db)

    assert stage.stage == "dense"
    assert len(stage.candidates) == 1
    assert stage.candidates[0].chunk_id == str(chunk1.id)
    assert stage.candidates[0].metadata["document_id"] == str(doc1_id)


def test_dense_retrieve_returns_empty_on_invalid_uuids():
    config = PipelineConfig(
        chunking_strategy="fixed",
        dense_top_k=3,
        document_ids=("invalid-uuid-string",),
    )

    mock_db = MagicMock(spec=["query"])

    with patch("hybrid_rag.ingest.embedder.embed_query") as mock_embed:
        import numpy as np

        mock_embed.return_value = np.zeros(384)
        stage = dense_retrieve("test query", config, mock_db)

    assert len(stage.candidates) == 0


def test_bm25_store_filters_by_document_ids():
    doc_a = str(uuid.uuid4())
    doc_b = str(uuid.uuid4())
    doc_c = str(uuid.uuid4())

    store = Bm25Store()
    store.rebuild(
        texts=[
            "machine learning deep neural networks",
            "relational database postgresql indexes",
            "machine learning statistical foundations",
        ],
        chunk_ids=["chunk_a1", "chunk_b1", "chunk_a2"],
        strategy="fixed",
        doc_ids=[doc_a, doc_b, doc_a],
    )

    # 1. Query with no document filter: finds matches across doc A
    results_all = store.search("machine learning", strategy="fixed")
    assert len(results_all) == 2
    assert {r[0] for r in results_all} == {"chunk_a1", "chunk_a2"}

    # 2. Query scoped to doc B only: no match because doc B is about databases
    results_b = store.search(
        "machine learning",
        strategy="fixed",
        document_ids=[doc_b],
    )
    assert len(results_b) == 0

    # 3. Query scoped to doc A only: finds both chunks from doc A
    results_a = store.search(
        "machine learning",
        strategy="fixed",
        document_ids=[doc_a],
    )
    assert len(results_a) == 2
    assert {r[0] for r in results_a} == {"chunk_a1", "chunk_a2"}

    # 4. Multi-document scope: doc A and doc C
    results_ac = store.search(
        "machine learning",
        strategy="fixed",
        document_ids=[doc_a, doc_c],
    )
    assert len(results_ac) == 2


def test_sparse_retrieve_honors_pipeline_config_document_ids():
    doc_a = str(uuid.uuid4())
    doc_b = str(uuid.uuid4())

    store = Bm25Store()
    store.rebuild(
        texts=["quantum computing qubits", "distributed systems consensus"],
        chunk_ids=["c_quantum", "c_distrib"],
        strategy="fixed",
        doc_ids=[doc_a, doc_b],
    )

    config_a = PipelineConfig(
        chunking_strategy="fixed",
        sparse_top_k=5,
        document_ids=(doc_a,),
    )
    stage_a = sparse_retrieve("quantum", config_a, store)
    assert len(stage_a.candidates) == 1
    assert stage_a.candidates[0].chunk_id == "c_quantum"

    # Query for quantum but scoped to doc_b -> 0 candidates
    config_b = PipelineConfig(
        chunking_strategy="fixed",
        sparse_top_k=5,
        document_ids=(doc_b,),
    )
    stage_b = sparse_retrieve("quantum", config_b, store)
    assert len(stage_b.candidates) == 0


def test_full_retrieve_defense_in_depth_filters_unselected_documents():
    doc_selected = str(uuid.uuid4())
    doc_other = str(uuid.uuid4())

    config = PipelineConfig(
        chunking_strategy="fixed",
        rerank_enabled=False,
        document_ids=(doc_selected,),
    )

    # Mock hybrid_retrieve to simulate a chunk leaking from an unselected document
    mock_hybrid_trace = QueryTrace(query="test", config=config)
    mock_hybrid_trace.add_stage(
        StageTrace(
            stage="rrf",
            candidates=[
                CandidateResult(
                    chunk_id="chunk_allowed",
                    score=0.9,
                    rank=1,
                    content="Allowed doc text",
                    metadata={"document_id": doc_selected},
                ),
                CandidateResult(
                    chunk_id="chunk_leaked",
                    score=0.8,
                    rank=2,
                    content="Leaked doc text",
                    metadata={"document_id": doc_other},
                ),
            ],
        )
    )

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.all.return_value = []

    with patch("hybrid_rag.retrieval.retriever.hybrid_retrieve", return_value=mock_hybrid_trace):
        trace = full_retrieve("test", config, db=mock_db)

    # Only chunk_allowed should remain in the rrf stage candidates
    final_candidates = trace.stages[-1].candidates
    assert len(final_candidates) == 1
    assert final_candidates[0].chunk_id == "chunk_allowed"
    assert final_candidates[0].metadata["document_id"] == doc_selected


def test_stream_query_passes_document_ids_to_pipeline_config(client):
    mock_trace = QueryTrace(query="What is AI?")
    mock_trace.add_stage(StageTrace(stage="rrf", candidates=[]))

    mock_gen = MagicMock()
    mock_gen.answer.answer = "AI answer"
    mock_gen.answer.citations = []
    mock_gen.answer.has_sufficient_context = True
    mock_gen.answer.fallback_reason = None
    mock_gen.verification = None
    mock_gen.confidence = None
    mock_gen.model = "test-model"
    mock_gen.cached = False

    target_doc_id = str(uuid.uuid4())

    with (
        patch("hybrid_rag.retrieval.retriever.full_retrieve", return_value=mock_trace) as mock_full,
        patch("hybrid_rag.generation.pipeline.generate_grounded_answer", return_value=mock_gen),
        patch("hybrid_rag.retrieval.bm25_store.get_bm25_store", return_value=MagicMock()),
        patch("hybrid_rag.core.trace_store.trace_store.save", MagicMock()),
    ):
        resp = client.post(
            "/api/query/stream",
            json={
                "question": "What is AI?",
                "document_ids": [target_doc_id],
            },
        )
        assert resp.status_code == 200
        assert mock_full.called
        called_config = mock_full.call_args[0][1]
        assert called_config.document_ids == (target_doc_id,)

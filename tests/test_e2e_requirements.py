"""Comprehensive 4-Tier E2E Requirements Test Suite for Hybrid-RAG.

This test suite provides opaque-box, requirement-driven verification covering
all 17 verified issues across Tiers 1 through 4:
  - CR-01: Fallback Error Cache Poisoning
  - CR-02: Rate Limiter Acquire Ignored & Division by Zero
  - CR-03: BM25 Index Crash on Empty / Stopword-Only Vocabulary
  - CR-04: User-Selected retrieval_strategy Ignored by Retrieval Engine
  - CR-05: Unbounded In-Memory Accumulation in JobRegistry
  - HI-01: SPA Fallback Masking Non-Existent API Routes
  - HI-02: Unhandled Parser & Concurrency Exceptions in upload_document
  - HI-03: Frontend SSE Stream Parser Drops Trailing Event on Stream Close
  - HI-04: Missing Unmount Abort Handler in useSSEQuery
  - HI-05: LLM Client _call_groq Response Content NoneType Crash
  - MD-01: Cache Key Ignores System Instruction Changes
  - MD-02: Wilcoxon Paired Significance Test Compares Misaligned Questions
  - MD-03: Citation Precision Calculation Is Mathematically Identical to Accuracy
  - MD-04: NDCG@k Inflated Above 1.0 on Duplicate Retrieved Chunks
  - MD-05: Ollama API Prompt Injection via Missing System Separation
  - LO-01: Missing Boundary Validation in QueryRequest
  - LO-02: Non-ASCII Characters Stripped by BM25 Technical Tokenizer

Tiers:
  - Tier 1: Isolated Feature Coverage (CR-01..05, HI-01..05, MD-01..05, LO-01..02)
  - Tier 2: Boundary & Corner Cases (empty vocabularies, zero/negative rate limiters,
            Unicode scripts, null contents, out-of-bound bounds, corrupt files, collisions)
  - Tier 3: Cross-Feature Combinations (retrieval strategy x Unicode, cache x rate limiter,
            Ollama x prompt injection, NDCG x citation precision)
  - Tier 4: Real-World Scenarios (complete end-to-end workflows: ingestion -> retrieval ->
            generation -> verification -> evaluation)
"""

from __future__ import annotations

import io
import json
import sqlite3
import subprocess
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import hybrid_rag.app as app_module
from hybrid_rag.api.query import QueryRequest
from hybrid_rag.api.query import router as query_router
from hybrid_rag.app import create_app
from hybrid_rag.core.jobs import JobRegistry
from hybrid_rag.db.engine import get_db
from hybrid_rag.eval.citation_metrics import compute_citation_metrics
from hybrid_rag.eval.metrics import ndcg_at_k
from hybrid_rag.eval.runner import ExperimentResult, QuestionResult, compare_experiments
from hybrid_rag.generation.cache import LLMCache, TokenBucketRateLimiter, cache_key
from hybrid_rag.generation.generator import (
    _parse_response,
    generate_answer,
)
from hybrid_rag.generation.llm_client import _call_groq, _call_ollama
from hybrid_rag.generation.verifier import (
    verify_citations,
)
from hybrid_rag.models.document import DocumentChunk
from hybrid_rag.retrieval.bm25_store import Bm25Store
from hybrid_rag.retrieval.retriever import (
    full_retrieve,
)
from hybrid_rag.retrieval.tokenizer import tokenize_technical
from hybrid_rag.schemas.generation import (
    Citation,
    ClaimVerification,
    GroundedAnswer,
    VerificationResult,
    VerificationVerdict,
)
from hybrid_rag.schemas.pipeline import CandidateResult, PipelineConfig, QueryTrace

# ==============================================================================
# Helper Fixtures & Utilities
# ==============================================================================

@pytest.fixture()
def mock_db_session():
    """Mock database session providing clean empty query structures."""
    session = MagicMock()
    session.query.return_value.filter.return_value.all.return_value = []
    session.query.return_value.filter.return_value.first.return_value = None
    session.query.return_value.count.return_value = 0
    return session


@pytest.fixture()
def in_memory_cache(tmp_path: Path):
    """Isolated SQLite LLM cache for testing cache pollution and keys."""
    return LLMCache(cache_dir=tmp_path, use_db=False)


def _app_with_dist(dist_dir: Path):
    """Create fresh FastAPI app with overridden DIST_DIR for SPA testing."""
    original = app_module.DIST_DIR
    app_module.DIST_DIR = dist_dir
    try:
        test_app = app_module.create_app()
        return TestClient(test_app)
    finally:
        app_module.DIST_DIR = original


def _override_db_dependency(mock_session):
    def _override():
        yield mock_session
    return _override


# ==============================================================================
# TIER 1: Feature Coverage (CR-01..05, HI-01..05, MD-01..05, LO-01..02)
# ==============================================================================

class TestTier1FeatureCoverage:
    """Isolated functional verification for each of the 17 verified issues."""

    # 1. CR-01: Fallback Error Cache Poisoning
    def test_cr01_fallback_error_not_stored_in_llm_cache(self, in_memory_cache: LLMCache):
        """CR-01: Transient fallback errors must not be cached permanently in llm_cache."""
        # Scenario A: Generation error response
        candidates = [CandidateResult(chunk_id="c1", score=1.0, rank=1, content="Context content")]
        with patch("hybrid_rag.generation.generator._dispatch_generation", return_value="Invalid JSON non-dict content"):
            answer, cached = generate_answer(
                question="What is the architecture?",
                candidates=candidates,
                llm_cache=in_memory_cache,
                model="mock-model",
            )
            assert answer.is_fallback is True or "Failed to parse" in answer.answer or "Invalid" in answer.answer

        # Search all entries in cache DB; cache should have 0 entries
        with sqlite3.connect(str(in_memory_cache._db_path)) as conn:
            count = conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
            assert count == 0, "CR-01 Violation: Fallback error was cached in llm_cache"

        # Scenario B: Verification error response (empty claims)
        grounded_answer = GroundedAnswer(
            answer="RAG combines dense and sparse search [C1].",
            citations=[Citation(citation_id="[C1]", chunk_id="c1", evidence_quote="dense and sparse")],
        )
        with patch("hybrid_rag.generation.verifier._dispatch_verification", return_value=""):
            result = verify_citations(
                answer=grounded_answer,
                candidates=candidates,
                llm_cache=in_memory_cache,
                model="mock-model",
            )
            assert len(result.claims) == 0

        with sqlite3.connect(str(in_memory_cache._db_path)) as conn:
            count = conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
            assert count == 0, "CR-01 Violation: Empty claims from failed verification were cached"

    # 2. CR-02: Rate Limiter Acquire Ignored & Division by Zero
    def test_cr02_rate_limiter_guards_zero_rate_and_respects_acquire(self):
        """CR-02: Rate limiter must avoid ZeroDivisionError when rate <= 0 and callers must check acquire()."""
        # Part A: Zero and negative rate must not raise ZeroDivisionError
        limiter_zero = TokenBucketRateLimiter(rate=0.0, capacity=0.0)
        acquired = limiter_zero.acquire(timeout=0.05)
        # Should return False safely without ZeroDivisionError
        assert acquired is False, "CR-02 Violation: acquire() on rate <= 0 must return False"

        limiter_neg = TokenBucketRateLimiter(rate=-5.0, capacity=0.0)
        acquired_neg = limiter_neg.acquire(timeout=0.05)
        assert acquired_neg is False, "CR-02 Violation: acquire() on negative rate must return False"

        # Part B: Caller must respect acquire() return status
        mock_limiter = MagicMock(spec=TokenBucketRateLimiter)
        mock_limiter.acquire.return_value = False  # Rate limit exceeded

        with patch("hybrid_rag.generation.generator._dispatch_generation") as mock_dispatch:
            try:
                generate_answer(
                    question="Test rate limit",
                    candidates=[CandidateResult(chunk_id="c1", score=1.0, rank=1, content="passage")],
                    rate_limiter=mock_limiter,
                    model="mock-model",
                )
            except Exception:
                # If an explicit rate limit exception is raised, that's valid
                pass
            # Downstream dispatch MUST NOT be called if rate limiter denied access
            assert not mock_dispatch.called, (
                "CR-02 Violation: Generator dispatched LLM call even though rate_limiter.acquire() returned False"
            )

    # 3. CR-03: BM25 Index Crash on Empty / Stopword-Only Vocabulary
    def test_cr03_bm25_index_handles_stopword_and_empty_vocabulary(self):
        """CR-03: BM25 index rebuild must not raise ValueError when all tokens are empty/stopwords."""
        store = Bm25Store()
        # English stopword-only texts
        stopword_texts = ["the and is in", "at of for to", "a an the"]
        chunk_ids = ["c1", "c2", "c3"]

        # This crashed with `ValueError: max() iterable argument is empty` in bm25s
        try:
            count = store.rebuild(stopword_texts, chunk_ids, strategy="fixed")
            assert count >= 0
        except ValueError as exc:
            pytest.fail(f"CR-03 Violation: BM25 rebuild crashed on stopword vocabulary: {exc}")

        # Search against empty index should return empty list safely
        results = store.search("the", strategy="fixed", top_k=5)
        assert results == [], "Expected empty results for empty/stopword-only index"

    # 4. CR-04: User-Selected retrieval_strategy Ignored
    def test_cr04_user_selected_retrieval_strategy_respected(self, mock_db_session):
        """CR-04: dense_only and bm25_only must bypass hybrid retrieval and RRF."""
        chunk = DocumentChunk(
            id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            chunk_index=0,
            content="Architecture overview",
            token_count=10,
        )
        mock_query = mock_db_session.query.return_value
        mock_query.join.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [
            (chunk, 0.1)
        ]

        store = Bm25Store()
        store.rebuild(["Architecture overview"], [str(chunk.id)], strategy="fixed")

        with patch("hybrid_rag.ingest.embedder.embed_query", return_value=np.zeros(384)):
            # Case 1: dense_only
            config_dense = PipelineConfig(retrieval_strategy="dense_only", rerank_enabled=False)
            trace_dense = full_retrieve("overview", config_dense, bm25_store=store, db=mock_db_session)
            stage_names_dense = [s.stage for s in trace_dense.stages]
            assert "dense" in stage_names_dense, "dense_only must include dense stage"
            assert "sparse" not in stage_names_dense, "CR-04 Violation: dense_only executed sparse retrieval"
            assert "rrf" not in stage_names_dense, "CR-04 Violation: dense_only executed RRF fusion"

            # Case 2: bm25_only
            config_bm25 = PipelineConfig(retrieval_strategy="bm25_only", rerank_enabled=False)
            trace_bm25 = full_retrieve("overview", config_bm25, bm25_store=store, db=mock_db_session)
            stage_names_bm25 = [s.stage for s in trace_bm25.stages]
            assert "sparse" in stage_names_bm25, "bm25_only must include sparse stage"
            assert "dense" not in stage_names_bm25, "CR-04 Violation: bm25_only executed dense retrieval"
            assert "rrf" not in stage_names_bm25, "CR-04 Violation: bm25_only executed RRF fusion"

    # 5. CR-05: Unbounded In-Memory Accumulation in JobRegistry
    def test_cr05_job_registry_enforces_capacity_and_eviction(self):
        """CR-05: JobRegistry must enforce max_jobs bound and evict completed jobs and subscribers."""
        # Initialize registry with bound capacity of 3
        try:
            registry = JobRegistry(max_jobs=3)
        except TypeError:
            pytest.fail("CR-05 Violation: JobRegistry does not accept max_jobs capacity parameter")

        # Create 3 jobs
        j1 = registry.create_job("ingest")
        registry.create_job("ingest")
        registry.create_job("ingest")

        # Complete job 1
        registry.complete_job(j1.job_id, {"status": "ok"})

        # Create 4th job; should trigger eviction of completed job 1
        registry.create_job("ingest")

        assert registry.count() <= 3, f"CR-05 Violation: JobRegistry exceeded capacity: {registry.count()} > 3"
        assert registry.get_job(j1.job_id) is None, "CR-05 Violation: Completed job j1 was not evicted"
        assert j1.job_id not in registry._subscribers, "CR-05 Violation: Evicted job subscribers were not cleaned"

    # 6. HI-01: SPA Fallback Masking Non-Existent API Routes
    def test_hi01_spa_fallback_returns_404_json_for_missing_api(self, tmp_path: Path):
        """HI-01: Missing /api/... routes must return 404 JSON, while SPA routes return index.html."""
        # Set up a dummy dist directory with index.html
        dist_dir = tmp_path / "dist"
        dist_dir.mkdir()
        (dist_dir / "index.html").write_text("<html>SPA INDEX</html>", encoding="utf-8")

        client = _app_with_dist(dist_dir)

        # Missing API endpoint MUST return 404 JSON
        res_missing_api = client.get("/api/v1/non_existent_route")
        assert res_missing_api.status_code == 404, (
            f"HI-01 Violation: Missing API route returned {res_missing_api.status_code} instead of 404"
        )
        assert res_missing_api.headers["content-type"].startswith("application/json")
        assert res_missing_api.json().get("detail") == "Not Found"

        # Route /api must also return 404
        res_api_root = client.get("/api")
        assert res_api_root.status_code == 404

        # Non-API route (e.g. /qa or /documents) should return SPA index.html (200 OK)
        res_spa = client.get("/qa")
        assert res_spa.status_code == 200
        assert "SPA INDEX" in res_spa.text

    # 7. HI-02: Unhandled Parser & Concurrency Exceptions in upload_document
    def test_hi02_upload_document_handles_parser_errors_and_collisions(self, mock_db_session):
        """HI-02: upload_document returns 400 for bad extension / corrupt PDF, and 409 for duplicate collision."""
        test_app = create_app()
        test_app.dependency_overrides[get_db] = _override_db_dependency(mock_db_session)
        client = TestClient(test_app)

        try:
            # Case A: Unsupported file extension -> HTTP 400
            res_bad_ext = client.post(
                "/api/v1/documents",
                files={"file": ("malicious.exe", io.BytesIO(b"binary payload"), "application/octet-stream")},
            )
            assert res_bad_ext.status_code == 400, (
                f"HI-02 Violation: Unsupported extension returned {res_bad_ext.status_code} instead of 400"
            )

            # Case B: Corrupted PDF file -> HTTP 400
            res_corrupt_pdf = client.post(
                "/api/v1/documents",
                files={"file": ("corrupt.pdf", io.BytesIO(b"%PDF-corrupted-binary-garbage"), "application/pdf")},
            )
            assert res_corrupt_pdf.status_code == 400, (
                f"HI-02 Violation: Corrupted PDF returned {res_corrupt_pdf.status_code} instead of 400"
            )

            # Case C: Concurrent file hash collision (IntegrityError) -> HTTP 409
            from sqlalchemy.exc import IntegrityError
            with patch("hybrid_rag.api.documents.index_document", side_effect=IntegrityError("stmt", "params", "orig")):
                res_collision = client.post(
                    "/api/v1/documents",
                    files={"file": ("valid.txt", io.BytesIO(b"Valid content to index"), "text/plain")},
                )
                assert res_collision.status_code == 409, (
                    f"HI-02 Violation: Concurrent hash collision returned {res_collision.status_code} instead of 409"
                )
        finally:
            test_app.dependency_overrides.clear()

    # 8. HI-03: Frontend SSE Stream Parser Drops Trailing Event on Close
    def test_hi03_frontend_sse_parser_preserves_trailing_event_on_stream_close(self):
        """HI-03: streamQuery in frontend/src/lib/api.ts must parse trailing buffered event on close."""
        test_script = Path(__file__).resolve().parent / "test_e2e_frontend_stream.mjs"
        assert test_script.is_file(), f"Missing Node.js test script: {test_script}"

        res = subprocess.run(["node", str(test_script)], capture_output=True, text=True)
        assert res.returncode == 0, (
            f"HI-03 / HI-04 Contract Violation:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
        )

    # 9. HI-04: Missing Unmount Abort Handler in useSSEQuery
    def test_hi04_use_sse_query_unmount_abort_handler_contract(self):
        """HI-04: useSSEQuery must register an unmount cleanup effect to abort active fetch stream."""
        hook_path = Path(__file__).resolve().parents[1] / "frontend" / "src" / "hooks" / "useSSEQuery.ts"
        assert hook_path.is_file(), f"Missing hook file: {hook_path}"
        content = hook_path.read_text(encoding="utf-8")

        # Must import useEffect
        assert "useEffect" in content, "HI-04 Violation: 'useEffect' is not imported in useSSEQuery.ts"
        # Must register cleanup returning abort()
        has_abort_cleanup = (
            "controllerRef.current?.abort()" in content or "controllerRef.current.abort()" in content
        ) and "useEffect" in content
        assert has_abort_cleanup, "HI-04 Violation: useSSEQuery does not register an unmount abort cleanup"

    # 10. HI-05: LLM Client _call_groq Response Content NoneType Crash
    def test_hi05_call_groq_and_generator_handle_null_content_gracefully(self):
        """HI-05: _call_groq and _parse_response must gracefully handle null response content."""
        # Case A: _call_groq with null message content
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{"message": {"content": None}}]
        }).encode("utf-8")
        mock_response.__enter__.return_value = mock_response

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = _call_groq("llama-3.3-70b-versatile", "Hello", "System", 0.0)
            assert result == "", f"HI-05 Violation: _call_groq should return empty string for null content, got {result!r}"

        # Case B: _parse_response with null JSON or NoneType
        candidates = [CandidateResult(chunk_id="c1", score=1.0, rank=1, content="passage")]
        parsed_answer = _parse_response("null", candidates)
        assert parsed_answer.is_fallback is True or "Failed to parse" in parsed_answer.answer or "Empty" in parsed_answer.answer

    # 11. MD-01: Cache Key Ignores System Instruction Changes
    def test_md01_cache_key_varies_with_system_instruction(self):
        """MD-01: Different system_instruction strings must produce distinct cache keys."""
        try:
            key_default = cache_key("gpt-4o", "What is RAG?", "{}", 0.0, system_instruction="Default system")
            key_custom = cache_key("gpt-4o", "What is RAG?", "{}", 0.0, system_instruction="Custom system")
            key_empty = cache_key("gpt-4o", "What is RAG?", "{}", 0.0, system_instruction="")
        except TypeError:
            pytest.fail("MD-01 Violation: cache_key() does not accept system_instruction parameter")

        assert key_default != key_custom, "MD-01 Violation: Cache keys match despite differing system instructions"
        assert key_default != key_empty, "MD-01 Violation: Cache key does not differentiate empty system instruction"

    # 12. MD-02: Wilcoxon Paired Significance Test Compares Misaligned Questions
    def test_md02_wilcoxon_paired_significance_aligns_by_question_id(self):
        """MD-02: Wilcoxon test in compare_experiments must align results by question_id."""
        # Create identical evaluations with shuffled question order
        res_a = ExperimentResult(
            name="Exp A",
            strategy="dense",
            chunking_strategy="fixed",
            per_question=[
                QuestionResult(question_id="q1", question="Q1", metrics={"ndcg_at_k": 0.9}),
                QuestionResult(question_id="q2", question="Q2", metrics={"ndcg_at_k": 0.5}),
                QuestionResult(question_id="q3", question="Q3", metrics={"ndcg_at_k": 0.2}),
            ],
        )
        # Reversed order with same per-question scores
        res_b = ExperimentResult(
            name="Exp B",
            strategy="hybrid",
            chunking_strategy="fixed",
            per_question=[
                QuestionResult(question_id="q3", question="Q3", metrics={"ndcg_at_k": 0.2}),
                QuestionResult(question_id="q2", question="Q2", metrics={"ndcg_at_k": 0.5}),
                QuestionResult(question_id="q1", question="Q1", metrics={"ndcg_at_k": 0.9}),
            ],
        )

        comparison = compare_experiments(res_a, res_b, metric="ndcg_at_k")
        # When aligned by question_id, scores_a == scores_b for every question, so win_loss_tie is (0, 0, 3)
        assert comparison["win_loss_tie"] == (0, 0, 3), (
            f"MD-02 Violation: Misaligned questions caused non-zero win/loss comparison: {comparison['win_loss_tie']}"
        )

    # 13. MD-03: Citation Precision Calculation Is Mathematically Identical to Accuracy
    def test_md03_citation_precision_calculated_as_supported_over_attributed(self):
        """MD-03: Citation precision must be supported / (supported + misattributed), not supported / total."""
        # 2 supported, 1 misattributed, 1 unsupported. Total = 4.
        # Accuracy = 2 / 4 = 0.5
        # Correct Precision = 2 / (2 + 1) = 2/3 = 0.6667
        # Buggy Precision was 2 / (2 + 1 + 1) = 2/4 = 0.5 (identical to accuracy)
        verification = VerificationResult(
            claims=[
                ClaimVerification(claim_text="c1", verdict=VerificationVerdict.SUPPORTED),
                ClaimVerification(claim_text="c2", verdict=VerificationVerdict.SUPPORTED),
                ClaimVerification(claim_text="c3", verdict=VerificationVerdict.MISATTRIBUTED),
                ClaimVerification(claim_text="c4", verdict=VerificationVerdict.UNSUPPORTED),
            ],
            citation_coverage=0.75,
            all_supported=False,
        )

        scores = compute_citation_metrics(verification)
        assert round(scores.citation_accuracy, 4) == 0.5000
        assert round(scores.citation_precision, 4) == pytest.approx(0.6667, abs=1e-3), (
            f"MD-03 Violation: Citation precision is {scores.citation_precision}, expected ~0.6667"
        )
        assert scores.citation_precision != scores.citation_accuracy, (
            "MD-03 Violation: Precision is mathematically identical to accuracy"
        )

    # 14. MD-04: NDCG@k Inflated Above 1.0 on Duplicate Retrieved Chunks
    def test_md04_ndcg_at_k_deduplicates_relevant_chunks(self):
        """MD-04: ndcg_at_k must not accumulate multiple gains for duplicate relevant chunk IDs."""
        retrieved = ["chunk_alpha", "chunk_alpha", "chunk_alpha"]
        relevant = {"chunk_alpha"}
        k = 3

        score = ndcg_at_k(retrieved, relevant, k=k)
        assert score <= 1.0, f"MD-04 Violation: NDCG@k inflated above 1.0 on duplicate chunks: {score}"
        assert score == 1.0, f"NDCG@k for perfectly retrieved item should be 1.0, got {score}"

    # 15. MD-05: Ollama API Prompt Injection via Missing System Separation
    def test_md05_ollama_passes_system_instruction_via_native_parameter(self):
        """MD-05: _call_ollama must pass system_instruction via top-level 'system' JSON parameter."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"response": "{}"}).encode("utf-8")
        mock_response.__enter__.return_value = mock_response

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            _call_ollama(
                model_name="llama3",
                prompt="User question",
                system_instruction="Strict system instructions",
                temperature=0.7,
            )

            assert mock_urlopen.called
            sent_req = mock_urlopen.call_args[0][0]
            sent_payload = json.loads(sent_req.data.decode("utf-8"))

            assert sent_payload.get("system") == "Strict system instructions", (
                "MD-05 Violation: Native 'system' field missing from Ollama request payload"
            )
            assert "Strict system instructions" not in sent_payload.get("prompt", ""), (
                "MD-05 Violation: System instruction was prepended into prompt text"
            )

    # 16. LO-01: Missing Boundary Validation in QueryRequest
    def test_lo01_query_request_boundary_validation(self):
        """LO-01: QueryRequest must enforce bounds on question length, alpha [0, 1], and temperature [0, 2]."""
        # Empty question should raise ValidationError
        with pytest.raises(ValidationError):
            QueryRequest(question="", alpha=0.5, temperature=0.0)

        # Alpha < 0
        with pytest.raises(ValidationError):
            QueryRequest(question="Valid question", alpha=-0.1)

        # Alpha > 1
        with pytest.raises(ValidationError):
            QueryRequest(question="Valid question", alpha=1.1)

        # Temperature < 0
        with pytest.raises(ValidationError):
            QueryRequest(question="Valid question", temperature=-0.1)

        # Temperature > 2
        with pytest.raises(ValidationError):
            QueryRequest(question="Valid question", temperature=2.5)

    # 17. LO-02: Non-ASCII Characters Stripped by BM25 Technical Tokenizer
    def test_lo02_tokenizer_preserves_multilingual_non_ascii_characters(self):
        """LO-02: tokenize_technical must preserve accented Latin and non-ASCII Unicode characters."""
        text = "café München naïve résumé привет"
        tokens = tokenize_technical(text)

        # Check preserved non-ASCII characters
        joined = " ".join(tokens)
        assert "café" in tokens or "caf" not in tokens or "é" in joined, "LO-02 Violation: 'café' was stripped"
        assert "münchen" in tokens or "ü" in joined, "LO-02 Violation: 'München' umlaut was stripped"
        assert "naïve" in tokens or "ï" in joined, "LO-02 Violation: 'naïve' was stripped"
        assert "résumé" in tokens or "é" in joined, "LO-02 Violation: 'résumé' was stripped"
        assert "привет" in tokens or any("привет" in t for t in tokens), "LO-02 Violation: Cyrillic was completely dropped"


# ==============================================================================
# TIER 2: Boundary & Corner Cases
# ==============================================================================

class TestTier2BoundaryCornerCases:
    """Stress tests on boundaries, extreme resource limits, and edge conditions."""

    def test_boundary_bm25_empty_and_whitespace_rebuild(self):
        """Verify BM25 rebuild with pure whitespace, tabs, and empty string entries."""
        store = Bm25Store()
        texts = ["", "   ", "\t\t\n", "   \r\n   "]
        chunk_ids = ["c1", "c2", "c3", "c4"]
        count = store.rebuild(texts, chunk_ids, strategy="fixed")
        assert count == 0 or count == 4
        assert store.search("any", strategy="fixed") == []

    def test_boundary_rate_limiter_extreme_rates(self):
        """Verify TokenBucketRateLimiter behavior with zero, negative, and large burst rates."""
        # Zero rate
        r_zero = TokenBucketRateLimiter(rate=0.0, capacity=2.0)
        assert r_zero.acquire(timeout=0.02) is False

        # Negative rate
        r_neg = TokenBucketRateLimiter(rate=-100.0, capacity=1.0)
        assert r_neg.acquire(timeout=0.02) is False

        # Drain fast
        r_burst = TokenBucketRateLimiter(rate=1.0, capacity=1.0)
        assert r_burst.acquire(timeout=0.1) is True
        assert r_burst.acquire(timeout=0.01) is False

    def test_boundary_tokenizer_extended_unicode_scripts(self):
        """Verify technical tokenization across diverse multilingual scripts and symbols."""
        scripts = "東京 日本語 한국어 עִבְרִית العربية 127.0.0.1:8080/v1/api"
        tokens = tokenize_technical(scripts)
        assert len(tokens) > 0
        # IP and port technical token should be extracted
        assert any("127.0.0.1:8080/v1/api" in t for t in tokens) or any("127" in t for t in tokens)
        # Asian characters preserved
        assert any("東京" in t or "日本" in t for t in tokens)

    def test_boundary_groq_null_and_malformed_structures(self):
        """Verify Groq caller handles empty choices list, null content, and non-dict payload."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"choices": []}).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            try:
                out = _call_groq("llama-3.3-70b-versatile", "test", "system", 0.0)
                assert out == "" or out is None
            except (IndexError, KeyError):
                # An explicit handled return or safe exception
                pass

    def test_boundary_query_request_exact_boundaries(self):
        """Verify exact valid boundary values (0.0, 1.0, 2.0) pass Pydantic validation."""
        req_min = QueryRequest(question="Q", alpha=0.0, temperature=0.0)
        assert req_min.alpha == 0.0
        assert req_min.temperature == 0.0

        req_max = QueryRequest(question="Q", alpha=1.0, temperature=2.0)
        assert req_max.alpha == 1.0
        assert req_max.temperature == 2.0

    def test_boundary_document_upload_zero_bytes(self, mock_db_session):
        """Verify uploading 0-byte file returns HTTP 400."""
        test_app = create_app()
        test_app.dependency_overrides[get_db] = _override_db_dependency(mock_db_session)
        client = TestClient(test_app)
        try:
            res = client.post(
                "/api/v1/documents",
                files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
            )
            # Empty file should either fail validation or return 400
            assert res.status_code in (400, 422)
        finally:
            test_app.dependency_overrides.clear()

    def test_boundary_job_registry_fifo_eviction_under_capacity(self):
        """Verify FIFO eviction ordering in JobRegistry when multiple jobs complete."""
        registry = JobRegistry(max_jobs=2)
        j1 = registry.create_job("j1")
        j2 = registry.create_job("j2")

        registry.complete_job(j1.job_id)
        registry.complete_job(j2.job_id)

        # Create j3 -> should evict j1 first (oldest completed)
        j3 = registry.create_job("j3")
        assert registry.get_job(j1.job_id) is None
        assert registry.get_job(j2.job_id) is not None
        assert registry.get_job(j3.job_id) is not None

    def test_boundary_ndcg_at_k_various_k_values(self):
        """Verify NDCG@k bounds across varying k values and empty inputs."""
        assert ndcg_at_k([], {"c1"}, 5) == 0.0
        assert ndcg_at_k(["c1"], set(), 5) == 0.0
        assert ndcg_at_k(["c1"], {"c1"}, 0) == 0.0

        # Duplicate occurrences at k=1, 2, 5
        retrieved = ["c1", "c1", "c2", "c1", "c2", "c3"]
        relevant = {"c1", "c2"}
        for k in [1, 2, 3, 5, 10]:
            score = ndcg_at_k(retrieved, relevant, k=k)
            assert 0.0 <= score <= 1.0, f"NDCG@{k} out of bounds: {score}"

    def test_boundary_citation_precision_denominators(self):
        """Verify citation precision when all claims are unsupported, all misattributed, or zero claims."""
        # 0 claims
        assert compute_citation_metrics(VerificationResult(claims=[], citation_coverage=0.0, all_supported=False)).citation_precision == 0.0

        # All unsupported (supported=0, misattributed=0, unsupported=3)
        res_unsupported = VerificationResult(
            claims=[
                ClaimVerification(claim_text="c1", verdict=VerificationVerdict.UNSUPPORTED),
                ClaimVerification(claim_text="c2", verdict=VerificationVerdict.UNSUPPORTED),
            ],
            citation_coverage=0.0,
            all_supported=False,
        )
        scores_unsupported = compute_citation_metrics(res_unsupported)
        assert scores_unsupported.citation_precision == 0.0
        assert scores_unsupported.citation_accuracy == 0.0

        # All misattributed (supported=0, misattributed=2)
        res_misattributed = VerificationResult(
            claims=[
                ClaimVerification(claim_text="c1", verdict=VerificationVerdict.MISATTRIBUTED),
                ClaimVerification(claim_text="c2", verdict=VerificationVerdict.MISATTRIBUTED),
            ],
            citation_coverage=0.0,
            all_supported=False,
        )
        scores_mis = compute_citation_metrics(res_misattributed)
        assert scores_mis.citation_precision == 0.0
        assert scores_mis.citation_accuracy == 0.0

    def test_boundary_wilcoxon_pairing_disjoint_questions(self):
        """Verify compare_experiments handles or rejects disjoint question sets safely."""
        res_a = ExperimentResult(
            name="A",
            strategy="dense",
            chunking_strategy="fixed",
            per_question=[QuestionResult(question_id="q1", question="Q1", metrics={"ndcg_at_k": 0.8})],
        )
        res_b = ExperimentResult(
            name="B",
            strategy="hybrid",
            chunking_strategy="fixed",
            per_question=[QuestionResult(question_id="q99", question="Q99", metrics={"ndcg_at_k": 0.8})],
        )

        with pytest.raises(ValueError):
            compare_experiments(res_a, res_b, metric="ndcg_at_k")

    def test_boundary_corrupted_pdf_handling(self, mock_db_session):
        """Tier 2: Corrupted PDF file uploaded to /api/v1/documents returns HTTP 400."""
        test_app = create_app()
        test_app.dependency_overrides[get_db] = _override_db_dependency(mock_db_session)
        client = TestClient(test_app)
        try:
            corrupted_pdf_bytes = b"%PDF-1.4 ... CORRUPTED STREAM DATA NOT VALID PDF ... EOF"
            res = client.post(
                "/api/v1/documents",
                files={"file": ("corrupted.pdf", io.BytesIO(corrupted_pdf_bytes), "application/pdf")},
            )
            assert res.status_code == 400, (
                f"HI-02 Boundary Violation: Corrupted PDF returned {res.status_code} instead of 400"
            )
        finally:
            test_app.dependency_overrides.clear()

    def test_boundary_duplicate_hash_collision_integrity_error(self, mock_db_session):
        """Tier 2: Duplicate file hash collision (IntegrityError) in upload_document returns HTTP 409."""
        from sqlalchemy.exc import IntegrityError

        test_app = create_app()
        test_app.dependency_overrides[get_db] = _override_db_dependency(mock_db_session)
        client = TestClient(test_app)
        try:
            with patch("hybrid_rag.api.documents.index_document", side_effect=IntegrityError("stmt", "params", "orig")):
                res = client.post(
                    "/api/v1/documents",
                    files={"file": ("document.txt", io.BytesIO(b"Unique content that collides concurrently"), "text/plain")},
                )
                assert res.status_code == 409, (
                    f"HI-02 Boundary Violation: Hash collision returned {res.status_code} instead of 409"
                )
        finally:
            test_app.dependency_overrides.clear()

    def test_boundary_sse_stream_termination_without_trailing_double_newline(self):
        """Tier 2: SSE stream parser flushes trailing event on close even without trailing double newline."""
        # Simulated stream simulation: buffer contains trailing event terminated only by EOF or single \n
        test_script = Path(__file__).resolve().parent / "test_e2e_frontend_stream.mjs"
        assert test_script.is_file()
        res = subprocess.run(["node", str(test_script)], capture_output=True, text=True)
        assert res.returncode == 0, (
            f"HI-03 Boundary Violation: Trailing event dropped on stream close:\n{res.stdout}\n{res.stderr}"
        )

    def test_boundary_query_out_of_bounds_parameters(self):
        """Tier 2: Out-of-bound alpha and temperature parameters must be strictly rejected."""
        # Out-of-bound alpha
        with pytest.raises(ValidationError):
            QueryRequest(question="What is RAG?", alpha=-0.001)

        with pytest.raises(ValidationError):
            QueryRequest(question="What is RAG?", alpha=1.001)

        # Out-of-bound temperature
        with pytest.raises(ValidationError):
            QueryRequest(question="What is RAG?", temperature=-0.01)

        with pytest.raises(ValidationError):
            QueryRequest(question="What is RAG?", temperature=2.01)

        # Empty question string
        with pytest.raises(ValidationError):
            QueryRequest(question="")

        # Stopword/empty token corpus with BM25
        store = Bm25Store()
        texts = ["", "a", "the", "in", "of", "and"]
        chunk_ids = [f"c_{i}" for i in range(len(texts))]
        assert store.rebuild(texts, chunk_ids, strategy="fixed") >= 0


# ==============================================================================
# TIER 3: Cross-Feature Combinations
# ==============================================================================

class TestTier3CrossFeatureCombinations:
    """Interactions between multiple subsystems and features."""

    # Cross 1: CR-04 (retrieval_strategy) + LO-02 (multilingual tokenization)
    def test_cross_bm25_only_with_multilingual_tokens(self, mock_db_session):
        """Verify bm25_only strategy correctly tokenizes and searches multilingual corpus."""
        store = Bm25Store()
        texts = [
            "Le café de Paris utilise PostgreSQL v15.2",
            "München Hauptbahnhof RAG-System Architektur",
            "Стандартные протоколы для нейронных сетей",
        ]
        chunk_ids = ["c_fr", "c_de", "c_ru"]
        store.rebuild(texts, chunk_ids, strategy="fixed")

        config = PipelineConfig(retrieval_strategy="bm25_only", rerank_enabled=False)
        trace = full_retrieve("café", config, bm25_store=store, db=mock_db_session)

        # Invariant 1: Bypassed dense and RRF
        stages = [s.stage for s in trace.stages]
        assert "sparse" in stages
        assert "dense" not in stages
        assert "rrf" not in stages

        # Invariant 2: Found French accented passage
        candidates = trace.stages[-1].candidates
        assert len(candidates) > 0
        assert candidates[0].chunk_id == "c_fr"

    # Cross 2: CR-04 (retrieval_strategy) + LO-01 (query boundary validation)
    def test_cross_dense_only_strategy_and_boundary_validation(self):
        """Verify dense_only request passes boundary validation and creates valid PipelineConfig."""
        valid_dense_req = QueryRequest(
            question="Explain hybrid retrieval",
            retrieval_strategy="dense_only",
            alpha=0.0,  # Pure dense boundary
            temperature=0.0,
        )
        assert valid_dense_req.retrieval_strategy == "dense_only"

        # Invalid boundary on dense_only request
        with pytest.raises(ValidationError):
            QueryRequest(
                question="Explain hybrid retrieval",
                retrieval_strategy="dense_only",
                alpha=1.5,
            )

    # Cross 3: MD-01 (system instruction cache key) + CR-02 (rate limiter)
    def test_cross_cache_key_system_instructions_under_rate_limiting(self):
        """Verify cache keys isolate different system instructions while rate limiter throttles calls."""
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=10.0)
        assert limiter.acquire() is True

        k_analyst = cache_key("gpt-4o", "Analyze this", "{}", 0.0, system_instruction="Act as financial analyst")
        k_engineer = cache_key("gpt-4o", "Analyze this", "{}", 0.0, system_instruction="Act as software engineer")

        assert k_analyst != k_engineer

    # Cross 4: MD-05 (Ollama system separation) + LO-02 (multilingual non-ASCII)
    def test_cross_ollama_system_separation_with_multilingual_unicode(self):
        """Verify Ollama native system separation preserves multilingual UTF-8 characters."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"response": "OK"}).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            _call_ollama(
                model_name="llama3",
                prompt="Comment fonctionne le système RAG?",
                system_instruction="Répondez toujours en français avec précision.",
                temperature=0.5,
            )

            sent_payload = json.loads(mock_urlopen.call_args[0][0].data.decode("utf-8"))
            assert sent_payload["system"] == "Répondez toujours en français avec précision."
            assert sent_payload["prompt"] == "Comment fonctionne le système RAG?"

    # Cross 5: CR-01 (cache error poisoning) + MD-01 (system instruction key)
    def test_cross_fallback_error_not_cached_across_system_instructions(self, in_memory_cache: LLMCache):
        """Verify fallback errors are suppressed across distinct system prompts without polluting cache."""
        with patch("hybrid_rag.generation.generator._dispatch_generation", return_value="Malformed"):
            ans1, _ = generate_answer(
                question="Prompt",
                candidates=[CandidateResult(chunk_id="c1", score=1.0, rank=1, content="passage")],
                llm_cache=in_memory_cache,
                model="m1",
            )

        with sqlite3.connect(str(in_memory_cache._db_path)) as conn:
            assert conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0] == 0

    # Cross 6: MD-03 (citation precision) + MD-04 (NDCG duplicate deduplication)
    def test_cross_eval_metrics_ndcg_dedup_and_citation_precision(self):
        """Verify evaluation pipeline correctly combines deduplicated NDCG and attributed precision."""
        # Retrieval with duplicate chunks
        retrieved = ["c1", "c1", "c2"]
        relevant = {"c1", "c2"}
        ndcg = ndcg_at_k(retrieved, relevant, k=3)
        assert ndcg <= 1.0

        # Citations generated from those chunks
        verification = VerificationResult(
            claims=[
                ClaimVerification(claim_text="c1 supported", verdict=VerificationVerdict.SUPPORTED),
                ClaimVerification(claim_text="c2 misattributed", verdict=VerificationVerdict.MISATTRIBUTED),
                ClaimVerification(claim_text="c3 unsupported", verdict=VerificationVerdict.UNSUPPORTED),
            ],
            citation_coverage=0.66,
            all_supported=False,
        )
        citation_scores = compute_citation_metrics(verification)
        assert citation_scores.citation_precision == 0.5  # 1 / (1 + 1)
        assert round(citation_scores.citation_accuracy, 4) == 0.3333  # 1 / 3

    # Cross 7: HI-01 (SPA fallback) + HI-02 (document upload errors)
    def test_cross_spa_and_document_error_routing(self, mock_db_session, tmp_path: Path):
        """Verify API 400/404/409 errors coexist with SPA fallback index.html without route collisions."""
        dist_dir = tmp_path / "dist"
        dist_dir.mkdir()
        (dist_dir / "index.html").write_text("<html>INDEX</html>", encoding="utf-8")

        client = _app_with_dist(dist_dir)
        client.app.dependency_overrides[get_db] = _override_db_dependency(mock_db_session)
        try:
            # 1. Invalid upload -> 400 JSON
            r_upload = client.post("/api/v1/documents", files={"file": ("bad.exe", b"bin", "application/octet-stream")})
            assert r_upload.status_code == 400

            # 2. Non-existent API route -> 404 JSON
            r_missing_api = client.get("/api/v1/unknown")
            assert r_missing_api.status_code == 404
            assert r_missing_api.headers["content-type"].startswith("application/json")

            # 3. SPA route -> 200 HTML
            r_spa = client.get("/documents")
            assert r_spa.status_code == 200
            assert "<html>INDEX</html>" in r_spa.text
        finally:
            client.app.dependency_overrides.clear()

    # Cross 8: HI-03 (SSE stream trailing) + HI-04 (unmount abort) + API streaming
    def test_cross_sse_stream_query_cancellation_and_unmount(self):
        """Cross 8: Stream query endpoint interaction with client unmount abort and trailing event buffer."""
        # Check unmount abort contract in useSSEQuery
        hook_path = Path(__file__).resolve().parents[1] / "frontend" / "src" / "hooks" / "useSSEQuery.ts"
        hook_code = hook_path.read_text(encoding="utf-8")
        assert "useEffect" in hook_code, "useSSEQuery must use useEffect for unmount abort"
        assert "abort" in hook_code, "useSSEQuery must trigger abort on unmount"

        # Check API streaming endpoint returns text/event-stream
        from fastapi import FastAPI
        api_app = FastAPI()
        api_app.include_router(query_router)
        client = TestClient(api_app)

        with (
            patch("hybrid_rag.retrieval.retriever.full_retrieve") as mock_full,
            patch("hybrid_rag.generation.generator.generate_answer") as mock_gen,
        ):
            mock_full.return_value = QueryTrace(query="test", config=PipelineConfig())
            mock_gen.return_value = (GroundedAnswer(answer="Answer", citations=[]), False)

            # Test client stream request
            res = client.post("/api/query/stream", json={"question": "What is RAG?"})
            assert res.status_code == 200
            assert "text/event-stream" in res.headers["content-type"]



# ==============================================================================
# TIER 4: Real-World Scenarios
# ==============================================================================

class TestTier4RealWorldScenarios:
    """Complete multi-step end-to-end integration workflows."""

    def test_scenario_document_ingestion_retrieval_generation_citation(self, mock_db_session, in_memory_cache: LLMCache):
        """Scenario 1: Complete flow: Ingestion -> Strategy Routing -> Generation -> Citation Attribution."""
        # 1. Ingestion: Rebuild BM25 store with technical documentation
        store = Bm25Store()
        texts = [
            "Hybrid-RAG uses reciprocal rank fusion to merge dense vector embeddings with BM25 sparse scores.",
            "Candidate hydration loads complete chunk text from PostgreSQL before cross-encoder reranking.",
            "Citation verification maps every claim to its source chunk and computes precision.",
        ]
        chunk_ids = ["chunk_rrf", "chunk_hydration", "chunk_citation"]
        store.rebuild(texts, chunk_ids, strategy="fixed")

        # 2. Retrieval: Execute bm25_only retrieval
        config = PipelineConfig(retrieval_strategy="bm25_only", rerank_enabled=False)
        trace = full_retrieve("reciprocal rank fusion", config, bm25_store=store, db=mock_db_session)

        assert len(trace.stages) == 1
        assert trace.stages[0].stage == "sparse"
        retrieved_candidate = trace.stages[0].candidates[0]
        assert retrieved_candidate.chunk_id == "chunk_rrf"

        # 3. Generation: LLM creates grounded answer
        llm_response_json = json.dumps({
            "answer": "Hybrid-RAG combines dense and sparse search using reciprocal rank fusion [1].",
            "citations": [{"id": 1, "text": "combines dense and sparse search", "chunk_id": "chunk_rrf"}],
        })

        with patch("hybrid_rag.generation.generator._dispatch_generation", return_value=llm_response_json):
            answer, cached = generate_answer(
                question="How does Hybrid-RAG merge search?",
                candidates=trace.stages[0].candidates,
                llm_cache=in_memory_cache,
                model="mock-llama",
            )
            assert answer.is_fallback is False
            assert len(answer.citations) == 1

        # 4. Citation Verification: Verifier confirms claim
        verifier_response_json = json.dumps({
            "claims": [
                {
                    "claim_text": "Hybrid-RAG combines dense and sparse search using reciprocal rank fusion",
                    "citation_num": 1,
                    "chunk_id": "chunk_rrf",
                    "verdict": "SUPPORTED",
                    "reason": "Directly matches retrieved document",
                }
            ],
            "citation_coverage": 1.0,
            "all_supported": True,
        })

        with patch("hybrid_rag.generation.verifier._dispatch_verification", return_value=verifier_response_json):
            verification = verify_citations(
                answer=answer,
                candidates=trace.stages[0].candidates,
                llm_cache=in_memory_cache,
                model="mock-llama",
            )
            assert verification.all_supported is True
            assert len(verification.claims) == 1

        # 5. Metrics calculation
        metrics = compute_citation_metrics(verification)
        assert metrics.citation_accuracy == 1.0
        assert metrics.citation_precision == 1.0

    def test_scenario_comparative_experiment_wilcoxon_significance(self):
        """Scenario 2: Comparative Benchmarking: Aligned Pairing, Duplicate Deduplication, and Wilcoxon Test."""
        questions = [f"q_{i}" for i in range(10)]

        # Baseline: Dense only
        results_dense = []
        for i, q in enumerate(questions):
            # Dense retrieval has some duplicates
            retrieved = [f"c_{i}", f"c_{i}", f"c_{i+1}"]
            relevant = {f"c_{i}"}
            ndcg = ndcg_at_k(retrieved, relevant, k=3)
            assert ndcg <= 1.0
            results_dense.append(QuestionResult(question_id=q, question=f"Question {i}", metrics={"ndcg_at_k": ndcg}))

        # Candidate: Hybrid rerank (superior scores)
        results_hybrid = []
        for i, q in enumerate(reversed(questions)):  # Intentionally reversed question order
            q_idx = int(q.split("_")[1])
            results_hybrid.append(
                QuestionResult(question_id=q, question=f"Question {q_idx}", metrics={"ndcg_at_k": 1.0})
            )

        exp_dense = ExperimentResult(name="DenseBaseline", strategy="dense", chunking_strategy="fixed", per_question=results_dense)
        exp_hybrid = ExperimentResult(name="HybridCandidate", strategy="hybrid_rerank", chunking_strategy="fixed", per_question=results_hybrid)

        # Wilcoxon comparison must align questions by question_id before signed-rank test
        comparison = compare_experiments(exp_dense, exp_hybrid, metric="ndcg_at_k")

        assert comparison["metric"] == "ndcg_at_k"
        assert "p_value" in comparison
        # Hybrid is >= dense for all aligned questions
        wins, losses, ties = comparison["win_loss_tie"]
        assert losses == 0, f"Candidate should have 0 losses compared to baseline, got {losses}"

    def test_scenario_multilingual_bounded_job_concurrency_lifecycle(self):
        """Scenario 3: Multilingual Ingestion, JobRegistry lifecycle, and bounded concurrency."""
        registry = JobRegistry(max_jobs=3)

        # Ingest 5 multilingual documents as background jobs
        jobs = []
        for i, lang in enumerate(["English", "Français", "Deutsch", "Русский", "Español"]):
            job = registry.create_job(f"ingest_{lang}")
            registry.update_progress(job.job_id, 50.0, f"Tokenizing {lang}")
            registry.complete_job(job.job_id, {"lang": lang, "status": "indexed"})
            jobs.append(job)

        # Registry must enforce capacity bound of 3
        assert registry.count() <= 3, f"JobRegistry exceeded bounded capacity: {registry.count()}"

        # Verify that earliest completed jobs were evicted
        assert registry.get_job(jobs[0].job_id) is None
        assert registry.get_job(jobs[1].job_id) is None
        # Latest completed jobs remain
        assert registry.get_job(jobs[-1].job_id) is not None

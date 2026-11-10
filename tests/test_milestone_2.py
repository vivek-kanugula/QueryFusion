"""Comprehensive unit and regression tests for Milestone 2: Generation & LLM Providers.

Covers:
  - CR-01: Fallback Error Cache Poisoning
  - CR-02: Rate Limiter Acquire Ignored & Division by Zero
  - HI-05: LLM Client _call_groq Content NoneType Crash & _parse_response Safety
  - MD-01: Cache Key Incorporates System Instructions
  - MD-05: Ollama API Native System Separation
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hybrid_rag.generation.cache import (
    LLMCache,
    TokenBucketRateLimiter,
    cache_key,
)
from hybrid_rag.generation.generator import (
    GENERATION_SYSTEM_INSTRUCTION,
    _parse_response,
    generate_answer,
)
from hybrid_rag.generation.llm_client import _call_groq, _call_ollama
from hybrid_rag.generation.verifier import (
    VERIFICATION_SYSTEM_INSTRUCTION,
    verify_citations,
)
from hybrid_rag.schemas.generation import (
    Citation,
    GroundedAnswer,
)
from hybrid_rag.schemas.pipeline import CandidateResult


@pytest.fixture()
def isolated_cache(tmp_path: Path) -> LLMCache:
    return LLMCache(cache_dir=tmp_path / "cache", use_db=False)


def _sample_candidates() -> list[CandidateResult]:
    return [
        CandidateResult(
            chunk_id="chunk_1",
            score=0.95,
            rank=1,
            content="Reciprocal Rank Fusion merges rankings from multiple retrieval algorithms.",
            metadata={"document_id": "doc_1"},
        ),
        CandidateResult(
            chunk_id="chunk_2",
            score=0.85,
            rank=2,
            content="BM25 scoring relies on term frequency and inverted document frequency.",
            metadata={"document_id": "doc_2"},
        ),
    ]


# ==============================================================================
# CR-01: Fallback Error Cache Poisoning
# ==============================================================================

class TestCR01CachePoisoningPrevention:
    """Ensure transient errors and empty verification claims are never cached."""

    def test_generator_empty_response_not_cached(self, isolated_cache: LLMCache):
        candidates = _sample_candidates()
        with patch("hybrid_rag.generation.generator._dispatch_generation", return_value=""):
            answer, cached = generate_answer(
                question="What is RRF?",
                candidates=candidates,
                llm_cache=isolated_cache,
                model="test-model",
            )
            assert answer.is_fallback is True
            assert not cached

        with sqlite3.connect(str(isolated_cache._db_path)) as conn:
            count = conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
            assert count == 0, "Empty LLM response was poisoned into llm_cache"

    def test_generator_malformed_json_not_cached(self, isolated_cache: LLMCache):
        candidates = _sample_candidates()
        with patch(
            "hybrid_rag.generation.generator._dispatch_generation",
            return_value="Internal Server Error: 500",
        ):
            answer, cached = generate_answer(
                question="What is RRF?",
                candidates=candidates,
                llm_cache=isolated_cache,
                model="test-model",
            )
            assert answer.is_fallback is True
            assert not cached

        with sqlite3.connect(str(isolated_cache._db_path)) as conn:
            count = conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
            assert count == 0, "Malformed JSON error was poisoned into llm_cache"

    def test_generator_rate_limit_fallback_not_cached(self, isolated_cache: LLMCache):
        candidates = _sample_candidates()
        mock_limiter = MagicMock(spec=TokenBucketRateLimiter)
        mock_limiter.acquire.return_value = False

        answer, cached = generate_answer(
            question="What is RRF?",
            candidates=candidates,
            llm_cache=isolated_cache,
            rate_limiter=mock_limiter,
            model="test-model",
        )
        assert answer.is_fallback is True
        assert answer.fallback_reason == "Rate limit exceeded."
        assert not cached

        with sqlite3.connect(str(isolated_cache._db_path)) as conn:
            count = conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
            assert count == 0, "Rate limit fallback was poisoned into llm_cache"

    def test_generator_valid_response_is_cached(self, isolated_cache: LLMCache):
        candidates = _sample_candidates()
        valid_json = json.dumps({
            "answer": "RRF merges rankings [C1].",
            "citations": [
                {
                    "citation_id": "[C1]",
                    "chunk_id": "chunk_1",
                    "evidence_quote": "merges rankings",
                }
            ],
            "has_sufficient_context": True,
        })
        with patch(
            "hybrid_rag.generation.generator._dispatch_generation", return_value=valid_json
        ) as mock_dispatch:
            ans1, cached1 = generate_answer(
                "What is RRF?", candidates, llm_cache=isolated_cache, model="test-model"
            )
            assert not cached1
            assert ans1.is_fallback is False
            assert mock_dispatch.call_count == 1

            ans2, cached2 = generate_answer(
                "What is RRF?", candidates, llm_cache=isolated_cache, model="test-model"
            )
            assert cached2
            assert ans2.is_fallback is False
            assert mock_dispatch.call_count == 1

        with sqlite3.connect(str(isolated_cache._db_path)) as conn:
            count = conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
            assert count == 1

    def test_verifier_empty_and_failed_claims_not_cached(self, isolated_cache: LLMCache):
        candidates = _sample_candidates()
        answer = GroundedAnswer(
            answer="RRF merges rankings [C1].",
            citations=[
                Citation(
                    citation_id="[C1]", chunk_id="chunk_1", evidence_quote="merges rankings"
                )
            ],
        )

        for bad_raw in ["", "   ", "not-a-json", "[]"]:
            with patch(
                "hybrid_rag.generation.verifier._dispatch_verification", return_value=bad_raw
            ):
                res = verify_citations(
                    answer=answer,
                    candidates=candidates,
                    llm_cache=isolated_cache,
                    model="test-model",
                )
                assert len(res.claims) == 0

            with sqlite3.connect(str(isolated_cache._db_path)) as conn:
                count = conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
                assert count == 0, f"Failed verification for {bad_raw!r} was poisoned into cache"

    def test_verifier_valid_claims_cached(self, isolated_cache: LLMCache):
        candidates = _sample_candidates()
        answer = GroundedAnswer(
            answer="RRF merges rankings [C1].",
            citations=[
                Citation(
                    citation_id="[C1]", chunk_id="chunk_1", evidence_quote="merges rankings"
                )
            ],
        )
        valid_verifier_raw = json.dumps([
            {
                "claim_text": "RRF merges rankings",
                "citation_id": "[C1]",
                "verdict": "SUPPORTED",
                "evidence_quote": "merges rankings",
                "explanation": "Exact match",
            }
        ])

        with patch(
            "hybrid_rag.generation.verifier._dispatch_verification",
            return_value=valid_verifier_raw,
        ) as mock_dispatch:
            res1 = verify_citations(
                answer, candidates, llm_cache=isolated_cache, model="test-model"
            )
            assert len(res1.claims) == 1
            assert mock_dispatch.call_count == 1

            res2 = verify_citations(
                answer, candidates, llm_cache=isolated_cache, model="test-model"
            )
            assert len(res2.claims) == 1
            assert mock_dispatch.call_count == 1


# ==============================================================================
# CR-02: Rate Limiter Acquire Ignored & Division by Zero
# ==============================================================================

class TestCR02RateLimiterGuards:
    """Ensure rate limiter prevents division by zero, negative sleep, and checks acquire()."""

    def test_rate_limiter_zero_rate(self):
        limiter = TokenBucketRateLimiter(rate=0.0, capacity=10.0)
        assert limiter.acquire(timeout=0.01) is False

    def test_rate_limiter_negative_rate(self):
        limiter = TokenBucketRateLimiter(rate=-50.0, capacity=10.0)
        assert limiter.acquire(timeout=0.01) is False

    def test_rate_limiter_token_depletion_and_refill(self):
        limiter = TokenBucketRateLimiter(rate=1.0, capacity=1.0)
        assert limiter.acquire(timeout=0.05) is True
        # Capacity depleted; immediate timeout should return False (needs 1.0s to refill 1 token)
        assert limiter.acquire(timeout=0.01) is False

    def test_generator_respects_rate_limiter_failure(self):
        mock_limiter = MagicMock(spec=TokenBucketRateLimiter)
        mock_limiter.acquire.return_value = False

        with patch("hybrid_rag.generation.generator._dispatch_generation") as mock_dispatch:
            answer, cached = generate_answer(
                question="What is RRF?",
                candidates=_sample_candidates(),
                rate_limiter=mock_limiter,
                model="test-model",
            )
            assert answer.is_fallback is True
            assert answer.fallback_reason == "Rate limit exceeded."
            assert not mock_dispatch.called

    def test_verifier_respects_rate_limiter_failure(self):
        mock_limiter = MagicMock(spec=TokenBucketRateLimiter)
        mock_limiter.acquire.return_value = False

        answer = GroundedAnswer(
            answer="RRF merges rankings [C1].",
            citations=[
                Citation(
                    citation_id="[C1]", chunk_id="chunk_1", evidence_quote="merges rankings"
                )
            ],
        )

        with patch("hybrid_rag.generation.verifier._dispatch_verification") as mock_dispatch:
            res = verify_citations(
                answer=answer,
                candidates=_sample_candidates(),
                rate_limiter=mock_limiter,
                model="test-model",
            )
            assert len(res.claims) == 0
            assert res.all_supported is False
            assert not mock_dispatch.called


# ==============================================================================
# HI-05: LLM Client _call_groq Content NoneType Crash & _parse_response Safety
# ==============================================================================

class TestHI05GroqContentNoneSafety:
    """Ensure null/malformed responses from Groq or LLMs do not crash parsing."""

    def test_call_groq_null_content(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": None}}]
        }).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            out = _call_groq("llama-3.3-70b-versatile", "Hello", "System", 0.0)
            assert out == ""

    def test_call_groq_empty_choices(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"choices": []}).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            out = _call_groq("llama-3.3-70b-versatile", "Hello", "System", 0.0)
            assert out == ""

    def test_call_groq_non_dict_message(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"choices": [{"message": None}]}).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            out = _call_groq("llama-3.3-70b-versatile", "Hello", "System", 0.0)
            assert out == ""

    def test_parse_response_null_literal(self):
        candidates = _sample_candidates()
        answer = _parse_response("null", candidates)
        assert answer.is_fallback is True
        assert answer.fallback_reason == "Failed to parse LLM response as JSON."

    def test_parse_response_non_dict_types(self):
        candidates = _sample_candidates()
        for raw in ["[1, 2, 3]", "12345", '"just a string"', "true"]:
            ans = _parse_response(raw, candidates)
            assert ans.is_fallback is True
            assert ans.fallback_reason == "Failed to parse LLM response as JSON."


# ==============================================================================
# MD-01: Cache Key Incorporates System Instructions
# ==============================================================================

class TestMD01SystemInstructionCacheKey:
    """Ensure changes to system instructions alter the cache key."""

    def test_cache_key_varies_with_system_instruction(self):
        k1 = cache_key("gpt-4o", "What is RAG?", "{}", 0.0, system_instruction="Sys 1")
        k2 = cache_key("gpt-4o", "What is RAG?", "{}", 0.0, system_instruction="Sys 2")
        k_empty = cache_key("gpt-4o", "What is RAG?", "{}", 0.0, system_instruction="")

        assert k1 != k2
        assert k1 != k_empty
        assert k2 != k_empty

    def test_cache_key_backward_compatibility(self):
        k_default = cache_key("gpt-4o", "What is RAG?", "{}", 0.0)
        k_explicit = cache_key("gpt-4o", "What is RAG?", "{}", 0.0, system_instruction="")
        assert k_default == k_explicit

    def test_generator_and_verifier_cache_keys_use_system_instructions(
        self, isolated_cache: LLMCache
    ):
        candidates = _sample_candidates()
        valid_json = json.dumps({
            "answer": "RRF combines rankings [C1].",
            "citations": [
                {
                    "citation_id": "[C1]",
                    "chunk_id": "chunk_1",
                    "evidence_quote": "combines rankings",
                }
            ],
            "has_sufficient_context": True,
        })
        with patch(
            "hybrid_rag.generation.generator._dispatch_generation", return_value=valid_json
        ):
            generate_answer("Q?", candidates, llm_cache=isolated_cache, model="m")

        with sqlite3.connect(str(isolated_cache._db_path)) as conn:
            rows = conn.execute("SELECT key FROM llm_cache").fetchall()
            assert len(rows) == 1
            stored_key = rows[0][0]

        assert len(stored_key) == 64
        assert isinstance(stored_key, str)
        assert GENERATION_SYSTEM_INSTRUCTION is not None
        assert VERIFICATION_SYSTEM_INSTRUCTION is not None


# ==============================================================================
# MD-05: Ollama API Native System Separation
# ==============================================================================

class TestMD05OllamaSystemSeparation:
    """Ensure Ollama uses native top-level system parameter and avoids prompt concatenation."""

    def test_ollama_native_system_parameter(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"response": '{"answer": "ok"}'}).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            _call_ollama(
                model_name="llama3.2",
                prompt="Explain vector search.",
                system_instruction="Strict system instructions: act as security analyst.",
                temperature=0.2,
            )

            assert mock_urlopen.called
            req = mock_urlopen.call_args[0][0]
            payload = json.loads(req.data.decode("utf-8"))

            assert payload.get("system") == "Strict system instructions: act as security analyst."
            assert payload.get("prompt") == "Explain vector search."
            assert "Strict system instructions" not in payload.get("prompt", "")

    def test_ollama_empty_system_instruction(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"response": '{"answer": "ok"}'}).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            _call_ollama(
                model_name="llama3.2",
                prompt="Explain vector search.",
                system_instruction="",
                temperature=0.2,
            )

            assert mock_urlopen.called
            req = mock_urlopen.call_args[0][0]
            payload = json.loads(req.data.decode("utf-8"))

            assert "system" not in payload
            assert payload.get("prompt") == "Explain vector search."

"""Adversarial stress-test suite for Milestone 2: Generation & LLM Providers.

Evaluates:
1. CR-02: Rate Limiter Acquire Ignored & Division by Zero
   - Non-positive rates: 0.0, -0.0, -1.0, -5.0, -100.0, -1e6.
   - Tiny rates: 1e-9 (does not hang or overflow).
   - Large rates: 1e6 (burst consumption).
   - Zero capacity (0.0) and non-positive timeouts (0.0, -0.01, -10.0).
   - Concurrency stress: 20 threads requesting tokens concurrently on rate=0.0 and rate=-5.0.
   - Concurrency token accounting integrity: 20 threads competing for 10 tokens
     (exactly 10 granted, 10 rejected).
   - Sustained multi-threaded contention (no deadlocks, valid token bounds).
   - Strict verification that when acquire() returns False, LLM dispatch is NOT executed in either
     generator or verifier, and no fallback entry is poisoned into LLMCache.

2. HI-05: Groq Malformed Payloads & Safe Generator Fallback
   - The 7 required Groq payload shapes:
     1. {"choices": []}
     2. {"choices": [{"message": {}}]}
     3. {"choices": [{"message": {"content": null}}]}
     4. {"choices": [{"message": {"content": "null"}}]}
     5. {"choices": [{"message": {"content": "[]"}}]}
     6. {"choices": [{"message": {"content": "123"}}]}
     7. {"choices": [{"message": {"content": "true"}}]}
   - Verification that generator and _call_groq handle all 7 without crashing,
     returning safe fallback answers.

3. HI-05 Deep Adversarial Defects Discovered:
   - Vulnerability 1: _call_groq crashes with AttributeError when choices[0] is not a dict.
   - Vulnerability 2: _parse_response crashes with TypeError/AttributeError/ValidationError when
     JSON keys have null or non-dict values (e.g. {"citations": null}, {"answer": null}).
   - Vulnerability 3: _parse_verification crashes with AttributeError when claim list contains
     non-dict items.
"""

from __future__ import annotations

import concurrent.futures
import json
import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hybrid_rag.generation.cache import LLMCache, TokenBucketRateLimiter
from hybrid_rag.generation.generator import _parse_response, generate_answer
from hybrid_rag.generation.llm_client import _call_groq
from hybrid_rag.generation.verifier import _parse_verification, verify_citations
from hybrid_rag.schemas.generation import Citation, GroundedAnswer
from hybrid_rag.schemas.pipeline import CandidateResult


def _sample_candidates() -> list[CandidateResult]:
    return [
        CandidateResult(
            chunk_id="chunk_alpha",
            score=0.92,
            rank=1,
            content="Alpha chunk contains vital context on reciprocal rank fusion.",
            metadata={"document_id": "doc_alpha"},
        ),
        CandidateResult(
            chunk_id="chunk_beta",
            score=0.81,
            rank=2,
            content="Beta chunk explains BM25 inverted indexing and tokenization.",
            metadata={"document_id": "doc_beta"},
        ),
    ]


@pytest.fixture()
def isolated_cache(tmp_path: Path) -> LLMCache:
    return LLMCache(cache_dir=tmp_path / "cache", use_db=False)


class FastTimeoutLimiter(TokenBucketRateLimiter):
    """Subclass defaulting timeout to 0.05s for quick adversarial rejection tests."""

    def acquire(self, timeout: float = 0.05) -> bool:
        return super().acquire(timeout=0.05)


# ==============================================================================
# 1. CR-02 ADVERSARIAL STRESS SUITE: Rate Limiter
# ==============================================================================


class TestCR02AdversarialRateLimiter:
    """Adversarial stress testing of TokenBucketRateLimiter edge cases and concurrency."""

    @pytest.mark.parametrize("rate", [0.0, -0.0, -1.0, -5.0, -100.0, -1e6])
    def test_non_positive_rates_immediately_fail_without_exception(self, rate: float):
        """Zero and negative rates must return False immediately without ZeroDivisionError."""
        start = time.monotonic()
        limiter = TokenBucketRateLimiter(rate=rate, capacity=10.0)
        result = limiter.acquire(timeout=1.0)
        elapsed = time.monotonic() - start

        assert result is False
        assert elapsed < 0.05, f"acquire() hung for {elapsed:.3f}s on non-positive rate {rate}"

    def test_tiny_positive_rate_does_not_hang_or_divide_by_zero(self):
        """Tiny positive rate (1e-9) should acquire initial tokens and timeout gracefully."""
        limiter = TokenBucketRateLimiter(rate=1e-9, capacity=1.0)
        assert limiter.acquire(timeout=0.05) is True

        start = time.monotonic()
        assert limiter.acquire(timeout=0.05) is False
        elapsed = time.monotonic() - start
        assert 0.04 <= elapsed <= 0.3, f"Unexpected sleep duration {elapsed:.3f}s"

    def test_large_rate_burst_consumption(self):
        """Large rate (1e6) refills rapidly without numerical instability."""
        limiter = TokenBucketRateLimiter(rate=1e6, capacity=1000.0)
        for _ in range(500):
            assert limiter.acquire(timeout=0.01) is True

    def test_zero_capacity_limiter(self):
        """Capacity 0.0 cannot ever satisfy tokens >= 1.0, must return False on timeout."""
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=0.0)
        start = time.monotonic()
        assert limiter.acquire(timeout=0.05) is False
        elapsed = time.monotonic() - start
        assert elapsed <= 0.2

    @pytest.mark.parametrize("timeout", [0.0, -0.01, -10.0])
    def test_non_positive_timeouts_with_depleted_bucket(self, timeout: float):
        """Non-positive timeout must return False immediately when bucket is empty."""
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=0.5)
        start = time.monotonic()
        assert limiter.acquire(timeout=timeout) is False
        elapsed = time.monotonic() - start
        assert elapsed < 0.05

    def test_concurrency_stress_20_threads_zero_rate(self):
        """20 concurrent threads on rate=0.0 must all receive False immediately."""
        limiter = TokenBucketRateLimiter(rate=0.0, capacity=10.0)
        results: list[bool] = []

        def worker() -> bool:
            return limiter.acquire(timeout=0.1)

        start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(worker) for _ in range(20)]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())
        elapsed = time.monotonic() - start

        assert len(results) == 20
        assert all(r is False for r in results)
        assert elapsed < 0.2, f"Concurrent acquire on rate=0 took {elapsed:.3f}s"

    def test_concurrency_stress_20_threads_negative_rate(self):
        """20 concurrent threads on rate=-5.0 must all receive False immediately."""
        limiter = TokenBucketRateLimiter(rate=-5.0, capacity=10.0)
        results: list[bool] = []

        def worker() -> bool:
            return limiter.acquire(timeout=0.1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(worker) for _ in range(20)]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())

        assert len(results) == 20
        assert all(r is False for r in results)

    def test_concurrency_stress_20_threads_token_accounting_integrity(self):
        """20 threads competing for 10 initial tokens with minimal refill.

        Must grant exactly 10 tokens and reject 10 without race condition double-spending.
        """
        limiter = TokenBucketRateLimiter(rate=0.01, capacity=10.0)
        results: list[bool] = []

        def worker() -> bool:
            return limiter.acquire(timeout=0.02)

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(worker) for _ in range(20)]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())

        acquired_count = sum(1 for r in results if r is True)
        rejected_count = sum(1 for r in results if r is False)

        assert acquired_count == 10, f"Expected 10 tokens granted, got {acquired_count}"
        assert rejected_count == 10, f"Expected 10 requests timed out, got {rejected_count}"

    def test_concurrency_stress_sustained_contention_no_deadlock(self):
        """High-contention stress: 20 threads performing repeated acquires concurrently."""
        limiter = TokenBucketRateLimiter(rate=1000.0, capacity=100.0)
        total_tokens_acquired = 0
        lock = concurrent.futures.thread.threading.Lock()

        def worker():
            nonlocal total_tokens_acquired
            local_count = 0
            for _ in range(50):
                if limiter.acquire(timeout=0.05):
                    local_count += 1
            with lock:
                total_tokens_acquired += local_count

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(worker) for _ in range(20)]
            concurrent.futures.wait(futures)

        assert total_tokens_acquired > 0
        with limiter._lock:
            assert 0.0 <= limiter._tokens <= limiter._capacity


class TestCR02AcquireEnforcement:
    """Verify that when acquire() returns False, LLM dispatch is strictly NOT executed."""

    @pytest.mark.parametrize(
        "rate_limiter_fixture",
        [
            ("mock_false", lambda: MagicMock(acquire=MagicMock(return_value=False))),
            ("zero_rate", lambda: TokenBucketRateLimiter(rate=0.0)),
            ("neg_rate", lambda: TokenBucketRateLimiter(rate=-5.0)),
            ("depleted_fast", lambda: FastTimeoutLimiter(rate=0.001, capacity=0.0)),
        ],
    )
    def test_generator_dispatch_blocked_on_acquire_false(
        self, isolated_cache: LLMCache, rate_limiter_fixture
    ):
        name, factory = rate_limiter_fixture
        limiter = factory()
        candidates = _sample_candidates()

        with patch("hybrid_rag.generation.generator._dispatch_generation") as mock_dispatch:
            answer, cached = generate_answer(
                question="What is RRF?",
                candidates=candidates,
                llm_cache=isolated_cache,
                rate_limiter=limiter,
                model="test-model",
            )
            assert not mock_dispatch.called, f"LLM dispatch was called under limiter {name}!"
            assert answer.is_fallback is True
            assert answer.fallback_reason == "Rate limit exceeded."
            assert not cached

        # Check cache was NOT poisoned
        with sqlite3.connect(str(isolated_cache._db_path)) as conn:
            count = conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
            assert count == 0, f"Rate-limited fallback was poisoned to cache for {name}!"

    @pytest.mark.parametrize(
        "rate_limiter_fixture",
        [
            ("mock_false", lambda: MagicMock(acquire=MagicMock(return_value=False))),
            ("zero_rate", lambda: TokenBucketRateLimiter(rate=0.0)),
            ("neg_rate", lambda: TokenBucketRateLimiter(rate=-5.0)),
            ("depleted_fast", lambda: FastTimeoutLimiter(rate=0.001, capacity=0.0)),
        ],
    )
    def test_verifier_dispatch_blocked_on_acquire_false(
        self, isolated_cache: LLMCache, rate_limiter_fixture
    ):
        name, factory = rate_limiter_fixture
        limiter = factory()
        candidates = _sample_candidates()
        answer = GroundedAnswer(
            answer="RRF is used in retrieval [C1].",
            citations=[
                Citation(
                    citation_id="[C1]",
                    chunk_id="chunk_alpha",
                    evidence_quote="reciprocal rank fusion",
                )
            ],
            has_sufficient_context=True,
        )

        with patch("hybrid_rag.generation.verifier._dispatch_verification") as mock_dispatch:
            res = verify_citations(
                answer=answer,
                candidates=candidates,
                llm_cache=isolated_cache,
                rate_limiter=limiter,
                model="test-model",
            )
            assert not mock_dispatch.called, (
                f"Verifier LLM dispatch called under limiter {name}!"
            )
            assert len(res.claims) == 0
            assert res.all_supported is False
            assert res.citation_coverage == 0.0

        with sqlite3.connect(str(isolated_cache._db_path)) as conn:
            count = conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
            assert count == 0, f"Verifier rate-limit empty claims poisoned cache for {name}!"


# ==============================================================================
# 2. HI-05 REQUIRED GROQ PAYLOADS (7 TARGET SCENARIOS)
# ==============================================================================

REQUIRED_GROQ_PAYLOADS = [
    ("empty_choices", {"choices": []}),
    ("empty_message", {"choices": [{"message": {}}]}),
    ("content_null", {"choices": [{"message": {"content": None}}]}),
    ("content_str_null", {"choices": [{"message": {"content": "null"}}]}),
    ("content_empty_list_str", {"choices": [{"message": {"content": "[]"}}]}),
    ("content_number_str", {"choices": [{"message": {"content": "123"}}]}),
    ("content_bool_str", {"choices": [{"message": {"content": "true"}}]}),
]


class TestHI05RequiredGroqPayloads:
    """Adversarial testing of the 7 specified Groq response shapes."""

    @pytest.mark.parametrize("name,payload", REQUIRED_GROQ_PAYLOADS)
    def test_call_groq_handles_required_malformed_payloads(self, name, payload):
        """_call_groq safely extracts content or returns empty string without exceptions."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = _call_groq("llama-3.3-70b-versatile", "Prompt", "System", 0.0)

            if name in {"empty_choices", "empty_message", "content_null"}:
                assert result == "", f"Expected empty string for {name}, got {result!r}"
            else:
                expected_content = payload["choices"][0]["message"]["content"]
                assert result == expected_content, f"Expected {expected_content!r} for {name}"

    @pytest.mark.parametrize("name,payload", REQUIRED_GROQ_PAYLOADS)
    def test_generator_end_to_end_with_groq_required_payloads(
        self, isolated_cache: LLMCache, name, payload
    ):
        """Full generator flow with provider='groq' handling required malformed payloads safely."""
        candidates = _sample_candidates()
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            answer, cached = generate_answer(
                question="Explain RRF and BM25",
                candidates=candidates,
                llm_cache=isolated_cache,
                provider="groq",
                model="llama-3.3-70b-versatile",
            )

            assert isinstance(answer, GroundedAnswer)
            assert answer.is_fallback is True, (
                f"Expected is_fallback=True for payload {name}, got {answer}"
            )
            assert not cached
            assert answer.citations == []

        # Confirm LLM cache is clean
        with sqlite3.connect(str(isolated_cache._db_path)) as conn:
            count = conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
            assert count == 0, f"Malformed Groq payload {name} caused cache poisoning!"

    @pytest.mark.parametrize(
        "raw_content",
        [
            "",
            "   \n  \t ",
            "null",
            "[]",
            "123",
            "true",
            "false",
            '"just a bare string"',
            "{",
            "{{bad json}}",
            "[1, 2, 3]",
        ],
    )
    def test_parse_response_robustness_on_adversarial_strings(self, raw_content: str):
        """_parse_response must never raise uncaught exceptions on arbitrary inputs."""
        candidates = _sample_candidates()
        ans = _parse_response(raw_content, candidates)
        assert isinstance(ans, GroundedAnswer)
        assert ans.is_fallback is True
        assert ans.citations == []


# ==============================================================================
# 3. HI-05 DEEP ADVERSARIAL DEFECTS (EMPIRICAL BUGS DISCOVERED)
# ==============================================================================


class TestHI05DiscoveredDefects:
    """Empirical proof of bugs discovered during adversarial stress-testing.

    These tests document unhandled exception crash vectors in generation and LLM parsing.
    """

    @pytest.mark.parametrize(
        "extreme_groq_payload",
        [
            pytest.param(
                {"choices": [None]},
                id="choices_contains_none",
            ),
            pytest.param(
                {"choices": [123]},
                id="choices_contains_int",
            ),
        ],
    )
    def test_bug_call_groq_crashes_on_non_dict_choice_element(self, extreme_groq_payload):
        """VULNERABILITY: _call_groq crashes with AttributeError when choices[0] is not a dict.

        Location: src/hybrid_rag/generation/llm_client.py:114
        Code: message = choices[0].get("message")
        Error: AttributeError: 'NoneType'/'int' object has no attribute 'get'
        """
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(extreme_groq_payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            try:
                out = _call_groq("llama-3.3-70b-versatile", "Prompt", "System", 0.0)
                assert out == ""
            except AttributeError as e:
                err_msg = f"CONFIRMED BUG: _call_groq crashed with {type(e).__name__}: {e}"
                pytest.fail(err_msg)

    @pytest.mark.parametrize(
        "schema_breaking_json,expected_err",
        [
            pytest.param(
                '{"citations": null}',
                "TypeError",
                id="citations_is_null",
            ),
            pytest.param(
                '{"citations": "not-a-list"}',
                "AttributeError",
                id="citations_is_string",
            ),
            pytest.param(
                '{"answer": null}',
                "ValidationError",
                id="answer_is_null",
            ),
            pytest.param(
                '{"citations": [123]}',
                "AttributeError",
                id="citation_item_is_int",
            ),
        ],
    )
    def test_bug_parse_response_crashes_on_schema_breaking_keys(
        self, schema_breaking_json: str, expected_err: str
    ):
        """VULNERABILITY: _parse_response crashes when LLM output has invalid types for keys.

        Locations:
          - src/hybrid_rag/generation/generator.py:225: data.get("citations", []) returns None
          - src/hybrid_rag/generation/generator.py:226: c.get(...) assumes c is dict
          - src/hybrid_rag/generation/generator.py:246: data.get("answer", "") returns None
        """
        candidates = _sample_candidates()
        try:
            ans = _parse_response(schema_breaking_json, candidates)
            assert isinstance(ans, GroundedAnswer)
        except Exception as e:
            err_msg = f"CONFIRMED BUG: _parse_response crashed with {type(e).__name__}: {e}"
            pytest.fail(err_msg)

    @pytest.mark.parametrize(
        "schema_breaking_verifier_json",
        [
            pytest.param('[null]', id="claims_contains_null"),
            pytest.param('["not-a-dict"]', id="claims_contains_str"),
            pytest.param('[123]', id="claims_contains_int"),
            pytest.param('{"claims": [null]}', id="nested_claims_contains_null"),
        ],
    )
    def test_bug_parse_verification_crashes_on_non_dict_claim_items(
        self, schema_breaking_verifier_json: str
    ):
        """VULNERABILITY: _parse_verification crashes when claims contain non-dict items.

        Location: src/hybrid_rag/generation/verifier.py:212
        Code: verdict_str = item.get("verdict", "UNSUPPORTED").upper()
        Error: AttributeError: 'NoneType'/'str'/'int' object has no attribute 'get'
        """
        try:
            res = _parse_verification(schema_breaking_verifier_json)
            assert len(res.claims) == 0
        except Exception as e:
            err_msg = f"CONFIRMED BUG: _parse_verification crashed with {type(e).__name__}: {e}"
            pytest.fail(err_msg)

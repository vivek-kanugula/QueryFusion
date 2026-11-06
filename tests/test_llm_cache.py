"""Tests for LLM cache, rate limiter, and retry backoff."""

import time

import pytest

from hybrid_rag.generation.cache import (
    LLMCache,
    TokenBucketRateLimiter,
    cache_key,
    retry_with_backoff,
)

# ── Cache key ──


def test_cache_key_deterministic():
    k1 = cache_key("model", "prompt", "schema", 0.0)
    k2 = cache_key("model", "prompt", "schema", 0.0)
    assert k1 == k2
    assert len(k1) == 64


def test_cache_key_different_model():
    k1 = cache_key("model-a", "prompt", "schema", 0.0)
    k2 = cache_key("model-b", "prompt", "schema", 0.0)
    assert k1 != k2


def test_cache_key_different_temperature():
    k1 = cache_key("model", "prompt", "schema", 0.0)
    k2 = cache_key("model", "prompt", "schema", 0.5)
    assert k1 != k2


def test_cache_key_different_prompt():
    k1 = cache_key("model", "prompt-a", "schema", 0.0)
    k2 = cache_key("model", "prompt-b", "schema", 0.0)
    assert k1 != k2


# ── LLMCache ──


def test_cache_miss(tmp_path):
    cache = LLMCache(cache_dir=tmp_path / "cache")
    assert cache.get("nonexistent") is None


def test_cache_put_get(tmp_path):
    cache = LLMCache(cache_dir=tmp_path / "cache")
    cache.put("key1", '{"answer": "test"}', "model-x")
    result = cache.get("key1")
    assert result == '{"answer": "test"}'


def test_cache_overwrite(tmp_path):
    cache = LLMCache(cache_dir=tmp_path / "cache")
    cache.put("key1", "old", "model")
    cache.put("key1", "new", "model")
    assert cache.get("key1") == "new"


def test_cache_clear(tmp_path):
    cache = LLMCache(cache_dir=tmp_path / "cache")
    cache.put("k1", "v1", "m")
    cache.put("k2", "v2", "m")
    cache.clear()
    assert cache.get("k1") is None
    assert cache.get("k2") is None


def test_cache_persistence(tmp_path):
    cache_dir = tmp_path / "cache"
    cache1 = LLMCache(cache_dir=cache_dir)
    cache1.put("k", "v", "m")
    cache2 = LLMCache(cache_dir=cache_dir)
    assert cache2.get("k") == "v"


# ── Rate limiter ──


def test_rate_limiter_basic():
    limiter = TokenBucketRateLimiter(rate=100.0, capacity=5.0)
    for _ in range(5):
        assert limiter.acquire(timeout=0.1)


def test_rate_limiter_exhaustion():
    limiter = TokenBucketRateLimiter(rate=0.1, capacity=1.0)
    assert limiter.acquire(timeout=0.1)
    assert not limiter.acquire(timeout=0.05)


def test_rate_limiter_refill():
    limiter = TokenBucketRateLimiter(rate=100.0, capacity=2.0)
    assert limiter.acquire(timeout=0.01)
    assert limiter.acquire(timeout=0.01)
    time.sleep(0.05)
    assert limiter.acquire(timeout=0.01)


# ── Retry with backoff ──


def test_retry_succeeds_first_try():
    result = retry_with_backoff(lambda: "ok", max_retries=3, base_delay=0.001)
    assert result == "ok"


def test_retry_succeeds_after_failure():
    call_count = [0]

    def flaky():
        call_count[0] += 1
        if call_count[0] < 3:
            e = Exception("rate limited")
            e.status_code = 429
            raise e
        return "ok"

    result = retry_with_backoff(flaky, max_retries=5, base_delay=0.001)
    assert result == "ok"
    assert call_count[0] == 3


def test_retry_exhausted():
    def always_fail():
        e = Exception("rate limited")
        e.status_code = 429
        raise e

    with pytest.raises(Exception, match="rate limited"):
        retry_with_backoff(always_fail, max_retries=2, base_delay=0.001)


def test_retry_non_retryable():
    def bad_request():
        e = Exception("bad request")
        e.status_code = 400
        raise e

    with pytest.raises(Exception, match="bad request"):
        retry_with_backoff(bad_request, max_retries=3, base_delay=0.001)


# ── DB-backed Cache Tests ──


def test_cache_db_mode_mock():
    from unittest.mock import MagicMock, patch

    from hybrid_rag.models.cache import LLMCacheEntry

    cache = LLMCache(use_db=True)
    mock_session = MagicMock()
    mock_entry = LLMCacheEntry(key="k1", response='{"val": 123}', model="test")
    mock_session.get.return_value = mock_entry

    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_session
    mock_cm.__exit__.return_value = False

    with patch("hybrid_rag.db.engine.SessionLocal", return_value=mock_cm):
        val = cache.get("k1")
        assert val == '{"val": 123}'
        mock_session.get.assert_called_once_with(LLMCacheEntry, "k1")

        mock_session.get.return_value = None
        cache.put("k2", '{"val": 456}', "model-x")
        assert mock_session.commit.called

        cache.clear()
        assert mock_session.query.return_value.delete.called

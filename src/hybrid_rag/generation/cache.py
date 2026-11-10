"""Deterministic LLM call cache and token-bucket rate limiter."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _default_cache_dir() -> Path:
    cache_env = os.getenv("LLM_CACHE_DIR")
    if cache_env:
        p = Path(cache_env)
        p.mkdir(parents=True, exist_ok=True)
        return p
    fallback = Path(".cache/gemini")
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def cache_key(
    model: str,
    prompt: str,
    schema_json: str,
    temperature: float,
    system_instruction: str = "",
) -> str:
    canonical = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "schema": schema_json,
            "system_instruction": system_instruction,
            "temperature": temperature,
        },
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


class LLMCache:
    def __init__(self, cache_dir: Path | None = None, use_db: bool | None = None):
        self._lock = threading.Lock()
        if use_db is None:
            self._use_db = cache_dir is None
        else:
            self._use_db = use_db

        self._db_path = None
        if not self._use_db or cache_dir is not None:
            self._dir = cache_dir or _default_cache_dir()
            self._dir.mkdir(parents=True, exist_ok=True)
            self._db_path = self._dir / "cache.db"
            self._init_sqlite()

    def _init_sqlite(self) -> None:
        if self._db_path is None:
            return
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS llm_cache ("
                "  key TEXT PRIMARY KEY,"
                "  response TEXT NOT NULL,"
                "  model TEXT,"
                "  created_at REAL"
                ")"
            )

    def get(self, key: str) -> str | None:
        if self._use_db:
            try:
                from hybrid_rag.db.engine import SessionLocal
                from hybrid_rag.models.cache import LLMCacheEntry

                with SessionLocal() as session:
                    entry = session.get(LLMCacheEntry, key)
                    if entry is not None:
                        return entry.response
            except Exception as e:
                logger.warning(
                    "PostgreSQL cache lookup failed: %s",
                    e,
                    exc_info=True,
                )
                if self._db_path is None:
                    return None

        if self._db_path:
            with self._lock, sqlite3.connect(str(self._db_path)) as conn:
                row = conn.execute(
                    "SELECT response FROM llm_cache WHERE key = ?", (key,)
                ).fetchone()
                return row[0] if row else None
        return None

    def put(self, key: str, response: str, model: str = "") -> None:
        if self._use_db:
            try:
                from hybrid_rag.db.engine import SessionLocal
                from hybrid_rag.models.cache import LLMCacheEntry

                with SessionLocal() as session:
                    entry = session.get(LLMCacheEntry, key)
                    if entry is not None:
                        entry.response = response
                        entry.model = model
                        entry.created_at = time.time()
                    else:
                        entry = LLMCacheEntry(
                            key=key,
                            response=response,
                            model=model,
                            created_at=time.time(),
                        )
                        session.add(entry)
                    session.commit()
                    return
            except Exception as e:
                logger.warning(
                    "PostgreSQL cache write failed: %s",
                    e,
                    exc_info=True,
                )
                if self._db_path is None:
                    # In DB-first mode on container platforms (HF Spaces), skip creating ephemeral SQLite files
                    return

        if self._db_path is None:
            self._dir = _default_cache_dir()
            self._dir.mkdir(parents=True, exist_ok=True)
            self._db_path = self._dir / "cache.db"
            self._init_sqlite()

        if self._db_path:
            with self._lock, sqlite3.connect(str(self._db_path)) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO llm_cache (key, response, model, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (key, response, model, time.time()),
                )

    def evict_stale(self, max_age_seconds: float = 7 * 86400) -> int:
        cutoff = time.time() - max_age_seconds
        evicted = 0
        if self._use_db:
            try:
                from hybrid_rag.db.engine import SessionLocal
                from hybrid_rag.models.cache import LLMCacheEntry

                with SessionLocal() as session:
                    evicted = (
                        session.query(LLMCacheEntry)
                        .filter(LLMCacheEntry.created_at < cutoff)
                        .delete()
                    )
                    session.commit()
            except Exception as e:
                logger.warning("PostgreSQL cache eviction failed: %s", e, exc_info=True)

        if self._db_path:
            with self._lock, sqlite3.connect(str(self._db_path)) as conn:
                cursor = conn.execute(
                    "DELETE FROM llm_cache WHERE created_at < ?", (cutoff,)
                )
                evicted += cursor.rowcount
        return evicted

    def clear(self) -> None:
        if self._use_db:
            try:
                from hybrid_rag.db.engine import SessionLocal
                from hybrid_rag.models.cache import LLMCacheEntry

                with SessionLocal() as session:
                    session.query(LLMCacheEntry).delete()
                    session.commit()
            except Exception as e:
                logger.warning("PostgreSQL cache clear failed: %s", e, exc_info=True)

        if self._db_path:
            with self._lock, sqlite3.connect(str(self._db_path)) as conn:
                conn.execute("DELETE FROM llm_cache")


class TokenBucketRateLimiter:
    def __init__(self, rate: float = 10.0, capacity: float = 10.0):
        self._rate = rate
        self._capacity = capacity
        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    def acquire(self, timeout: float = 60.0) -> bool:
        if self._rate <= 0:
            return False
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
                now = time.monotonic()
                if now >= deadline:
                    return False
                needed = (1.0 - self._tokens) / self._rate
                wait_time = min(needed, deadline - now)
                if wait_time <= 0:
                    wait_time = 0.001
                self._condition.wait(timeout=wait_time)

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        if self._rate > 0:
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now


def retry_with_backoff(
    fn,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_codes: tuple[int, ...] = (429, 503),
):
    import random

    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exception = e
            status_code = getattr(e, "status_code", None) or getattr(e, "code", None)
            if status_code not in retryable_codes or attempt == max_retries:
                raise
            delay = min(base_delay * (2**attempt), max_delay)
            jitter = random.uniform(0, delay * 0.5)
            time.sleep(delay + jitter)
    raise last_exception  # type: ignore[misc]

# Code Review — Deep Audit Pass

**Date:** 2026-11-09  
**Scope:** Full codebase — backend, frontend, DB, APIs, pipelines, background tasks  
**Focus:** Logic & Contract Errors, Concurrency & Resources, Resilience & Error Handling, Security  

---

## Verified Findings

### CR-01 — RRF Confidence Normalization Is Non-Discriminative [High]

**Location:** `src/hybrid_rag/generation/confidence.py:20-28`

**Root Cause:** RRF scores are inherently tightly clustered near `1/(rrf_k+1)`. With default `rrf_k=60`, the theoretical max score is `1/61 ≈ 0.0164`, and the second-best candidate (rank 1 in one list, rank 2 in the other) scores `≈ 0.0161`. After dividing by `max_rrf`, top scores all map to 0.95–1.0.

**Impact:** The `retrieval_score` component (40% of composite confidence) is always near 1.0 for any query that returns results. Confidence scores never reflect genuinely poor retrieval quality, making the composite score misleading for CRAG trigger decisions and user-facing confidence badges.

**Fix:**

```python
def retrieval_score(trace: QueryTrace) -> float:
    if not trace.stages:
        return 0.0
    last_stage = trace.stages[-1]
    if not last_stage.candidates:
        return 0.0

    top_score = max(c.score for c in last_stage.candidates)

    if last_stage.stage in ("rrf", "multi_query_rrf"):
        n = len(last_stage.candidates)
        if n <= 1:
            return 0.3
        scores = sorted((c.score for c in last_stage.candidates), reverse=True)
        median_score = scores[n // 2]
        if median_score > 0:
            spread = min(1.0, (scores[0] / median_score - 1.0) * 2)
        else:
            spread = 1.0
        fill_ratio = min(1.0, n / max(1, trace.config.fusion_top_k if trace.config else 10))
        return max(0.0, min(1.0, 0.3 + 0.4 * spread + 0.3 * fill_ratio))

    return max(0.0, min(1.0, top_score))
```

---

### CR-02 — LLMCache Silently Swallows All Exceptions [Medium]

**Location:** `src/hybrid_rag/generation/cache.py:84` (get), `117-118` (put)

**Root Cause:** Both PostgreSQL `get` and `put` paths catch bare `Exception` and `pass`. On a transient DB error (connection timeout, pool exhausted), `get` returns `None` (cache miss), and `put` silently fails.

**Impact:**
- Cache miss on transient error → unnecessary LLM API call → burns free-tier quota
- `put` failure → response never cached → same query repeats the API call every time
- No logging means the failure is invisible in production

**Fix:**

```python
import logging
logger = logging.getLogger(__name__)

# In get():
except Exception:
    logger.warning("PostgreSQL cache lookup failed; falling back to SQLite", exc_info=True)

# In put():
except Exception:
    logger.warning("PostgreSQL cache write failed; falling back to SQLite", exc_info=True)
```

---

### CR-03 — BM25 Rebuild Fallback Uses Rolled-Back Session [Medium]

**Location:** `src/hybrid_rag/ingest/index_manager.py:87-88`

**Root Cause:** After a failed `index_document`, the compensating rollback creates a fresh session for BM25 rebuild. If that also fails, the code falls back to the original `db` session — which has been rolled back and may be in an invalid state.

**Impact:** BM25 index could be rebuilt from an inconsistent session state, leading to stale or corrupt sparse retrieval results until next app restart.

**Fix:**

```python
try:
    with SessionLocal() as fresh_db:
        store.rebuild_from_db(fresh_db, strategy)
except Exception:
    logger.warning(
        "BM25 index rebuild failed during compensating rollback for document %s; "
        "sparse retrieval may be stale until next restart",
        document.id,
        exc_info=True,
    )
```

Remove the `store.rebuild_from_db(db, strategy)` fallback on line 88 entirely.

---

### CR-04 — Cross-Thread SQLAlchemy Session in hybrid_retrieve [Medium]

**Location:** `src/hybrid_rag/retrieval/retriever.py:266-267`

**Root Cause:** `hybrid_retrieve` submits `dense_retrieve(query, config, db)` to a `ThreadPoolExecutor`. If `db` is not `None`, a SQLAlchemy session is shared across threads. SQLAlchemy `Session` is explicitly documented as not thread-safe.

**Impact:** Currently mitigated because the SSE query endpoint passes `db=None`. However, any future caller passing a real session triggers concurrent access causing potential data corruption or `OperationalError`.

**Fix:**

```python
with ThreadPoolExecutor(max_workers=2) as executor:
    dense_future = executor.submit(dense_retrieve, query, config, None)
    sparse_future = executor.submit(sparse_retrieve, query, config, bm25_store)
```

Always force `db=None` in the thread pool path so `dense_retrieve` creates its own session.

---

### CR-05 — Evaluation Dashboard Shows Hardcoded Fabricated Data [Medium]

**Location:** `frontend/src/views/Evaluation.tsx:66-181`

**Root Cause:** All benchmark data (COMPARISON_METRICS, TRADEOFF_DATA, LATENCY_BREAKDOWN, CATEGORY_BENCHMARKS, WIN_LOSS_CASES) are hardcoded constants with specific numbers (87.0% Hit Rate, 0.810 MRR, 155ms latency) not sourced from any evaluation run.

**Impact:** Users see specific metric numbers presented as "Empirical Retrieval & RAG Benchmark" results that do not correspond to any real experiment. Misleading for evaluation-driven research claims.

**Fix:** Either fetch real results from a backend evaluation API, or add a prominent disclaimer that these are placeholder values for layout demonstration.

---

### CR-06 — Token Bucket Rate Limiter Busy-Wait [Low]

**Location:** `src/hybrid_rag/generation/cache.py:153-165`

**Root Cause:** `TokenBucketRateLimiter.acquire()` polls in a tight loop with `time.sleep(min(0.1, 1/rate))`. Under contention, multiple threads spin-wait simultaneously.

**Impact:** Minor CPU waste. Acceptable for single-worker Heroku but would become problematic at scale.

**Fix (optional):** Replace `threading.Lock` with `threading.Condition` and use `condition.wait(timeout)` instead of polling.

---

## Previously Identified — Now Fixed

These findings from the first review pass have been verified as fixed in the current codebase:

| ID | Issue | Status |
|----|-------|--------|
| CR-01 (old) | `response.text` None crash in generator/verifier | **Fixed** — `or ""` + `TypeError` catch |
| CR-02 (old) | Cross-thread SQLAlchemy session in SSE endpoint | **Fixed** — SSE endpoint passes `db=None` |
| CR-03 (old) | Unvalidated batch delete body | **Fixed** — `BatchDeleteRequest` Pydantic model + UUID validation |
| CR-05 (old) | Error message leaks credentials in SSE | **Fixed** — `_format_error_message` redacts in dev, generic in prod |
| WR-01 | CRAG broadening was no-op | **Fixed** — multipliers applied |
| WR-02 | Unescaped XML in prompt context | **Fixed** — `quoteattr` + `escape_delimiter_tags` |
| WR-07 | CORS wildcard `["*"]` | **Fixed** — per-environment origins list |

## Summary

| Severity | Count | IDs |
|----------|-------|-----|
| High | 1 | CR-01 |
| Medium | 4 | CR-02, CR-03, CR-04, CR-05 |
| Low | 1 | CR-06 |
| **Total** | **6** | |

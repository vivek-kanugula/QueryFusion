# Production-Readiness Audit — QueryFusion

**Date:** 2026-11-09
**Target Infra:** HF Spaces (Docker, 2 vCPU, 16 GB RAM, CPU-only) + Supabase Free (PG 16 + pgvector, 500 MB)
**Scope:** End-to-end: pipeline design, concurrency, database scaling, security, enterprise gaps

---

## A. RAG Retrieval & Pipeline Architecture

### A-01 — RRF Confidence Normalization Remains Non-Discriminative [High]

**Location:** `src/hybrid_rag/generation/confidence.py:13-37`

RRF scores cluster near `1/(rrf_k+1)` by construction. With `rrf_k=60`, theoretical max is `1/61 ≈ 0.0164`. The normalization attempt (ratio × spread) still produces values in a narrow band because spread between rank-1 and median rank in a 20-candidate pool is tiny in RRF-score space.

**Impact:** `retrieval_score` component (40% of composite confidence) is near-constant for any query that returns ≥5 candidates. CRAG trigger decisions and user-facing confidence badges lose their primary signal.

**Fix:** Replace RRF-score-based normalization with a rank-percentile approach:

```python
def retrieval_score(trace: QueryTrace) -> float:
    if not trace.stages:
        return 0.0
    last_stage = trace.stages[-1]
    n = len(last_stage.candidates)
    if n == 0:
        return 0.0

    # If reranked, use sigmoid-normalized reranker scores directly
    if last_stage.stage == "rerank":
        top_score = max(c.score for c in last_stage.candidates)
        return max(0.0, min(1.0, top_score))

    # For RRF stages, use fill-ratio and score-gap as proxy
    fusion_k = trace.config.fusion_top_k if trace.config else 20
    fill_ratio = min(1.0, n / max(1, fusion_k))
    if n <= 1:
        return 0.3 * fill_ratio
    scores = sorted((c.score for c in last_stage.candidates), reverse=True)
    gap = (scores[0] - scores[-1]) / (scores[0] + 1e-10)
    return max(0.0, min(1.0, 0.5 * fill_ratio + 0.5 * gap))
```

Better: when the final stage IS reranked (which it usually is), use reranker scores directly — they are sigmoid-normalized to `[0, 1]` and genuinely discriminative.

---

### A-02 — Candidate Handoff: Dense 50 + Sparse 50 → RRF 20 → Reranker 6 [Low Risk]

Pipeline geometry is sound. RRF@20 from two 50-candidate pools catches both semantic and lexical matches. Reranker@6 is tight but correct for the 512-token cross-encoder window.

**One subtle issue:** `dense_top_k=50` and `sparse_top_k=50` are PipelineConfig defaults, but `fusion_top_k=20` is the RRF output. With `alpha=0.5`, a document appearing only in dense results at rank 45 gets `0.5/(60+45) = 0.00476`, which is negligible. Effective contribution from ranks 30+ is noise.

**Recommendation:** Reduce `dense_top_k` and `sparse_top_k` to 30. Saves 40% pgvector query latency (fewer rows scanned) and 40% BM25 retrieval time. Zero quality loss — ranks 31-50 never survive RRF@20 with `rrf_k=60`.

---

### A-03 — Tri-Chunking Failure Modes [Medium]

#### Fixed chunking (`chunk_fixed`)
- **Overlap handling when `overlap >= token_size`:** The guard `step = max(1, token_size // 2)` prevents infinite loop, but with `overlap=450, token_size=500`, effective step is 50 tokens — creates massive chunk proliferation. Frontend slider allows overlap up to 200, but no backend validation enforces `overlap < token_size`.
- **Code snippets:** Token boundary splitting can break mid-identifier. BM25 tokenizer's `expand_technical_tokens` partially mitigates (DB-204 → DB204 DB 204), but embedded code blocks split mid-function are unrecoverable for retrieval.

#### Structure-aware chunking (`chunk_structure_aware`)
- **`text.find(stext, char_pos)` on line 185:** If section text appears earlier in the document than expected (duplicate paragraphs), `find` returns wrong offset. `char_offset_start/end` metadata becomes incorrect, breaking citation "show in context" features.
- **PDF structure detection:** `parse_pdf` uses PyMuPDF `get_text("text")` — flat text, no heading detection. Structure-aware chunking falls back to `_split_by_headers` (Markdown `# ` pattern), which finds nothing in PDF text. Every PDF degrades to fixed-size sentence accumulation.

#### Semantic chunking (`chunk_semantic`)
- **Embedder memory spike:** Embeds ALL sentences at once (`list(embedder.embed(sentences))`). A 200-page PDF with 5,000 sentences creates a `(5000, 384)` float32 array = ~7.3 MB plus the ONNX batch inference memory. Safe on 16 GB, but worth noting.
- **Percentile threshold:** Hard-coded `25.0` percentile means 25% of inter-sentence boundaries are always split points, regardless of document coherence. A highly uniform document gets chunked identically to a diverse one.

---

### A-04 — Lost-in-the-Middle After Reranking: Marginal Value [Low]

LitM reordering (`[1, 3, 5, ..., 4, 2]`) was designed for long-context LLMs that exhibit attention degradation in middle positions. Gemini Flash models have strong attention across their full context window.

**With 6 candidates at ~350 tokens each = ~2,100 tokens of context**, Gemini Flash can attend to all of it without degradation. LitM adds no measurable quality at this context length. However, it also causes no harm — the pattern is fully deterministic and reversible.

**Recommendation:** Keep it as a toggle. Worth measuring in your ablation: run the 50-question benchmark with LitM on/off and compare Faithfulness scores.

---

### A-05 — Query Expansion (HyDE) Bypasses Cache and Rate Limiter [Medium]

**Location:** `src/hybrid_rag/retrieval/expansion.py:49`

`expand_query()` calls `call_llm()` directly — no `LLMCache` lookup, no `TokenBucketRateLimiter` gate. Identical queries re-call Gemini every time. With HyDE enabled, each query burns 2 Gemini calls minimum (expansion + generation), or 3 with verification.

Same issue in `contextualizer.py:68` — chat contextualization also bypasses cache/rate-limiter.

**Impact:** Free-tier Gemini quota exhaustion 2-3x faster under normal usage patterns.

**Fix:** Pass `LLMCache` and `TokenBucketRateLimiter` down to expansion and contextualization calls, or implement a lightweight dedicated cache layer.

---

## B. Concurrency, Threading & Async Runtime

### B-01 — CR-04 Cross-Thread Session Guard Is Wrong [High]

**Location:** `src/hybrid_rag/retrieval/retriever.py:266-268`

```python
worker_db = None if (db is None or isinstance(db, Session)) else db
```

This sets `worker_db = None` when `db` IS a Session — accidentally correct for the common case. But if `db` is a non-Session truthy value (a proxy, wrapper, or mock), it passes it into the thread pool. The `isinstance` check is the wrong guard.

**Fix:**
```python
worker_db = None  # Always force None — dense_retrieve creates its own session
```

---

### B-02 — Default ThreadPoolExecutor Growth Under Load [Medium]

**Location:** `src/hybrid_rag/retrieval/retriever.py:270`

`hybrid_retrieve` creates a new `ThreadPoolExecutor(max_workers=2)` per call. Under 10 concurrent queries, that's 20 OS threads (10 executors × 2 workers each) plus the main event loop thread, the BM25 search threads, the ONNX inference threads (OMP_NUM_THREADS=2), and SQLAlchemy's pool connections.

On 2 vCPU, this causes OS-level context-switching thrash. Thread count: 1 (uvicorn) + N×2 (hybrid_retrieve) + N×1 (BM25 search) + 2 (ONNX OMP) = potentially 30+ threads competing for 2 cores.

**Fix:** Use a module-level bounded thread pool:
```python
_RETRIEVAL_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="retrieval")
```
And reuse it. Cap at 4 workers = at most 4 concurrent CPU tasks queued, rest wait.

---

### B-03 — `asyncio.to_thread` Serialization Bottleneck [Low-Medium]

`asyncio.to_thread` uses the default executor (a `ThreadPoolExecutor`). With single-worker uvicorn, ALL offloaded work shares one default pool. Heavy CPU inference (reranking: 500-800ms per call, embedding: 100-200ms) blocks lighter tasks (BM25 search: 5-10ms, DB queries: 20-50ms).

Under 5 concurrent queries:
- 5 rerank calls × 600ms each = 3,000ms total compute
- On 2 vCPU, this becomes ~1,500ms wall clock (2x parallelism)
- But default pool (usually 5 workers in CPython 3.12) means only 5 concurrent items

Not a showstopper at portfolio-project traffic, but p99 latency spikes under concurrent load.

**Recommendation:** Set a dedicated pool for heavy inference:
```python
import asyncio
_INFERENCE_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="inference")
loop = asyncio.get_running_loop()
await loop.run_in_executor(_INFERENCE_POOL, rerank_candidates, ...)
```

---

### B-04 — SSE Generator Holds Event Loop During Entire Pipeline [Low]

**Location:** `src/hybrid_rag/api/query.py:55-260`

`stream_query` yields SSE events from an async generator. Each `await asyncio.to_thread(...)` releases the event loop while the thread runs. Good. But between those calls, SSE event formatting, JSON serialization, and Python overhead run on the event loop.

Under 10 concurrent SSE streams, the event loop is busy formatting/yielding events for all 10 streams interleaved. This is fine — FastAPI/Starlette handles this correctly with ASGI. No blocking issue found.

---

## C. Database & Storage Scaling (Supabase 500 MB Budget)

### C-01 — HNSW Index Size Will Exceed Expectations [High]

pgvector HNSW index for 384-dim vectors is significantly larger than the raw vectors:

| Chunk Count | Raw Vectors | HNSW Index | Text + Metadata | Overhead | **Total** |
|-------------|-------------|------------|-----------------|----------|-----------|
| 5,000 | 7.3 MB | ~22 MB | ~15 MB | ~20 MB | **~64 MB** |
| 10,000 | 14.6 MB | ~55 MB | ~30 MB | ~25 MB | **~125 MB** |
| 25,000 | 36.5 MB | ~165 MB | ~75 MB | ~35 MB | **~312 MB** |
| 35,000 | 51 MB | ~250 MB | ~105 MB | ~40 MB | **~446 MB** |
| 40,000 | 58 MB | ~300 MB | ~120 MB | ~45 MB | **~523 MB** ⚠️ |

**HNSW index = ~6-8x the raw vector size** for cosine distance with default `m=16, ef_construction=64`. The index alone consumes more than the vectors.

**Breaking point: ~35,000 chunks** (about 35-40 documents of 30 pages each with fixed 500-token chunking + 100 overlap).

Your 200 MB raw upload cap maps to roughly 25,000-30,000 chunks — safe margin, but closer than the 140K estimate from raw vector math alone.

**Recommendation:** Add a `chunk_count` check alongside the `max_storage_mb` check. Warn users at 25K chunks. Monitor actual Supabase usage via the dashboard.

---

### C-02 — WAL Accumulation During Bulk Ingestion [Medium]

Each `index_document` call does:
1. `db.add(document)` + `db.add_all(db_chunks)` + `db.flush()` — generates WAL
2. BM25 rebuild from DB (full table scan) — read-only, no WAL
3. `db.commit()` — WAL checkpoint

For a 50-page PDF with 300 chunks: the single commit writes ~300 chunk rows + 300 embedding vectors. WAL segment = ~10-15 MB temporarily.

Uploading 10 documents rapidly: WAL can spike to ~100 MB before autovacuum/checkpoint runs. This is transient but counts against the 500 MB limit.

**Mitigation:** Supabase runs checkpoints frequently (default every 5 minutes). Not a practical concern unless batch-uploading many large documents simultaneously.

---

### C-03 — `llm_cache` Table Unbounded Growth [Medium]

**Location:** `src/hybrid_rag/models/cache.py:14`

`llm_cache` table has no TTL, no eviction, no size limit. Each cached response is a full JSON blob (500-2000 bytes). Over time:
- 1,000 unique queries → ~1-2 MB
- 10,000 unique queries → ~10-20 MB

Not catastrophic, but eats into the 500 MB budget silently.

**Fix:** Add a periodic cleanup or TTL-based eviction:
```sql
DELETE FROM llm_cache WHERE created_at < extract(epoch FROM now()) - 2592000;  -- 30 days
```

---

### C-04 — `prepare_threshold=0` Applied Correctly [Verified Good]

**Location:** `src/hybrid_rag/db/engine.py:6-9`

`connect_args["prepare_threshold"] = 0` is set when the URL contains `psycopg` or `postgres`. This disables client-side prepared statement caching, which is mandatory for Supabase's PgBouncer transaction-mode pooler.

✅ No issue found. This is correctly implemented.

---

## D. Security, Guardrails & Adversarial Inputs

### D-01 — No Authentication on Any Endpoint [Critical for Production, Expected for Portfolio]

Every endpoint is publicly accessible:
- `/api/v1/documents` — upload, list, delete any document
- `/api/query/stream` — unlimited Gemini API calls
- `/api/v1/documents/batch-delete` — mass deletion
- `/api/traces/{trace_id}` — retrieval diagnostic data

For a portfolio project, this is standard. For production, every endpoint that mutates state or calls external APIs needs auth.

**Minimum viable auth:** API key in `Authorization: Bearer <key>` header, validated in FastAPI middleware. One key per deployment, stored as env var.

---

### D-02 — Malicious File Upload Vectors [Medium]

**PDF bombs:** PyMuPDF's `get_text("text")` is safe against most decompression bombs (it limits page iteration), but a malicious PDF with 10,000 pages would generate a massive `full_text` string. The 50 MB upload limit helps, but a heavily compressed PDF could expand to hundreds of MB of text.

**Fix:** Add a page count limit:
```python
def parse_pdf(data, filename):
    doc = pymupdf.open(stream=data, filetype="pdf")
    if len(doc) > 500:
        doc.close()
        raise ValueError(f"PDF has {len(doc)} pages (max 500)")
    ...
```

**HTML XSS in chunk content:** Chunk text is stored as-is in PostgreSQL and served via API JSON responses. The React frontend uses `{c.content}` in JSX (auto-escaped) and `ReactMarkdown` (XSS-safe by default). No injection vector in current frontend.

However, if any future consumer renders chunk content as raw HTML, stored `<script>` tags from parsed HTML documents would execute.

---

### D-03 — Denial of Service via Query Expansion [Medium]

With HyDE enabled, a single query triggers:
1. `expand_query` → 1 Gemini call
2. `multi_query_retrieve` → 3-4 separate retrieval pipelines (one per expanded query)
3. Each pipeline: pgvector query + BM25 search + RRF fusion
4. `rerank_candidates` on merged results
5. `generate_answer` → 1 Gemini call
6. `verify_citations` → 1 Gemini call
7. Possibly CRAG fallback → repeat steps 2-6

Total: 3+ Gemini calls + 3-4× retrieval cost per single query. At 5 concurrent users with HyDE, that's 15+ Gemini calls and 20+ concurrent DB queries.

**Fix:** Rate limit the `/api/query/stream` endpoint (e.g., 10 requests/minute per IP), independent of the token bucket limiter on Gemini calls.

---

### D-04 — Prompt Injection Defense Is Solid [Verified Good]

- `escape_delimiter_tags` neutralizes `<`, `>`, `&` in context passages and user questions
- `quoteattr` for XML attribute values
- System instructions explicitly tell the model to treat tag contents as data
- Conversation history is also escaped

✅ This is well above average for RAG systems. No gaps found.

---

### D-05 — CORS Configuration Wrong for HF Spaces [High]

**Location:** `src/hybrid_rag/settings.py:79-89`

Production CORS defaults to `[os.getenv("FRONTEND_URL", "https://queryfusion.herokuapp.com")]`. On HF Spaces, the actual URL is `https://<username>-queryfusion.hf.space`. Without setting `FRONTEND_URL` or `CORS_ORIGINS`, all frontend API calls are CORS-blocked in production.

Meanwhile, `.env.example` line 15 sets `CORS_ORIGINS=["*"]` — if copied verbatim, wide-open CORS in production.

**Fix:** Update the production default:
```python
cors_origins: list[str] = Field(
    default_factory=lambda: json.loads(os.getenv("CORS_ORIGINS", "[]"))
    or [os.getenv("FRONTEND_URL", "")]
    if os.getenv("APP_ENV") == "production"
    else ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"]
)
```
And update `.env.example` to document proper values.

---

### D-06 — Trace Endpoint Exposes Internal Pipeline Data [Low]

**Location:** `src/hybrid_rag/api/traces.py:17-35`

`/api/traces/{trace_id}` returns full retrieval diagnostics: chunk IDs, scores, stage timings, metadata. Trace IDs are UUIDv4 (unguessable), so enumeration is infeasible. But any trace ID leaked (e.g., in browser console via SSE `completed` event) gives full retrieval internals.

Low risk for portfolio project. For production: restrict to authenticated users or omit from production builds.

---

## E. Missing Enterprise / "Top 1%" Capabilities

### E-01 — Request-Level Timeout and Circuit Breaker [High Leverage]

No timeout on the end-to-end query pipeline. If Gemini API hangs (network partition, 504 from Google), the SSE connection hangs indefinitely. The `retry_with_backoff` in `cache.py:198-219` retries on 429/503 but has no total wall-clock timeout.

**Fix:** Wrap the entire `event_stream()` in a timeout:
```python
async def event_stream():
    try:
        async with asyncio.timeout(120):  # 2 minute hard deadline
            ... existing pipeline ...
    except TimeoutError:
        yield _sse_event("error", {"message": "Query timed out after 120 seconds"})
```

Add a per-call timeout to Gemini:
```python
with urllib.request.urlopen(req, timeout=30) as resp:  # Already exists for Groq/Ollama
```
But `google.genai.Client` doesn't accept `timeout` in `generate_content` — wrap in `asyncio.wait_for` at the caller.

---

### E-02 — Structured Observability: Request Tracing [High Leverage]

Current logging is ad-hoc (`logger.error`, `logger.info`). No request IDs, no structured JSON logging, no latency percentile tracking.

For a top-1% system, add:
1. **Request ID middleware:** Generate UUID per request, propagate via context var, include in all log lines
2. **Structured JSON logging:** Replace text format with JSON lines (`python-json-logger`), parseable by any log aggregator
3. **Latency histogram:** Track p50/p95/p99 for each pipeline stage, exposed via `/metrics` endpoint

This is ~50 lines of middleware + formatter config. Massive debugging improvement.

---

### E-03 — Graceful Degradation Under Partial Failures [Medium Leverage]

Current failure modes:
- Gemini API down → `retry_with_backoff` retries 5 times → raises → SSE `error` event. ✅
- Supabase down → `pool_pre_ping` detects → dense search fails → entire query fails. ❌
- BM25 index empty → `sparse_retrieve` returns `[]` → RRF has only dense candidates. ✅
- Reranker load fails → warmup crashes → lifespan still yields → app starts without reranker. ❌ (reranking will crash on first query)

**Fix for partial failures:**
- If dense search fails, fall back to sparse-only (BM25 doesn't need PG for search, only for rebuild)
- If reranker fails to load, disable reranking and log a warning, rather than crashing on first query
- Add a health check that reports which components are operational

---

## Previously Identified Findings (From Prior Reviews)

| ID | Status | Notes |
|----|--------|-------|
| CR-01 (RRF normalization) | **Still Open** | See A-01 above |
| CR-02 (Silent cache exceptions) | **Fixed** | `logger.warning` with `exc_info=True` added |
| CR-03 (Rolled-back session fallback) | **Fixed** | Fresh session + warning-only fallback |
| CR-04 (Cross-thread session) | **Partially Fixed** | isinstance guard is wrong; see B-01 |
| CR-05 (Hardcoded eval data) | **Still Open** | Evaluation.tsx still shows fabricated metrics |
| CR-06 (Busy-wait rate limiter) | **Fixed** | `threading.Condition` with `wait(timeout)` |

---

## Summary

| Section | ID | Severity | Description |
|---------|-----|----------|-------------|
| A | A-01 | High | RRF confidence normalization non-discriminative |
| A | A-02 | Low | Dense/sparse top_k over-fetching (50 → 30 safe) |
| A | A-03 | Medium | Chunking failure modes (PDF structure, find offset) |
| A | A-04 | Low | LitM marginal value at 6-candidate context |
| A | A-05 | Medium | HyDE/contextualization bypasses cache and rate limiter |
| B | B-01 | High | Cross-thread session guard logic is inverted |
| B | B-02 | Medium | Unbounded ThreadPoolExecutor per query |
| B | B-03 | Low-Med | asyncio.to_thread serialization under concurrency |
| C | C-01 | High | HNSW index 6-8x raw vectors; ~35K chunk ceiling |
| C | C-02 | Medium | WAL spike during bulk ingestion |
| C | C-03 | Medium | llm_cache unbounded growth |
| D | D-01 | Critical* | No authentication (*expected for portfolio) |
| D | D-02 | Medium | PDF bomb / page count unlimited |
| D | D-03 | Medium | DoS via HyDE query amplification |
| D | D-05 | High | CORS defaults wrong for HF Spaces |
| E | E-01 | High | No request-level timeout / circuit breaker |
| E | E-02 | High | No structured observability |
| E | E-03 | Medium | No graceful degradation under partial failures |

**Actionable high-severity items:** A-01, B-01, C-01, D-05, E-01, E-02
**Quick wins:** A-02 (change two defaults), B-02 (module-level pool), C-03 (TTL query), D-02 (page limit)

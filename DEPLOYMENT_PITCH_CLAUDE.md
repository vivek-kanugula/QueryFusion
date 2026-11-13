# QueryFusion: Production Deployment Pitch & Architecture Risk Audit

**Target Audience:** Cloud Architect / Claude Reviewer  
**System:** QueryFusion (Hybrid Dense-Sparse RAG + Cross-Encoder Reranking + React SPA)  
**Date:** November 2026  
**Status:** Pre-Deployment / Infrastructure Hardening  

---

## 1. Executive Summary & Architecture Topology

QueryFusion is a unified full-stack application designed to empirically compare and benchmark dense vector search (`bge-small` via pgvector HNSW), sparse lexical search (`bm25s`), reciprocal rank fusion ($\alpha$-weighted RRF), and cross-encoder reranking (`bge-reranker-base` via ONNX).

```
                            [ Web Client / Browser ]
                                       │
                                       ▼ (HTTPS / WSS / SSE)
               ┌───────────────────────────────────────────────┐
               │    Unified Docker Container (Port 8080)       │
               │                                               │
               │  FastAPI (Single Worker, Async Runtime)       │
               │  ├─ Static SPA Mount (frontend/dist)          │
               │  ├─ In-Memory BM25 Store (Thread-safe Lock)   │
               │  ├─ In-Memory Job Registry & Trace LRU        │
               │  └─ CPU ONNX Runtime (fastembed, OMP=1)       │
               └───────────┬───────────────────────┬───────────┘
                           │                       │
              SQLAlchemy   │                       │ boto3 SigV4
              (Psycopg 3)  ▼                       ▼ (Path-Style)
       ┌───────────────────────────────┐   ┌───────────────────────────┐
       │   Neon Serverless Postgres    │   │  Neon S3 Object Storage   │
       │   (pgvector 0.8.6, PG 18)     │   │  (Private Buckets)        │
       │   ├─ Pooled (-pooler: 5432)   │   │  ├─ 'uploads' (raw docs)  │
       │   └─ Direct (Unpooled: 5432)  │   │  └─ 'assets'              │
       └───────────────────────────────┘   └───────────────────────────┘
```

### Deployed Artifacts
- **Docker Image:** Multi-stage build (`node:22-slim` + `python:3.12-slim`).
- **Base Footprint:** ~1.4 GB image; pre-bakes model weights for `bge-small-en-v1.5` and `bge-reranker-base` in `/app/.cache/fastembed`.
- **Runtime Topology:** Single container holding FastAPI backend + pre-built React 19 SPA.
- **External State:** Neon Postgres (`hybrid_rag` database) + Neon S3 Object Storage (`uploads` bucket).

---

## 2. Key Architectural Bottlenecks & Critical Hidden Traps

### Trap 1: The "Multi-Worker / Multi-Replica" Illusion (State Invalidation)
- **The Trap:** Common cloud platforms (e.g., Render, Railway, AWS ECS, GCP Cloud Run) encourage horizontal autoscaling or setting `uvicorn --workers 4`.
- **Impact in QueryFusion:**
  1. **BM25 Split-Brain:** The BM25 index is completely **in-memory** (`bm25s`). Uploading a document on Worker A rebuilds Worker A's index. Workers B, C, and D remain stale unless restarted.
  2. **SSE Job Registry Desync:** Async ingest jobs and SSE query traces are managed in memory via `JobRegistry` and `TraceStore`. If a client initiates a streaming query that hits Worker A, but the browser reconnects or inspects `/api/traces/{trace_id}` on Worker B, it returns `404 Not Found`.
  3. **RAM Explosion:** Pre-loaded FastEmbed ONNX models (`bge-small` + `bge-reranker-base`) consume ~350 MB RSS per process. 4 workers = 1.4 GB RAM consumed before serving a single query.
- **Hard Rule:** Deployment **must be pinned to exactly 1 replica / 1 uvicorn worker**.

---

### Trap 2: Aggressive "Scale-to-Zero" Cold-Start Cascades
- **The Trap:** Hosting on serverless container platforms (e.g., Google Cloud Run, Fly.io auto-stop, Render Free Tier) with `min-instances = 0`.
- **Compounding Latency Breakdown:**
  1. **Container Cold Boot:** Container pull/start + Python import overhead: ~4–8s.
  2. **Neon DB Wake-up:** Neon Free Tier compute suspends after 5 min idle. Wake-up latency: ~1.5–3s.
  3. **BM25 Rebuild on Lifespan:** `lifespan` in `app.py` queries all existing document chunks across all three strategies (`fixed`, `structure`, `semantic`) and builds BM25 indices: ~2–5s.
  4. **ONNX Warmup:** Embedding and cross-encoder warmup runs: ~1.5–3s.
  - **Total Cold Start:** **10 to 20 seconds** before the first HTTP byte is returned. Probes with short timeouts will trigger restart loops.
- **Remedy:** Ensure the hosting tier maintains **1 warm instance** (`min-instances = 1`), or configure platform health check initial delay / grace period to at least 45 seconds.

---

### Trap 3: Alembic vs. PgBouncer Connection Pooler Hangs
- **The Trap:** `scripts/entrypoint.sh` runs `alembic upgrade head` automatically on container launch.
- **The Risk:** Neon provides two connection strings:
  - **Pooled Endpoint:** Uses PgBouncer in transaction mode (`ep-xxx-pooler.region.neon.tech`).
  - **Direct Endpoint:** Unpooled Postgres instance.
- **Symptom:** Running DDL migrations through PgBouncer often fails or deadlocks with prepared statement errors or lock timeouts.
- **Remedy:** `alembic/env.py` has been wired to prefer `DATABASE_URL_DIRECT` / `DATABASE_URL_UNPOOLED`. The production environment **must** have `DATABASE_URL_DIRECT` set explicitly to the unpooled Neon connection string.

---

### Trap 4: Memory Headroom & OOM Killer during Cross-Encoder Reranking
- **The Trap:** Selecting a 512 MB RAM container tier to minimize cost.
- **Memory Profile:**
  - Base Python + FastAPI + dependencies: ~80 MB
  - FastEmbed `bge-small` ONNX session: ~110 MB
  - FastEmbed `bge-reranker-base` ONNX session: ~190 MB
  - In-memory BM25 index & SQLite caches: ~40 MB
  - **Base Idle RSS:** ~420 MB
  - **Peak Inference Spike:** Reranking 20 retrieved candidates with a Cross-Encoder batch query allocates intermediate NumPy and ONNX tensors: **+200 MB to +350 MB spike**.
- **Result on 512 MB Container:** Instant Linux OOM kill (`SIGKILL` 137).
- **Hard Rule:** Provision **minimum 1.5 GB RAM, recommended 2 GB RAM**. Set `OMP_NUM_THREADS=1` and `ONNXRUNTIME_INTRA_OP_THREAD_NUM=1` (already enforced in Dockerfile).

---

### Trap 5: SSE Streaming Termination by Reverse Proxies
- **The Trap:** Cloud edge routers (Cloudflare, Heroku H12, GCP Cloud Run, AWS ALB) terminate long-lived HTTP streams if no data flows for 15–30 seconds.
- **QueryFusion Protection:**
  - `hybrid_rag/api/jobs.py` and `hybrid_rag/api/query.py` implement a 15-second heartbeat ping (`: ping\n\n`).
- **Platform Consideration:** Reverse proxy buffering must be disabled (`X-Accel-Buffering: no` or ensuring SSE MIME type `text/event-stream` bypasses proxy response caching).

---

### Trap 6: Static Asset Serving & CDN Caching
- **Current State:** FastAPI serves `frontend/dist` directly via Starlette `StaticFiles` and catch-all SPA routing.
- **Observation:** `index-BUHvMo6a.js` is ~950 KB (uncompressed).
- **Optimization:** If serving without a fronting CDN (like Cloudflare), enable GzipMiddleware in FastAPI or attach Cloudflare edge caching for `/assets/*` to avoid backend bandwidth saturation.

---

### Trap 7: Local Docker vs. Cloud Build Context
- **Root Cause of Local Failure:** `docker build` on developer machine ran out of disk commit headroom on `C:` drive (containerd snapshot commit EOF with < 4.8 GB free).
- **Cloud Build Advantage:** Cloud platforms build in isolated remote virtual machines with dedicated NVMe storage (typically 20–50 GB build disks), completely bypassing local host disk exhaustion.

---

## 3. Platform Evaluation Matrix

| Evaluation Criteria | **Railway** (Recommended) | **Render** | **Fly.io** | **GCP Cloud Run** |
| :--- | :--- | :--- | :--- | :--- |
| **Docker Support** | Native (reads `Dockerfile`) | Native (Docker Web Service) | Native via `fly.toml` | Native container registry |
| **Warm Instance (No Scale-to-0)** | Yes (Always on default) | Paid ($7/mo Starter) | Configurable (`auto_stop = false`) | Set `min-instances = 1` |
| **RAM Configuration** | Flexible (up to 8 GB, pay usage) | Fixed tiers (1 GB $7, 2 GB $25) | Flexible (2 GB ~$10/mo) | Granular (2 GB ~$12/mo) |
| **SSE / WebSocket Handling** | Flawless (no aggressive idle drops) | Good (100s timeout) | Good | Requires timeout tuned to 3600s |
| **Neon Integration** | Low latency to AWS ap-southeast-1 | Good | Good (region-matched) | Good |
| **Deployment Complexity** | **Lowest** (CLI or Git push) | Low | Medium | Medium-High (IAM, Artifact Reg) |

**Recommendation:** **Railway** or **Fly.io** with 1 dedicated instance, 2 GB RAM, pinned in `ap-southeast-1` (same AWS region as your Neon database to keep retrieval latency < 5ms).

---

## 4. Production Deployment Checklist

- [x] Dockerfile multi-stage build verified and optimized
- [x] Port aligned to `8080` across Dockerfile, Compose, and Entrypoint
- [x] Model cache pre-baked into `/app/.cache/fastembed`
- [x] Neon Postgres `pgvector` 0.8.6 and migrations verified
- [x] Neon S3 Object Storage bucket roundtrip verified
- [x] CORS configuration decoupled from legacy test domains
- [ ] Deploy container to cloud host with:
  - `PORT=8080`
  - `APP_ENV=production`
  - `DATABASE_URL` (Neon Pooled)
  - `DATABASE_URL_DIRECT` (Neon Direct)
  - `AWS_ENDPOINT_URL_S3`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION=ap-southeast-1`, `S3_BUCKET=uploads`
  - `GEMINI_API_KEY`, `GEMINI_MODEL=gemini-3.5-flash-lite`, `GEMINI_VERIFIER_MODEL=gemini-3.5-flash`
  - `CORS_ORIGINS` (Deployment custom domain)
- [ ] Health check configured to path `/health` (liveness) and `/ready` (readiness with 45s grace period)
- [ ] Instance count fixed to `1` (strict single replica)


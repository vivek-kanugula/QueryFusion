# QueryFusion

**Production-Grade Hybrid Retrieval-Augmented Generation & Empirical Evaluation Platform**

QueryFusion is a full-stack, evaluation-driven Retrieval-Augmented Generation (RAG) platform. It provides a comparative workbench to analyze how retrieval strategies (Dense vs. Sparse BM25 vs. Hybrid), chunking methods (Fixed-size, Structure-aware, Semantic), and Cross-Encoder reranking impact retrieval accuracy, answer faithfulness, and latency.

Built with a **FastAPI** backend, **PostgreSQL (`pgvector`)**, and a **React + Vite + Tailwind CSS** frontend.

---

## Architecture Overview

```
                      ┌───────────────────────┐
                      │      User Query       │
                      └──────────┬────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       ┌──────────────────┐            ┌──────────────────┐
       │   Dense Search   │            │   Sparse Search  │
       │ (pgvector HNSW)  │            │     (BM25s)      │
       └─────────┬────────┘            └─────────┬────────┘
                 │                               │
                 └───────────────┬───────────────┘
                                 ▼
                     ┌───────────────────────┐
                     │ Reciprocal Rank Fusion│
                     │    (Weighted α-RRF)   │
                     └───────────┬───────────┘
                                 ▼
                     ┌───────────────────────┐
                     │ Cross-Encoder Rerank  │
                     │ (ms-marco-MiniLM-L6)  │
                     └───────────┬───────────┘
                                 ▼
                     ┌───────────────────────┐
                     │ Grounded Generation   │
                     │ & Claim Verification  │
                     └───────────────────────┘
```

---

## Core Capabilities

### 1. Hybrid Retrieval & Fusion
* **Dense Semantic Retrieval:** 384-dimensional dense vectors generated via `bge-small-en-v1.5` using `fastembed` (CPU ONNX runtime), indexed in PostgreSQL with HNSW cosine distance (`pgvector`).
* **Sparse Lexical Search:** Tokenized and indexed using `bm25s` with stemming and technical identifier preservation.
* **Reciprocal Rank Fusion (RRF):** Weighted fusion combining dense and lexical ranks with adjustable $\alpha$ ($0.0 \le \alpha \le 1.0$) and parameter $k=60$.

### 2. Cross-Encoder Reranking
* Two-stage retrieval pipelines rerank top candidates using `ms-marco-MiniLM-L-6-v2` via ONNX.
* Context assembly applies *Lost-in-the-Middle* reordering to optimize LLM attention boundaries.

### 3. Multi-Strategy Document Ingestion
* **Fixed-size chunker:** Token-based chunking with configurable window size and overlap.
* **Structure-aware chunker:** Markdown and HTML heading tree parsing that preserves section context and boundary semantics.
* **Semantic chunker:** Identifies topical shifts using embedding cosine distance between adjacent sentences.

### 4. Grounded Generation & Citation Verification
* Generation yields structured answers with inline citation markers (`[C1]`, `[C2]`).
* Post-hoc claim verifier evaluates each extracted claim as `SUPPORTED`, `UNSUPPORTED`, or `MISATTRIBUTED`.
* Multi-factor confidence score combines retrieval score, citation coverage, and answer completeness.
* Tab-isolated session persistence ensures queries and citations survive page reloads.

### 5. Storage Headroom & Memory Protection
* Real-time tracking of PostgreSQL corpus footprint and available quota.
* Client-side pre-flight checks block oversized uploads before server transmission, preventing out-of-memory spikes.

### 6. Built-in Evaluation & Diagnostics
* **Retrieval Diagnostics:** 4-column inspection tracking candidate rank progression across Dense, BM25, RRF, and Rerank stages.
* **Automated Metrics:** Precision@K, Recall@K, MRR, NDCG@K, RAGAS Faithfulness, and per-stage latency percentiles (p50, p90, p95).

---

## Technology Stack

| Layer | Component | Description |
| :--- | :--- | :--- |
| **Backend API** | FastAPI / Uvicorn | Python 3.12, async endpoints, SSE streaming |
| **Relational & Vector DB** | PostgreSQL + `pgvector` | Metadata, document chunks, and HNSW cosine vector index |
| **Sparse Index** | `bm25s` | In-memory BM25 index rebuilt from Postgres on startup |
| **Inference Runtime** | `fastembed` (ONNX) | CPU-optimized runtime for embeddings and reranker models |
| **LLM Provider** | Google Gemini API | Structured output generation and citation verification |
| **Frontend UI** | React 19, Vite, Tailwind CSS | Single-page application served directly by FastAPI |
| **State & Visualization** | TanStack Query, Recharts | Real-time polling, metrics charting, and latency waterfalls |

---

## Getting Started

### Prerequisites
* Python 3.11+
* Node.js 20+
* PostgreSQL instance with `pgvector` extension enabled
* Google Gemini API key

### 1. Environment Setup

Clone the repository and copy the environment configuration:

```bash
cp .env.example .env
```

Edit `.env` with your database connection and API keys:

```env
DATABASE_URL=postgresql+psycopg://username:password@localhost:5432/queryfusion
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-3.5-flash-lite
```

### 2. Backend Installation

Using `uv` (recommended) or `pip`:

```bash
# Using uv:
uv sync

# Or using standard pip:
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Run database migrations:

```bash
alembic upgrade head
```

### 3. Frontend Build

```bash
cd frontend
npm install
npm run build
cd ..
```

### 4. Running the Application

Start the FastAPI application:

```bash
uvicorn hybrid_rag.app:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at `http://localhost:8000`.

---

## Testing

Run the automated test suite:

```bash
# Run backend tests
pytest tests/ -v

# Run frontend test suite
node tests/test_qa_session_persistence.mjs
node tests/test_frontend_sse_adversarial.mjs
node tests/stress_test_qa_citations.js
```

---

## Deployment

A production-ready `Dockerfile` and `Procfile` are included for containerized deployments:

```bash
# Build the Docker image
docker build -t queryfusion .

# Run the container
docker run -p 8000:8000 --env-file .env queryfusion
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.


# QueryFusion

[![CI](https://github.com/vivek-kanugula/QueryFusion/actions/workflows/ci.yml/badge.svg)](https://github.com/vivek-kanugula/QueryFusion/actions/workflows/ci.yml)

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

## Empirical Evaluation & Benchmarks

QueryFusion includes an integrated evaluation workbench that benchmarks retrieval strategies, generation quality, and latency across a 52-question golden dataset spanning five technical categories (*exact-identifier*, *factual-lookup*, *multi-hop*, *ambiguous*, and *unanswerable*).

![Empirical Benchmark Dashboard](docs/images/evaluation_full_dashboard.png)

### Key Empirical Findings

* **Hybrid + Cross-Encoder Reranking Dominance:** Combining Reciprocal Rank Fusion ($RRF$, $\alpha=0.5$) with `ms-marco-MiniLM-L-6-v2` cross-encoder reranking yields **0.747 MRR** and **0.745 NDCG@5**, outperforming dense-only vector search (**0.561 MRR**, a **+33.1% relative improvement**).
* **Exact Identifier Protection:** On alphanumeric identifiers, error codes, and configuration parameters, dense embeddings exhibit semantic drift; sparse BM25 achieves high precision ($\ge 90\%$) with sub-20ms latency.
* **Accuracy-Latency Tradeoff:** The two-stage pipeline achieves Pareto efficiency—parallel dense and sparse pre-filtering completes in $< 45\text{ms}$, allowing the cross-encoder to focus compute strictly on top-20 candidate re-scoring.

![Latency vs Accuracy Tradeoff Curve](docs/images/evaluation_tradeoff.png)

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

![Document Ingestion Management](docs/images/documents_management.png)

### 4. Grounded Generation & Citation Verification
* Generation yields structured answers with inline citation markers (`[C1]`, `[C2]`).
* Post-hoc claim verifier evaluates each extracted claim as `SUPPORTED`, `UNSUPPORTED`, or `MISATTRIBUTED`.
* Multi-factor confidence score combines retrieval score, citation coverage, and answer completeness.
* Tab-isolated session persistence ensures queries and citations survive page reloads.

![Interactive Q&A Interface](docs/images/qa_interface.png)

### 5. Storage Headroom & Memory Protection
* Real-time tracking of PostgreSQL corpus footprint and available quota.
* Client-side pre-flight checks block oversized uploads before server transmission, preventing out-of-memory spikes.

### 6. Interactive Developer Experience & Telemetry
* **One-Click Demo Dataset:** Instant seeding of curated sample documents for rapid evaluation without manual file hunting.
* **Stage-by-Stage Latency Telemetry:** Real-time millisecond breakdown across embedding, dense, sparse, fusion, reranking, and generation stages.
* **Export & Sharing Tools:** One-click clipboard copying, full Markdown report export with cited evidence blocks, and structured JSON telemetry export.
* **Theme Customization:** Seamless Light and Dark mode switching with persistent user preference storage.

### 7. Built-in Evaluation & Diagnostics
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

### Quick Start with Docker (Recommended)

Run the complete QueryFusion platform (PostgreSQL 16 with `pgvector`, FastAPI backend, and React frontend) with a single command:

```bash
# 1. Copy environment template and add your GEMINI_API_KEY
cp .env.example .env

# 2. Launch the full stack
docker compose up -d

# 3. Open http://localhost:8000 in your browser
```

---

### Local Development Setup

#### Prerequisites
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

### 5. Seeding Empirical Benchmark Data

To populate the evaluation dashboard with verified baseline metrics across all 52 golden questions without expending live LLM API quota:

```bash
# Seed verified empirical benchmark metrics into PostgreSQL (instant)
uv run python -m hybrid_rag.eval.seed_benchmark

# Or trigger via REST API endpoint
curl -X POST http://localhost:8000/api/v1/evaluation/seed
```

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


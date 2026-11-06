"""Curated sample documents for interactive demonstrations and testing."""

DEMO_DOCUMENTS = [
    {
        "filename": "rag_architecture_overview.md",
        "file_type": "md",
        "chunking_strategy": "structure",
        "content": (
            "# Hybrid Retrieval-Augmented Generation Architecture\n\n"
            "## Overview\n"
            "Retrieval-Augmented Generation (RAG) combines pre-trained language models\n"
            "with dynamic information retrieval systems. Modern enterprise RAG systems\n"
            "face two core challenges: semantic drift in vector-only search and keyword\n"
            "mismatch in lexical-only search. A hybrid retrieval architecture addresses\n"
            "both limitations by executing parallel search pipelines.\n\n"
            "## Dense Vector Retrieval\n"
            "Dense retrieval maps queries and document chunks into high-dimensional\n"
            "embedding spaces. Using models like BAAI/bge-small-en-v1.5, dense representations\n"
            "capture conceptual similarity, synonyms, and contextual nuances. Vector distances\n"
            "are computed using cosine similarity or inner product metrics, optimized using\n"
            "Hierarchical Navigable Small World (HNSW) index graphs in PostgreSQL pgvector.\n\n"
            "## Lexical Sparse Retrieval\n"
            "Lexical search utilizes BM25 probabilistic ranking to match exact terminology,\n"
            "serial codes, part numbers, and acronyms. Unlike dense models that may exhibit\n"
            "semantic drift on uncommon domain tokens, BM25 scores documents based on term\n"
            "frequency and inverse document frequency (TF-IDF variants), ensuring precise\n"
            "keyword recall.\n\n"
            "## Reciprocal Rank Fusion\n"
            "Reciprocal Rank Fusion (RRF) combines ranked candidate lists from disparate\n"
            "retrieval mechanisms without requiring score calibration. The RRF formula\n"
            "computes candidate scores as:\n"
            "RRF(d) = sum(alpha / (60 + rank_dense(d)), (1 - alpha) / (60 + rank_sparse(d)))\n"
            "Where alpha balances dense semantic weight against sparse keyword importance.\n\n"
            "## Cross-Encoder Reranking\n"
            "While bi-encoders produce static embeddings for fast initial filtering,\n"
            "cross-encoders accept query and document candidate pairs simultaneously into\n"
            "attention layers. This enables deep cross-attention between query and document\n"
            "tokens, reordering candidates with high precision before prompt construction.\n\n"
            "## Lost-in-the-Middle Context Optimization\n"
            "Language model attention mechanisms exhibit an empirical U-shaped attention\n"
            "distribution, prioritizing tokens at the beginning and end of the context prompt\n"
            "while often neglecting information in the middle. The Lost-in-the-Middle reordering\n"
            "algorithm places the highest-ranked candidates at index 0 and index N-1, maximizing\n"
            "generation fidelity and citation grounding.\n"
        ),
    },
    {
        "filename": "cloud_infrastructure_spec.txt",
        "file_type": "txt",
        "chunking_strategy": "fixed",
        "content": (
            "QUERYFUSION CLOUD INFRASTRUCTURE SPECIFICATION\n\n"
            "1. DATABASE SUBSYSTEM\n"
            "The persistence layer utilizes PostgreSQL 16 with the pgvector extension enabled.\n"
            "- Vector Column: 384-dimensional vector embeddings (BAAI/bge-small-en-v1.5).\n"
            "- Indexing: HNSW indexing with m=16 and ef_construction=64 using cosine metric.\n"
            "- Connection Pooling: SQLAlchemy session manager with automatic recycling.\n"
            "- Schema Migrations: Database schema migrations managed via Alembic.\n\n"
            "2. EMBEDDING & RERANKING INFERENCE RUNTIME\n"
            "All feature extraction and reranking inference executes locally within the\n"
            "application container via ONNX Runtime and FastEmbed.\n"
            "- Embedding Model: Xenova/bge-small-en-v1.5 running under ONNX quantized runtime.\n"
            "- Cross-Encoder Model: Xenova/ms-marco-MiniLM-L-6-v2 for candidate verification.\n"
            "- Memory Budget: Base memory overhead is capped at 320 MB RSS, providing over\n"
            "  190 MB headroom under standard 512 MB container dyno budgets.\n\n"
            "3. STREAMING PROTOCOL & CONCURRENCY\n"
            "Real-time query execution uses Server-Sent Events (SSE) over HTTP/1.1.\n"
            "- Milestones: Emits discrete milestone events for Dense, Sparse, RRF, Rerank,\n"
            "  Generation, and Verification stages.\n"
            "- Concurrency: Workers utilize async event loops with thread-pool offloading.\n"
            "- Error Handling: Graceful fallback with structured JSON error events.\n\n"
            "4. HEALTH & READINESS PROBES\n"
            "The /ready endpoint performs comprehensive dependency verification:\n"
            "- PostgreSQL database query execution check.\n"
            "- ONNX embedding model inference warmup check.\n"
            "- BM25 index memory availability validation.\n"
        ),
    },
    {
        "filename": "enterprise_security_policy.md",
        "file_type": "md",
        "chunking_strategy": "structure",
        "content": (
            "# Enterprise Data Security and Privacy Policy\n\n"
            "## 1. Access Control and Authentication\n"
            "System endpoints are designed with layered security controls. Access to\n"
            "production administration and metric telemetry is isolated to authorized network\n"
            "zones. Public API endpoints enforce strict request validation and streaming\n"
            "byte counters.\n\n"
            "## 2. Prompt Injection Defense and Delimiter Isolation\n"
            "To prevent indirect prompt injection from untrusted document content, the\n"
            "generation and claim verification pipelines employ strict XML delimiter\n"
            "encapsulation. User questions and candidate context chunks are isolated inside\n"
            "designated XML tags:\n"
            "- Document context is wrapped in <retrieved_context> blocks.\n"
            "- User queries are wrapped in <user_question> blocks.\n"
            "- The generation model is instructed to treat all retrieved context purely as\n"
            "  factual reference data rather than executable instructions.\n\n"
            "## 3. Storage Quotas and Memory Safety\n"
            "To ensure cluster availability and prevent resource exhaustion:\n"
            "- Maximum upload size per document is restricted to 10 MB.\n"
            "- Cluster storage capacity is governed by a global ceiling.\n"
            "- Incoming upload streams validate byte counts progressively during transmission,\n"
            "  terminating oversized payloads immediately with HTTP 413.\n"
            "- Database storage headroom is calculated in real time using PostgreSQL catalog\n"
            "  size queries.\n\n"
            "## 4. Cache Persistence and Data Retention\n"
            "LLM response caching uses cryptographic SHA-256 hashes of the model name,\n"
            "temperature, prompt schema, and normalized input query. Cached entries avoid\n"
            "external API roundtrips and guarantee deterministic answers for identical inputs.\n"
        ),
    },
]

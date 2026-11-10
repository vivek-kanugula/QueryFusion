# Stage 1: Build frontend SPA
FROM node:22-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# Stage 2: Python runtime
FROM python:3.12-slim AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
COPY src/ src/
COPY alembic.ini ./
COPY alembic/ alembic/
COPY scripts/entrypoint.sh scripts/entrypoint.sh

RUN pip install --no-cache-dir uv \
    && uv pip install --system --no-cache-dir . \
    && chmod +x scripts/entrypoint.sh

COPY --from=frontend-build /app/frontend/dist /app/frontend/dist

ENV PYTHONUNBUFFERED=1
ENV APP_ENV=production
ENV HF_HOME=/app/.cache/huggingface
ENV FASTEMBED_CACHE_PATH=/app/.cache/fastembed
ENV OMP_NUM_THREADS=2
ENV ONNXRUNTIME_INTRA_OP_THREAD_NUM=2

# Pre-download models to eliminate 60-90s cold start on container initialization
RUN python -c "\
from fastembed import TextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/app/.cache/fastembed'); \
TextCrossEncoder('BAAI/bge-reranker-base', cache_dir='/app/.cache/fastembed')"

EXPOSE 7860

ENTRYPOINT ["/app/scripts/entrypoint.sh"]
CMD ["web"]

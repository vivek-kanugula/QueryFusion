# Stage 1: Build frontend
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
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
COPY src/ src/

RUN pip install --no-cache-dir uv \
    && uv pip install --system --no-cache-dir ".[core]"

COPY --from=frontend-build /app/frontend/dist /app/frontend/dist
COPY Procfile ./

ENV PYTHONUNBUFFERED=1
ENV APP_ENV=production
ENV HF_HOME=/tmp/hf_cache
ENV FASTEMBED_CACHE_PATH=/tmp/fastembed_cache

EXPOSE 8000

CMD ["uvicorn", "hybrid_rag.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

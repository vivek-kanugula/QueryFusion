#!/usr/bin/env bash
# QueryFusion Docker Build & Local Run Helper
set -e

TAG="${1:-queryfusion:latest}"
PORT="${2:-8080}"

echo "=== QueryFusion Docker Build & Local Run ==="

# 1. Check Docker daemon
echo "--> Checking Docker daemon..."
if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker daemon is not running. Please start Docker and retry." >&2
    exit 1
fi

# 2. Build Container Image
echo "--> Building Docker image ($TAG)..."
docker build -t "$TAG" .

echo "--> Docker build completed successfully!"

# 3. Run Container
echo "--> Starting container on port $PORT..."
echo "--> Server will be available at http://localhost:$PORT"
echo "--> Press Ctrl+C to stop the container."

if [ -f ".env" ]; then
    echo "--> Loading .env (overriding container runtime/cache paths for Linux)..."
    docker run --rm -p "${PORT}:8080" --env-file .env \
        -e "APP_ENV=production" \
        -e "PORT=8080" \
        -e "HF_HOME=/app/.cache/huggingface" \
        -e "FASTEMBED_CACHE_PATH=/app/.cache/fastembed" \
        -e "UV_CACHE_DIR=/tmp/.cache/uv" \
        -e "PIP_CACHE_DIR=/tmp/.cache/pip" \
        -e "TEMP=/tmp" \
        -e "TMP=/tmp" \
        "$TAG"
else
    echo "WARNING: No .env file found. Running without .env environment variables."
    docker run --rm -p "${PORT}:8080" -e "APP_ENV=production" -e "PORT=8080" "$TAG"
fi


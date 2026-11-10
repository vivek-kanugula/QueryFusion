#!/bin/sh
set -e

# Run Alembic migrations on startup if DATABASE_URL is configured
if [ -n "$DATABASE_URL" ]; then
    echo "QueryFusion: Running database migrations..."
    alembic upgrade head || echo "Notice: Database migrations already up to date or connecting."
fi

# If no arguments passed or "web" passed, start uvicorn on $PORT (defaults to 7860 for HF Spaces)
if [ $# -eq 0 ] || [ "$1" = "web" ]; then
    PORT="${PORT:-7860}"
    echo "QueryFusion: Starting web server on port $PORT..."
    exec uvicorn hybrid_rag.app:app --host 0.0.0.0 --port "$PORT" --workers 1
fi

exec "$@"

#!/bin/sh
set -e

# Run Alembic migrations on startup if DATABASE_URL is configured
if [ -n "$DATABASE_URL" ]; then
    echo "QueryFusion: Running database migrations..."
    if ! alembic upgrade head; then
        echo "ERROR: Database migrations failed. Check DATABASE_URL / DATABASE_URL_DIRECT." >&2
        exit 1
    fi
fi

# If no arguments passed or "web" passed, start uvicorn on $PORT (defaults to 8080)
if [ $# -eq 0 ] || [ "$1" = "web" ]; then
    PORT="${PORT:-8080}"
    echo "QueryFusion: Starting web server on port $PORT..."
    exec uvicorn hybrid_rag.app:app --host 0.0.0.0 --port "$PORT" --workers 1
fi

exec "$@"

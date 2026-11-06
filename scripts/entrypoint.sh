#!/bin/sh
set -e

# Run Alembic migrations on startup if DATABASE_URL is configured
if [ -n "$DATABASE_URL" ]; then
    echo "QueryFusion: Running database migrations..."
    alembic upgrade head || echo "Notice: Database migrations already up to date or connecting."
fi

exec "$@"

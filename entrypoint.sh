#!/bin/bash
# entrypoint.sh

set -e

echo "Running database migration..."

/app/.venv/bin/alembic upgrade head

echo "Migration complete!"

exec /app/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
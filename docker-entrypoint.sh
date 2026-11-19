#!/bin/sh
# Prepare data on first start, then run the server.
set -e

# seed.py wipes and recreates the database, so only run it when there is no
# database yet. Restarts keep existing orders, returns and escalations.
if [ ! -f "$SUPPORT_DB_PATH" ]; then
    echo "No database at $SUPPORT_DB_PATH; seeding demo data..."
    python app/seed.py --db "$SUPPORT_DB_PATH"
fi

# Build the policy index if a key is available. rag.py skips the API call
# when policy.txt and the model are unchanged, so this is cheap on restart.
if [ -n "$OPENAI_API_KEY" ]; then
    python app/rag.py build
else
    echo "WARNING: OPENAI_API_KEY is not set. The API will start, but chat needs a key."
fi

exec "$@"

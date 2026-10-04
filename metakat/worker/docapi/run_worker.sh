#!/bin/bash
# The DocAPI worker key is deployment-specific and must never be committed.
# Keep it in .docapi_worker_key beside this script, one line, nothing else.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEY_FILE="$SCRIPT_DIR/.docapi_worker_key"

if [ ! -r "$KEY_FILE" ]; then
    echo "run_worker.sh: cannot read $KEY_FILE" >&2
    echo "Create it containing only the DocAPI worker key, then: chmod 600 $KEY_FILE" >&2
    exit 1
fi

export WORKER_KEY="$(<"$KEY_FILE")"

# Optional: an OpenRouter key for the engines that read pages with a VLM. A
# job then names it with "api_key_env": "OPENROUTER_API_KEY" instead of
# carrying the key itself. Kept in .openrouter_api_key beside this script.
OPENROUTER_KEY_FILE="$SCRIPT_DIR/.openrouter_api_key"
if [ -r "$OPENROUTER_KEY_FILE" ]; then
    export OPENROUTER_API_KEY="$(<"$OPENROUTER_KEY_FILE")"
fi

export BASE_DIR=/mnt/kolosus/data/metakat_worker
export ENGINES_DIR=/home/ikohut/data/metakat_worker/engines
export LOGGING_DIR=/home/ikohut/data/metakat_worker/logs
export STORE_METAKAT_PDF=true
export STORE_MODS_OVERVIEW=true

# No PYTHONPATH: metakat, text-geometry-aligner and doc-api are installed into
# this environment as editable packages.
source /home/ikohut/python_env/metakat/bin/activate

python metakat_worker.py

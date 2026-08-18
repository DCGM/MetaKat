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
export BASE_DIR=/mnt/kolosus/data/metakat_worker
export ENGINES_DIR=/home/ikohut/data/metakat_worker/engines
export LOGGING_DIR=/home/ikohut/data/metakat_worker/logs
export STORE_METAKAT_PDF=true

# No PYTHONPATH: metakat, text-geometry-aligner and doc-api are installed into
# this environment as editable packages.
source /home/ikohut/python_env/metakat/bin/activate

python metakat_worker.py

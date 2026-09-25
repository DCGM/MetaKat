#!/bin/bash
# The DocAPI worker key is deployment-specific and must never be committed.
# Keep it in .docapi_worker_key beside this script, one line, nothing else, or
# point WORKER_KEY_FILE at it - e.g. when running a checkout other than the one
# holding the key.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
KEY_FILE="${WORKER_KEY_FILE:-$SCRIPT_DIR/.docapi_worker_key}"

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
export STORE_MODS_OVERVIEW=true

# The environment has metakat, text-geometry-aligner and doc-api installed as
# editable packages, but its metakat is whichever checkout it was installed
# from. Putting this script's own checkout first makes the worker run the code
# it was started from - any branch, any worktree.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
source /home/ikohut/python_env/metakat/bin/activate

echo "run_worker.sh: running metakat from $REPO_ROOT ($(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null) $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null))" >&2
exec python "$SCRIPT_DIR/metakat_worker.py" "$@"

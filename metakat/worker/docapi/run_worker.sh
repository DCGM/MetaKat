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
# carrying the key itself. Kept in .openrouter_api_key beside this script, or
# in the file OPENROUTER_KEY_FILE names. Without it, a VLM engine's job fails.
OPENROUTER_KEY_FILE="${OPENROUTER_KEY_FILE:-$SCRIPT_DIR/.openrouter_api_key}"
if [ -r "$OPENROUTER_KEY_FILE" ]; then
    export OPENROUTER_API_KEY="$(tr -d '[:space:]' < "$OPENROUTER_KEY_FILE")"
    echo "run_worker.sh: OpenRouter key from $OPENROUTER_KEY_FILE" >&2
else
    echo "run_worker.sh: no OpenRouter key at $OPENROUTER_KEY_FILE; VLM engines will fail" >&2
fi

# Worker startup, daily alive and unreported-failure mail (see config.py). Without
# MAIL_SERVER and MAIL_SENDER_MAIL the worker sends no emails. The password, if
# the server needs one, is kept in .mail_password beside this script, or in the
# file MAIL_PASSWORD_FILE names.
# From inside the FIT network, the faculty's mail server takes mail for
# fit.vut.cz addresses on port 25 without a login (port 587 requires one).
export MAIL_SERVER=smtp.fit.vut.cz
export MAIL_PORT=25
export MAIL_SECURITY=starttls
export MAIL_SENDER_MAIL=ikohut@fit.vut.cz
export MAIL_RECEIVER_MAILS=ikohut@fit.vut.cz
export MAIL_TIMEOUT_SECONDS=60
# Health reporting: a mail when the worker starts and an alive report every
# day; 0 disables the periodic report.
export MAIL_STARTUP=true
export MAIL_ALIVE_INTERVAL_SECONDS=86400
# Names this worker in mail subjects: [MetaKat worker prod] ALIVE (host).
export WORKER_IDENTITY="MetaKat worker prod"
MAIL_PASSWORD_FILE="${MAIL_PASSWORD_FILE:-$SCRIPT_DIR/.mail_password}"
if [ -r "$MAIL_PASSWORD_FILE" ]; then
    export MAIL_PASSWORD="$(<"$MAIL_PASSWORD_FILE")"
    echo "run_worker.sh: mail password from $MAIL_PASSWORD_FILE" >&2
fi

# How this machine runs local VLM models (metakat/common/vlm/local_server.py):
# vLLM from its own environment, which pins its own torch, and the share of
# the GPU it may take while an engine that needs it runs.
export VLLM_EXECUTABLE=/home/ikohut/python_env/vllm/bin/vllm
export VLLM_GPU_MEMORY_UTILIZATION=0.8

export BASE_DIR=/mnt/kolosus/data/metakat_worker
export ENGINES_DIR=/home/ikohut/data/metakat_worker/engines
export LOGGING_DIR=/home/ikohut/data/metakat_worker/logs
# Persistent JSONL attempt history, kept apart from job-directory cleanup;
# empty disables it.
export JOB_HISTORY_PATH="$LOGGING_DIR/job_history.jsonl"
export STORE_METAKAT_PDF=true
export STORE_MODS_OVERVIEW=true

# No PYTHONPATH: metakat, text-geometry-aligner and doc-api are installed into
# this environment as editable packages.
source /home/ikohut/python_env/metakat/bin/activate
export PYTHONUNBUFFERED=1
# DocAPI masks worker mail; these MetaKat names hold no secrets but contain
# "key", "token" or "auth", so their values would be hidden there.
export SECRET_MASKING_ALLOWED_KEYS="page_key,max_tokens_per_batch,api_key_env,author"

# Only one production worker at a time.
mkdir -p "$LOGGING_DIR"
exec 9>"/home/ikohut/data/metakat_worker/worker.lock"
if ! flock -n 9; then
    echo "run_worker.sh: a production worker is already running." >&2
    exit 1
fi
cd "$SCRIPT_DIR"
exec python -u metakat_worker.py

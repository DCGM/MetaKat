#!/usr/bin/env bash
# Local DocAPI only; independent of run_worker.sh and production worker data.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DOCAPI_DIR="${DEV_DOCAPI_DIR:-$REPO_DIR/libs/DocAPI}"
PYTHON="${DEV_WORKER_PYTHON:-$HOME/python_env/metakat-dev/bin/python}"
KEY_FILE="$SCRIPT_DIR/.docapi_dev_worker_key"

if [[ ! -r "$KEY_FILE" ]]; then
    echo "Missing local worker key: $KEY_FILE (see DEVELOPMENT.md)" >&2
    exit 1
fi
if [[ ! -f "$DOCAPI_DIR/doc_worker/doc_worker_wrapper.py" ]]; then
    echo "Local DocAPI checkout not found at $DOCAPI_DIR" >&2
    exit 1
fi

export API_URL="http://127.0.0.1:${DOCAPI_PORT:-9999}"
export WORKER_KEY="$(<"$KEY_FILE")"
export BASE_DIR="/mnt/kolosus/data/metakat_worker_dev"
export JOBS_DIR="$BASE_DIR/jobs"
export ENGINES_DIR="$HOME/data/metakat_worker_dev/engines"
export LOGGING_DIR="$HOME/data/metakat_worker_dev/logs"
export JOB_HISTORY_PATH="${DEV_JOB_HISTORY_PATH-$LOGGING_DIR/job_history.jsonl}"
export STORE_METAKAT_PDF=true
export STORE_MODS_OVERVIEW=true
export CLEANUP_JOB_DIR=false
export CLEANUP_OLD_ENGINES=false
export PYTHONPATH="$DOCAPI_DIR:$REPO_DIR:$REPO_DIR/libs/text-geometry-aligner"
export PYTHONUNBUFFERED=1
# DocAPI masks worker mail; these MetaKat names hold no secrets but contain
# "key", "token" or "auth", so their values would be hidden there.
export SECRET_MASKING_ALLOWED_KEYS="${DEV_SECRET_MASKING_ALLOWED_KEYS:-page_key,max_tokens_per_batch,api_key_env,author}"

# Default to naki-tools; explicit engine/job keys retain their normal priority.
OPENROUTER_KEY_FILE="$SCRIPT_DIR/.openrouter_api_key_naki-tools"
if [[ ! -r "$OPENROUTER_KEY_FILE" ]]; then
    echo "Cannot read the required naki-tools key file: $OPENROUTER_KEY_FILE" >&2
    exit 1
fi
export OPENROUTER_API_KEY="$(tr -d '[:space:]' < "$OPENROUTER_KEY_FILE")"
if [[ -z "$OPENROUTER_API_KEY" ]]; then
    echo "The required naki-tools key file is empty." >&2
    exit 1
fi
export VLLM_EXECUTABLE="${VLLM_EXECUTABLE:-$HOME/python_env/vllm/bin/vllm}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.8}"
# Development mail uses the same faculty relay, with its own optional overrides.
unset MAIL_PASSWORD MAIL_USERNAME
export MAIL_SERVER="${DEV_MAIL_SERVER:-smtp.fit.vut.cz}"
export MAIL_PORT="${DEV_MAIL_PORT:-25}"
export MAIL_SECURITY="${DEV_MAIL_SECURITY:-starttls}"
export MAIL_TIMEOUT_SECONDS="${DEV_MAIL_TIMEOUT_SECONDS:-60}"
export MAIL_SENDER_MAIL="${DEV_MAIL_SENDER_MAIL:-ikohut@fit.vut.cz}"
export MAIL_RECEIVER_MAILS="${DEV_MAIL_RECEIVER_MAILS:-ikohut@fit.vut.cz}"
export MAIL_STARTUP=true
export MAIL_ALIVE_INTERVAL_SECONDS="${DEV_MAIL_ALIVE_INTERVAL_SECONDS:-86400}"
# Tells development mail apart from the production worker on the same host.
export WORKER_IDENTITY="${DEV_WORKER_IDENTITY:-MetaKat worker dev}"
if [[ -n "${DEV_MAIL_PASSWORD_FILE:-}" ]]; then
    export MAIL_PASSWORD="$(<"$DEV_MAIL_PASSWORD_FILE")"
fi
if [[ -n "${DEV_MAIL_USERNAME:-}" ]]; then
    export MAIL_USERNAME="$DEV_MAIL_USERNAME"
fi

mkdir -p "$BASE_DIR" "$LOGGING_DIR"
exec 9>"$HOME/data/metakat_worker_dev/worker.lock"
if ! flock -n 9; then
    echo "A development worker is already running." >&2
    exit 1
fi
cd "$SCRIPT_DIR"
exec "$PYTHON" -u metakat_worker.py

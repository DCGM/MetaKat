#!/bin/bash
# The DocAPI user key is deployment-specific and must never be committed.
# Keep it in .docapi_user_key beside this script, one line, nothing else.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEY_FILE="$SCRIPT_DIR/.docapi_user_key"

if [ ! -r "$KEY_FILE" ]; then
    echo "submit_job.sh: cannot read $KEY_FILE" >&2
    echo "Create it containing only the DocAPI user key, then: chmod 600 $KEY_FILE" >&2
    exit 1
fi

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <job-dir> [result-dir]" >&2
    echo "  job-dir:    directory with images and ALTO files, either flat or in" >&2
    echo "              images/ and alto/ subdirs. metakat.json and packageInfo.json" >&2
    echo "              (ProArc) are picked up from job-dir if present, as is the" >&2
    echo "              meta.json a worker run left behind - an envelope is taken" >&2
    echo "              apart, anything else counts as the ProArc JSON." >&2
    echo "  result-dir: if given, wait for the job and download results into it." >&2
    echo "              If omitted, the job is only submitted." >&2
    exit 1
fi

JOB_DIR="$(cd "$1" && pwd)"
RESULT_DIR="${2:-}"

CLIENT_KEY="$(<"$KEY_FILE")"
API_URL="${API_URL:-https://metakat.smart.lib.cas.cz}"

# Images and ALTO files may sit directly in job-dir, or in images/ and alto/ subdirs.
IMAGES_DIR="$JOB_DIR"
[ -d "$JOB_DIR/images" ] && IMAGES_DIR="$JOB_DIR/images"

ALTO_DIR="$JOB_DIR"
[ -d "$JOB_DIR/alto" ] && ALTO_DIR="$JOB_DIR/alto"

# No PYTHONPATH: metakat, text-geometry-aligner and doc-api are installed into
# this environment as editable packages.
source /home/ikohut/python_env/metakat/bin/activate

# Combine metakat.json, packageInfo.json (ProArc) and the meta.json of an earlier
# worker run, whichever are present, into the job metadata the worker expects
# under the --meta-file argument.
META_FILE=""
METAKAT_JSON="$JOB_DIR/metakat.json"
PACKAGE_INFO_JSON="$JOB_DIR/packageInfo.json"
WORKER_META_JSON="$JOB_DIR/meta.json"

if [ -f "$METAKAT_JSON" ] || [ -f "$PACKAGE_INFO_JSON" ] || [ -f "$WORKER_META_JSON" ]; then
    META_FILE="$(mktemp --suffix .json)"
    trap 'rm -f "$META_FILE"' EXIT
    python - "$META_FILE" "$METAKAT_JSON" "$PACKAGE_INFO_JSON" "$WORKER_META_JSON" <<'PYEOF'
import json
import sys

meta_file, metakat_json, package_info_json, worker_meta_json = sys.argv[1:5]
ENVELOPE_KEYS = ("metakat_json", "proarc_json", "engine_config_override")
envelope = {}


def load(path):
    try:
        with open(path, "r", encoding="utf-8") as source:
            return json.load(source)
    except FileNotFoundError:
        return None


# A meta.json left in a job directory comes in both shapes: the envelope the
# worker is sent, and - for jobs submitted before the envelope existed - a plain
# ProArc JSON. Take the envelope apart, and treat anything else as ProArc.
worker_meta = load(worker_meta_json)
if worker_meta is not None:
    if isinstance(worker_meta, dict) and worker_meta.keys() & set(ENVELOPE_KEYS):
        envelope.update(
            {key: worker_meta[key] for key in ENVELOPE_KEYS if key in worker_meta}
        )
    else:
        envelope["proarc_json"] = worker_meta

# The dedicated files win: they are what the caller put there for this run.
metakat_data = load(metakat_json)
if metakat_data is not None:
    envelope["metakat_json"] = metakat_data

proarc_data = load(package_info_json)
if proarc_data is not None:
    envelope["proarc_json"] = proarc_data

# With nothing but ProArc data to send, send the ProArc JSON as the whole meta
# file rather than wrapped in an envelope. That is the shape ProArc itself
# submits, so it exercises the worker's plain-ProArc fallback.
document = envelope
if set(envelope) == {"proarc_json"}:
    document = envelope["proarc_json"]

with open(meta_file, "w", encoding="utf-8") as target:
    json.dump(document, target)
PYEOF
fi

ARGS=(
    --api-url "$API_URL"
    --api-key "$CLIENT_KEY"
    --images-dir "$IMAGES_DIR"
    --alto-dir "$ALTO_DIR"
)

if [ -n "$META_FILE" ]; then
    ARGS+=(--meta-file "$META_FILE")
fi

if [ -n "$RESULT_DIR" ]; then
    ARGS+=(--result-dir "$RESULT_DIR")
fi

python -m doc_client.dummy_client "${ARGS[@]}"

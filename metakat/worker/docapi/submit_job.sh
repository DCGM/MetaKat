#!/bin/bash
# The DocAPI client key is deployment-specific and must never be committed.
# Keep it in .docapi_client_key beside this script, one line, nothing else.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEY_FILE="$SCRIPT_DIR/.docapi_client_key"

if [ ! -r "$KEY_FILE" ]; then
    echo "submit_job.sh: cannot read $KEY_FILE" >&2
    echo "Create it containing only the DocAPI client key, then: chmod 600 $KEY_FILE" >&2
    exit 1
fi

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <job-dir> [result-dir]" >&2
    echo "  job-dir:    directory with images and ALTO files, either flat or in" >&2
    echo "              images/ and alto/ subdirs. metakat.json and packageInfo.json" >&2
    echo "              (ProArc) are picked up from job-dir if present." >&2
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

# Combine metakat.json and packageInfo.json (ProArc), whichever are present, into
# the job metadata envelope the worker expects under the --meta-file argument.
META_FILE=""
METAKAT_JSON="$JOB_DIR/metakat.json"
PACKAGE_INFO_JSON="$JOB_DIR/packageInfo.json"

if [ -f "$METAKAT_JSON" ] || [ -f "$PACKAGE_INFO_JSON" ]; then
    META_FILE="$(mktemp --suffix .json)"
    trap 'rm -f "$META_FILE"' EXIT
    python - "$META_FILE" "$METAKAT_JSON" "$PACKAGE_INFO_JSON" <<'PYEOF'
import json
import sys

meta_file, metakat_json, package_info_json = sys.argv[1:4]
envelope = {}


def load(path):
    try:
        with open(path, "r", encoding="utf-8") as source:
            return json.load(source)
    except FileNotFoundError:
        return None


metakat_data = load(metakat_json)
if metakat_data is not None:
    envelope["metakat_json"] = metakat_data

proarc_data = load(package_info_json)
if proarc_data is not None:
    envelope["proarc_json"] = proarc_data

with open(meta_file, "w", encoding="utf-8") as target:
    json.dump(envelope, target)
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

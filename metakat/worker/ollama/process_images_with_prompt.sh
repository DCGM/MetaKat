#!/usr/bin/env bash
# Process all images in a directory with Ollama using a given prompt and model.
# Each image produces a corresponding JSON response file in the output directory.
#
# Usage:
#   ./process_images_with_prompt.sh <host> <model> <prompt-file> <input-image-dir> <output-json-dir>

set -euo pipefail

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <host> <model> <prompt-file> <input-image-dir> <output-json-dir>"
    exit 1
fi

HOST="$1"
MODEL="$2"
PROMPT_FILE="$3"
INPUT_DIR="$4"
OUTPUT_DIR="$5"

# Resolve path to ollama_worker.py relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OLLAMA_WORKER="$SCRIPT_DIR/ollama_worker.py"

# Validate inputs
if [ ! -f "$PROMPT_FILE" ]; then
    echo "Error: Prompt file not found: $PROMPT_FILE"
    exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input image directory not found: $INPUT_DIR"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Supported image extensions
IMAGE_EXTENSIONS="jpg jpeg png gif webp"

# Build a find pattern for supported extensions
FIND_ARGS=()
first=true
for ext in $IMAGE_EXTENSIONS; do
    if [ "$first" = true ]; then
        first=false
    else
        FIND_ARGS+=("-o")
    fi
    FIND_ARGS+=("-iname" "*.$ext")
done

# Count total images for progress reporting
TOTAL=$(find "$INPUT_DIR" -maxdepth 1 -type f \( "${FIND_ARGS[@]}" \) | wc -l)
CURRENT=0

echo "Found $TOTAL image(s) in $INPUT_DIR"

# Process each image
find "$INPUT_DIR" -maxdepth 1 -type f \( "${FIND_ARGS[@]}" \) | sort | while read -r IMAGE_PATH; do
    CURRENT=$((CURRENT + 1))
    BASENAME="$(basename "$IMAGE_PATH")"
    NAME_NO_EXT="${BASENAME%.*}"
    OUTPUT_FILE="$OUTPUT_DIR/${NAME_NO_EXT}.json"

    # Skip if output already exists
    if [ -f "$OUTPUT_FILE" ]; then
        echo "[$CURRENT/$TOTAL] Skipping $BASENAME (output already exists)"
        continue
    fi

    echo "[$CURRENT/$TOTAL] Processing $BASENAME ..."

    python "$OLLAMA_WORKER" \
        --host "$HOST" \
        --model "$MODEL" \
        --prompt-file "$PROMPT_FILE" \
        --images "$IMAGE_PATH" \
        --json-mode \
        --output-file "$OUTPUT_FILE" \
        --log-level INFO

    echo "[$CURRENT/$TOTAL] Saved $OUTPUT_FILE"
done

echo "Done. Processed $TOTAL image(s)."

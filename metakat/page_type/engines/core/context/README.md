# Context page-type inference

Four entry modules live beside the original ViT engine:

- `dino.py`: LightlyTrain DINOv2-with-registers ViT-B/14, final normalized CLS
  token, 224 × 224 whole-image resize with OpenCV INTER_AREA and ImageNet scaling.
- `mmbert.py`: mmBERT-base attention-masked mean pooling, head truncation that
  preserves the final special token, and the padding-NaN fix used for the dump.
- `alto_stats.py`: ALTO/PAGE XML text and layout extraction, the full 3,324 raw
  statistics, then all 147 grouped normalized features.
- `context.py`: feature extraction, fusion and context model, overlapping-window
  scanning, merging, semantic outputs, and a JSON inference command.

`models.py`, `merging.py`, `labels.py`, `_page_stats.py` and `_ocr_xml.py` freeze
the experiment's architecture and feature contract locally. There is no runtime
dependency on the training repository, SQLite index, LMDB, or database dumps.
The supported architecture is BERT with separate image, text, statistics and
pagination MLPs, page-type and side heads, the adjacent-pair boundary head, and
independent linear page-start/page-end heads. Legacy architecture variants are
rejected. The binder and feature encoders remain independent of the context model.

`pagination.py` freezes the pagination parser and normalization from training.
The binder reads existing `MetakatPage.pageNumber.text` into `PageTypeInput.page_number`
in scan order. Run the page-number engine before page-type processing to supply this
signal; page-type inference does not fill, alter, or infer page numbers itself.
Artifacts (`metakat_context_v3`) include a separate five-input pagination MLP:
log magnitude capped at the checkpoint's fixed cap (normally 10000), odd parity,
Arabic flag, Roman flag, and availability. Features round through FP16 exactly as
in training. Missing or rejected numbers become five zeros, and the projected branch
is masked after its MLP. Entire scans without page numbers remain supported.

Filtering matches policy `db_candidate_single_v1`: accept positive Arabic integers,
canonical Roman numerals, or the unbracketed part of `55 [58]` (55). Reject bracket-only
`[12]` / `[XIV]`, suffixes, ranges, spreads/multiple numbers, zero, and malformed values.
These remain noisy candidates; unbracketed metadata does not prove physical printing.
Confidence and page indices are not added to the input. There is no inference-time
dropout. Only v3 artifacts with start/end heads and pagination support are loaded;
v1/v2 artifacts and missing head weights fail before loading feature encoders.
Missing page numbers remain supported through the availability mask.

## Model files

Export a **trusted local** checkpoint from `classification-with-context`:

```bash
python -m page_sequence.export --checkpoint /path/to/best.pt --output /models/context.pt
```

The exporter reads the training pickle once and writes tensors plus primitive
metadata only. MetaKat loads the artifact with `weights_only=True`. It includes
architecture, class order, feature normalization/schema and inference settings;
optimizer state, RNG and data-source paths are omitted. Loading validates class
order, all five heads (strict weight loading), feature schema and encoder dimensions
before inference. Export the selected `best.pt` checkpoint for deployment, rather
than assuming the last training update is best. No training checkpoint is bundled
with the code.

Use the same feature encoders as the training dump. For the current feature contract:

- DINO checkpoint SHA256:
  `d08f6d05da58efff5748274f487c2583d94cad5ee9e0490a57e40c3a3832f80e`.
- `jhu-clsp/mmBERT-base` revision
  `c5955035435e2bf121cde7f3c8863ef52ff35d82`.

The mmBERT model and tokenizer must already be in a local Hugging Face model
directory; inference never downloads them. The DINO file is the original
LightlyTrain exported state dictionary, not a generic pretrained DINO model.
The context artifact is not a substitute for either encoder's weights.

Example **core** configuration (paths are resolved relative to its file):

```yaml
name: page_type_core_engine_context
checkpoint_path: /models/context.pt
image_checkpoint_path: /models/dino/exported_last.pt
text_model_dir: /models/mmbert
# Optional: verify the expected DINO weights before loading.
image_checkpoint_sha256: d08f6d05da58efff5748274f487c2583d94cad5ee9e0490a57e40c3a3832f80e
device: cuda
feature_chunk_size: 128
image_batch_size: 32
text:
  batch_size: 8
  max_tokens_per_batch: 16384
  max_length: 8192
  max_chars: 4000
inference:
  window_size: 256
  stride: 128
  merge: center_logits
  edge_ramp: 64
  start_threshold: 0.5
  batch_size: 64
```

`max_chars: 4000` mirrors the pilot's database text cap. Text is reconstructed
from OCR lines in top-to-bottom order, joined with newlines, before this cap.
When training from a different text export, check its text reconstruction and
cap against this preprocessing rather than assuming equivalent OCR formats
produce identical text embeddings. Statistics always use the whole OCR page.
Dense vectors and raw statistics are rounded through FP16 as in the training
stores; dense vectors are then L2 normalized, statistics are bounded with the
saved normalizer and divided by `sqrt(147)`. CPU encoders use FP32 computation;
CUDA uses the dump's FP16 DINO and BF16 mmBERT computation. The context model
runs in FP32. CPU/GPU feature extraction can have small numerical differences.

## Scanning and output

BERT attends bidirectionally within each window, with learned absolute position
embeddings reset for every window. Window size, stride and edge weighting are
loaded from the exported checkpoint;
the example above uses 256-page windows with stride 128. Only override them for
an evaluated deployment setup. Centre-weighted logits are accumulated over overlaps,
followed by softmax
or sigmoid. All merge strategies from the experiment are available, including
uniform logits/probabilities, centre probabilities, and central-window choice.

Both independent start/end heads are loaded and computed from each page
representation, matching training exactly. They strengthen the shared encoder
during training; their probabilities are not fed into the pair head or combined
with its score at inference. Scanning and the public three-task output remain
unchanged. Consistency regularization is a training loss and requires no extra
inference parameters or decoding rule.

The boundary head compares adjacent contextual page representations. Each
window's first boundary output is excluded; an overlapping window supplies it.
Only the entire batch's first page has no valid predecessor, so its
`startsDocument` is `None`. Other pages return true or false with confidence.
No document boundaries or existing page labels are used as model inputs.

Feature extraction processes bounded chunks and retains compact vectors and
merged outputs proportional to page count, never all decoded images or OCR
texts at once. Window and token batch sizes are separately bounded. A missing
OCR mapping masks both text and statistics; a genuinely blank OCR page remains
present. A missing image mapping masks the image modality. A page with neither
input fails, as do unreadable/corrupt files: pages are never silently dropped
or reordered.

For standalone inference, make an ordered JSON list:

```json
[
  {"image_path": "scan1.jpg", "ocr_path": "scan1.xml", "page_number": "12"},
  {"image_path": "scan2.jpg", "ocr_path": null, "page_number": "[13]"}
]
```

```bash
python -m metakat.page_type.engines.core.context.context \
  --engine-config context.yaml --pages pages.json --output predictions.json
```

Page paths are relative to `pages.json`. Output is a JSON list of semantic
`pageType`, `side`, and `startsDocument` values with confidences. For the normal
pipeline, use this core with `page_type_bind_engine_base`; it propagates the
results into `MetakatPage`. `process()` follows model inference with the shared
[bottom-document resolver](../../../README.md#bottom-document-resolution).
It scores whole-scan partitions using start-head log odds and page-type policies.
The resulting `startsDocument` decisions are consumed by bibliography binding.
`predict()` exposes raw model outputs without the resolver; `scan_features()`
continues to expose raw merged probabilities for model evaluation.

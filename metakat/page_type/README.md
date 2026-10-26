# Page-type processing

The `page_type` component classifies an ordered scan batch. Both core engines
accept `list[PageTypeInput]` and return an equally long `list[PageTypePrediction]`.
The position in the list identifies a page, so repeated image paths remain
separate scan positions. Each input has `image_path` and optional `ocr_path`
(ALTO or PAGE XML), plus optional `page_number` text. Missing inputs use `None`.
The binder passes existing `MetakatPage.pageNumber.text` unchanged; run page-number
processing first to supply it. Only pagination-aware context checkpoints consume it.

The output contains:

- `pageType`: `(PageType, confidence)`.
- `side`: optional `(PageSideType, confidence)`, canonically `left`, `right`,
  or `singlePage` in the MetaKat schema.
- `startsDocument`: optional `(bool, confidence)`, meaning **this** page starts
  a bottom-level document, such as an issue. Confidence belongs to the returned
  boolean. After resolution it measures a score margin, not calibrated probability.

## ViT (`page_type_core_engine_vit`)

This engine loads the existing image classifier, handles all pages internally,
and returns page types. Its side output is unset; document boundaries are resolved
from page-type evidence without a learned start head. Every
page requires an image.

```yaml
name: page_type_core_engine_vit
model_dir: path/to/model
labels:
  titlePage: TitlePage
```

The checkpoint supplies the numeric class-to-model-label mapping. The optional
`labels` configuration maps semantic `PageType` values to model labels;
unspecified values use the enum value itself. Class IDs must be contiguous
from zero, labels unique, and every checkpoint label must resolve to a page type.
The standalone ViT command retains its full model-label probability output
for each image.

## Context (`page_type_core_engine_context`)

The [context engine](engines/core/context/README.md) combines DINOv2 image
vectors, mmBERT OCR text vectors, and grouped normalized OCR statistics with
the experiment's fusion layers and contextual BERT. It returns all three tasks
and merges overlapping windows over arbitrarily long scan batches.

## Base binder (`page_type_bind_engine_base`)

```yaml
name: page_type_bind_engine_base
```

The binder passes **all** `MetakatPage` objects to the core in ascending
`batch_index`, the schema's scan order. It does not sort image filenames or
split the batch into documents. It resolves image and OCR paths against the
batch directory, preferring the ALTO mapping and falling back to PAGE XML.
Duplicate scan indices or missing core results are errors.

The binder writes the results back by page position on a copy of `MetakatIO`.
An absent optional prediction preserves any existing value, so the ViT engine
continues to work without providing sides. `startsDocument` survives internal JSON
serialization and is consumed by bibliography grouping. MODS records the resulting
hierarchy rather than exporting this internal flag directly.

## Bottom-document resolution

Both engines have two stages: `predict(pages)` exposes raw model outputs and
`process(pages)` applies the standalone policy in `core/boundaries.py` afterward.
Page types and sides are preserved. Position zero keeps `startsDocument=None`;
it implicitly begins the first scan segment, without claiming a predecessor.

The resolver finds the highest-scoring partition of the entire scan with a
three-phase dynamic program: opening pages, content, and closing pages.
Start-head log odds support or oppose each split. Page types penalize misplaced
front/back matter and cuts inside opening sequences or before closing pages.
All penalties are finite: neither a page type nor a model score is an absolute
constraint. Low-confidence page types contribute less evidence. Missing start
scores leave a type-only fallback. Title-less documents can be separated by the
start head; repeated title pages alone do not force separate documents.

The following **core** configuration sets the initial defaults:

```yaml
boundary_policy:
  model_weight: 1.0
  split_penalty: 1.5
  misplaced_type_penalty: 8.0
  content_in_margins_penalty: 0.25
  opening_run_cut_penalty: 5.0
  before_closing_cut_penalty: 6.0
```

These weights require development-set tuning. The output confidence is the
sigmoid of the score difference between the best complete solutions with and
without that boundary. It is not a measured accuracy or probability. Resolver
changes to model decisions are logged by scan index. Use `predict()` to inspect
the original model evidence separately. The resolver uses linear time and memory
in the number of pages and has no training/encoder dependency.

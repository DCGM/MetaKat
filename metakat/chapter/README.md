# Chapter processing

## Composable OCR + YOLO pipeline

The production chapter core can compose three independently replaceable
stages. A core engine directory uses this configuration:

```json
{
  "name": "chapter_core_engine_pipeline",
  "stages": {
    "toc_page_analysis": "toc_page_analysis",
    "toc_extraction": "toc_extraction",
    "toc_alignment": "toc_alignment"
  }
}
```

Each path is relative to the core engine directory unless it is absolute.
The initial implementations use these respective `name` values in their own
`metakat_engine_config.json` files:

- `toc_page_analysis_engine_yolo_alto`
- `toc_extraction_engine_yolo_alto`
- `toc_alignment_engine_fuzzy`

The first two stage directories contain their corresponding YOLO `.pt`
models and may override their `labels` and normal `EngineYOLOALTO` settings.
The fuzzy stage accepts `title_match_threshold` (default `0.7`) and
`offset_tolerance` (default `2`). It first builds a monotonic chain of physical
page-number anchors, then resolves remaining titles using a consistent anchor
offset. When the surrounding anchors imply different offsets, title matches
are constrained only to their physical page interval. With only one unusable
anchor the search uses its one-sided bound; without anchors it falls back to
document-wide title matching. Input image and ALTO filenames must have
matching, unique stems.

`ChapterCoreEngine.process()` accepts an optional `page_numbers` sequence
aligned with the image sequence. Each item is the printed page number or
`None`. When supplied, these values take precedence over physical page numbers
detected during TOC page analysis; when omitted, internal detection remains
available to the alignment stage. Physical page numbers are internal alignment
inputs and are not returned in `ChapterCoreResult`.

## Experimental source

The experimental `xshele02` pipeline is contained in
`metakat/chapter/xshele02`. Its command-line entry point and supporting Python
modules live alongside the `toc_only` extraction package.

Run it from the repository root:

```bash
python -m metakat.chapter.xshele02.main INPUT_PATH
```

Use `--llm gpt` or `--llm gemini` to select an LLM-backed TOC extractor;
without `--llm`, the pipeline uses PERO OCR.

The implementation was synchronized from the authoritative `src` supplied
with the xshele02 project. Its Python dependencies are recorded in
`xshele02/requirements.txt`. The large trained models are deliberately not
stored in this repository; provide them below
`metakat/chapter/xshele02/modely` using the layout expected by `main.py`.

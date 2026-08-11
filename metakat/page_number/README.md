# Page-number processing

Page-number processing is split into two engines:

- a core engine that detects and aligns printed page-number regions, parses
  their OCR text, filters candidates, and selects one physical number per
  page;
- a bind engine that only writes the selected core results and their geometry
  mappings to the MetaKat representation.

Each engine is supplied as a directory containing a
`metakat_engine_config.json` file. The initial implementations use the names
`page_number_core_engine_yolo` and `page_number_bind_engine_base`.

The reusable `PhysicalPageNumberResolver` lives in the core package. Its
standard mode retains a single valid candidate anywhere on the page. When
several valid candidates exist, only candidates whose centers are in the top
or bottom edge band are eligible; edge proximity and geometry confidence are
combined to select the winner. Its edge-only mode applies the edge condition
even to a single candidate and is used by chapter TOC-page analysis.

The core configuration accepts:

| Parameter | Default | Meaning |
|---|---:|---|
| `page_number_edge_band_ratio` | `0.15` | Height of each edge band as a fraction of page height. |
| `page_number_edge_score_weight` | `0.65` | Weight assigned to edge proximity; the remaining weight is assigned to detection confidence. |

`PageNumberCoreEngine.process()` returns `PageNumberCoreResult`, whose
`page_numbers` mapping contains one `PhysicalPageNumberEvidence` for each
selected page. The model extends
`metakat.common.models.DetectionEvidence`, retaining its complete OCR string,
confidence, MetaKat bounding box, and page key. It adds the parsed token,
integer value, and numeral system. `normalized_text(case=None)` exposes the
parsed token;
`output_text(case=None)` returns that token when parsing succeeded and the
original OCR string otherwise. The optional case is `"lowercase"` or
`"uppercase"`; `None` preserves the parsed Roman-token case. The binder
creates MetaKat detection UUIDs only for evidence it actually writes.

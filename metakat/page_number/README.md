# Page-number processing

## Navigation

- [Purpose](#purpose)
- [Page-number core engine contract](#page-number-core-engine-contract)
  - [Implementing another core engine](#implementing-another-core-engine)
- [Core and bind orchestration](#core-and-bind-orchestration)
  - [Engine directory convention](#engine-directory-convention)
  - [Processing handoff](#processing-handoff)
- [Available core implementation](#available-core-implementation)
  - [YOLO + ALTO](#engine-yolo--alto-page_number_core_engine_yolo)
- [Document postprocessing](#document-postprocessing)
- [Page-number parsers](#page-number-parsers)
  - [Decorated page-number parser](#decorated-page-number-parser)
- [Page-number resolvers](#page-number-resolvers)
  - [Physical page-number resolver](#physical-page-number-resolver)
- [Available bind implementation](#available-bind-implementation)
  - [Base](#engine-base-page_number_bind_engine_base)
- [Observability and revision](#observability-and-revision)

## Purpose

The `metakat.page_number` package detects the physical page number printed on
each document page and writes it to `MetakatPage.pageNumber`.

Physical page numbers are distinct from the destination-page references read
from a table of contents. A physical number belongs to the page on which it is
printed. A TOC reference belongs to a chapter entry and is processed by the
[`metakat.chapter`](../chapter/README.md) package.

Page-number processing has two boundaries:

1. The **[core engine](#page-number-core-engine-contract)** analyzes ordered
   page images and ALTO files. It returns at most one parsed physical-number
   evidence object per page, with an explicit unresolved entry where needed.
2. The **[bind engine](#available-bind-implementation)** selects processable
   MetaKat pages, invokes the core, applies confidence precedence, creates
   detection UUIDs, and writes the selected evidence and its provenance into
   `MetakatIO`.

The core owns detection, OCR alignment, parsing, candidate filtering,
document-level postprocessing, and physical-candidate selection. The binder does not repeat or alter those
decisions.

```mermaid
flowchart LR
    P[Ordered page images + ALTO files]
    C[Page-number core engine]
    R[PageNumberCoreResult]
    B[Page-number bind engine]
    M[MetakatIO pageNumber + geometry maps]

    P --> C --> R --> B --> M
```

## Page-number core engine contract

Every page-number core engine subclasses `PageNumberCoreEngine` and exposes:

```python
process(
    images: List[str],
    alto_files: List[str],
) -> PageNumberCoreResult
```

The two input sequences represent the same pages in the same physical order.
An image filename stem is the page key used in the result. Page keys returned
by the core must be unique and must identify one of the input pages.

| Argument | Contract |
|---|---|
| `images` | Ordered image paths. Each filename stem identifies the page in core output. |
| `alto_files` | Ordered ALTO paths corresponding position by position to `images`. |

The method returns:

```python
PageNumberCoreResult(
    page_numbers: Mapping[str, PhysicalPageNumberEvidence | None],
    decisions: Mapping[str, PageNumberDecision],
    chunks: tuple[PageNumberChunk, ...],
)
```

`page_numbers` contains every input page key, in input order. Its value is
`None` when no physical number can be resolved, otherwise one evidence object.
Unnumbered pages are retained in the physical sequence. The final NDK stage
may supply a cataloguing label when its pagination context is sufficiently clear;
these labels have `supplied=True`, empty OCR text, and no invented geometry. Each value is a
`PhysicalPageNumberEvidence`:

```python
@dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class AltoRefs:
    blocks: tuple[str, ...] = ()
    lines: tuple[str, ...] = ()
    words: tuple[str, ...] = ()


@dataclass(frozen=True)
class DetectionEvidence:
    text: str
    confidence: float
    bbox: BoundingBox | None
    page_key: str
    alto: AltoRefs = AltoRefs()  # keyword-only


@dataclass(frozen=True)
class PhysicalPageNumberEvidence(DetectionEvidence):
    normalized: str | None
    value: int | None
    numeral_system: PageNumberNumeralSystem | None
    supplied: bool = False  # keyword-only
    suffix: str = ""  # keyword-only
    bracketed: bool = False  # keyword-only
```

It extends the common `DetectionEvidence` provenance with a parsed
physical-number representation.

| Attribute | Core result contract |
|---|---|
| `text` | Complete OCR text covered by an observed detection; empty for a supplied NDK label. |
| `confidence` | Document support score used for binding precedence and review. With postprocessing enabled, this replaces raw detector confidence; see [Document postprocessing](#document-postprocessing). |
| `bbox` | `None` for supplied labels. Otherwise selected detection geometry in the aligned page coordinate system, represented by the top-left coordinates `x` and `y` plus `width` and `height`. |
| `page_key` | Image filename stem of the source page. |
| `alto` | ID attributes of the ALTO elements holding the detection's words: `TextBlock` (`blocks`), `TextLine` (`lines`) and `String` (`words`), each once, in reading order. IDs are unique only within the page's ALTO file. A level is empty where the ALTO gives its elements no ID - producers differ, and most write IDs on blocks, many on lines, few on words. |
| `normalized` | Bare parsed or supplied numeral token; use `output_text()` for the complete archive label. |
| `value` | Integer value of the parsed Arabic or Roman numeral, or `None`. |
| `numeral_system` | `PageNumberNumeralSystem.ARABIC`, `PageNumberNumeralSystem.ROMAN`, or `None`. |
| `supplied` | True for a cataloguing label inferred or supplied from explicit pagination facts. |
| `suffix`, `bracketed` | NDK letter suffix and bracket notation. `value` remains the numeral base; a suffix does not identify a new ordinary numeric page. |

`normalized_text(case=None)` returns the normalized token. `output_text()`
adds any NDK suffix and brackets to that token, or returns the complete OCR text
if no normalized token exists. Letter suffixes stay lowercase during case conversion.
Both methods accept `case="lowercase"` or `case="uppercase"`; `None` preserves
the parsed Roman token's case. Observed detections are successfully parsed, but the optional fields allow evidence to retain
an unsuccessful parse at other processing boundaries.

`PageNumberCoreResult.decisions` records each postprocessing action, positional
support, local sequence support, sequence agreement before correction, original
reading, and source confidence. These diagnostics are available to core callers and actions are
logged; the binder writes the selected value and confidence to the existing
schema. It has no MetaKat detection UUIDs: UUID creation and schema mutation belong to the
[bind engine](#available-bind-implementation).

### Implementing another core engine

Every core engine receives its configuration mapping directly. To add one:

1. subclass `PageNumberCoreEngine` and call its constructor with the core
   configuration mapping;
2. implement `process()` while preserving the complete
   [core contract](#page-number-core-engine-contract); the base class does not
   prescribe detection, parsing, or resolution;
3. return common `PhysicalPageNumberEvidence` objects or `None` for every
   input page key, in order;
4. register the config `name` in `page_number_core_engines` in core
   `definitions.py`;
5. test engine loading, unresolved results, parsing, candidate selection, and
   page key validation. Older sparse results remain accepted by the binder.

`PageNumberBindEngineBase` can bind any implementation satisfying this
contract. A core that returns a different result model requires a matching
bind engine and explicit bind-engine registration.

## Core and bind orchestration

### Pipeline configuration

The complete MetaKat pipeline configuration nests the page-number core and
bind mappings under `page_number`. A minimal configuration is:

```json
{
  "page_number": {
    "core": {
      "name": "page_number_core_engine_yolo",
      "model_path": "page_number/core/model.pt",
      "labels": {
        "PageNumber": "cislo strany"
      }
    },
    "bind": {
      "name": "page_number_bind_engine_base"
    }
  }
}
```

The central pipeline loader resolves relative `*_path` and `*_dir` values
before construction. Core and bind loaders receive these prepared mappings,
read `name`, and resolve it through their respective registries. An unknown
name is an error; placing an implementation in the package does not register
it automatically. Per-engine configuration files and directory-based loader
arguments are not supported.

### Processing handoff

When both page-number mappings are configured,
`process_batch()` runs page-number processing before page-type,
bibliographic, and chapter processing. This order allows later processing,
including chapter destination alignment, to consume the physical numbers
written to `MetakatPage`.

The page-number bind engine owns the complete handoff:

1. select MetaKat pages that have both image and ALTO mappings;
2. resolve their paths relative to the batch directory;
3. invoke the core once with the ordered inputs;
4. map returned image-stem page keys back to `MetakatPage` objects;
5. bind each selected result according to
   [field precedence](#field-mapping-and-precedence).

## Available core implementation

The registered core implementation is:

| Config `name` | Implementation |
|---|---|
| `page_number_core_engine_yolo` | YOLO geometry aligned with ALTO text, followed by numeral parsing and physical-candidate selection. |

### Engine: YOLO + ALTO (`page_number_core_engine_yolo`)

The engine detects page-number geometry with YOLO, aligns each detection with
ALTO words, parses the covered text, and selects initial physical page numbers.
It then runs reusable document postprocessing across all ordered input pages.

#### Configuration

```json
{
  "name": "page_number_core_engine_yolo",
  "model_path": "page_number/core/model.pt",
  "labels": {
    "PageNumber": "cislo strany"
  },
  "page_number_edge_band_ratio": 0.15,
  "page_number_edge_score_weight": 0.65,
  "postprocess": {
    "enabled": true,
    "position_bandwidth_x": 0.04,
    "position_bandwidth_y": 0.025,
    "minimum_support": 3,
    "firm_support_count": 5,
    "firm_cluster_spread": 0.5,
    "maximum_firm_positions": 2,
    "additional_position_support_cap": 0.65,
    "position_window": 40,
    "sequence_window": 8,
    "drop_threshold": 0.1,
    "segmentation": {
      "enabled": true,
      "minimum_run_anchors": 3,
      "maximum_anchor_gap": 12,
      "minimum_offset_change": 4,
      "maximum_number_step": 2
    },
    "ndk": {
      "enabled": true,
      "profile": "shared",
      "max_gap_pages": 8,
      "minimum_anchor_confidence": 0.6,
      "complete_document": false,
      "wholly_unnumbered": false,
      "tail_mode": "unknown"
    }
  }
}
```

`labels` maps the supported `PageNumberType` value `PageNumber` to the raw
YOLO model label. When omitted, `PageNumber` defaults to `cislo strany`.
Unknown keys, non-string values, and empty values are rejected. The two
`page_number_*` settings are passed to `PhysicalPageNumberResolver`; their
defaults and validation are documented with the
[physical page-number resolver](#physical-page-number-resolver).

#### Geometry and text loading

`model_path` must identify the YOLO `.pt` model explicitly.

For every input page, the shared `EngineYOLOALTO`:

1. runs YOLO on the image;
2. reads the corresponding ALTO document;
3. aligns ALTO words to detected geometry using bidirectional containment and
   greatest-coverage word assignment;
4. exposes the resulting aligned regions to the page-number core engine.

Optional shared settings are `yolo_batch_size` (default `32`),
`yolo_confidence_threshold` (`0.25`), `yolo_image_size` (`640`),
`yolo_device` (`0`), and `minimum_overlap_coverage` (`0.65`). The YOLO batch
size and image size must be positive, and its confidence threshold must be in
`[0, 1]`. Configuration values override the corresponding constructor
defaults.

The page-number core then visits every aligned page. Duplicate page keys are
an error. A region is retained only when its exported label matches the raw
YOLO label configured for `PageNumber`.

The implementation parses each retained region's aligned OCR text with
`DecoratedPageNumberParser`. When a page has multiple valid parsed detections,
`PhysicalPageNumberResolver` resolves them to at most one physical page number
using `STANDARD` mode. The parser and resolver are documented independently
below.

After initial selection, all candidates and page dimensions are passed to
`PageNumberPostprocessor`. Input image and ALTO lists must have equal lengths,
matching corresponding filename stems, and unique image stems. Aligner output
is reassembled in input order, and pages omitted by a detector are retained.
Finite positive ALTO page dimensions are required on every page.

## Document postprocessing

Postprocessing is enabled by default. `postprocess.enabled: false` preserves
initial resolver selections and their detector confidence, while still returning
an explicit entry for every input page.

### Numbering-based scan chunks

A batch can contain several books or issues. Before learning any position
model, `PageNumberChunker` separates sustained numbering regimes using the
initial parsed detector selections. Each resulting chunk gets independent
position learning, recovery windows, confidence scoring, sequence correction,
and NDK pagination. Explicit `NdkPageContext.unit_key` changes are hard
boundaries; the chunker may refine a known unit into several numbering regimes
while preserving that unit's identity.

By default, a candidate transition needs a supported run of at least three
observed numbers on **each** side. A run requires one numeral system, increasing
values, at most 12 ordinary scan positions between anchors, and no step larger
than two numbers per intervening scan. Known excluded scans and foldout
segments do not advance that scan count. Detector confidence is not used to
establish the boundary, because those scores are not reliable indicators of a
correct reading. Supplied numbers, invalid geometry, and recovered OCR never
become boundary anchors.

The chunker splits supported runs at a numbering restart or numeral-system
change. A large forward discontinuity is split when the jump exceeds the
progression predicted by the two runs' median slopes by at least four numbers.
This can also isolate a missing block of scans: a processing boundary does not
assert that a different bibliographic document was found. Short interruptions,
single OCR errors, repeated header numbers, ordinary insertions, and small scan
omissions do not establish a boundary on their own. Set
`postprocess.segmentation.enabled` to false to retain only explicit unit
boundaries.

For example, a supported run ending at 122, several unnumbered scans, and a
supported new run starting at 1 produce three scopes: the old run, an ambiguous
boundary interval, and the new run. Numbering alone cannot say whether the
intervening scans are old end matter or new front matter. Those ambiguous
scans receive neither recovery windows from another chunk nor NDK labels.
An existing valid reading in that interval is retained at review confidence
0.35; otherwise the result stays unresolved. Explicit same-page segment links
attach a scan to its parent's side where possible, and prevent cuts that would
separate a known parent and child.

A chunk is a **processing hypothesis**, not a confirmed whole document. Roman
front matter and an Arabic body may belong to the same book. Consequently,
inferred splits do not enable NDK leading/trailing inference, even when the
batch was marked `complete_document=True`. Interior NDK rules still apply
inside each nonambiguous chunk. Explicit complete units without inferred
internal splits retain their configured edge policy. `NdkPaginator` also
respects chunk scopes when invoked separately on a chunked result.

`PageNumberCoreResult.chunks` covers every input page exactly once in order.
Each `PageNumberChunk` records `page_keys`, its start `reason`, whether its scope
was `inferred`, whether it is `ambiguous`, and the original `unit_key`.
`PageNumberDecision.chunk_index` links each page to that scope. Boundary reasons
are `input_start`, `explicit_unit`, `numbering_restart`, `numeral_system_change`,
`numbering_offset_change`, or `uncertain_boundary`; ambiguous pages use the
`boundary_uncertain` decision action. Decisions and boundaries are logged and
available to callers without adding shared schema fields.

Numbering cannot reveal every document border. Two books with uninterrupted
compatible pagination, or too few reliable observed numbers, may remain one
chunk. The local position window still limits distant influence in that case.
The thresholds are conservative heuristics and need evaluation on representative
batches; no image, page-type, or bibliographic boundary classifier is added.

### Position estimation and recovery

`PageNumberPostprocessor` estimates distinct page-number positions within each
numbering chunk, using the initial selected detections and coordinates normalized
by each page's dimensions. Each scan supplies at most one seed. Spatial grouping
uses a radius of two bandwidths, merges nearby representations of the same
region, and trims isolated tails outside a squared bandwidth distance of 2 from
the coordinate-wise median. This prevents one displaced detection from spoiling
an otherwise tight cluster.

Each location earns support **independently**, from its count and tightness.
There is no normalization against the most frequent location. Thus 30 tight
bottom-left detections and 40 tight bottom-right detections can both establish
firm positions; missing detections do not count against either location.

The defaults are:

| Setting | Default | Meaning |
|---|---:|---|
| `minimum_support` | 3 | Independent nearby seeds needed to estimate a local cluster at all. |
| `firm_support_count` | 5 | Independent seeds in the chunk needed for full count support at a firm position. |
| `firm_cluster_spread` | 0.5 | Maximum RMS dispersion, in bandwidth units, for full tightness support. |
| `maximum_firm_positions` | 2 | Maximum firm positions per entire chunk; accepts 1 or 2. |
| `additional_position_support_cap` | 0.65 | Maximum location strength for any position outside the firm shortlist. |
| `position_window` | 40 | Ordinary scans considered on each side for local center refinement and weak-location evidence. |

RMS dispersion is computed from normalized x/y distances divided by their
respective bandwidths. With defaults, a spread of 0.5 corresponds to 2% of page
width if dispersion is purely horizontal, or 1.25% of page height if purely
vertical; combined x/y dispersion uses their squared distances together.

At most two sufficiently populated and compact clusters enter a fixed firm
shortlist for the chunk. They are ranked by count weighted by compactness. The
shortlist is not chosen separately for each parity or moving window, so a third
location cannot become firm merely by being the only local alternative. Other
repeated locations remain available for recovery and candidate selection at
reduced confidence.

Each page is scored using independent peers **excluding its own detection**.
Firm positions use the other supporting detections in the chunk. Nearby peers
refine the center when available; otherwise the established chunk-wide estimate
is used. A run of missed detections does not erase an already firm position.
Additional weak positions still require local evidence. The per-location strength is:

```text
count_support = min(1, independent_peer_count / firm_support_count)
tightness = min(1, (firm_cluster_spread / RMS_spread)^2)
location_strength = count_support * tightness
```

Zero spread gives tightness 1. A location outside the two-position
shortlist is capped at `additional_position_support_cap`. Full support therefore
requires both enough independent peers and a tight cluster. The tightness check
uses the greater of the overall supporting spread and any locally estimated
spread. A detected number
needs five other seeds under the default settings; it cannot be its own fifth
supporting observation. A recovered number is evaluated by the same rule.

Candidate position support is the greatest independent location strength,
multiplied by Gaussian proximity to that location's center:

```text
support = max(location_strength * exp(-squared_bandwidth_distance / 2))
```

Contributions beyond a distance of three bandwidths are ignored. This keeps a
candidate in a distribution tail uncertain even in a very large document, and
prevents nearby modes from inflating support by adding their scores together.
The two firm positions can each reach support 1 without competing. With the
default cap, additional positions have final observed-number confidence at most
0.8045, even after the sequence-agreement bonus. Scores remain heuristic review
scores and need calibration on representative scans.

Alternating left/right placement is enforced only when the two firm clusters
show opposing parity preferences: each needs at least six seeds and at least
80% of its seeds on the same parity. Otherwise parity is not used to reject a
location. This avoids mistaking a single position detected only on even scans
for evidence that odd scans need a different position. Parity is counted inside
each chunk, excluding known covers and foldout segments.

Every locally supported location produces a search window extending three
bandwidths in each direction, clipped to the page. All pages with supported
locations are searched, including pages already carrying a detection. Weak
additional locations are searched too. The ALTO adapter looks for short
numeral-only lines or geometrically separated spans in a running header, allows
split digits and simple decoration, and retains the original ALTO references.
It reads existing OCR at its original coordinates; it does **not** run image OCR
or recover numerals absent from ALTO. A different callback can inspect image
crops using the same windows.

Recovered candidates never retrain the position estimates or firm shortlist.
Candidates are ranked by position support, using the initial selection to break
ties. Without an established firm position or enough local peers to estimate a weak
position, the initial selection is retained when available at the existing uncertain starting confidence of 0.5.

### Sequence smoothing and review confidence

The sequence stage uses a fixed snapshot of the selected evidence; inferred
corrections cannot become anchors for more corrections. Within
`sequence_window` physical pages on each side (default 8), it requires at least
two reliable anchors on each side sharing the same numeral system and
`value - physical_index` offset. The nearest reliable anchor on each side must
also agree. Anchors with learned position support below 0.6 are excluded.
This avoids extrapolation at document ends and preserves ordinary numbering
resets, Roman-to-Arabic transitions, missing scans, and unnumbered insertions
when the surrounding offsets disagree.

When that consensus exists, an attested candidate with the expected number
and position support at least 0.6 is preferred (without a learned mode, an
attested candidate is also eligible). Otherwise an existing, well-positioned
reading in the same numeral system may have its normalized number corrected.
Raw OCR text, bounding box and ALTO references are retained. Roman case and
Arabic zero padding are preserved. This observation stage leaves pages without
evidence unresolved; the separate NDK stage may then supply archival labels.

Confidence is a heuristic review score, **not a calibrated probability**:

- Without a supported position model: 0.5, regardless of YOLO confidence.
- With positional support `s`: `0.25 + 0.73 * s` (a typical position gets 0.98).
- Agreement with a strong sequence adds 0.08, capped at 0.99.
- Inferred corrections are capped at 0.85 so they remain reviewable.
- An unresolved sequence contradiction caps confidence at 0.35. With at least
  four reliable neighbors in the same numeral system, fewer than 25% supporting
  the candidate's physical-index offset or a compatible positive progression
  also caps confidence at 0.35. Slower monotonic progression remains compatible
  with extra unnumbered leaves and does not reduce folio confidence by itself. This keeps
  repeated running-header years from receiving folio-level certainty.
- Support below `drop_threshold` (default 0.1) drops the candidate after recovery
  and sequence reconsideration. Set this threshold to 0 to retain outliers at
  low confidence.

Small documents or sparse layouts may not supply enough seeds to learn a
position model; their scores deliberately remain uncertain. The defaults need
evaluation on representative scanned documents before treating them as review
thresholds. Sequence inference cannot distinguish every genuine printing error
from an OCR error, so inferred corrections remain less certain.

### NDK archival pagination

The final, detector-independent `NdkPaginator` follows the pagination notation in
[NDK monograph description rules 2.5](https://standardy.ndk.cz/ndk/standardy-digitalizace/ppp_mono_2.5_final.pdf)
(§§1.1.3–1.1.8) and
[NDK periodical description rules 8.8](https://standardy.ndk.cz/ndk/standardy-digitalizace/ppp_perio_8.8_final.pdf)
(§§1.1.3–1.1.7), both published 21 October 2026 and checked 26 October 2026.
The notation is prescribed by NDK; automatic inference criteria and confidence
thresholds are implementation policy, not requirements of the standard.

It is **not** correct to put every absent page number in brackets. The supported
cases are:

| Context | Result |
|---|---|
| A missing interior number in a continuous sequence | Supply the ordinary number without brackets. |
| Extra unnumbered pages between consecutive observed numbers | Append lowercase letter suffixes to the preceding number, in brackets. |
| A gap containing both extra leaves and missing ordinary numbers | Use NDK's permitted convention of placing suffixed inserts first, then bracketed ordinary numbers. |
| Roman pagination | Preserve the numeral system and case, with lowercase letter suffixes. |
| Known complete document, missing leading ordinary numbers | Supply bracketed numbers, if backward counting stays positive. |
| Known complete document, unnumbered pages before page 1 | Supply bracketed letter labels; use the permitted double-letter prefix if later labels would collide. |
| Known complete document, trailing pages | Require explicit counted/extra-leaf facts or a chosen `tail_mode`; then supply bracketed continuations or suffixed inserts. |
| Confirmed wholly unnumbered monograph | Start at 1 with every supplied number bracketed. |
| Known segments of one foldout | Reuse the parent's number with bracketed letter suffixes. Segment scans do not advance ordinary pagination. |

Default inference handles **bounded interior gaps only**, with at most 8 missing
scans per gap. Each bounding anchor must have confidence at least 0.6 and a
nearby independent attested continuation on its own side (two anchors on each
side in total). Searches stop at the first observed number and reject resets
or jumps exceeding the intervening scan count. Supplied numbers never become anchors. Resets, numeral-system changes,
number jumps indicating absent scans, and insufficient context remain unresolved.
For mixed insert/number gaps, the chosen assignment is only one NDK-permitted
arrangement; it receives confidence at most 0.55 because page content may justify
another arrangement. Known counting facts are respected; contradictions leave
the gap unresolved.

Inferred plain interior labels have confidence at most 0.9; inserted leaves 0.8;
leading/trailing labels 0.7–0.75. Anchor-based scores never exceed their source
anchor confidence. These are review scores, not claims that image OCR found the
number. The original observation action and reading remain in
`PageNumberDecision.observation_action` and `original_reading`, and
`pagination_rule` records the inference rule. `supplied=True` labels carry empty
`text`, `bbox=None`, and empty ALTO references. The binder writes their complete
`output_text()` and confidence without creating geometry or OCR references.

`profile` defaults to `shared`, covering rules common to both document kinds.
`monograph` additionally allows an explicitly confirmed wholly unnumbered volume.
It must be combined with `complete_document=True` and `wholly_unnumbered=True`;
failed OCR alone never establishes that condition. Document edges are processed
only with `complete_document=True`. A trailing run is ambiguous in both profiles
without information distinguishing a continuation from added leaves, so
`tail_mode` defaults to `unknown`. Its explicit alternatives are `continue` and
`insert`. No cover/blank/page-type classifier or shared schema is changed.

Source facts can be supplied through `NdkPageContext`:

```python
contexts = {
    "scan-001": NdkPageContext(unit_key="issue-1", excluded=True),
    "scan-002": NdkPageContext(unit_key="issue-1"),
    "scan-003": NdkPageContext(
        unit_key="issue-1", segment_of="scan-002", segment_index=1,
    ),
}
result = yolo_core.process(images, alto_files, page_contexts=contexts)
```

Reusable callers set `PageNumberPostprocessInput.pagination` to the same context.
`unit_key` separates contiguous issues/volumes; position learning and sequence
correction also respect those boundaries. `excluded=True` excludes covers or
other known scans from numeric progression and label inference. `counted=True`
or `False` distinguishes ordinary pagination from extra leaves. `insert_anchor`
can be `previous` (default) or `next` when source content associates an insert
with the following page. Supplied suffixes are kept distinct within a processing chunk,
using ASCII letters and omitting the Czech digraph and diacritics. Extensions
beyond a single alphabet use a documented spreadsheet-style sequence.

`segment_of` must name a parent scan in the same unit, and `segment_index` must
be a unique positive index for that parent. Both are explicit source facts;
repeated numbers or missed detections are not proof of a foldout. Recto/verso
folio numbering and two-page/column labels require physical-layout evidence;
this implementation does not infer those relationships from missing OCR. Such
source material must be identified before choosing its pagination convention.
No pages are removed, reordered, or silently classified as blank.

### Reusing the step with another detector

The module `engines/core/postprocessing.py` imports neither YOLO nor the ALTO
aligner. Another engine supplies ordered `PageNumberPostprocessInput` objects:

```python
pages = [
    PageNumberPostprocessInput(
        page_key=key,
        dimensions=PageDimensions(width, height),
        candidates=tuple(detected_candidates),
        selected=initial_number,  # PhysicalPageNumberEvidence or None
    )
    # ... every page in physical order, including pages without detections
]
result = PageNumberPostprocessor().process(pages, recover=recovery_callback)
```

`selected` is the detector's initial per-page choice used for position learning;
`candidates` keeps alternatives available. Every candidate must identify its
own page and contain parsed numeral fields and a finite positive-area bounding
box inside that page. Invalid geometry is ignored; foreign and duplicate page
keys are errors. Dimensions and bounding boxes must use the same coordinate
system. The optional recovery callback receives one page input and a tuple of
`BoundingBox` search windows, and returns more `PhysicalPageNumberEvidence`
objects. Omit it to use existing evidence only. Set `postprocess.ndk.enabled`
to false to disable archival label inference while retaining position and sequence processing.

`PageNumberCoreResult.decisions` uses actions `detected`, `recovered`,
`corrected`, `dropped`, `unresolved`, `excluded`, `supplied`, and
`boundary_uncertain`. Disabled postprocessing leaves this
mapping empty. The original detector score remains diagnostic information and
does not contribute to postprocessed confidence. No new shared schema fields
are required.

## Page-number parsers

Parsers interpret OCR text and produce normalized page-number tokens. They do
not select between competing detections. Supporting helpers can combine a
parse result with detection provenance to create physical-number evidence. A
parser is a reusable supporting mechanism rather than an independently loaded
core engine, and the [core contract](#page-number-core-engine-contract) does
not require a particular parser.

The available parser is:

| Parser | Responsibility |
|---|---|
| `DecoratedPageNumberParser` | Extract one normalized Arabic or Roman page-number token from decorated OCR text. |

### Decorated page-number parser

`DecoratedPageNumberParser` extracts one unambiguous Arabic or Roman numeral
from OCR text. Its main input and output are:

```python
parse(page_number: str) -> str | bool
```

`page_number` is the complete OCR text associated with a page-number
detection. On success, `parse()` returns the normalized numeral token as a
string. It returns `False` when the text does not contain exactly one
unambiguous valid number.

| Input form | Result |
|---|---|
| One Arabic digit sequence | Digits are normalized to ASCII; surrounding decoration is ignored. |
| Arabic digit groups separated only by whitespace | Groups are joined into one number. |
| One valid Roman-numeral word | Unicode numeral glyphs are normalized to ASCII while letter case is preserved. |
| No numeral, an invalid Roman numeral, or multiple ambiguous numerals | `False`. |

Leading Arabic zeros and Roman letter case are preserved in the returned
token.

#### Evidence creation helpers

The parser inherits two helpers from `PhysicalPageNumberParser` for wrapping
its parsing result in `PhysicalPageNumberEvidence`. The direct evidence
factory has this input and output:

```python
create(
    *,
    page_key: str,
    text: str,
    confidence: float,
    bbox: BoundingBox,
) -> PhysicalPageNumberEvidence
```

`create()` parses `text` and combines the result with the explicitly supplied
detection provenance. It always returns evidence. A successful parse sets
`normalized`, the integer `value`, and `numeral_system`; leading Arabic zeros
are retained in `normalized` but not in `value`. If parsing fails, all three
fields are `None`. The complete unmodified input remains in `text`.

`parse_region()` adapts an aligned detection to `create()`:

```python
parse_region(
    *,
    page_key: str,
    region: AlignmentRegion,
) -> PhysicalPageNumberEvidence | None
```

`page_key` identifies the page that owns the detection. `region` must be
matched and must contain `alto_text`, `input_geometry`, and
`input_geometry_confidence`. The helper maps them to evidence as follows:

| Evidence attribute | Source |
|---|---|
| `page_key` | The `page_key` argument. |
| `text` | `region.alto_text`, unchanged. |
| `confidence` | `region.input_geometry_confidence`. |
| `bbox` | The bounds of `region.input_geometry`. |
| `alto` | `AltoRefs.from_words(region.words)`: the aligned words' `alto_block_id`, `alto_line_id` and `alto_word_id`. |
| `normalized`, `value`, `numeral_system` | The parsed Arabic or Roman number. |

`parse_region()` returns `None` when the region is unmatched, required region
data is missing, or its OCR text cannot be parsed. Every non-`None` result is
therefore complete, successfully parsed evidence suitable for the resolver.

## Page-number resolvers

Resolvers choose between competing physical-number detections. They do not
define the page-number core contract and are not independently loaded core
engines. A core engine may use the available resolver, implement another
resolver, or perform selection internally.

The available resolver is:

| Resolver | Responsibility |
|---|---|
| `PhysicalPageNumberResolver` | Apply physical-position eligibility and select at most one item from parsed physical-number evidence. |

### Physical page-number resolver

`PhysicalPageNumberResolver` accepts a sequence of already parsed
`PhysicalPageNumberEvidence`, together with the page width and height.
Its complete input and output are:

```python
resolve(
    candidates: Iterable[PhysicalPageNumberEvidence],
    *,
    page_width: float | None,
    page_height: float | None,
    mode: PageNumberSelectionMode = PageNumberSelectionMode.STANDARD,
) -> PhysicalPageNumberEvidence | None
```

The resolver does not inspect aligned regions and does not invoke a parser.
The caller owns candidate parsing and passes only the evidence to be ranked.
The return value is the winning evidence, or `None` when there are no
candidates or none satisfies the selected mode's eligibility rules. The
resolver does not return the input candidates. When at least one candidate is
provided, `page_width` and `page_height` must both be finite positive values;
otherwise resolution raises `ValueError`.

All candidates in one call must have the same `page_key`. The resolver derives
the page key from the candidates for logging and validation errors. It raises
`ValueError` when candidates from different pages are supplied. An empty
candidate sequence returns `None` and requires no page key.

`PhysicalPageNumberResolver` supports two modes:

| Mode | Behavior |
|---|---|
| `STANDARD` | A single valid candidate is retained anywhere on the page. With multiple candidates, edge-based selection is applied. This is the mode used for initial selection by the page-number core engine. |
| `EDGE_ONLY` | Edge-based selection is applied even to a single candidate. |

The resolver accepts these settings directly or through `from_config()`:

| Parameter | Default | Meaning |
|---|---:|---|
| `page_number_edge_band_ratio` | `0.15` | Maximum normalized vertical distance from a candidate bounding-box center to the nearest horizontal page edge. This defines equal-height bands next to the top and bottom edges. |
| `page_number_edge_score_weight` | `0.65` | Weight assigned to proximity between the candidate bounding-box center and the nearest horizontal page edge; the remaining weight is assigned to detection confidence. |

`page_number_edge_band_ratio` must be greater than `0` and smaller than `0.5`.
`page_number_edge_score_weight` must be in `[0, 1]`.

When at least one candidate is supplied, resolution requires finite positive
page width and height. Missing or invalid page dimensions are input errors.

Before applying either selection mode, the resolver requires the candidate's
entire bounding box to have finite coordinates, positive width and height, and
to be contained within the page:

```text
0 <= bbox.x
0 <= bbox.y
bbox.x + bbox.width <= page_width
bbox.y + bbox.height <= page_height
```

A candidate that crosses any page boundary is not eligible for selection.

All resolver distances are vertical distances measured from the center of the
candidate bounding box. They are not measured from a bounding-box edge, and
they are not distances between candidates. For a candidate with bounding box
`bbox` and a page with height `page_height`, the resolver calculates:

```text
center_y = bbox.y + bbox.height / 2
center_y_ratio = center_y / page_height

distance_to_top_edge = center_y_ratio
distance_to_bottom_edge = 1 - center_y_ratio
edge_distance = min(distance_to_top_edge, distance_to_bottom_edge)
```

A candidate is eligible when:

```text
0 <= edge_distance <= page_number_edge_band_ratio
```

With the default ratio `0.15`, this admits bounding-box centers in the top 15%
or bottom 15% of the page. If neither edge band contains a candidate, the
physical page number remains unresolved.

The following table is the complete tie-breaking order. A lower rank is
considered only when every higher-ranked value ties:

| Rank | Physical page-number candidate |
|---:|---|
| 1 | Greatest weighted score: `page_number_edge_score_weight * edge_proximity_score + (1 - page_number_edge_score_weight) * confidence` |
| 2 | Highest confidence |
| 3 | Smallest normalized vertical distance from its bounding-box center to the nearest horizontal page edge: top or bottom |
| 4 | First candidate in resolver input order |

For an eligible candidate, the resolver converts `edge_distance` into
`edge_proximity_score`:

```text
edge_proximity_score =
    1 - edge_distance / page_number_edge_band_ratio
```

The score decreases linearly while moving inward from the nearest page edge.
A bounding-box center exactly on the top or bottom edge has
`edge_proximity_score = 1`. A center on the inner boundary of either edge band
has `edge_proximity_score = 0`. The resolver prefers the greatest weighted
score, so—when confidence is unchanged—a center closer to the top or bottom
page edge is preferred.

## Available bind implementation

The registered bind implementation is:

| Config `name` | Implementation |
|---|---|
| `page_number_bind_engine_base` | Invoke any compatible page-number core engine and bind its result into `MetakatIO`. |

### Engine: Base (`page_number_bind_engine_base`)

The Base bind engine deep-copies the supplied `MetakatIO`, invokes its
configured core engine, and returns the modified copy. `ProarcIO` is accepted
by the bind contract but is not used by this implementation.

#### Page selection and core invocation

The binder sorts all `MetakatPage` elements by `batch_index`. It skips the core
when every page number is annotated, the batch is empty, or no processable page
needs a number. In mixed batches, annotated pages stay in the core input to retain
sequence context; their values are preserved when results are bound. Only pages with
both `page_to_image_mapping` and `page_to_alto_mapping` entries are passed to
the core; skipped pages are logged. Image and ALTO paths are resolved relative
to `batch_dir`.

The image filename stem is used as the page key. Image mappings must therefore
have unique stems. A duplicate stem is an error because it would make the core
result ambiguous. A result containing a page key that cannot be mapped back to
a MetaKat page is also an error.

#### Field mapping and precedence

Explicit unresolved (`None`) results are skipped without creating detections or
erasing existing values. For every returned evidence object, the binder compares its confidence with
the page's existing `pageNumber` value:

| Condition | Binding decision |
|---|---|
| Existing value has `annotated: true` | Retain it regardless of model confidence. |
| No existing `MetakatPage.pageNumber` | Bind the core result. |
| Core confidence is greater than the existing confidence | Replace the existing value with the core result. |
| Core confidence is equal to or lower than the existing confidence | Retain the existing value. |

`annotated` is optional. Only an explicitly present boolean `true` protects a
value; absent, null and false use the confidence rules above. It marks externally
authoritative input, including DB-supplied numbers, without implying human review.
Annotated values retain their identity and evidence mappings. Pagination used by
page-type models still filters bracketed/inferred numbers independently.

When a core result is bound, the binder writes:

| MetaKat destination | Source |
|---|---|
| `MetakatPage.pageNumber` | `Value(text=evidence.output_text(), confidence=evidence.confidence, id=detection_uuid)` |
| `MetakatIO.detection_to_bbox[detection_uuid]` | `(bbox.x, bbox.y, bbox.width, bbox.height)` |
| `MetakatIO.detection_to_page_mapping[detection_uuid]` | Source `MetakatPage.id` |
| `MetakatIO.detection_to_alto[detection_uuid]` | `MetakatAltoRefs(blocks, lines, words)` from `evidence.alto`; written only when the ALTO provides at least one ID |

The detection UUID is created only for evidence that is actually written.
Candidate and selection metadata are not added to the MetaKat schema.

## Observability and revision

The available implementation logs engine loading, YOLO batch progress,
detection counts, skipped or invalid aligned regions, unresolved edge-band
selection, selected candidates, skipped MetaKat pages, and final core and bind
counts. Detailed YOLO detections and existing-number precedence decisions are
available at `DEBUG`.

Initial candidate-selection behavior is centralized in
`PhysicalPageNumberResolver`, and its complete priority order is documented in
[Physical page-number resolver](#physical-page-number-resolver).
Changes to the edge band, combined score, eligibility, or ties should update
that resolver, its focused tests, and the ranking table together.

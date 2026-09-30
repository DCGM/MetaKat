# MODS export

`export_mods(metakat_io, output_dir)` writes one MODS 3.8 record per title,
volume, issue, supplement, chapter, article and page of a `MetakatIO`, each to
`<uuid>.xml`. It runs inside the pipeline (`process_batch(output_mods_dir=…)`,
which the worker writes to `mods/` next to `metakat.json`) or on its own:

```
python -m metakat.io_exporters.mods_exporter --metakat-json metakat.json \
    --output-dir mods [--overview metakat.mods.txt] [--no-provenance]
```

`--overview` also writes every record into one text file in the order of the
MetaKat JSON's `elements`, each under a header naming its position, type, title
or page number, and uuid, for reading side by side with the JSON. It is for
inspection only: the worker writes it, as `metakat.mods.txt`, beside
`result.zip` like the interactive PDF - never into the uploaded result - and
only when `STORE_MODS_OVERVIEW` is enabled.

## What a record holds

A record describes one unit. How units nest is not written: that stays in the
MetaKat JSON, and a record names its unit only by
`<identifier type="uuid">`. Every record is valid MODS 3.8; the tests check each
one against the Library of Congress schema.

| MetaKat | MODS |
|---|---|
| element type | `<genre>`; an article's `articleGenre` as `genre@type` |
| `title`, `subTitle`, `partNumber`, `partName` | `<titleInfo>` |
| agents | `<name><namePart>` whole, with `role/roleTerm` marcrelator code: `aut`, `ill`, `pht`, `trl`, `edt` - a redaktor is `edt`, as in Czech records |
| `affiliation` | `<name><affiliation>` |
| `publisher`, `placeTerm`, `dateIssued`, `edition`, `frequency` | `<originInfo eventType="publication">`, publisher as `agent/namePart` with role `publisher` |
| `copyrightDate` | `<originInfo eventType="copyright">` |
| manufacture fields | `<originInfo eventType="manufacture">`, printer as `agent` with role `manufacturer`, date as `dateOther type="manufacture"` |
| series fields | `<relatedItem type="series"><titleInfo>` |
| `statementOfResponsibility` | `<note type="statement of responsibility">` |
| `language`, `form` | `<language><languageTerm authority="iso639-2b">`, `<physicalDescription><form authority="marcform">` |
| internal part `abstract`, `keywords` | `<abstract>`, `<subject><topic>` |
| reviewed work | `<relatedItem type="reviewOf">` |
| page runs | `<part type="pageIndex">` and `<part type="pageNumber">` with `extent/start`, `end` |
| internal part `dateIssued` | provenance only: an internal part has no `<originInfo>` |
| page | `part@type`/`genre@type` = page type; `detail type="pageNumber"`; `detail type="pageIndex"`; side as `<note>`; `genre` is `reprePage` when `MetakatPage.representative` is set, otherwise `page` |

`email` and `hierarchy` have no MODS element and are not written yet.

### Order

A record's elements come in the order its model declares their fields, so a
MetaKat record and its MODS read in the same order, and the schema is the
only place the order is defined. `section_order()` walks the model's fields
and writes each MODS section where its first field sits; values inside a
container follow field order too.

The identity fields - `type`, `id`, `parent_id`, `hierarchy` - lead every
model but not a DMF record, so their elements take the DMF's places: `<genre>`
after `<name>`, `<identifier>` before the `<part>` elements (last if there are
none), then `<recordInfo>` and the provenance extension. Fields without a MODS
element write nothing and take no place.

The tests hold the schema order to the NDK DMF tables - titleInfo, name,
genre, originInfo (publication, manufacture, copyright), language,
physicalDescription, abstract, note, subject, relatedItem, identifier, part
(pageNumber, pageIndex), recordInfo - including the order of the leaves inside
`titleInfo` and `originInfo`. They also require every field of both bases to
name its MODS section or declare it has none, so a new field cannot be added
without deciding where it goes.

### Containers come only from groups

The code that creates values groups what it knows belongs together. A group
becomes one container; every ungrouped value gets its own. The exporter never
pairs or merges anything itself: a page run is written with an end only when a
`pageRange` group pairs them, so an ungrouped start stands alone and an
ungrouped end is not written. A container carries `lang` when its values share
one language.

### Two readings of one field

A chapter's title, subtitle and part number can be read on its own page and in
its TOC entry. The own-page reading is written; the TOC reading becomes a
second event in the same provenance assertion. A TOC reading with no
own-page counterpart is written itself.

## Provenance

`<extension type="metakatProvenance">` holds one `<mkp:provenance
version="1.0">`, in namespace
`https://github.com/DCGM/MetaKat/ns/provenance/1.0`, with one `assertion` per
written value:

- `target` - `#ID` of the MODS container; absent for a value that has no place
  in the record (an internal part's date). `property` - the MODS element
  holding the value; `index` - its position among such elements in the
  container. `selectedEvent` - the event whose value was written.
- `event` - `id` is the MetaKat `Value.id` (for a classification, the unit id,
  field and label); `action` is `extract` for read text and `classify` for a
  label chosen by a classifier; `field` is the MetaKat field. Its children
  follow the PREMIS event: when, the outcome - the value and how sure - who,
  and on what:
  - `eventDateTime` - when the event happened: the value read, imported or
    annotated. It comes first, as PREMIS `eventDateTime` does, and holds an
    `xs:dateTime` in UTC to the second, `2026-09-30T10:12:00Z` - the form of
    `recordCreationDate`, but fixed by this format, so it takes no `encoding`.
    It is defined but not written yet: `MetakatIO` holds no time to fill it
    from. `recordCreationDate` is when the MODS was written, not when a value
    was read;
  - `observedValue`, with `lang` when known;
  - `confidence` - a number from 0 to 1, how confident the producer is in
    the value. How it is computed is the producer's business; the format
    fixes only the range, which the MetaKat schema enforces on every
    confidence it holds;
  - `source` - who produced the value; one per agent, since an event can
    have several, as PREMIS links several agents to one event. `type` is
    from the PREMIS agentType vocabulary (`software`, `person`,
    `organization`, `hardware`), `role` says what part the agent played, and
    every `software` source is identified the same way, by `name` and
    `version`. The exporter writes two:
    - `role="application"` - MetaKat itself, from `MetakatIO.application`.
      Its version is a fixed `1.0.0` until the worker exposes the running
      version;
    - `role="engine"` - the engine MetaKat ran, from `MetakatIO.engine`,
      when set;
  - `evidence` - `page` is the page it was read on as a `urn:uuid:` URN
    (RFC 4122), and `xywh` the region as a W3C Media Fragments spatial
    dimension, `pixel:x,y,w,h`: whole pixels of the page image from its
    top-left corner, the space `MetakatIO.bbox_coordinates` declares. Media
    Fragments admits only whole pixels, so a box on half pixels is widened
    to the smallest one enclosing it. Inside it, one `altoRef` per ALTO
    element the value was read from, from `MetakatIO.detection_to_alto`:
    `element` is the ALTO element name (`TextBlock`, `TextLine`, `String`)
    and `id` its `ID` attribute, copied verbatim - producers name IDs as
    they like, so the kind of element is never read from the ID. Blocks come
    first, then lines, then words, each in reading order; a level the ALTO
    gave no IDs is absent, and a value without any has no `altoRef`. The IDs
    are unique only within the page's ALTO file, which is why they sit
    inside the evidence naming the page. `altoRef` is MetaKat's own element,
    but it maps directly onto METS: `id` is what `mets:area/@BEGIN` holds
    with `BETYPE="IDREF"`, and the page's ALTO file is its `FILEID`.

`--no-provenance` writes plain MODS without the extension.

### Versioning

The format is versioned on two levels, so a reader always knows what it is
reading:

- **Namespace - the major version.** A breaking change - an element or
  attribute renamed or removed, or its meaning changed - gets a new namespace
  (`…/provenance/2.0`). A reader that knows only 1.0 then does not mistake
  the new content for something it understands, and one reader can support
  both by recognising both namespaces.
- **`version` on `<mkp:provenance>` - the minor version.** An additive change,
  such as a new optional element, attribute or `action` value, keeps the
  namespace and raises `version` (1.0 → 1.1). Readers that do not know the
  addition skip it.

The `<mkp:provenance>` wrapper exists for this attribute: `<mods:extension>`
admits no `version` of its own, and `type="metakatProvenance"` is a label, not
a version. Both are set in `mods_exporter.py` (`PROVENANCE_NS` and the
`version` written with the wrapper); a change to the format updates them, the
tests and this README together.

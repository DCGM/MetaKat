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
or page number, and uuid, for reading side by side with the JSON.

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

`<extension type="metakatProvenance">` holds, in namespace
`https://github.com/DCGM/MetaKat/ns/provenance/1.0`, one `assertion` per
written value:

- `target` - `#ID` of the MODS container; absent for a value that has no place
  in the record (an internal part's date). `property` - the MODS element
  holding the value; `index` - its position among such elements in the
  container. `selectedEvent` - the event whose value was written.
- `event` - `id` is the MetaKat `Value.id` (for a classification, the unit id,
  field and label); `action` is `extract` for read text and `classify` for a
  label chosen by a classifier; `field` is the MetaKat field.
  - `observedValue`, with `lang` when known;
  - `source` - `type="software"`, the `engine` and `version` from
    `MetakatIO.engine` when set, and the pipeline `stage`;
  - `confidence scheme="model-score"` - detector and classifier scores are
    not calibrated probabilities;
  - `evidence pageRef` - the page it was read on, with `roi` in the
    coordinate system declared by `MetakatIO.bbox_coordinates`.

`--no-provenance` writes plain MODS without the extension.

# Bibliographic processing

## Purpose

The `biblio` pipeline component extracts bibliographic evidence from selected
document pages and binds the resulting title, volume, issue, and hierarchy
information into `MetakatIO`.

## Available core implementation

### Engine: YOLO + ALTO (`biblio_core_engine_yolo`)

This implementation detects bibliographic regions with YOLO and aligns their
geometry with ALTO text. The `labels` mapping connects semantic `BiblioType`
values to the raw labels exposed by the model.

```yaml
name: biblio_core_engine_yolo
model_path: path/to/model.pt
labels:
  <BiblioType>: <model-label>
```

`model_path` and a non-empty `labels` mapping are required. Semantic types and
model labels must be valid and unique. Numeric model class IDs do not form
part of the pipeline configuration.

## Available bind implementation

### Engine: Base (`biblio_bind_engine_base`)

The base binder selects title pages, invokes the configured core engine,
constructs MetaKat bibliographic elements from the aligned detections, and
binds pages, issues, and volumes into the resulting document hierarchy.

```yaml
name: biblio_bind_engine_base
```

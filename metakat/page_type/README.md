# Page-type processing

## Purpose

The `page_type` pipeline component classifies document pages. Its core engine
returns a probability vector for each page image, and its bind engine selects
the winning semantic page type and writes it to the corresponding
`MetakatPage`.

## Available core implementation

### Engine: ViT (`page_type_core_engine_vit`)

This implementation loads a ViT image-classification checkpoint and processes
the ordered page images. The checkpoint supplies the numeric class-to-model
label mapping. The optional `labels` configuration maps `PageType` values to
those model labels; unspecified values use the `PageType` value itself.

```yaml
name: page_type_core_engine_vit
model_dir: path/to/model
labels:
  <PageType>: <model-label>
```

`model_dir` is required. Model labels and class IDs must be unique, class IDs
must be contiguous from zero, and every checkpoint label must resolve to one
configured semantic `PageType`.

## Available bind implementation

### Engine: Base (`page_type_bind_engine_base`)

The base binder processes every page having an image mapping, selects the
highest-probability class returned by the core engine, and stores its semantic
`PageType` and probability in `MetakatPage.pageType`.

```yaml
name: page_type_bind_engine_base
```

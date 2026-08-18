# MetaKat

MetaKat enriches an ordered batch of document pages with structured metadata.
It initializes a shared `MetakatIO` object from the batch and any supplied
metadata, then passes that object through each enabled pipeline component in
pipeline order. Every pipeline component has a core engine, which produces
component-specific results, and a bind engine, which integrates those results
into `MetakatIO`. The bind engine envelopes the core engine: it may preprocess
the current `MetakatIO` and other inputs, invokes the core with the prepared
source data, and postprocesses the core result into an updated `MetakatIO`.
That updated object becomes the input to the next component.

The engine pipeline is configured independently of the document input and
output. Configuration files and command-line overrides are combined first;
all configured paths are then resolved centrally before any engine is loaded.

```mermaid
flowchart TD
    Base[Pipeline configuration] --> Merge[Merge configuration]
    Override[Optional override] --> Merge
    Set[Optional assignments] --> Merge
    Merge --> Resolve[Resolve configured paths]
    Resolve --> Prepared[Prepared engine configuration]

    Batch[Ordered page batch] --> Initialize[Initialize shared IO]
    Metadata[Optional MetaKat and source metadata] --> Initialize
    Initialize --> Current[Current MetakatIO]

    Prepared --> Pair
    Current --> Pair

    subgraph Component[For each enabled pipeline component]
        Pair[Load core and bind engine pair] --> Preprocess

        subgraph Bind[Bind engine]
            Preprocess[Preprocess MetakatIO and source inputs]
            Core[Core engine]
            Postprocess[Postprocess core result into MetakatIO]

            Preprocess -->|Prepared source data| Core
            Core -->|Component result| Postprocess
        end

        Postprocess --> Updated[Updated MetakatIO]
    end

    Updated --> More{Another enabled component?}
    More -->|Yes| Pair
    More -->|No| Validate[Validate final MetakatIO]
    Validate --> Result[Return or write configured outputs]
```

## Installation

MetaKat requires Python 3.12 or newer. Runtime versions are pinned to a
known-good set rather than left to the resolver; upgrading them is a deliberate,
wholesale change rather than something that happens on a fresh install.

Two dependencies are git submodules rather than published packages, so they are
installed from the checkout. Install all three in one command, so that the
declarations naming `text-geometry-aligner` and `doc-api` are satisfied locally
instead of being looked for on an index:

```bash
git submodule update --init --recursive

pip install -e libs/text-geometry-aligner \
            -e libs/DocAPI \
            -e ".[all]"
```

Editable installs place a pointer to the checkout rather than a copy, so edits
take effect without reinstalling, and no `PYTHONPATH` is needed.

### Installation modes

The base install carries the schemas, the pipeline configuration handling and
the IO layer, but no engine. Engine implementations are imported only when a
pipeline selects them, so an environment installs only the tiers it uses. A
component whose dependencies are absent is reported before any page is read,
naming the missing module and the extra that provides it.

| Extra | Adds | Use for |
| --- | --- | --- |
| *(none)* | schemas, configuration, IO | consumers that only read or write `MetakatIO` |
| `torch` | torch, torchvision, transformers | the `page_type` ViT engine |
| `yolo` | ultralytics, and `torch` with it | `page_number`, `biblio`, and the `chapter` page-analysis and extraction stages |
| `pdf` | PyMuPDF | interactive PDF output |
| `worker` | `yolo`, `pdf`, and the DocAPI client layer | running `metakat/worker/docapi` |
| `train` | accelerate, scikit-learn, safe-gpu, and `torch` | training and evaluation in `metakat/page_type/nets` |
| `dev` | pytest, pytest-cov | running the test suite |
| `all` | `worker`, `train`, `dev` | a full development machine |

`ultralytics` depends on torch, so `yolo` is never lighter than `torch`. The
`worker` extra is the complete set needed to process a job: every engine a job
may select, plus the PDF exporter the worker writes when `STORE_METAKAT_PDF` is
set. `doc-api` is requested through its own `worker` extra, which keeps the API
service's database and ASGI stack out of the installation.

```bash
pip install -e ".[worker]"    # a worker deployment
pip install -e ".[train]"     # a training machine, no DocAPI, no PDF export
pip install -e ".[yolo,pdf]"  # the pipeline through process_batch, no worker
pip install -e ".[dev]"       # the tests that need no model runtime
```

## Tests

Tests live beside the code they exercise, in a `tests` directory within each
package, so a path selects a subset and no separate mapping has to be
remembered. `testpaths` is configured, so an invocation without arguments runs
everything:

```bash
pytest                                                  # the whole suite
pytest metakat/chapter                                  # one component
pytest metakat/chapter/engines/core/chapter_alignment   # one pipeline stage
pytest metakat/page_number/engines/core                 # parsers, resolver, loaders
pytest metakat/worker/docapi/tests                      # the DocAPI worker
```

Single files, single tests and keyword selection work as usual, the last being
the simplest way to reach the cases of a parametrized test:

```bash
pytest metakat/chapter/engines/core/chapter_alignment/tests/test_engine_fuzzy.py
pytest metakat/tests/test_engine_config.py::test_deep_merge_replaces_only_leaves_and_lists
pytest -k "anchor"
```

Which tests can run depends on the installed tier, and the directory layout is
what separates them. Most of the suite needs only the base install and `dev`;
the tests that exercise a model runtime, the PDF exporter or the DocAPI worker
fail to collect without the corresponding extra. `metakat[all]` runs all of
them.

The tests directories carry no `__init__.py`, which keeps them out of an
installed distribution. Because several test modules share a file name across
different packages, pytest is configured with `--import-mode=importlib`, which
imports each file under a name derived from its path.

## Pipeline components

Components run in the order shown below. The hierarchy mirrors their pipeline
configuration: each top-level component selects one `core` and one `bind`
implementation. An omitted component is skipped. Links lead to the detailed
documentation for each available implementation.

1. [`page_number`](page_number/README.md#purpose)
   - [`core`](page_number/README.md#page-number-core-engine-contract)
     - [`page_number_core_engine_yolo`](page_number/README.md#engine-yolo--alto-page_number_core_engine_yolo)
   - [`bind`](page_number/README.md#available-bind-implementation)
     - [`page_number_bind_engine_base`](page_number/README.md#engine-base-page_number_bind_engine_base)
2. [`page_type`](page_type/README.md#purpose)
   - [`core`](page_type/README.md#available-core-implementation)
     - [`page_type_core_engine_vit`](page_type/README.md#engine-vit-page_type_core_engine_vit)
   - [`bind`](page_type/README.md#available-bind-implementation)
     - [`page_type_bind_engine_base`](page_type/README.md#engine-base-page_type_bind_engine_base)
3. [`biblio`](biblio/README.md#purpose)
   - [`core`](biblio/README.md#available-core-implementation)
     - [`biblio_core_engine_yolo`](biblio/README.md#engine-yolo--alto-biblio_core_engine_yolo)
   - [`bind`](biblio/README.md#available-bind-implementation)
     - [`biblio_bind_engine_base`](biblio/README.md#engine-base-biblio_bind_engine_base)
4. [`chapter`](chapter/README.md#purpose)
   - [`core`](chapter/README.md#chapter-core-engine-contract)
     - [`chapter_core_engine_pipeline`](chapter/README.md#replaceable-three-stage-pipeline-processing)
       - [`page_analysis`](chapter/README.md#stage-1-contract-chapter-page-analysis)
         - [`chapter_page_analysis_engine_yolo_alto`](chapter/README.md#engine-yolo--alto-chapter_page_analysis_engine_yolo_alto)
       - [`extraction`](chapter/README.md#stage-2-contract-chapter-extraction)
         - [`chapter_extraction_engine_yolo_alto`](chapter/README.md#engine-yolo--alto-chapter_extraction_engine_yolo_alto)
       - [`alignment`](chapter/README.md#stage-3-contract-chapter-alignment)
         - [`chapter_alignment_engine_fuzzy`](chapter/README.md#engine-fuzzy-alignment-chapter_alignment_engine_fuzzy)
   - [`bind`](chapter/README.md#available-bind-implementation)
     - [`chapter_bind_engine_base`](chapter/README.md#engine-base-chapter_bind_engine_base)

## Pipeline invocation

`process_batch.py` is configured by one complete YAML or JSON engine pipeline.
Input/output paths remain separate command-line arguments.

```bash
python -m metakat.process_batch \
  --batch-dir /data/batch \
  --engine-config /engines/pipeline.yaml \
  --engine-config-override /jobs/override.json \
  --set component:core:option 0.5 \
  --output-metakat-json /data/result/metakat.json
```

`--engine-config` is required. `--engine-config-override` is optional and
`--set PATH VALUE` may be repeated. A `--set` path uses colons between mapping
keys. Its value is interpreted as JSON when possible (`0.5`, `10`, `true`,
`null`, lists, and objects); otherwise it remains a string.

Configuration precedence, from lowest to highest, is:

1. the main engine configuration;
2. the override file;
3. repeated `--set` assignments in command-line order.

Mappings are merged recursively. A leaf replaces the corresponding leaf,
lists are replaced as complete values, and new keys are allowed. Replacing a
mapping with a non-mapping or the reverse is an error.

## Pipeline configuration

The pipeline configuration is a mapping of pipeline components. Each enabled
component contains both a `core` and `bind` mapping, and each of those mappings
selects an engine through `name`. Omitting a component disables it; providing
only one half is an error.

Engine-specific options belong beside `name` in the relevant mapping. Their
meaning and accepted values are defined by that engine and are documented in
the corresponding package, not in this pipeline-level overview.

```yaml
<component>:
  core:
    name: <core-engine-name>
    <core-option>: <value>
  bind:
    name: <bind-engine-name>
    <bind-option>: <value>
```

After all overrides are applied, fields ending in `_path`, `_dir`, `_paths`,
or `_dirs` are resolved centrally. Relative paths use the directory containing
the main pipeline configuration; absolute paths remain absolute. Engines
receive only their nested, prepared mapping and never resolve paths or load a
separate configuration file themselves.

The Python `process_batch()` boundary similarly accepts only the final
prepared `engine_config`, plus decoded `metakat_data` and `proarc_data`
objects. YAML/JSON loading, merging, assignments, and path resolution belong
outside that function.

Before IO initialization and engine loading, `process_batch()` logs the
complete final pipeline configuration at `INFO`. The logged value is a
separate recursively sanitized view; the configuration passed to engines is
not modified. Values under potential secret keys are replaced with
`<redacted>`. Detection is case-insensitive, recognizes snake case, kebab
case, and camel case, and covers passwords, passphrases, secrets, credentials,
authorization, cookies, connection strings, DSNs, tokens, and API, access,
private, signing, encryption, or SSH keys.

## Interactive PDF metadata

When `output_metakat_pdf` or `--output-metakat-pdf` is provided, the generated
PDF uses compact chapter outline labels in this order:

```text
part number | TOC title | page number
```

Missing values are omitted. The destination title replaces the TOC title only
when the TOC title is unavailable. Document-level outline entries use
`monograph | title`, or `monograph` when the title is unavailable.

The clickable rectangle over a detected TOC entry carries the complete chapter
description and opens the resolved destination page. When destination-title
geometry is available on that page, the title rectangle carries the same
description, links back to the TOC entry, and has a visible sticky note.

The PDF also adds visible sticky notes for:

- a detected physical page number, beside its detection geometry;
- every bibliographic detection, beside its detection geometry;
- the complete bibliographic information for each issue or volume, in the
  upper-left corner of the first page classified as `TitlePage`;
- each page type, in the upper-right corner of the page.

Geometry-specific annotations are omitted when their detection-to-page or
bounding-box mapping is unavailable. Sticky-note contents are standard PDF
text annotations; their popup presentation depends on the PDF viewer.

## DocAPI worker

`metakat/worker/docapi/metakat_worker.py` runs MetaKat as a long-lived worker
against a DocAPI server. It polls for a job, downloads the engine files the job
names, runs the pipeline over the job's pages, and uploads the result. It then
polls again; it does not exit after a job.

Running it needs the `worker` installation tier and a worker key issued by the
DocAPI server:

```bash
pip install -e ".[worker]"
```

That tier is the complete set: every engine a job may select, the interactive
PDF exporter, and the DocAPI client layer. A base installation is not enough —
a worker would accept a job and then fail to process it. Should a job select an
engine whose dependencies are missing anyway, the pipeline reports it before
reading any page, naming the missing module and the extra that supplies it.

A job must carry ALTO files alongside its images; the worker fails the job
otherwise. Images whose extension is outside `ALLOWED_IMAGE_EXTENSIONS` are
rejected in the same way.

### Configuration

Every setting is read from an environment variable, and most can also be given
on the command line, in which case the command line wins.

| Variable | Flag | Default | Meaning |
| --- | --- | --- | --- |
| `API_URL` | `--api-url` | `https://metakat.smart.lib.cas.cz` | DocAPI server to poll |
| `WORKER_KEY` | `--api-key` | a placeholder that will not authenticate | key issued by the server |
| `BASE_DIR` | `--base-dir` | `./metakat_worker_data` | parent of the directories below |
| `JOBS_DIR` | `--jobs-dir` | `$BASE_DIR/jobs` | per-job working data |
| `ENGINES_DIR` | `--engines-dir` | `$BASE_DIR/engines` | downloaded engine files |
| `LOGGING_DIR` | — | `$BASE_DIR/logs` | `worker.log`, rotated at UTC midnight |
| `POLLING_INTERVAL` | `--polling-interval` | `5` | seconds between job requests |
| `STORE_METAKAT_PDF` | `--store-metakat-pdf` | `false` | also write the interactive PDF |
| `CLEANUP_JOB_DIR` | `--cleanup-job-dir` | `false` | delete the job directory once uploaded |
| `CLEANUP_OLD_ENGINES` | `--cleanup-old-engines` | `false` | delete superseded engine versions |
| `ALLOWED_IMAGE_EXTENSIONS` | — | `.jpg,.jpeg,.png,.tif,.tiff` | comma-separated |
| `LOGGING_CONSOLE_LEVEL` | `--log-level` | `INFO` | console verbosity |
| `LOGGING_FILE_LEVEL` | — | `INFO` | file verbosity |

`WORKER_KEY` has a placeholder default that will not authenticate, so it must be
supplied. Either `BASE_DIR`, or both `JOBS_DIR` and `ENGINES_DIR`, must resolve;
the worker exits with a usage error otherwise. The three directories are created
at startup if missing.

Each of the three boolean settings has both forms, so the environment sets the
baseline and the command line overrides it in either direction:
`--store-metakat-pdf` and `--no-store-metakat-pdf`, `--cleanup-job-dir` and
`--no-cleanup-job-dir`, `--cleanup-old-engines` and `--no-cleanup-old-engines`.
An option that is not given leaves the environment value alone, which for the
non-boolean settings means a legitimate `0` is honoured rather than mistaken for
an absent argument.

### Running it

`run_worker.sh` beside the worker is the reference invocation. It reads the
worker key from `.docapi_worker_key` in the same directory, exports the rest of
the settings, activates the environment and starts the worker:

```bash
cd metakat/worker/docapi
printf '%s' 'metakat.<the key issued to you>' > .docapi_worker_key
chmod 600 .docapi_worker_key
./run_worker.sh
```

The key file is gitignored and never appears in the script, so rotating the key
does not touch a tracked file. A missing or unreadable key file stops the script
before the worker starts. The script sets no `PYTHONPATH`: MetaKat and its two
submodule dependencies are expected to be installed into the environment.

The equivalent without the script:

```bash
WORKER_KEY=... BASE_DIR=/data/metakat_worker STORE_METAKAT_PDF=true \
  python -m metakat.worker.docapi.metakat_worker
```

### Results

For each job the worker writes `metakat.json` into the job's result directory,
which is uploaded as `result.zip`. With `STORE_METAKAT_PDF` enabled it also
writes `result.pdf` beside that archive, built as described in
[Interactive PDF metadata](#interactive-pdf-metadata).

Enabling `CLEANUP_JOB_DIR` together with `STORE_METAKAT_PDF` is contradictory:
the PDF is written into the directory that is then deleted after upload. The
worker logs a warning at startup rather than refusing to run.

### Worker metadata envelope

The DocAPI worker treats `job.engine_definition` as the base pipeline mapping.
When `meta_file` is present, it must be a JSON object containing only these
optional object-or-null values:

```json
{
  "metakat_json": null,
  "proarc_json": null,
  "engine_config_override": null
}
```

The worker merges `engine_config_override` into `job.engine_definition`, then
resolves paths against the downloaded engine directory. Any relative path,
absolute path, or symlink that resolves outside that directory raises
`ValueError`, causing the job to finish in the error state. The worker does
not support command-line-style `--set` assignments.

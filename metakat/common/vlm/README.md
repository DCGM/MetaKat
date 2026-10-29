# Vision-language model client

`metakat.common.vlm` asks a vision-language model for JSON. It is the shared
part of every engine that reads pages with a model, so such an engine only
brings its prompts, its JSON schema and the mapping of the reply to its
result. The first are the biblio cores
[`biblio_core_engine_vlm_monograph` and `biblio_core_engine_vlm_periodical`](../../biblio/README.md#engine-vision-language-model-biblio_core_engine_vlm_monograph-biblio_core_engine_vlm_periodical).

It needs the `[vlm]` extra (`pip install -e ".[vlm]"`), which the worker
extra includes.

## Endpoints

The client uses the `openai` SDK against any OpenAI-compatible endpoint:

| Provider | `api_url` |
|---|---|
| OpenAI | `https://api.openai.com/v1` |
| OpenRouter - also Anthropic, Google, Qwen, Mistral, Meta and others | `https://openrouter.ai/api/v1` |
| vLLM or another local server | e.g. `http://localhost:8000/v1` |

## Configuration

`VLMConfig.from_config()` reads the engine's `vlm` mapping; an unknown key is
an error.

| Key | Default | Meaning |
|---|---|---|
| `api_url` | required | The endpoint. |
| `model` | required | The model name as the endpoint knows it, e.g. `qwen/qwen3-vl-8b-instruct`. |
| `api_key` | - | The key itself. |
| `api_key_env` | - | The name of an environment variable holding the key, so it can stay with the worker instead of travelling in the job. One of the two is required. |
| `api` | `completions` | `completions` (Chat Completions, which every compatible endpoint serves) or `responses` (OpenAI's Responses API). |
| `response_format` | `schema` | How the model is asked for JSON, see below. |
| `max_attempts` | `3` | How many times an invalid reply is asked for again. |
| `timeout` | `300` | Seconds per request. |
| `max_tokens`, `temperature`, `reasoning_effort`, `verbosity`, `service_tier` | - | Passed to the endpoint when set. `reasoning_effort` is also sent as OpenRouter's `reasoning.effort`. |
| `zdr` | `false` | OpenRouter only: route the request only to Zero Data Retention endpoints (`provider.zdr`); a model without one fails instead of falling back. |
| `image_detail` | - | OpenAI's image `detail` (`low`, `high`, `auto`). |
| `image_preprocessing` | `{}` | How large images are sent, see below. |

Keys are kept out of the logs: the pipeline configuration is logged with
`api_key` and similar keys redacted, and `VLMConfig`'s repr omits the key.

## Asking for JSON

Models differ in how they can be held to JSON, and asking a model for
something it does not support fails the request, so `response_format` is set
per model:

| `response_format` | Request | For |
|---|---|---|
| `schema` | Structured output: the reply is constrained to the schema (`strict`). | Models and providers that support JSON schema output - most current ones. |
| `json` | JSON mode: the reply is constrained to some JSON object. | Models that accept JSON mode but not a schema. |
| `raw` | Nothing; the prompt alone asks for JSON. | Models that accept neither. |

Every reply is checked the same way, whatever the mode: a Markdown code fence
around it is removed, it must parse as JSON, must contain no lone Unicode
surrogates, and must validate against the schema. A reply that fails any of
these - or comes without any content, as some providers report errors - is
asked for again, up to `max_attempts` times, since smaller models do not
return valid JSON every time; then `VLMResponseError` is raised. An error from
the endpoint itself - a wrong key, an unknown model, an unsupported option -
is a configuration error and is raised at once, not retried. Connection
errors, rate limits and server errors are retried by the `openai` SDK itself
before they surface.

## Prompts and images

`load_prompts()` reads prompt templates with their roles (`system`,
`developer`, `user`) and `render_prompts()` fills their `{{ variables }}` with
Jinja; a variable a template uses but is not given is an error. With Chat
Completions a `developer` prompt is sent as `system`. Images are attached to
the last prompt as data URLs: JPEG, PNG and WebP as they are, anything else
(TIFF, JPEG 2000) re-encoded as JPEG, and any image scaled down as
`image_preprocessing` sets.

## Image size

Every VLM engine sends its images through one resizer
(`image_preprocessing.py`), set by the `image_preprocessing` section of the
`vlm` config. An image is only ever scaled down, keeping its aspect ratio, and
re-encoded as JPEG when it is.

| Key | Default | Meaning |
|---|---|---|
| `max_pixels` | `25000000` | Width x height at most. Always applies: it cannot be `null`, so sending more takes stating a larger number. Endpoints reject images past their own limit - OpenAI 30000 32 px patches per image, about 30.7M pixels - and the default stays below it. |
| `max_w`, `max_h` | - | Width and height at most. |
| `text_height` | - | Pixels the smallest text on the page is scaled down to, measured in its ALTO. |
| `min_pixels` | - | How far `text_height` may scale a page down: a guard for text the OCR missed. Never scales up. |

The maxes win over `min_pixels`. The smallest text is the lowest line, each
line measured as the median height of its words (its own box when they have
none). Only lines with at least three letters or digits and, when the ALTO
gives confidences, a mean word confidence of at least 0.5 count: PERO OCR
reads cloth covers, paper texture and blank paper as low-confidence lines,
and specks as one or two characters. A page without such a line is scaled by
the maxes only.

## Local models

`local_server.serve(LocalModelConfig)` runs a model with vLLM for the duration
of a `with` block and yields its OpenAI-compatible URL, so a local model is
called with the same client as an API one. The model's part - `model_dir`,
`served_model_name`, `vllm_args` - comes with the engine; the machine's part
from the environment:

| Variable | Default | Meaning |
|---|---|---|
| `VLLM_EXECUTABLE` | `vllm` | The `vllm` command, typically from vLLM's own environment (it pins its own torch, so it does not live in MetaKat's). Its directory is put first on `PATH`, as activating that environment would, since vLLM runs tools installed beside it such as `ninja`. |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.85` | The fraction of the GPU's memory vLLM reserves. |
| `VLLM_STARTUP_TIMEOUT` | `900` | Seconds to wait until the server answers `/health`. |
| `VLLM_CUDA_VISIBLE_DEVICES` | - | The GPU(s) to run on; unset leaves `CUDA_VISIBLE_DEVICES` as it is. |

The server runs as a process group of its own on a free port of 127.0.0.1;
on leaving the block, normally or by an error, the whole group is terminated
and all its GPU memory returns to the system. A server that exits while
starting raises with the end of its log. `vllm_args` may not set what the
server sets itself: `--host`, `--port`, `--gpu-memory-utilization`,
`--served-model-name`, `--api-key`.

A local model is part of an engine and its settings start a process, so the
worker rejects a job override that sets a `local` key anywhere
(`local_model_locations`).

## Adding a VLM engine to another stage

1. Put the prompt templates and the JSON schema next to the engine, shaped
   like that stage's core result, and list them in `pyproject.toml`'s
   package data.
2. Build a `VLMClient` from the engine's `vlm` config - inside
   `local_server.serve()` for a local model, entered in `process()` so the
   model holds the GPU only while the stage runs - render the prompts per
   page and call `request_json()`.
3. Map the reply onto the stage's core result; locate values in the ALTO with
   the text aligner as the biblio engine does when the result needs boxes.
   `BiblioCoreEngineVLMBase` shows the split: everything but the mapping in a
   base class, the mapping of one output schema per subclass.
4. Register the engine with `requires` including `openai`, `jsonschema` and
   `jinja2`, and `extra='vlm'`.
5. Test it with a fake client: `VLMClient(config, client=...)` takes any
   object with `chat.completions.create` (or `responses.create`); the
   `fake_vllm` fixture in the root `conftest.py` stands in for a vLLM server.

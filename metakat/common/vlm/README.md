# Vision-language model client

`metakat.common.vlm` asks a vision-language model for JSON. It is the shared
part of every engine that reads pages with a model, so such an engine only
brings its prompts, its JSON schema and the mapping of the reply to its
result. The first one is the biblio core
[`biblio_core_engine_vlm`](../../biblio/README.md#engine-vision-language-model-biblio_core_engine_vlm).

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
| `image_detail` | - | OpenAI's image `detail` (`low`, `high`, `auto`). |
| `image_max_side` | - | Longest image side in pixels; larger images are scaled down before sending, which saves tokens on large scans. |

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
(TIFF, JPEG 2000) re-encoded as JPEG, and any image scaled down to
`image_max_side` when set.

## Adding a VLM engine to another stage

1. Put the prompt templates and the JSON schema next to the engine, shaped
   like that stage's core result, and list them in `pyproject.toml`'s
   package data.
2. Build a `VLMClient` from the engine's `vlm` config, render the prompts per
   page and call `request_json()`.
3. Map the reply onto the stage's core result; locate values in the ALTO with
   the text aligner as the biblio engine does when the result needs boxes.
4. Register the engine with `requires` including `openai`, `jsonschema` and
   `jinja2`, and `extra='vlm'`.
5. Test it with a fake client: `VLMClient(config, client=...)` takes any
   object with `chat.completions.create` (or `responses.create`).

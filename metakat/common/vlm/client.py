"""Prompting a vision-language model for JSON, shared by the VLM engines.

One client talks to any OpenAI-compatible endpoint - OpenAI, OpenRouter (which
also serves Anthropic, Google, Qwen, Mistral and others), or a local vLLM - so
an engine names only the endpoint, the model and how the model is asked for
JSON. Models differ in that last part, and asking a model for something it
does not support fails the request, so it is configured per model:

- `schema` - the reply is constrained to the JSON schema (structured output);
- `json` - the reply is constrained to some JSON object, for models that
  accept JSON mode but not a schema;
- `raw` - nothing is requested, for models that accept neither; the prompt
  alone asks for JSON.

Whatever the mode, the reply is parsed the same way - a Markdown code fence
around it is removed, it must be valid JSON without lone Unicode surrogates,
and it must validate against the schema - and a reply that fails is asked
for again, up to `max_attempts` times: smaller models do not return valid JSON
every time. A request the endpoint rejects - a wrong key, an unknown model, a
parameter the model does not support - is a configuration error and is
raised at once rather than retried.

Every call is reported (metakat.common.vlm.report): its attempts, their
tokens and their cost.

Shared prompts (`SHARED_PROMPTS`, in prompts/): a template inserts one by its
name, `{{ ocr_rules }}` - how to read the OCR given with the page images -
so that every engine treats the OCR alike; what an engine extracts and how
it writes the values stays in its own prompts.

Prompt caching (`prompt_cache`, OpenAI models and OpenRouter): the leading
system/developer prompts are meant to be the same in every request of an
engine and the user prompt to hold what differs - the page data and the
images. `explicit` marks the end of the leading prompts as the one cache
breakpoint, so only that part is written to the cache (OpenAI: at 1.25 times
the input price) and read back by the next request (0.1 times); what follows
is billed as usual and never written. A `prompt_cache_key` derived from the
model, the leading prompts and the schema keeps the requests on one provider
endpoint (OpenRouter's sticky routing) and one cache (OpenAI). `off` caches
nothing; unset leaves the endpoint's default, which for current OpenAI models
writes the whole prompt, page data included, at the write price.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from metakat.common.vlm.image_preprocessing import ImagePreprocessing, smallest_text_height
from metakat.common.vlm.report import VLMCallReport, VLMImageSize, VLMPricing, attempt

logger = logging.getLogger(__name__)

_APIS = ("completions", "responses")
_RESPONSE_FORMATS = ("schema", "json", "raw")
_ROLES = ("system", "developer", "user")
_PROMPT_CACHE_MODES = ("explicit", "off")
# Texts every prompt template may insert by name: the rules for reading the
# OCR given with the page images are the same for every engine.
SHARED_PROMPTS = {"ocr_rules": Path(__file__).parent / "prompts" / "ocr_rules.txt"}
# Image types every OpenAI-compatible vision endpoint accepts as a data URL;
# anything else (TIFF, JPEG 2000) is re-encoded as JPEG.
_SENT_AS_IS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
_SURROGATE = re.compile(r"[\ud800-\udfff]")


@dataclass(frozen=True)
class VLMConfig:
    """Which model to ask, where, and how.

    The key is given either directly (`api_key`) or as the name of an
    environment variable holding it (`api_key_env`), so a worker can keep the
    key out of the job. It is never part of the repr.
    """

    api_url: str
    model: str
    api_key: Optional[str] = field(default=None, repr=False)
    api_key_env: Optional[str] = None
    api: str = "completions"
    response_format: str = "schema"
    # Total attempts, including the first request; 1 disables repeated attempts.
    max_attempts: int = 3
    timeout: float = 300.0
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    reasoning_effort: Optional[str] = None
    verbosity: Optional[str] = None
    service_tier: Optional[str] = None
    # OpenRouter only: route only to Zero Data Retention endpoints
    # (provider.zdr); a model without one fails instead of falling back.
    zdr: bool = False
    image_detail: Optional[str] = None
    # How large the images are sent (image_preprocessing); its max_pixels
    # always applies.
    image_preprocessing: ImagePreprocessing = field(default_factory=ImagePreprocessing)
    # The cost of the requests when the endpoint does not return it (VLMPricing).
    pricing: Optional[VLMPricing] = None
    # Prompt caching: explicit, off, or unset for the endpoint's default.
    prompt_cache: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("api_url", "model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"VLM {name} must be a non-empty string")
        if self.api not in _APIS:
            raise ValueError(f"VLM api must be one of {', '.join(_APIS)}: {self.api!r}")
        if self.response_format not in _RESPONSE_FORMATS:
            raise ValueError(
                f"VLM response_format must be one of {', '.join(_RESPONSE_FORMATS)}: "
                f"{self.response_format!r}"
            )
        if not isinstance(self.max_attempts, int) or self.max_attempts < 1:
            raise ValueError("VLM max_attempts must be a positive integer")
        if not isinstance(self.zdr, bool):
            raise ValueError("VLM zdr must be true or false")
        if not isinstance(self.image_preprocessing, ImagePreprocessing):
            raise ValueError("VLM image_preprocessing must be an ImagePreprocessing")
        if self.prompt_cache is not None and self.prompt_cache not in _PROMPT_CACHE_MODES:
            raise ValueError(
                f"VLM prompt_cache must be one of {', '.join(_PROMPT_CACHE_MODES)}: {self.prompt_cache!r}")
        if self.pricing is not None and not isinstance(self.pricing, VLMPricing):
            raise ValueError("VLM pricing must be a VLMPricing")
        if self.api_key is None and self.api_key_env is None:
            raise ValueError("VLM config needs api_key or api_key_env")

    @classmethod
    def from_config(cls, config: Mapping[str, Any], location: str = "VLM config") -> "VLMConfig":
        if not isinstance(config, Mapping):
            raise ValueError(f"{location} must be an object")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(config) - known)
        if unknown:
            raise ValueError(f"{location} has unknown keys: {', '.join(unknown)}")
        config = dict(config)
        if "image_preprocessing" in config:
            config["image_preprocessing"] = ImagePreprocessing.from_config(
                config["image_preprocessing"], f"{location} image_preprocessing")
        if config.get("pricing") is not None:
            config["pricing"] = VLMPricing.from_config(config["pricing"], f"{location} pricing")
        return cls(**config)

    def resolve_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        key = os.environ.get(self.api_key_env or "")
        if not key:
            raise ValueError(
                f"VLM api_key_env names {self.api_key_env!r}, which is not set in the environment"
            )
        return key


@dataclass(frozen=True)
class VLMPrompt:
    """One message: its role and text. Images are attached to the last one."""

    role: str
    text: str

    def __post_init__(self) -> None:
        if self.role not in _ROLES:
            raise ValueError(f"Prompt role must be one of {', '.join(_ROLES)}: {self.role!r}")


@dataclass(frozen=True)
class VLMImage:
    """An image as sent: a data URL, and for one read from a file its sizes."""

    data_url: str
    size: Optional[VLMImageSize] = None

    @classmethod
    def from_file(
        cls,
        path: str | os.PathLike[str],
        preprocessing: Optional[ImagePreprocessing] = None,
        alto: Optional[str | os.PathLike[str]] = None,
    ) -> "VLMImage":
        """The image at the size `preprocessing` allows; `alto`, when given, measures its text."""
        from PIL import Image

        path = Path(path)
        preprocessing = preprocessing or ImagePreprocessing()
        with Image.open(path) as image:
            image_format, original = image.format, image.size
            text_height = None
            if preprocessing.text_height is not None and alto is not None:
                text_height = smallest_text_height(alto, image.size)
            size = preprocessing.scaled_size(*image.size, text_height)
            scaled = size != image.size
            if scaled:
                logger.info("Image %s scaled from %dx%d to %dx%d (smallest text %s px)",
                            path.name, *image.size, *size, "-" if text_height is None else f"{text_height:.0f}")
            if image_format in _SENT_AS_IS and not scaled:
                data, mime = path.read_bytes(), _SENT_AS_IS[image_format]
            else:
                image = image.convert("RGB")
                if scaled:
                    image = image.resize(size, Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=90)
                data, mime = buffer.getvalue(), "image/jpeg"
        return cls(f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}",
                   VLMImageSize(original=original, sent=size, smallest_text_px=text_height))


class VLMResponseError(RuntimeError):
    """No valid JSON reply within `max_attempts`; `report` holds the attempts made."""

    def __init__(self, message: str, report: Optional[VLMCallReport] = None):
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class VLMReply:
    """A valid reply's JSON and the report of the call that got it."""

    data: Any
    report: VLMCallReport


def load_prompts(prompt_files: Sequence[Mapping[str, str]]) -> list[VLMPrompt]:
    """Prompt templates from `[{"role": ..., "path": ...}, ...]`, in order."""
    prompts = []
    for index, entry in enumerate(prompt_files):
        if not isinstance(entry, Mapping) or set(entry) != {"role", "path"}:
            raise ValueError(f"Prompt {index} must be an object with exactly 'role' and 'path'")
        prompts.append(VLMPrompt(role=entry["role"], text=Path(entry["path"]).read_text(encoding="utf-8")))
    if not prompts:
        raise ValueError("At least one prompt is required")
    return prompts


def prompt_sources(prompt_files: Sequence[Mapping[str, str]]) -> list[tuple[str, str]]:
    """The (role, path) of each prompt file, each followed by the shared prompts it inserts, under its role."""
    from jinja2 import Environment, meta

    sources = []
    for entry in prompt_files:
        sources.append((entry["role"], entry["path"]))
        used = meta.find_undeclared_variables(Environment().parse(Path(entry["path"]).read_text(encoding="utf-8")))
        sources.extend((entry["role"], str(path)) for name, path in SHARED_PROMPTS.items() if name in used)
    return sources


def render_prompts(templates: Sequence[VLMPrompt], variables: Mapping[str, Any]) -> list[VLMPrompt]:
    """Fill each template's {{ variables }} and shared prompts; an unknown variable is an error."""
    from jinja2 import StrictUndefined, Template

    variables = {**{name: path.read_text(encoding="utf-8").rstrip("\n") for name, path in SHARED_PROMPTS.items()},
                 **variables}
    return [
        VLMPrompt(role=template.role,
                  text=Template(template.text, undefined=StrictUndefined).render(**variables))
        for template in templates
    ]


class VLMClient:
    def __init__(self, config: VLMConfig, client: Any = None, *, local: bool = False):
        # local: the model is served by MetaKat itself, so its requests cost nothing.
        self.config = config
        self.local = local
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=config.api_url, api_key=config.resolve_api_key(),
                            timeout=config.timeout, max_retries=0)
        self._client = client

    def request_json(
        self,
        prompts: Sequence[VLMPrompt],
        images: Sequence[VLMImage] = (),
        json_schema: Optional[Mapping[str, Any]] = None,
        request_id: str = "",
    ) -> Any:
        """The model's reply as validated JSON; raises VLMResponseError when none comes."""
        return self.request(prompts, images, json_schema, request_id).data

    def request(
        self,
        prompts: Sequence[VLMPrompt],
        images: Sequence[VLMImage] = (),
        json_schema: Optional[Mapping[str, Any]] = None,
        request_id: str = "",
    ) -> VLMReply:
        """The model's reply as validated JSON with the call's report; raises VLMResponseError when none comes.

        A request the endpoint rejects is raised as it is: the job fails, and
        the attempts made before it are only logged.
        """
        if self.config.api == "responses":
            request, send, reply_text = self._responses_request(prompts, images, json_schema)
        else:
            request, send, reply_text = self._completions_request(prompts, images, json_schema)

        started_utc, started = datetime.now(timezone.utc), time.monotonic()
        attempts = []

        def report() -> VLMCallReport:
            return VLMCallReport(
                api=self.config.api, api_url=self.config.api_url, model=self.config.model,
                reasoning_effort=self.config.reasoning_effort, started_utc=started_utc,
                duration_s=round(time.monotonic() - started, 3), attempts=tuple(attempts),
                images=tuple(image.size for image in images),
            )

        errors = []
        for number in range(1, self.config.max_attempts + 1):
            attempt_started = time.monotonic()
            try:
                response = send(**request)
            except Exception:
                if attempts:
                    logger.error("[%s] %s rejected the request after %d attempt(s): %s",
                                 request_id, self.config.model, len(attempts), report().describe())
                raise
            try:
                # A reply without choices or output (some providers return an
                # error object this way) counts as an invalid reply.
                reply = parse_reply(reply_text(response), json_schema)
            except (AttributeError, IndexError, TypeError, ValueError) as error:
                errors.append(f"attempt {number}: {error}")
                attempts.append(attempt(response, "invalid_reply", time.monotonic() - attempt_started,
                                        error=str(error), pricing=self.config.pricing, local=self.local))
                logger.warning(
                    "[%s] %s returned no valid JSON (attempt %d/%d): %s",
                    request_id, self.config.model, number, self.config.max_attempts, error,
                )
                continue
            attempts.append(attempt(response, "ok", time.monotonic() - attempt_started,
                                    pricing=self.config.pricing, local=self.local))
            call = report()
            logger.info("[%s] %s returned valid JSON (attempt %d): %s",
                        request_id, self.config.model, number, call.describe())
            return VLMReply(reply, call)
        call = report()
        logger.warning("[%s] %s gave no valid JSON: %s", request_id, self.config.model, call.describe())
        raise VLMResponseError(
            f"[{request_id}] {self.config.model} returned no valid JSON in "
            f"{self.config.max_attempts} attempt(s): " + "; ".join(errors),
            call,
        )

    def _completions_request(self, prompts, images, json_schema):
        image_part = {"type": "image_url", "image_url": {"url": None}}
        if self.config.image_detail is not None:
            image_part["image_url"]["detail"] = self.config.image_detail
        messages = []
        for prompt in prompts:
            # Chat Completions has no developer role on most endpoints.
            role = "system" if prompt.role == "developer" else prompt.role
            content = [{"type": "text", "text": prompt.text}] if prompt.text else []
            messages.append({"role": role, "content": content})
        messages[-1]["content"] += [
            {**image_part, "image_url": {**image_part["image_url"], "url": image.data_url}}
            for image in images
        ]

        request: dict[str, Any] = {"model": self.config.model, "messages": messages}
        if self.config.response_format == "json":
            request["response_format"] = {"type": "json_object"}
        elif self.config.response_format == "schema" and json_schema is not None:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response_schema", "strict": True, "schema": json_schema},
            }
        if self.config.max_tokens is not None:
            request["max_completion_tokens"] = self.config.max_tokens
        if self.config.temperature is not None:
            request["temperature"] = self.config.temperature
        if self.config.reasoning_effort is not None:
            request["reasoning_effort"] = self.config.reasoning_effort
            # OpenRouter reads the effort from its own reasoning object.
            request["extra_body"] = {"reasoning": {"effort": self.config.reasoning_effort}}
        if self.config.verbosity is not None:
            request["verbosity"] = self.config.verbosity
        if self.config.service_tier is not None:
            request["service_tier"] = self.config.service_tier
        if self.config.zdr:
            request.setdefault("extra_body", {})["provider"] = {"zdr": True}
        self._prompt_cache(request, prompts, [message["content"] for message in messages], json_schema)

        def reply_text(response):
            return response.choices[0].message.content

        return request, self._client.chat.completions.create, reply_text

    def _responses_request(self, prompts, images, json_schema):
        inputs = []
        for prompt in prompts:
            content = [{"type": "input_text", "text": prompt.text}] if prompt.text else []
            inputs.append({"role": prompt.role, "content": content})
        for image in images:
            part = {"type": "input_image", "image_url": image.data_url}
            if self.config.image_detail is not None:
                part["detail"] = self.config.image_detail
            inputs[-1]["content"].append(part)

        request: dict[str, Any] = {"model": self.config.model, "input": inputs}
        text: dict[str, Any] = {}
        if self.config.response_format == "json":
            text["format"] = {"type": "json_object"}
        elif self.config.response_format == "schema" and json_schema is not None:
            text["format"] = {"type": "json_schema", "name": "response_schema", "strict": True,
                              "schema": json_schema}
        if self.config.verbosity is not None:
            text["verbosity"] = self.config.verbosity
        if text:
            request["text"] = text
        if self.config.max_tokens is not None:
            request["max_output_tokens"] = self.config.max_tokens
        if self.config.temperature is not None:
            request["temperature"] = self.config.temperature
        if self.config.reasoning_effort is not None:
            request["reasoning"] = {"effort": self.config.reasoning_effort}
        if self.config.service_tier is not None:
            request["service_tier"] = self.config.service_tier
        if self.config.zdr:
            request.setdefault("extra_body", {})["provider"] = {"zdr": True}
        self._prompt_cache(request, prompts, [item["content"] for item in inputs], json_schema)

        def reply_text(response):
            return response.output_text

        return request, self._client.responses.create, reply_text

    def _prompt_cache(self, request, prompts, contents, json_schema) -> None:
        """Set the request's prompt caching; `contents` are the content blocks of each prompt's message."""
        if self.config.prompt_cache is None:
            return
        extra = request.setdefault("extra_body", {})
        # Explicit mode caches only at marked blocks; none marked, nothing.
        extra["prompt_cache_options"] = {"mode": "explicit"}
        if self.config.prompt_cache == "off":
            return
        leading = 0
        while leading < len(prompts) and prompts[leading].role in ("system", "developer"):
            leading += 1
        blocks = [block for content in contents[:leading] for block in content
                  if block["type"] in ("text", "input_text")]
        if not blocks:
            return
        blocks[-1]["prompt_cache_breakpoint"] = {"mode": "explicit"}
        cached = [self.config.model, [prompt.text for prompt in prompts[:leading]], json_schema]
        extra["prompt_cache_key"] = hashlib.sha256(
            json.dumps(cached, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:32]


def parse_reply(text: Optional[str], json_schema: Optional[Mapping[str, Any]] = None) -> Any:
    """A reply's JSON: unfenced, parsed, free of lone surrogates, schema-valid."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("the reply is empty")
    try:
        value = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError as error:
        raise ValueError(f"the reply is not valid JSON: {error}") from error
    _reject_surrogates(value)
    if json_schema is not None:
        from jsonschema import ValidationError, validate

        try:
            validate(instance=value, schema=json_schema)
        except ValidationError as error:
            raise ValueError(f"the reply does not match the schema: {error.message}") from error
    return value


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    lines = text.splitlines()
    if (
        len(lines) >= 2
        and lines[0].strip().lower() in ("```", "```json")
        and lines[-1].strip().startswith("```")
    ):
        return "\n".join(lines[1:-1]).strip()
    return text


def _reject_surrogates(value: Any, path: str = "$") -> None:
    if isinstance(value, str):
        if _SURROGATE.search(value):
            raise ValueError(f"invalid Unicode surrogate at {path}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_surrogates(item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            if _SURROGATE.search(key):
                raise ValueError(f"invalid Unicode surrogate in a key at {path}")
            _reject_surrogates(item, f"{path}.{key}")

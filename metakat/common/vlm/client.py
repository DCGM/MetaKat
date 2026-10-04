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
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_APIS = ("completions", "responses")
_RESPONSE_FORMATS = ("schema", "json", "raw")
_ROLES = ("system", "developer", "user")
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
    max_attempts: int = 3
    timeout: float = 300.0
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    reasoning_effort: Optional[str] = None
    verbosity: Optional[str] = None
    service_tier: Optional[str] = None
    image_detail: Optional[str] = None
    # Longest image side in pixels sent to the model; larger images are
    # scaled down. None sends the image as it is.
    image_max_side: Optional[int] = None

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
        if self.image_max_side is not None and self.image_max_side < 1:
            raise ValueError("VLM image_max_side must be positive")
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
        return cls(**dict(config))

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
    """An image as sent: a data URL."""

    data_url: str

    @classmethod
    def from_file(cls, path: str | os.PathLike[str], max_side: Optional[int] = None) -> "VLMImage":
        from PIL import Image

        path = Path(path)
        with Image.open(path) as image:
            image_format = image.format
            too_large = max_side is not None and max(image.size) > max_side
            if image_format in _SENT_AS_IS and not too_large:
                data, mime = path.read_bytes(), _SENT_AS_IS[image_format]
            else:
                image = image.convert("RGB")
                if too_large:
                    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=90)
                data, mime = buffer.getvalue(), "image/jpeg"
        return cls(f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}")


class VLMResponseError(RuntimeError):
    """No valid JSON reply within `max_attempts`."""


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


def render_prompts(templates: Sequence[VLMPrompt], variables: Mapping[str, Any]) -> list[VLMPrompt]:
    """Fill each template's {{ variables }}; an unknown variable is an error."""
    from jinja2 import StrictUndefined, Template

    return [
        VLMPrompt(role=template.role,
                  text=Template(template.text, undefined=StrictUndefined).render(**variables))
        for template in templates
    ]


class VLMClient:
    def __init__(self, config: VLMConfig, client: Any = None):
        self.config = config
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=config.api_url, api_key=config.resolve_api_key(),
                            timeout=config.timeout)
        self._client = client

    def request_json(
        self,
        prompts: Sequence[VLMPrompt],
        images: Sequence[VLMImage] = (),
        json_schema: Optional[Mapping[str, Any]] = None,
        request_id: str = "",
    ) -> Any:
        """The model's reply as validated JSON; raises VLMResponseError when none comes."""
        if self.config.api == "responses":
            request, send, reply_text = self._responses_request(prompts, images, json_schema)
        else:
            request, send, reply_text = self._completions_request(prompts, images, json_schema)

        errors = []
        for attempt in range(1, self.config.max_attempts + 1):
            response = send(**request)
            try:
                # A reply without choices or output (some providers return an
                # error object this way) counts as an invalid reply.
                reply = parse_reply(reply_text(response), json_schema)
            except (AttributeError, IndexError, TypeError, ValueError) as error:
                errors.append(f"attempt {attempt}: {error}")
                logger.warning(
                    "[%s] %s returned no valid JSON (attempt %d/%d): %s",
                    request_id, self.config.model, attempt, self.config.max_attempts, error,
                )
                continue
            logger.info("[%s] %s returned valid JSON (attempt %d)", request_id, self.config.model, attempt)
            return reply
        raise VLMResponseError(
            f"[{request_id}] {self.config.model} returned no valid JSON in "
            f"{self.config.max_attempts} attempt(s): " + "; ".join(errors)
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

        def reply_text(response):
            return response.output_text

        return request, self._client.responses.create, reply_text


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

"""What a VLM call cost: tokens, price and attempts, reported per call.

The client reports every call it makes (VLMCallReport): each attempt with the
tokens and the cost the endpoint gave, failed attempts included, since they
are paid too. Endpoints differ in what they return. The token counts are read
from the usage fields both OpenAI APIs define (Chat Completions and
Responses), and whatever an endpoint leaves out stays unknown. The cost is
taken from the endpoint when it gives one (OpenRouter's `usage.cost`), else
computed from the tokens and the `pricing` of the VLM config, else unknown; a
model MetaKat serves itself costs nothing.

An engine adds what only it knows - the pages it sent, its name and the
prompt files - and a bind engine turns the report into the
MetakatVLMProcessingReport of the output, naming the elements the call
concerned.
"""
from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional
from uuid import UUID

from metakat.schemas.base_objects import (
    MetakatCost,
    MetakatEngine,
    MetakatPromptRef,
    MetakatTokenUsage,
    MetakatVLMAttempt,
    MetakatVLMImage,
    MetakatVLMProcessingReport,
)

METAKAT_ROOT = Path(__file__).resolve().parents[2]
# OpenRouter states its cost in credits, which are US dollars.
PROVIDER_CURRENCY = "USD"


@dataclass(frozen=True)
class VLMPricing:
    """What the endpoint charges, per million tokens: the cost of an endpoint that does not return one.

    `cached_input` is the price of input tokens read from the prompt cache,
    the `input` price when not given. Reasoning tokens are output tokens.
    """

    input: float
    output: float
    cached_input: Optional[float] = None
    currency: str = "USD"

    def __post_init__(self) -> None:
        for name in ("input", "output", "cached_input"):
            value = getattr(self, name)
            if value is None and name == "cached_input":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"VLM pricing {name} must be a non-negative number")
        if not isinstance(self.currency, str) or not self.currency.strip():
            raise ValueError("VLM pricing currency must be a non-empty string")

    @classmethod
    def from_config(cls, config: Mapping[str, Any], location: str = "VLM pricing") -> "VLMPricing":
        if not isinstance(config, Mapping):
            raise ValueError(f"{location} must be an object")
        unknown = sorted(set(config) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f"{location} has unknown keys: {', '.join(unknown)}")
        return cls(**dict(config))

    def cost(self, tokens: MetakatTokenUsage) -> Optional[MetakatCost]:
        """The cost of these tokens; None without input and output counts."""
        if tokens.input is None or tokens.output is None:
            return None
        cached = tokens.cached_input or 0
        cached_price = self.input if self.cached_input is None else self.cached_input
        amount = ((tokens.input - cached) * self.input + cached * cached_price + tokens.output * self.output) / 1e6
        return MetakatCost(amount=amount, currency=self.currency, source="config")


@dataclass(frozen=True)
class VLMImageSize:
    """An image's size (width, height) before and after image_preprocessing, and its smallest text."""

    original: tuple[int, int]
    sent: tuple[int, int]
    smallest_text_px: Optional[float] = None


def prompt_refs(files: Iterable[tuple[str, str | Path]]) -> tuple[MetakatPromptRef, ...]:
    """The (role, path) files a request is made from, each with its absolute path and its hash."""
    refs = []
    for role, path in files:
        path = Path(path).resolve()
        refs.append(MetakatPromptRef(role=role, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return tuple(refs)


def _plain(value: Any) -> Any:
    """An SDK object as plain JSON data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return _plain(dump())
    if hasattr(value, "__dict__"):
        return _plain(vars(value))
    return str(value)


def _count(*values: Any) -> Optional[int]:
    """The first value that is a token count."""
    return next((value for value in values if isinstance(value, int) and not isinstance(value, bool)), None)


def _get(data: Any, key: str) -> Any:
    return data.get(key) if isinstance(data, Mapping) else None


def tokens_from_usage(usage: Optional[Mapping[str, Any]]) -> MetakatTokenUsage:
    """The token counts of a usage object of either API; None for each one it does not give."""
    if not isinstance(usage, Mapping):
        return MetakatTokenUsage()
    input_details = _get(usage, "prompt_tokens_details") or _get(usage, "input_tokens_details")
    output_details = _get(usage, "completion_tokens_details") or _get(usage, "output_tokens_details")
    return MetakatTokenUsage(
        input=_count(_get(usage, "prompt_tokens"), _get(usage, "input_tokens")),
        cached_input=_count(_get(input_details, "cached_tokens")),
        cache_write=_count(_get(input_details, "cache_write_tokens")),
        output=_count(_get(usage, "completion_tokens"), _get(usage, "output_tokens")),
        reasoning=_count(_get(output_details, "reasoning_tokens")),
    )


def attempt(
    response: Any,
    status: str,
    duration_s: float,
    *,
    error: Optional[str] = None,
    pricing: Optional[VLMPricing] = None,
    local: bool = False,
) -> MetakatVLMAttempt:
    """One attempt's report from the endpoint's response, whatever of it the endpoint filled."""
    usage = _plain(getattr(response, "usage", None))
    usage = usage if isinstance(usage, dict) else None
    tokens = tokens_from_usage(usage)
    provider_cost = _get(usage, "cost")
    if local:
        cost = MetakatCost(amount=0.0, source="local")
    elif isinstance(provider_cost, (int, float)) and not isinstance(provider_cost, bool):
        cost = MetakatCost(amount=float(provider_cost), currency=PROVIDER_CURRENCY, source="provider")
    else:
        cost = (pricing.cost(tokens) if pricing is not None else None) or MetakatCost()

    def text(name: str) -> Optional[str]:
        value = getattr(response, name, None)
        return value if isinstance(value, str) and value else None

    return MetakatVLMAttempt(
        status=status, error=error, duration_s=round(duration_s, 3),
        generation_id=text("id"), served_model=text("model"), provider=text("provider"),
        tokens=tokens, cost=cost, usage=usage,
    )


def total_tokens(attempts: Sequence[MetakatVLMAttempt]) -> MetakatTokenUsage:
    """The attempts' tokens summed per kind; unknown where any attempt's is."""
    totals = {}
    for name in MetakatTokenUsage.model_fields:
        values = [getattr(item.tokens, name) for item in attempts]
        totals[name] = None if not values or None in values else sum(values)
    return MetakatTokenUsage(**totals)


def total_cost(costs: Sequence[MetakatCost]) -> MetakatCost:
    """The costs summed; unknown when any of them is, so a sum never understates."""
    if not costs or any(cost.amount is None for cost in costs):
        return MetakatCost()
    currencies = {cost.currency for cost in costs if cost.currency is not None}
    if len(currencies) > 1:
        return MetakatCost()
    sources = {cost.source for cost in costs}
    return MetakatCost(amount=sum(cost.amount for cost in costs),
                       currency=next(iter(currencies), None),
                       source=sources.pop() if len(sources) == 1 else "mixed")


def describe_cost(cost: MetakatCost) -> str:
    if cost.amount is None:
        return "cost unknown"
    return f"cost {cost.amount:.6f} {cost.currency or ''} ({cost.source})".replace("  ", " ")


@dataclass(frozen=True)
class VLMCallReport:
    """One call of the client: every attempt until a valid reply or the last one.

    The client fills what it knows; the engine adds the page keys of the
    images in sending order, its name and its prompt files (`for_engine`).
    """

    api: str
    api_url: str
    model: str
    reasoning_effort: Optional[str]
    started_utc: datetime
    duration_s: float
    attempts: tuple[MetakatVLMAttempt, ...]
    images: tuple[Optional[VLMImageSize], ...] = ()
    page_keys: tuple[str, ...] = ()
    core_engine: Optional[str] = None
    prompts: tuple[MetakatPromptRef, ...] = ()

    @property
    def status(self) -> str:
        return "ok" if self.attempts and self.attempts[-1].status == "ok" else "failed"

    @property
    def tokens(self) -> MetakatTokenUsage:
        return total_tokens(self.attempts)

    @property
    def cost(self) -> MetakatCost:
        return total_cost([item.cost for item in self.attempts])

    def describe(self) -> str:
        tokens = self.tokens
        return (f"{len(self.attempts)} attempt(s), {describe_cost(self.cost)}, tokens "
                f"{'?' if tokens.input is None else tokens.input} in "
                f"({'?' if tokens.cached_input is None else tokens.cached_input} cached), "
                f"{'?' if tokens.output is None else tokens.output} out")

    def for_engine(
        self, core_engine: str, prompts: Sequence[MetakatPromptRef], page_keys: Sequence[str],
    ) -> "VLMCallReport":
        return dataclasses.replace(self, core_engine=core_engine, prompts=tuple(prompts), page_keys=tuple(page_keys))

    def to_metakat(
        self,
        page_ids: Sequence[UUID],
        *,
        stage: str,
        engine: Optional[MetakatEngine],
        also: Iterable[UUID] = (),
    ) -> MetakatVLMProcessingReport:
        """The output report; `page_ids` are the pages of `page_keys`, `also` the elements their reading went into."""
        if len(page_ids) != len(self.page_keys):
            raise ValueError(f"VLM report of {len(self.page_keys)} page(s) given {len(page_ids)} page id(s)")
        images = [
            MetakatVLMImage(page_id=page_id, original=size.original, sent=size.sent,
                            smallest_text_px=None if size.smallest_text_px is None
                            else round(size.smallest_text_px, 1))
            for page_id, size in zip(page_ids, self.images)
            if size is not None
        ]
        return MetakatVLMProcessingReport(
            element_ids=list(dict.fromkeys([*page_ids, *also])),
            status=self.status,
            stage=stage,
            metakat_engine=engine,
            core_engine=self.core_engine or "",
            prompts=list(self.prompts),
            started_utc=self.started_utc,
            duration_s=self.duration_s,
            cost=self.cost,
            api=self.api,
            api_url=self.api_url,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            images=images,
            tokens=self.tokens,
            attempts=list(self.attempts),
        )


def localize_prompt_paths(
    reports: Iterable[MetakatVLMProcessingReport], engine_dir: Optional[str | Path] = None,
) -> None:
    """Give each absolute prompt path its origin: within the metakat package or the engine directory."""
    roots = [("metakat", METAKAT_ROOT)]
    if engine_dir is not None:
        # First: an engine directory may lie within a metakat checkout.
        roots.insert(0, ("engine", Path(engine_dir).resolve()))
    for report in reports:
        for prompt in report.prompts:
            if prompt.origin is not None:
                continue
            path = Path(prompt.path)
            for origin, root in roots:
                if path.is_relative_to(root):
                    prompt.origin, prompt.path = origin, str(path.relative_to(root))
                    break

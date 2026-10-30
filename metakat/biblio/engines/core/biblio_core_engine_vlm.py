"""Biblio core engines reading title pages with a vision-language model.

For each span - the consecutive title pages of one document - the model gets
all of its page images and their ALTO text in one request and returns the
bibliographic statements it can read as JSON, one record for the span. The
model names no regions, so each value is located afterwards by finding its
text among the ALTO words of the span's pages (text-geometry-aligner's text
alignment, exact then fuzzy, sharing words across top-level metadata fields), on the
first page, in span order, where it is found. A value found there gets that
page, the box of its words and their ALTO IDs; one that is not - the model read
it from the image differently from the OCR, or no page has ALTO - keeps only
the span's first page.
Equally good repeated matches prefer nearby sibling fields, such as the date
and publisher of an imprint. Within a top-level field words remain exclusive;
inferred publishers are never given geometry. Ambiguous text can still be located
on the wrong occurrence. The model gives no per-value confidence, so every value
carries the configured `confidence` (0.95 unless set).

The model is either behind an API (`vlm.api_url`) or a local one that vLLM
serves only while `process()` runs (`local`, see
metakat.common.vlm.local_server). Both run the same engine: what differs
between engines is the JSON the model is asked for. `BiblioCoreEngineVLMBase`
does everything but that; a subclass names its output schema and implements
`page_from_reply`, the mapping of a reply in that schema onto the core result.
The monograph and periodical engines (biblio_core_engine_vlm_monograph,
biblio_core_engine_vlm_periodical) are the implementations, each with its
schema and prompts in vlm/<kind>/; a model trained for another output schema
needs its own subclass.

Configuration, besides `name`:

- `vlm` - how to call the model (metakat.common.vlm.VLMConfig); for a local
  model without `api_url`, `model` and the key, which the local server sets;
- `local` - a local model: `model_dir`, `served_model_name`, `vllm_args`
  (metakat.common.vlm.local_server.LocalModelConfig);
- `system_prompt_path`, `user_prompt_path` - the prompts, when not the
  engine's own. The system prompt is what every request shares, so that it
  can be cached (`vlm.prompt_cache`); the user prompt holds the page data.
  Both receive the variables `ocr` (the ALTO text
  of the span: of its one page as it is, of several pages each under a
  `--- Page k of n ---` line), `page_count` (the pages of the span) and
  `schema` (the output schema as JSON), and the page images are attached to
  the user prompt in page order;
- `schema_path` - the output schema, for an engine that has none of its own;
- `confidence` - the confidence given to every value.
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
from abc import abstractmethod
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterable, Optional, Sequence

from metakat.biblio.engines.core.biblio_core_engine import BiblioCoreEngine
from metakat.biblio.engines.core.models import (
    BiblioCoreResult,
    BiblioPage,
    BiblioSpan,
    BiblioSpanResult,
    container_values,
)
from metakat.common.models import AltoRefs, BoundingBox, DetectionEvidence
from metakat.common.vlm import (
    VLMCallReport,
    VLMClient,
    VLMConfig,
    VLMImage,
    VLMResponseError,
    load_prompts,
    prompt_refs,
    render_prompts,
)
from metakat.common.vlm.local_server import LocalModelConfig, serve

logger = logging.getLogger(__name__)

RESOURCES = Path(__file__).parent / "vlm"
_CONFIG_KEYS = {"name", "vlm", "local", "system_prompt_path", "user_prompt_path", "schema_path", "confidence"}
# Set by the local server, not by the configuration of a local model.
_LOCAL_VLM_KEYS = {"api_url", "model", "api_key", "api_key_env"}
DEFAULT_CONFIDENCE = 0.95
# The CP-SAT selection is exact but may take long on a page with many
# candidates; past this limit the best selection found so far is used.
_SELECTION_TIME_LIMIT_SECONDS = 10.0

# evidence(path, text): the text at a JSON path of the reply as evidence, or
# None for an empty value.
EvidenceFactory = Callable[[tuple, Optional[str]], Optional[DetectionEvidence]]


class BiblioCoreEngineVLMBase(BiblioCoreEngine):
    """A VLM biblio core; a subclass gives its output schema and its mapping.

    `DEFAULT_SCHEMA_PATH` is the schema the subclass's `page_from_reply`
    reads. An engine that has one maps only that schema, so a configured
    `schema_path` is rejected; an engine without one takes its schema from
    `schema_path`. The prompts default to the subclass's own when it has them.
    """

    DEFAULT_SCHEMA_PATH: ClassVar[Optional[Path]] = None
    DEFAULT_SYSTEM_PROMPT_PATH: ClassVar[Optional[Path]] = None
    DEFAULT_USER_PROMPT_PATH: ClassVar[Optional[Path]] = None
    # Keys of the reply whose values are labels rather than text printed on
    # the page, so are not looked for in the ALTO.
    LABEL_KEYS: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, config: Mapping[str, Any], client: Any = None):
        super().__init__(config=config)
        unknown = sorted(set(self.config) - _CONFIG_KEYS)
        if unknown:
            raise ValueError(f"Biblio core config has unknown keys: {', '.join(unknown)}")

        vlm = dict(self.config.get("vlm") or {})
        self.local: Optional[LocalModelConfig] = None
        if "local" in self.config:
            self.local = LocalModelConfig.from_config(self.config["local"], "Biblio core local config")
            given = sorted(set(vlm) & _LOCAL_VLM_KEYS)
            if given:
                raise ValueError(
                    f"Biblio core vlm config of a local model must not set {', '.join(given)}; "
                    "the local server does"
                )
            # The URL is a placeholder until the server runs and says where it
            # listens; the server takes any key.
            vlm.update(api_url="http://127.0.0.1", model=self.local.served_model_name, api_key="local")
        self.vlm_config = VLMConfig.from_config(vlm, "Biblio core vlm config")
        self._client = client

        prompt_files = [
            {"role": "developer", "path": self._path("system_prompt_path", self.DEFAULT_SYSTEM_PROMPT_PATH)},
            {"role": "user", "path": self._path("user_prompt_path", self.DEFAULT_USER_PROMPT_PATH)},
        ]
        self.prompts = load_prompts(prompt_files)
        if self.DEFAULT_SCHEMA_PATH is not None and "schema_path" in self.config:
            raise ValueError(
                f"{type(self).__name__} maps only its own output schema; a model with another "
                "schema needs an engine implementing that schema's mapping"
            )
        schema_path = self._path("schema_path", self.DEFAULT_SCHEMA_PATH)
        self.schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        self.prompt_refs = prompt_refs([*((entry["role"], entry["path"]) for entry in prompt_files),
                                        ("schema", schema_path)])

        self.confidence = self.config.get("confidence", DEFAULT_CONFIDENCE)
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("Biblio core confidence must be a number from 0 to 1")
        self.aligner = _text_aligner()
        if self.local is not None:
            logger.info("Biblio core VLM: local model %s, response format %s",
                        self.local.model_dir, self.vlm_config.response_format)
        else:
            logger.info("Biblio core VLM: model %s at %s, response format %s",
                        self.vlm_config.model, self.vlm_config.api_url, self.vlm_config.response_format)

    @abstractmethod
    def page_from_reply(
        self,
        reply: Any,
        evidence: EvidenceFactory,
        page_keys: Sequence[str],
    ) -> Optional[BiblioSpanResult]:
        """The span result for a reply in this engine's schema; None if nothing was read.

        Every value goes through `evidence` with its JSON path in the reply,
        which locates it in the ALTO.
        """

    def process(self, spans: Sequence[BiblioSpan]) -> BiblioCoreResult:
        keys = [page.page_key for span in spans for page in span]
        if len(set(keys)) != len(keys):
            raise ValueError("Biblio core got a page key in more than one place")
        spans = list(spans)
        if not any(spans):
            return BiblioCoreResult(spans=tuple(None for _ in spans))
        results, reports = [], []
        with self._model() as client:
            for span in spans:
                result, report = self.read_span(client, span) if span else (None, None)
                results.append(result)
                if report is not None:
                    reports.append(report)
        logger.info("Biblio core read bibliographic information on %d of %d span(s)",
                    sum(result is not None for result in results), len(spans))
        return BiblioCoreResult(spans=tuple(results), reports=tuple(reports))

    @contextlib.contextmanager
    def _model(self) -> Iterator[VLMClient]:
        """A client for the model; a local one is served only inside the block."""
        if self.local is None:
            yield VLMClient(self.vlm_config, self._client)
            return
        with serve(self.local) as api_url:
            yield VLMClient(dataclasses.replace(self.vlm_config, api_url=api_url), self._client, local=True)

    def read_span(
        self, client: VLMClient, span: Sequence[BiblioPage],
    ) -> tuple[Optional[BiblioSpanResult], Optional[VLMCallReport]]:
        """One span's readings, None when the model gave none or no valid reply, and the report of its call."""
        from text_geometry_aligner import ALTOReader

        page_keys = [page.page_key for page in span]
        request_id = page_keys[0] if len(span) == 1 else f"{page_keys[0]}+{len(span) - 1}"
        alto_pages = []
        for page in span:
            alto_page = ALTOReader().read(page.alto) if page.alto is not None else None
            if alto_page is None:
                logger.warning("Page %s has no ALTO: its values will have no box", page.page_key)
            alto_pages.append(alto_page)
        prompts = render_prompts(self.prompts, {
            "ocr": span_text([None if alto_page is None else self.page_text(page, alto_page)
                              for page, alto_page in zip(span, alto_pages)]),
            "page_count": len(span),
            "schema": json.dumps(self.schema, ensure_ascii=False, indent=2),
        })
        try:
            answer = client.request(
                prompts,
                [VLMImage.from_file(page.image, self.vlm_config.image_preprocessing, page.alto) for page in span],
                self.schema,
                request_id=request_id,
            )
        except VLMResponseError as error:
            logger.error("Biblio core skips span %s: %s", request_id, error)
            return None, self._report(error.report, page_keys)
        reply, report = answer.data, self._report(answer.report, page_keys)

        # Per JSON path, the first page of the span whose ALTO holds the value.
        located: dict[tuple, tuple[str, Any]] = {}
        for page, alto_page in zip(span, alto_pages):
            if alto_page is None:
                continue
            for path, region in self._locate(alto_page, reply).items():
                if region.alto_geometry is not None and path not in located:
                    located[path] = (page.page_key, region)

        def evidence(path: tuple, text: Optional[str]) -> Optional[DetectionEvidence]:
            if text is None or not text.strip():
                return None
            page_key, region = located.get(path, (page_keys[0], None))
            geometry = None if region is None else region.alto_geometry
            return DetectionEvidence(
                text=text.strip(),
                confidence=self.confidence,
                bbox=None if geometry is None else BoundingBox(
                    geometry.x, geometry.y, geometry.width, geometry.height),
                page_key=page_key,
                alto=AltoRefs() if region is None else AltoRefs.from_words(region.words),
            )

        result = self.page_from_reply(reply, evidence, page_keys)
        logger.info("Span %s: located %d value(s) in the ALTO", request_id, len(located))
        return result, report

    def _report(self, report: Optional[VLMCallReport], page_keys: Sequence[str]) -> Optional[VLMCallReport]:
        return None if report is None else report.for_engine(self.name, self.prompt_refs, page_keys)

    def page_text(self, page: BiblioPage, alto_page) -> str:
        """The OCR text of one page for the prompt: a line per TextLine, a blank line between blocks."""
        return alto_text(alto_page.words)

    def _path(self, key: str, default: Optional[Path]) -> str:
        path = self.config.get(key) or default
        if path is None:
            raise ValueError(f"{type(self).__name__} needs {key}")
        return str(path)

    def _locate(self, alto_page, reply: Any) -> dict:
        """The aligned region of every text value in the reply, by JSON path."""
        document = self.aligner.align_data(alto_page, _alignable(reply, self.LABEL_KEYS))
        return {
            tuple(region.json_text_path): region
            for page in document.pages
            for region in page.regions
            if region.json_text_path is not None
        }


def kept_containers(containers: Iterable) -> tuple:
    """The containers that hold at least one value."""
    return tuple(container for container in containers if next(container_values(container), None) is not None)


class ReplyValues:
    """The text values of a reply as evidence, each by its JSON path (see EvidenceFactory)."""

    def __init__(self, evidence: EvidenceFactory):
        self.evidence = evidence

    def one(self, path: tuple, value: Any) -> Optional[DetectionEvidence]:
        return self.evidence(path, value) if isinstance(value, str) else None

    def many(self, path: tuple, values: Any) -> tuple:
        return tuple(
            item for index, value in enumerate(values if isinstance(values, list) else ())
            if (item := self.one(path + (index,), value)) is not None
        )

    @staticmethod
    def items(reply: Mapping[str, Any], key: str) -> Iterator[tuple[tuple, Mapping[str, Any]]]:
        """(JSON path, object) of every object in the list under `key`."""
        for index, item in enumerate(reply.get(key) or ()):
            if isinstance(item, Mapping):
                yield (key, index), item


def _alignable(reply: Any, label_keys: frozenset[str] = frozenset()) -> Any:
    """The reply with only its text values left to align.

    A label is not text on the page, and an empty string has nothing to find;
    both become None, which the aligner skips, so every other value keeps its
    JSON path.
    """
    if isinstance(reply, Mapping):
        return {key: None if key in label_keys or (key == "publisher" and reply.get("publisherInferred") is True)
                else _alignable(value, label_keys)
                for key, value in reply.items()}
    if isinstance(reply, list):
        return [_alignable(value, label_keys) for value in reply]
    # Only text is printed on the page: a flag or a number is not looked for.
    if not isinstance(reply, str) or not reply.strip():
        return None
    return reply


def span_text(page_texts: Sequence[Optional[str]]) -> str:
    """The OCR of a span: one page's text as it is, several pages each under its own heading."""
    if len(page_texts) == 1:
        return page_texts[0] or ""
    return "\n\n".join(f"--- Page {index} of {len(page_texts)} ---\n{text or ''}"
                        for index, text in enumerate(page_texts, start=1))


def alto_text(words) -> str:
    """The page's ALTO text: a line per TextLine, a blank line between blocks."""
    lines: list[str] = []
    current_line = current_block = object()
    for word in words:
        if word.block_index != current_block and lines:
            lines.append("")
        if word.line_index != current_line or word.block_index != current_block:
            lines.append(word.text)
        else:
            lines[-1] += " " + word.text
        current_line, current_block = word.line_index, word.block_index
    return "\n".join(lines)


def _text_aligner():
    from text_geometry_aligner import CPSATCandidateSelector, TextAligner
    from text_geometry_aligner.normalization import TextNormalizationPipeline
    from text_geometry_aligner.text_matching.candidate_generators import (
        AnchoredFuzzyTextCandidateGenerator,
        CompositeCandidateGenerator,
        ExactTextCandidateGenerator,
        FuzzyCandidateConfig,
    )

    return TextAligner(
        candidate_generator=CompositeCandidateGenerator((
            ExactTextCandidateGenerator(),
            AnchoredFuzzyTextCandidateGenerator(FuzzyCandidateConfig()),
        )),
        candidate_selector=CPSATCandidateSelector(
            time_limit_seconds=_SELECTION_TIME_LIMIT_SECONDS,
            require_optimal=False,
        ),
        normalizer=TextNormalizationPipeline.from_optional_names(["lowercase", "strip-punctuation"]),
        min_similarity=0.5,
        share_words_depth=1,
        prefer_sibling_proximity=True,
    )

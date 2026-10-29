"""Biblio core engines reading title pages with a vision-language model.

For each span - the consecutive title pages of one document - the model gets
all of its page images and their ALTO text in one request and returns the
bibliographic statements it can read as JSON, one record for the span. The
model names no regions, so each value is located afterwards by finding its
text among the ALTO words of the span's pages (text-geometry-aligner's text
alignment, exact then fuzzy, every word given to at most one value), on the
first page, in span order, where it is found. A value found there gets that
page, the box of its words and their ALTO IDs; one that is not - the model read
it from the image differently from the OCR, or no page has ALTO - keeps only
the span's first page.
Ambiguous text, such as a place printed twice, may occasionally be located on
the wrong occurrence. The model gives no per-value confidence, so every value
carries the configured `confidence` (0.95 unless set).

The model is either behind an API (`vlm.api_url`) or a local one that vLLM
serves only while `process()` runs (`local`, see
metakat.common.vlm.local_server). Both run the same engine: what differs
between engines is the JSON the model is asked for. `BiblioCoreEngineVLMBase`
does everything but that; a subclass names its output schema and implements
`page_from_reply`, the mapping of a reply in that schema onto the core result.
`BiblioCoreEngineVLM` is the one implementation so far, with its schema,
prompts and mapping in vlm/; a model trained for another output schema needs
its own subclass.

Configuration, besides `name`:

- `vlm` - how to call the model (metakat.common.vlm.VLMConfig); for a local
  model without `api_url`, `model` and the key, which the local server sets;
- `local` - a local model: `model_dir`, `served_model_name`, `vllm_args`
  (metakat.common.vlm.local_server.LocalModelConfig);
- `system_prompt_path`, `user_prompt_path` - the prompts, when not the
  engine's own; the system prompt receives the variables `ocr` (the ALTO text
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
    AgentRole,
    BiblioAgent,
    BiblioCoreResult,
    BiblioManufacture,
    BiblioPage,
    BiblioPublication,
    BiblioReading,
    BiblioSeries,
    BiblioSpan,
    BiblioSpanResult,
    BiblioTitleInfo,
    container_values,
)
from metakat.common.models import AltoRefs, BoundingBox, DetectionEvidence
from metakat.common.vlm import (
    VLMClient,
    VLMConfig,
    VLMImage,
    VLMResponseError,
    load_prompts,
    render_prompts,
)
from metakat.common.vlm.local_server import LocalModelConfig, serve

logger = logging.getLogger(__name__)

_RESOURCES = Path(__file__).parent / "vlm"
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

        self.prompts = load_prompts([
            {"role": "developer", "path": self._path("system_prompt_path", self.DEFAULT_SYSTEM_PROMPT_PATH)},
            {"role": "user", "path": self._path("user_prompt_path", self.DEFAULT_USER_PROMPT_PATH)},
        ])
        if self.DEFAULT_SCHEMA_PATH is not None and "schema_path" in self.config:
            raise ValueError(
                f"{type(self).__name__} maps only its own output schema; a model with another "
                "schema needs an engine implementing that schema's mapping"
            )
        self.schema = json.loads(
            Path(self._path("schema_path", self.DEFAULT_SCHEMA_PATH)).read_text(encoding="utf-8"))

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
        with self._model() as client:
            results = tuple(self.read_span(client, span) if span else None for span in spans)
        logger.info("Biblio core read bibliographic information on %d of %d span(s)",
                    sum(result is not None for result in results), len(spans))
        return BiblioCoreResult(spans=results)

    @contextlib.contextmanager
    def _model(self) -> Iterator[VLMClient]:
        """A client for the model; a local one is served only inside the block."""
        if self.local is None:
            yield VLMClient(self.vlm_config, self._client)
            return
        with serve(self.local) as api_url:
            yield VLMClient(dataclasses.replace(self.vlm_config, api_url=api_url), self._client)

    def read_span(self, client: VLMClient, span: Sequence[BiblioPage]) -> Optional[BiblioSpanResult]:
        """One span's readings; None when the model gave none or no valid reply."""
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
            "ocr": span_text([None if a is None else alto_text(a.words) for a in alto_pages]),
            "page_count": len(span),
            "schema": json.dumps(self.schema, ensure_ascii=False, indent=2),
        })
        try:
            reply = client.request_json(
                prompts,
                [VLMImage.from_file(page.image, self.vlm_config.image_max_side) for page in span],
                self.schema,
                request_id=request_id,
            )
        except VLMResponseError as error:
            logger.error("Biblio core skips span %s: %s", request_id, error)
            return None

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
        return result

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


class BiblioCoreEngineVLM(BiblioCoreEngineVLMBase):
    """The output schema grouped like the core result (vlm/schema.json).

    One object per title, imprint, printer statement, series and name, plus
    what the page says about a periodical volume or issue, each JSON key named
    after the MetaKat field it fills. Used by API and local models alike; a
    local model's prompts may replace the ones in vlm/, its schema may not.
    """

    DEFAULT_SCHEMA_PATH = _RESOURCES / "schema.json"
    DEFAULT_SYSTEM_PROMPT_PATH = _RESOURCES / "prompt_system.txt"
    DEFAULT_USER_PROMPT_PATH = _RESOURCES / "prompt_user.txt"
    # An agent's role is a label, not text on the page.
    LABEL_KEYS = frozenset({"role"})

    def page_from_reply(self, reply, evidence, page_keys):
        return page_from_reply(reply, evidence, page_keys)


def page_from_reply(reply: Mapping[str, Any], evidence: EvidenceFactory,
                    page_keys: Sequence[str]) -> Optional[BiblioSpanResult]:
    """Build a span result from a reply in vlm/schema.json.

    Containers left without any value are dropped.
    """

    def one(path: tuple, value: Any) -> Optional[DetectionEvidence]:
        return evidence(path, value) if isinstance(value, str) else None

    def many(path: tuple, values: Any) -> tuple:
        return tuple(
            item for index, value in enumerate(values or ())
            if (item := one(path + (index,), value)) is not None
        )

    def items(key: str) -> Iterable[tuple[tuple, Mapping[str, Any]]]:
        for index, item in enumerate(reply.get(key) or ()):
            if isinstance(item, Mapping):
                yield (key, index), item

    title_infos = tuple(
        BiblioTitleInfo(
            title=one(path + ("title",), item.get("title")),
            sub_title=one(path + ("subTitle",), item.get("subTitle")),
            part_number=one(path + ("partNumber",), item.get("partNumber")),
            part_name=one(path + ("partName",), item.get("partName")),
        )
        for path, item in items("titleInfo")
    )
    publications = tuple(
        BiblioPublication(
            places=many(path + ("placeTerm",), item.get("placeTerm")),
            publishers=many(path + ("publisher",), item.get("publisher")),
            date_issued=one(path + ("dateIssued",), item.get("dateIssued")),
            edition=one(path + ("edition",), item.get("edition")),
            frequency=one(path + ("frequency",), item.get("frequency")),
        )
        for path, item in items("publication")
    )
    manufactures = tuple(
        BiblioManufacture(
            places=many(path + ("manufacturePlaceTerm",), item.get("manufacturePlaceTerm")),
            manufacturers=many(path + ("manufacturePublisher",), item.get("manufacturePublisher")),
            date=one(path + ("manufactureDate",), item.get("manufactureDate")),
        )
        for path, item in items("manufacture")
    )
    series = tuple(
        BiblioSeries(
            name=one(path + ("seriesName",), item.get("seriesName")),
            part_number=one(path + ("seriesPartNumber",), item.get("seriesPartNumber")),
            part_name=one(path + ("seriesPartName",), item.get("seriesPartName")),
        )
        for path, item in items("series")
    )
    agents = []
    for path, item in items("agents"):
        name = one(path + ("name",), item.get("name"))
        try:
            role = AgentRole(item.get("role"))
        except ValueError:
            logger.warning("Span %s: skipping a name with unknown role %r", page_keys[0], item.get("role"))
            continue
        if name is None:
            continue
        agents.append(BiblioAgent(
            role=role,
            name=name,
            affiliations=many(path + ("affiliation",), item.get("affiliation")),
            emails=many(path + ("email",), item.get("email")),
        ))

    reading = BiblioReading(
        title_infos=_kept(title_infos),
        publications=_kept(publications),
        manufactures=_kept(manufactures),
        series=_kept(series),
        agents=tuple(agents),
    )
    periodical_volume = _periodical(reply, "periodicalVolume", one)
    periodical_issue = _periodical(reply, "periodicalIssue", one)
    if reading.is_empty() and periodical_volume is None and periodical_issue is None:
        return None
    return BiblioSpanResult(
        page_keys=tuple(page_keys),
        reading=reading,
        periodical_volume=periodical_volume,
        periodical_issue=periodical_issue,
    )


def _periodical(reply: Mapping[str, Any], key: str, one) -> Optional[BiblioReading]:
    # As the YOLO core does: the level's number is a titleInfo part number,
    # its date the date of an imprint.
    level = reply.get(key)
    if not isinstance(level, Mapping):
        return None
    reading = BiblioReading(
        title_infos=_kept((BiblioTitleInfo(part_number=one((key, "partNumber"), level.get("partNumber"))),)),
        publications=_kept((BiblioPublication(date_issued=one((key, "dateIssued"), level.get("dateIssued"))),)),
    )
    return None if reading.is_empty() else reading


def _kept(containers: Iterable) -> tuple:
    return tuple(container for container in containers if next(container_values(container), None) is not None)


def _alignable(reply: Any, label_keys: frozenset[str] = frozenset()) -> Any:
    """The reply with only its text values left to align.

    A label is not text on the page, and an empty string has nothing to find;
    both become None, which the aligner skips, so every other value keeps its
    JSON path.
    """
    if isinstance(reply, Mapping):
        return {key: None if key in label_keys else _alignable(value, label_keys)
                for key, value in reply.items()}
    if isinstance(reply, list):
        return [_alignable(value, label_keys) for value in reply]
    if isinstance(reply, str) and not reply.strip():
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
    )

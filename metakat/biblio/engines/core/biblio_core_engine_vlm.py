"""Biblio core engines reading title pages with a vision-language model.

For each page the model gets the page image and its ALTO text and returns the
bibliographic statements it can read as JSON. The model names no regions, so
each value is located afterwards by finding its text among the page's ALTO
words (text-geometry-aligner's text alignment, exact then fuzzy, every word
given to at most one value). A value found there gets the box of its words
and their ALTO IDs; one that is not - the model read it from the image
differently from the OCR, or the page has no ALTO - keeps only its page.
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
  engine's own; the system prompt receives the variables `ocr` (the page's
  ALTO text) and `schema` (the output schema as JSON), and the page image is
  attached to the user prompt;
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
from typing import Any, Callable, ClassVar, Iterable, List, Optional

from metakat.biblio.engines.core.biblio_core_engine import BiblioCoreEngine
from metakat.biblio.engines.core.models import (
    AgentRole,
    BiblioAgent,
    BiblioCoreResult,
    BiblioManufacture,
    BiblioPageResult,
    BiblioPublication,
    BiblioReading,
    BiblioSeries,
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
        page_key: str,
    ) -> Optional[BiblioPageResult]:
        """The page result for a reply in this engine's schema; None if nothing was read.

        Every value goes through `evidence` with its JSON path in the reply,
        which locates it in the ALTO.
        """

    def process(
        self,
        images: List[str],
        alto_files: List[str],
    ) -> BiblioCoreResult:
        pages: dict[str, BiblioPageResult] = {}
        if not images:
            return BiblioCoreResult(pages=pages)
        alto_by_key = {Path(alto_file).stem: alto_file for alto_file in alto_files}
        with self._model() as client:
            for image in images:
                page_key = Path(image).stem
                if page_key in pages:
                    raise ValueError(f"Biblio core got duplicate page key: {page_key}")
                page_result = self.read_page(client, page_key, image, alto_by_key.get(page_key))
                if page_result is not None:
                    pages[page_key] = page_result
        logger.info("Biblio core read bibliographic information on %d of %d page(s)",
                    len(pages), len(images))
        return BiblioCoreResult(pages=pages)

    @contextlib.contextmanager
    def _model(self) -> Iterator[VLMClient]:
        """A client for the model; a local one is served only inside the block."""
        if self.local is None:
            yield VLMClient(self.vlm_config, self._client)
            return
        with serve(self.local) as api_url:
            yield VLMClient(dataclasses.replace(self.vlm_config, api_url=api_url), self._client)

    def read_page(
        self,
        client: VLMClient,
        page_key: str,
        image: str,
        alto_file: Optional[str],
    ) -> Optional[BiblioPageResult]:
        """One page's readings; None when the model gave none or no valid reply."""
        from text_geometry_aligner import ALTOReader

        alto_page = ALTOReader().read(alto_file) if alto_file is not None else None
        if alto_page is None:
            logger.warning("Page %s has no ALTO: its values will have no box", page_key)
        prompts = render_prompts(self.prompts, {
            "ocr": alto_text(alto_page.words) if alto_page is not None else "",
            "schema": json.dumps(self.schema, ensure_ascii=False, indent=2),
        })
        try:
            reply = client.request_json(
                prompts,
                [VLMImage.from_file(image, self.vlm_config.image_max_side)],
                self.schema,
                request_id=page_key,
            )
        except VLMResponseError as error:
            logger.error("Biblio core skips page %s: %s", page_key, error)
            return None

        regions = self._locate(alto_page, reply) if alto_page is not None else {}

        def evidence(path: tuple, text: Optional[str]) -> Optional[DetectionEvidence]:
            if text is None or not text.strip():
                return None
            region = regions.get(path)
            geometry = None if region is None else region.alto_geometry
            return DetectionEvidence(
                text=text.strip(),
                confidence=self.confidence,
                bbox=None if geometry is None else BoundingBox(
                    geometry.x, geometry.y, geometry.width, geometry.height),
                page_key=page_key,
                alto=AltoRefs() if region is None else AltoRefs.from_words(region.words),
            )

        page_result = self.page_from_reply(reply, evidence, page_key)
        found = sum(1 for region in regions.values() if region.alto_geometry is not None)
        logger.info("Page %s: located %d of %d value(s) in the ALTO", page_key, found, len(regions))
        return page_result

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

    def page_from_reply(self, reply, evidence, page_key):
        return page_from_reply(reply, evidence, page_key)


def page_from_reply(reply: Mapping[str, Any], evidence: EvidenceFactory, page_key: str) -> Optional[BiblioPageResult]:
    """Build a page result from a reply in vlm/schema.json.

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
            logger.warning("Page %s: skipping a name with unknown role %r", page_key, item.get("role"))
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
    return BiblioPageResult(
        page_key=page_key,
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

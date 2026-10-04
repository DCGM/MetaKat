"""Biblio core engine reading title pages with a vision-language model.

For each page the model gets the page image and its ALTO text and returns the
bibliographic statements it can read as JSON (vlm/schema.json), grouped the
way the core result groups them: one object per title, imprint, printer
statement, series and name, plus what the page says about a periodical volume
or issue. The model names no regions, so each value is located afterwards by
finding its text among the page's ALTO words (text-geometry-aligner's text
alignment, exact then fuzzy, every word given to at most one value). A value
found there gets the box of its words and their ALTO IDs; one that is not -
the model read it from the image differently from the OCR, or the page has no
ALTO - keeps only its page. Ambiguous text, such as a place printed twice, may
occasionally be located on the wrong occurrence.

The model gives no per-value confidence, so every value carries the configured
`confidence` (0.95 unless set).

Configuration, besides `name`:

- `vlm` - the model to call (metakat.common.vlm.VLMConfig);
- `system_prompt_path`, `user_prompt_path`, `schema_path` - optional
  replacements of the prompts and schema in vlm/; the system prompt receives
  the variables `ocr` (the page's ALTO text) and `schema` (the schema as
  JSON), and the page image is attached to the user prompt;
- `confidence` - the confidence given to every value.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable, List, Optional

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

logger = logging.getLogger(__name__)

_RESOURCES = Path(__file__).parent / "vlm"
_CONFIG_KEYS = {"name", "vlm", "system_prompt_path", "user_prompt_path", "schema_path", "confidence"}
DEFAULT_CONFIDENCE = 0.95
# The CP-SAT selection is exact but may take long on a page with many
# candidates; past this limit the best selection found so far is used.
_SELECTION_TIME_LIMIT_SECONDS = 10.0


class BiblioCoreEngineVLM(BiblioCoreEngine):
    def __init__(self, config: Mapping[str, Any], client: Any = None):
        super().__init__(config=config)
        unknown = sorted(set(self.config) - _CONFIG_KEYS)
        if unknown:
            raise ValueError(f"Biblio core config has unknown keys: {', '.join(unknown)}")
        self.vlm_config = VLMConfig.from_config(self.config.get("vlm"), "Biblio core vlm config")
        self.prompts = load_prompts([
            {"role": "developer",
             "path": self.config.get("system_prompt_path") or str(_RESOURCES / "prompt_system.txt")},
            {"role": "user",
             "path": self.config.get("user_prompt_path") or str(_RESOURCES / "prompt_user.txt")},
        ])
        schema_path = self.config.get("schema_path") or str(_RESOURCES / "schema.json")
        self.schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        self.confidence = self.config.get("confidence", DEFAULT_CONFIDENCE)
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("Biblio core confidence must be a number from 0 to 1")
        self.client = VLMClient(self.vlm_config, client)
        self.aligner = _text_aligner()
        logger.info("Biblio core VLM: model %s at %s, response format %s",
                    self.vlm_config.model, self.vlm_config.api_url, self.vlm_config.response_format)

    def process(
        self,
        images: List[str],
        alto_files: List[str],
    ) -> BiblioCoreResult:
        alto_by_key = {Path(alto_file).stem: alto_file for alto_file in alto_files}
        pages = {}
        for image in images:
            page_key = Path(image).stem
            if page_key in pages:
                raise ValueError(f"Biblio core got duplicate page key: {page_key}")
            page_result = self.read_page(page_key, image, alto_by_key.get(page_key))
            if page_result is not None:
                pages[page_key] = page_result
        logger.info("Biblio core read bibliographic information on %d of %d page(s)",
                    len(pages), len(images))
        return BiblioCoreResult(pages=pages)

    def read_page(self, page_key: str, image: str, alto_file: Optional[str]) -> Optional[BiblioPageResult]:
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
            reply = self.client.request_json(
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

        page_result = page_from_reply(reply, evidence, page_key)
        found = sum(1 for region in regions.values() if region.alto_geometry is not None)
        logger.info("Page %s: located %d of %d value(s) in the ALTO", page_key, found, len(regions))
        return page_result

    def _locate(self, alto_page, reply: Mapping[str, Any]) -> dict:
        """The aligned region of every text value in the reply, by JSON path."""
        document = self.aligner.align_data(alto_page, _alignable(reply))
        return {
            tuple(region.json_text_path): region
            for page in document.pages
            for region in page.regions
            if region.json_text_path is not None
        }


def page_from_reply(reply: Mapping[str, Any], evidence, page_key: str) -> Optional[BiblioPageResult]:
    """Build a page result from the model's JSON.

    `evidence(path, text)` turns the text at a JSON path into evidence, or
    None for an empty value. Containers left without any value are dropped.
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


def _alignable(reply: Any) -> Any:
    """The reply with only its text values left to align.

    An agent's role is a label, not text on the page, and an empty string has
    nothing to find; both become None, which the aligner skips, so every other
    value keeps its JSON path.
    """
    if isinstance(reply, Mapping):
        return {key: None if key == "role" else _alignable(value) for key, value in reply.items()}
    if isinstance(reply, list):
        return [_alignable(value) for value in reply]
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

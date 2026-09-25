import logging
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple

from text_geometry_aligner import AlignmentPage

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
from metakat.common.engines.engine_yolo_alto import EngineYOLOALTO
from metakat.common.models import BoundingBox, DetectionEvidence
from metakat.schemas.base_objects import BiblioType

logger = logging.getLogger(__name__)

# Where each label's detections go: which reading of the page (see
# BiblioPageResult), which slot in it, and whether the slot keeps only the
# most confident detection on the page or every one, in region order.
_BEST, _ALL = "best", "all"
_READING, _VOLUME, _ISSUE = "reading", "periodical_volume", "periodical_issue"
_SLOTS: Dict[BiblioType, Tuple[str, str, str]] = {
    BiblioType.TITLE: (_READING, "title", _BEST),
    BiblioType.SUBTITLE: (_READING, "sub_title", _BEST),
    BiblioType.PART_NUMBER: (_READING, "part_number", _BEST),
    BiblioType.PART_NAME: (_READING, "part_name", _BEST),
    BiblioType.PLACE_TERM: (_READING, "places", _BEST),
    BiblioType.PUBLISHER: (_READING, "publishers", _ALL),
    BiblioType.DATE_ISSUED: (_READING, "date_issued", _BEST),
    BiblioType.EDITION: (_READING, "edition", _BEST),
    BiblioType.MANUFACTURE_PLACE_TERM: (_READING, "manufacture_places", _ALL),
    BiblioType.MANUFACTURE_PUBLISHER: (_READING, "manufacturers", _ALL),
    BiblioType.SERIES_NAME: (_READING, "series_names", _ALL),
    BiblioType.SERIES_NUMBER: (_READING, "series_numbers", _ALL),
    BiblioType.AUTHOR: (_READING, AgentRole.AUTHOR.value, _ALL),
    BiblioType.ILLUSTRATOR: (_READING, AgentRole.ILLUSTRATOR.value, _ALL),
    BiblioType.PHOTOGRAPHER: (_READING, AgentRole.PHOTOGRAPHER.value, _ALL),
    BiblioType.TRANSLATOR: (_READING, AgentRole.TRANSLATOR.value, _ALL),
    BiblioType.EDITOR: (_READING, AgentRole.EDITOR.value, _ALL),
    BiblioType.REDAKTOR: (_READING, AgentRole.REDAKTOR.value, _ALL),
    BiblioType.PERIODICAL_VOLUME_PART_NUMBER: (_VOLUME, "part_number", _BEST),
    BiblioType.PERIODICAL_VOLUME_DATE_ISSUED: (_VOLUME, "date_issued", _BEST),
    BiblioType.PERIODICAL_ISSUE_PART_NUMBER: (_ISSUE, "part_number", _BEST),
    BiblioType.PERIODICAL_ISSUE_DATE_ISSUED: (_ISSUE, "date_issued", _BEST),
}


def parse_biblio_labels(config: Mapping[str, Any]) -> Dict[BiblioType, str]:
    """The configured BiblioType -> model label mapping, validated."""
    if "id2label" in config:
        raise ValueError("id2label is not supported; use the labels mapping")
    configured_labels = config.get("labels")
    if not isinstance(configured_labels, dict) or not configured_labels:
        raise ValueError("labels must be a non-empty object")

    labels: Dict[BiblioType, str] = {}
    for raw_type, label in configured_labels.items():
        try:
            biblio_type = BiblioType(raw_type)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Unknown bibliographic label type: {raw_type!r}"
            ) from error
        if not isinstance(label, str) or not label.strip():
            raise ValueError(
                f"Label for {biblio_type.value!r} must be a non-empty string"
            )
        if label in labels.values():
            raise ValueError(
                f"Bibliographic model label is assigned more than once: {label!r}"
            )
        labels[biblio_type] = label
    return labels


class BiblioCoreEngineYOLO(BiblioCoreEngine):
    def __init__(self, config: Mapping[str, Any],
                 yolo_batch_size=32,
                 yolo_confidence_threshold=0.25,
                 yolo_image_size=640,
                 minimum_overlap_coverage=0.65):
        super().__init__(config=config)
        self.labels = parse_biblio_labels(self.config)
        self.biblio_type_by_label = {label: biblio_type for biblio_type, label in self.labels.items()}
        logger.info("Loaded %d bibliographic label(s)", len(self.labels))
        self.engine_yolo_alto = EngineYOLOALTO(
            config=self.config,
            yolo_batch_size=yolo_batch_size,
            yolo_confidence_threshold=yolo_confidence_threshold,
            yolo_image_size=yolo_image_size,
            minimum_overlap_coverage=minimum_overlap_coverage,
        )

    def process(
        self,
        images: List[str],
        alto_files: List[str],
    ) -> BiblioCoreResult:
        alignment_pages = self.engine_yolo_alto.process(
            images=images,
            alto_files=alto_files
        ).pages
        pages = {}
        for alignment_page in alignment_pages:
            if alignment_page.page_key in pages:
                raise ValueError(
                    f"Biblio core returned duplicate page key: {alignment_page.page_key}"
                )
            page_result = read_page(alignment_page, self.biblio_type_by_label)
            if page_result is not None:
                pages[alignment_page.page_key] = page_result
        logger.info(
            "Biblio core read bibliographic information on %d of %d page(s)",
            len(pages), len(alignment_pages),
        )
        return BiblioCoreResult(pages=pages)


def read_page(
    alignment_page: AlignmentPage,
    biblio_type_by_label: Mapping[str, BiblioType],
) -> Optional[BiblioPageResult]:
    """Group one aligned page's detections into its readings; None if it has none."""
    slots: Dict[str, Dict[str, Any]] = {_READING: {}, _VOLUME: {}, _ISSUE: {}}
    agents: List[BiblioAgent] = []
    for region in alignment_page.regions:
        evidence = _evidence(alignment_page.page_key, region)
        if evidence is None:
            continue
        model_label = region.label_for_export
        biblio_type = biblio_type_by_label.get(model_label)
        if biblio_type is None:
            logger.warning(
                "Model label %r (raw=%r, export=%r) not found in the engine "
                "configuration's labels mapping; skipping detection",
                model_label, region.label, region.label_export,
            )
            continue
        reading, slot, keep = _SLOTS[biblio_type]
        if reading == _READING and slot in AgentRole._value2member_map_:
            agents.append(BiblioAgent(role=AgentRole(slot), name=evidence))
        elif keep == _BEST:
            # A later detection replaces the kept one only when it is more
            # confident, so the first of equally confident ones stays.
            kept = slots[reading].get(slot)
            if kept is None or kept.confidence < evidence.confidence:
                slots[reading][slot] = evidence
        else:
            slots[reading].setdefault(slot, []).append(evidence)

    reading = _reading(slots[_READING], agents)
    periodical_volume = _reading(slots[_VOLUME], [])
    periodical_issue = _reading(slots[_ISSUE], [])
    if reading.is_empty() and periodical_volume.is_empty() and periodical_issue.is_empty():
        return None
    return BiblioPageResult(
        page_key=alignment_page.page_key,
        reading=reading,
        periodical_volume=None if periodical_volume.is_empty() else periodical_volume,
        periodical_issue=None if periodical_issue.is_empty() else periodical_issue,
    )


def _evidence(page_key: str, region) -> Optional[DetectionEvidence]:
    if not region.matched:
        return None
    if (
        region.input_geometry is None
        or region.input_geometry_confidence is None
        or region.alto_text is None
    ):
        logger.warning(
            "Matched region %s on page %s is missing YOLO metadata; skipping detection",
            region.region_id, page_key,
        )
        return None
    bounds = region.input_geometry.bounds
    return DetectionEvidence(
        text=region.alto_text,
        confidence=region.input_geometry_confidence,
        bbox=BoundingBox(bounds.x, bounds.y, bounds.width, bounds.height),
        page_key=page_key,
    )


def _reading(slots: Dict[str, Any], agents: List[BiblioAgent]) -> BiblioReading:
    # Everything one page prints about a record is read as one statement per
    # container type: one title, one imprint, one manufacture statement. The
    # detector does not say which place goes with which publisher, and one
    # container does not claim it either.
    title_info = BiblioTitleInfo(
        title=slots.get("title"),
        sub_title=slots.get("sub_title"),
        part_number=slots.get("part_number"),
        part_name=slots.get("part_name"),
    )
    publication = BiblioPublication(
        places=_as_tuple(slots.get("places")),
        publishers=_as_tuple(slots.get("publishers")),
        date_issued=slots.get("date_issued"),
        edition=slots.get("edition"),
    )
    manufacture = BiblioManufacture(
        places=_as_tuple(slots.get("manufacture_places")),
        manufacturers=_as_tuple(slots.get("manufacturers")),
    )
    return BiblioReading(
        title_infos=_if_any(title_info),
        publications=_if_any(publication),
        manufactures=_if_any(manufacture),
        series=_series(slots.get("series_names") or [], slots.get("series_numbers") or []),
        agents=tuple(agents),
    )


def _series(names: List[DetectionEvidence], numbers: List[DetectionEvidence]) -> tuple:
    # A <relatedItem type="series"> holds one series title. One name with at
    # most one number is one series statement; with several, which number
    # belongs to which name is unknown, so each is written on its own.
    if not names and not numbers:
        return ()
    if len(names) <= 1 and len(numbers) <= 1:
        return (BiblioSeries(name=names[0] if names else None,
                             part_number=numbers[0] if numbers else None),)
    return (
        *(BiblioSeries(name=name) for name in names),
        *(BiblioSeries(part_number=number) for number in numbers),
    )


def _as_tuple(value) -> tuple:
    if value is None:
        return ()
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _if_any(container) -> tuple:
    return (container,) if next(container_values(container), None) is not None else ()

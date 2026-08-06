from __future__ import annotations

import copy
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
from uuid import UUID, uuid4

from text_geometry_aligner import AlignmentPage

from metakat.page_number.engines.bind.page_number_bind_engine import (
    PageNumberBindEngine,
)
from metakat.page_number.engines.bind.page_number_parsers import (
    DecoratedPageNumberParser,
)
from metakat.schemas.base_objects import (
    DocumentType,
    MetakatElement,
    MetakatIO,
    MetakatPage,
    PageNumberType,
    ProarcIO,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PageNumberCandidate:
    text: str
    confidence: float
    detection_id: UUID
    bbox: tuple[float, float, float, float]

    @property
    def center_y(self) -> float:
        return self.bbox[1] + self.bbox[3] / 2


class PageNumberBindEngineBase(PageNumberBindEngine):
    DEFAULT_EDGE_BAND_RATIO = 0.15
    DEFAULT_EDGE_SCORE_WEIGHT = 0.65

    def __init__(self, bind_engine_dir: str, core_engine_dir: str):
        super().__init__(bind_engine_dir, core_engine_dir)
        self.edge_band_ratio = self._validated_config_number(
            "page_number_edge_band_ratio",
            self.DEFAULT_EDGE_BAND_RATIO,
            minimum=0.0,
            maximum=0.5,
            maximum_inclusive=False,
        )
        self.edge_score_weight = self._validated_config_number(
            "page_number_edge_score_weight",
            self.DEFAULT_EDGE_SCORE_WEIGHT,
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=True,
        )

    def _validated_config_number(
        self,
        key: str,
        default: float,
        *,
        minimum: float,
        maximum: float,
        minimum_inclusive: bool = False,
        maximum_inclusive: bool = True,
    ) -> float:
        value = self.config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a number")
        value = float(value)
        maximum_valid = (
            value <= maximum if maximum_inclusive else value < maximum
        )
        minimum_valid = (
            value >= minimum if minimum_inclusive else value > minimum
        )
        if not math.isfinite(value) or not minimum_valid or not maximum_valid:
            minimum_operator = ">=" if minimum_inclusive else ">"
            maximum_operator = "<=" if maximum_inclusive else "<"
            raise ValueError(
                f"{key} must satisfy value {minimum_operator} {minimum} "
                f"and value {maximum_operator} {maximum}"
            )
        return value

    def process(
        self,
        batch_dir: str,
        metakat_io: MetakatIO,
        proarc_io: ProarcIO = None,
    ) -> MetakatIO:
        metakat_io = copy.deepcopy(metakat_io)
        pages = sorted(
            (
                element
                for element in metakat_io.elements
                if element.type == DocumentType.PAGE.value
            ),
            key=lambda page: page.batch_index,
        )
        image_mapping = metakat_io.page_to_image_mapping or {}
        alto_mapping = metakat_io.page_to_alto_mapping or {}
        processable_pages = [
            page
            for page in pages
            if page.id in image_mapping and page.id in alto_mapping
        ]
        skipped = len(pages) - len(processable_pages)
        if skipped:
            logger.warning(
                "Skipping %d page(s) without both image and ALTO mappings",
                skipped,
            )

        images = [
            os.path.join(batch_dir, image_mapping[page.id])
            for page in processable_pages
        ]
        alto_files = [
            os.path.join(batch_dir, alto_mapping[page.id])
            for page in processable_pages
        ]
        logger.info(
            "Processing %d images with page number core engine",
            len(images),
        )
        alignment_pages = self.core_engine.process(images, alto_files)
        logger.info(
            "Page number core engine returned %d matched detections",
            sum(page.matched_count for page in alignment_pages),
        )

        page_by_key = self._page_by_image_key(pages, image_mapping)
        _, bbox_by_id, page_by_detection = (
            self.extract_metakat_elements_from_alignment(
                alignment_pages,
                page_by_key,
            )
        )
        if metakat_io.detection_to_bbox is None:
            metakat_io.detection_to_bbox = {}
        if metakat_io.detection_to_page_mapping is None:
            metakat_io.detection_to_page_mapping = {}
        metakat_io.detection_to_bbox.update(bbox_by_id)
        metakat_io.detection_to_page_mapping.update(page_by_detection)
        return metakat_io

    @staticmethod
    def _page_by_image_key(
        pages: List[MetakatPage],
        image_mapping: dict,
    ) -> dict[str, MetakatPage]:
        result: dict[str, MetakatPage] = {}
        page_by_id = {page.id: page for page in pages}
        for page_id, image_filename in image_mapping.items():
            if page_id not in page_by_id:
                continue
            page_key = Path(image_filename).stem
            if page_key in result:
                raise ValueError(
                    f"Page image mappings must have unique stems: {page_key}"
                )
            result[page_key] = page_by_id[page_id]
        return result

    def extract_metakat_elements_from_alignment(
        self,
        alignment_pages: List[AlignmentPage],
        alignment_page_key_to_metakat_page: dict,
    ) -> Tuple[List[MetakatElement], dict, dict]:
        elements = []
        detection_id_to_detection_bbox = {}
        detection_id_to_page_id = {}
        for alignment_page in alignment_pages:
            metakat_page = alignment_page_key_to_metakat_page[
                alignment_page.page_key
            ]
            page_elements, page_id_to_detection_bbox = (
                self.get_metakat_elements_from_page(
                    alignment_page,
                    metakat_page,
                )
            )
            elements.extend(page_elements)
            detection_id_to_detection_bbox.update(page_id_to_detection_bbox)
            for detection_id in page_id_to_detection_bbox:
                detection_id_to_page_id[detection_id] = metakat_page.id
        return (
            elements,
            detection_id_to_detection_bbox,
            detection_id_to_page_id,
        )

    def get_metakat_elements_from_page(
        self,
        alignment_page: AlignmentPage,
        metakat_page: MetakatPage,
    ) -> Tuple[List[MetakatElement], dict]:
        elements = []
        detection_id_to_detection_bbox = {}
        candidates: list[_PageNumberCandidate] = []
        for region in alignment_page.regions:
            if not region.matched:
                continue
            if (
                region.category_id is None
                or region.input_geometry is None
                or region.input_geometry_confidence is None
                or region.alto_text is None
            ):
                logger.warning(
                    "Matched region %s on page %s is missing YOLO metadata; "
                    "skipping detection",
                    region.region_id,
                    alignment_page.page_key,
                )
                continue

            class_id = str(region.category_id)
            bbox = region.input_geometry.bounds
            detection_bbox = (bbox.x, bbox.y, bbox.width, bbox.height)
            detection_id = uuid4()
            detection_text = region.alto_text
            detection_confidence = region.input_geometry_confidence

            if class_id not in self.core_engine.id2label:
                logger.warning(
                    "CLASS_ID %s (label=%r, label_export=%r) not in "
                    "id2label, skipping - val: %s, conf: %s, bbox: %s, "
                    "page_key: %s",
                    class_id,
                    region.label,
                    region.label_export,
                    detection_text,
                    detection_confidence,
                    detection_bbox,
                    alignment_page.page_key,
                )
                continue
            page_number_type = PageNumberType(
                self.core_engine.id2label[class_id]
            )
            if page_number_type != PageNumberType.PAGE_NUMBER:
                continue

            detection_text_parsed = DecoratedPageNumberParser.parse(
                detection_text
            )
            if not detection_text_parsed:
                logger.warning(
                    "Invalid PAGE_NUMBER, skipping - val: %s, conf: %s, "
                    "bbox: %s, page_key: %s",
                    detection_text,
                    detection_confidence,
                    detection_bbox,
                    alignment_page.page_key,
                )
                continue
            candidates.append(
                _PageNumberCandidate(
                    text=detection_text_parsed,
                    confidence=detection_confidence,
                    detection_id=detection_id,
                    bbox=detection_bbox,
                )
            )
            detection_id_to_detection_bbox[detection_id] = detection_bbox

        selected = self._select_page_number_candidate(
            candidates,
            alignment_page,
        )
        if selected is not None and (
            metakat_page.pageNumber is None
            or metakat_page.pageNumber[1] < selected.confidence
        ):
            metakat_page.pageNumber = (
                selected.text,
                selected.confidence,
                selected.detection_id,
            )
        return elements, detection_id_to_detection_bbox

    def _select_page_number_candidate(
        self,
        candidates: list[_PageNumberCandidate],
        alignment_page: AlignmentPage,
    ) -> _PageNumberCandidate | None:
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        page_height = alignment_page.alto_height
        if (
            page_height is None
            or not math.isfinite(page_height)
            or page_height <= 0
        ):
            selected = max(candidates, key=lambda candidate: candidate.confidence)
            logger.warning(
                "Page %s has %d valid PAGE_NUMBER candidates but no usable "
                "ALTO page height; selecting highest-confidence candidate %r "
                "(confidence=%.4f)",
                alignment_page.page_key,
                len(candidates),
                selected.text,
                selected.confidence,
            )
            return selected

        positioned_candidates: list[
            tuple[_PageNumberCandidate, float, float]
        ] = []
        for candidate in candidates:
            normalized_center_y = candidate.center_y / page_height
            if not math.isfinite(normalized_center_y):
                continue
            edge_distance = min(
                normalized_center_y,
                1.0 - normalized_center_y,
            )
            if 0.0 <= edge_distance <= self.edge_band_ratio:
                edge_score = 1.0 - edge_distance / self.edge_band_ratio
                positioned_candidates.append(
                    (candidate, edge_distance, edge_score)
                )

        candidate_summary = self._format_candidate_summary(
            candidates,
            page_height,
        )
        if not positioned_candidates:
            logger.warning(
                "Page %s has %d valid PAGE_NUMBER candidates but none in "
                "the top/bottom %.1f%% bands; leaving page number "
                "unresolved. Candidates: %s",
                alignment_page.page_key,
                len(candidates),
                self.edge_band_ratio * 100,
                candidate_summary,
            )
            return None

        def selection_key(
            positioned: tuple[_PageNumberCandidate, float, float],
        ) -> tuple[float, float, float]:
            candidate, edge_distance, edge_score = positioned
            combined_score = (
                self.edge_score_weight * edge_score
                + (1.0 - self.edge_score_weight) * candidate.confidence
            )
            return combined_score, candidate.confidence, -edge_distance

        selected, selected_edge_distance, selected_edge_score = max(
            positioned_candidates,
            key=selection_key,
        )
        selected_score = (
            self.edge_score_weight * selected_edge_score
            + (1.0 - self.edge_score_weight) * selected.confidence
        )
        logger.info(
            "Page %s has %d valid PAGE_NUMBER candidates; selected %r "
            "from an edge band (confidence=%.4f, edge_distance=%.4f, "
            "combined_score=%.4f). Candidates: %s",
            alignment_page.page_key,
            len(candidates),
            selected.text,
            selected.confidence,
            selected_edge_distance,
            selected_score,
            candidate_summary,
        )
        return selected

    @staticmethod
    def _format_candidate_summary(
        candidates: list[_PageNumberCandidate],
        page_height: float,
    ) -> str:
        return ", ".join(
            (
                f"text={candidate.text!r} confidence={candidate.confidence:.4f} "
                f"center_y_ratio={candidate.center_y / page_height:.4f} "
                f"bbox={candidate.bbox!r}"
            )
            for candidate in candidates
        )

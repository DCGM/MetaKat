from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping

from text_geometry_aligner import AlignmentPage, AlignmentRegion

from metakat.common.models import BoundingBox
from metakat.page_number.engines.core.models import (
    PhysicalPageNumberEvidence,
)
from metakat.page_number.engines.core.page_number_parsers import (
    DecoratedPageNumberParser,
)


logger = logging.getLogger(__name__)


class PageNumberSelectionMode(str, Enum):
    STANDARD = "standard"
    EDGE_ONLY = "edge_only"


@dataclass(frozen=True)
class PageNumberCandidate:
    region_id: int
    evidence: PhysicalPageNumberEvidence

    @property
    def center_y(self) -> float:
        return self.evidence.bbox.y + self.evidence.bbox.height / 2


@dataclass(frozen=True)
class PageNumberResolution:
    candidates: tuple[PageNumberCandidate, ...]
    selected: PageNumberCandidate | None

    @property
    def selected_evidence(self) -> PhysicalPageNumberEvidence | None:
        if self.selected is None:
            return None
        return self.selected.evidence


class PhysicalPageNumberResolver:
    DEFAULT_EDGE_BAND_RATIO = 0.15
    DEFAULT_EDGE_SCORE_WEIGHT = 0.65

    def __init__(
        self,
        *,
        edge_band_ratio: float = DEFAULT_EDGE_BAND_RATIO,
        edge_score_weight: float = DEFAULT_EDGE_SCORE_WEIGHT,
    ) -> None:
        self.edge_band_ratio = self._validated_number(
            "page_number_edge_band_ratio",
            edge_band_ratio,
            minimum=0.0,
            maximum=0.5,
            maximum_inclusive=False,
        )
        self.edge_score_weight = self._validated_number(
            "page_number_edge_score_weight",
            edge_score_weight,
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=True,
        )

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> PhysicalPageNumberResolver:
        return cls(
            edge_band_ratio=config.get(
                "page_number_edge_band_ratio",
                cls.DEFAULT_EDGE_BAND_RATIO,
            ),
            edge_score_weight=config.get(
                "page_number_edge_score_weight",
                cls.DEFAULT_EDGE_SCORE_WEIGHT,
            ),
        )

    def resolve(
        self,
        page: AlignmentPage,
        regions: Iterable[AlignmentRegion],
        *,
        mode: PageNumberSelectionMode = PageNumberSelectionMode.STANDARD,
        page_height: float | None = None,
    ) -> PageNumberResolution:
        candidates = tuple(
            candidate
            for region in regions
            if (candidate := self._candidate(page.page_key, region))
            is not None
        )
        selected = self._select(
            candidates,
            page_key=page.page_key,
            page_height=(
                page.alto_height if page_height is None else page_height
            ),
            mode=mode,
        )
        return PageNumberResolution(candidates, selected)

    @staticmethod
    def _candidate(
        page_key: str,
        region: AlignmentRegion,
    ) -> PageNumberCandidate | None:
        if not region.matched:
            return None
        if (
            region.input_geometry is None
            or region.input_geometry_confidence is None
            or region.alto_text is None
        ):
            logger.warning(
                "Matched page-number region %s on page %s is missing "
                "geometry, confidence, or ALTO text; skipping detection",
                region.region_id,
                page_key,
            )
            return None

        evidence = DecoratedPageNumberParser.create(
            page_key=page_key,
            text=region.alto_text,
            confidence=region.input_geometry_confidence,
            bbox=BoundingBox(
                x=region.input_geometry.bounds.x,
                y=region.input_geometry.bounds.y,
                width=region.input_geometry.bounds.width,
                height=region.input_geometry.bounds.height,
            ),
        )
        if evidence.normalized is None:
            logger.warning(
                "Invalid PAGE_NUMBER, skipping - val: %s, conf: %s, "
                "bbox: %s, page_key: %s",
                region.alto_text,
                region.input_geometry_confidence,
                region.input_geometry.bounds,
                page_key,
            )
            return None
        return PageNumberCandidate(
            region_id=region.region_id,
            evidence=evidence,
        )

    def _select(
        self,
        candidates: tuple[PageNumberCandidate, ...],
        *,
        page_key: str,
        page_height: float | None,
        mode: PageNumberSelectionMode,
    ) -> PageNumberCandidate | None:
        if not candidates:
            return None
        if len(candidates) == 1 and mode is PageNumberSelectionMode.STANDARD:
            return candidates[0]

        usable_height = (
            page_height is not None
            and math.isfinite(page_height)
            and page_height > 0
        )
        if not usable_height:
            if mode is PageNumberSelectionMode.EDGE_ONLY:
                logger.warning(
                    "Page %s requires edge-only page-number selection but "
                    "has no usable page height; leaving page number unresolved",
                    page_key,
                )
                return None
            selected = max(
                candidates,
                key=lambda candidate: candidate.evidence.confidence,
            )
            logger.warning(
                "Page %s has %d valid PAGE_NUMBER candidates but no usable "
                "page height; selecting highest-confidence candidate %r "
                "(confidence=%.4f)",
                page_key,
                len(candidates),
                selected.evidence.output_text(),
                selected.evidence.confidence,
            )
            return selected

        assert page_height is not None
        positioned: list[tuple[PageNumberCandidate, float, float]] = []
        for candidate in candidates:
            normalized_center_y = candidate.center_y / page_height
            if not math.isfinite(normalized_center_y):
                continue
            edge_distance = min(normalized_center_y, 1.0 - normalized_center_y)
            if 0.0 <= edge_distance <= self.edge_band_ratio:
                edge_score = 1.0 - edge_distance / self.edge_band_ratio
                positioned.append((candidate, edge_distance, edge_score))

        summary = self._format_candidate_summary(candidates, page_height)
        if not positioned:
            logger.warning(
                "Page %s has %d valid PAGE_NUMBER candidates but none in "
                "the top/bottom %.1f%% bands; leaving page number unresolved. "
                "Selection mode=%s. Candidates: %s",
                page_key,
                len(candidates),
                self.edge_band_ratio * 100,
                mode.value,
                summary,
            )
            return None

        def selection_key(
            item: tuple[PageNumberCandidate, float, float],
        ) -> tuple[float, float, float]:
            candidate, edge_distance, edge_score = item
            combined_score = (
                self.edge_score_weight * edge_score
                + (1.0 - self.edge_score_weight)
                * candidate.evidence.confidence
            )
            return (
                combined_score,
                candidate.evidence.confidence,
                -edge_distance,
            )

        selected, edge_distance, edge_score = max(
            positioned,
            key=selection_key,
        )
        combined_score = (
            self.edge_score_weight * edge_score
            + (1.0 - self.edge_score_weight)
            * selected.evidence.confidence
        )
        logger.info(
            "Page %s has %d valid PAGE_NUMBER candidate(s); selected %r "
            "from an edge band (mode=%s, confidence=%.4f, "
            "edge_distance=%.4f, combined_score=%.4f). Candidates: %s",
            page_key,
            len(candidates),
            selected.evidence.output_text(),
            mode.value,
            selected.evidence.confidence,
            edge_distance,
            combined_score,
            summary,
        )
        return selected

    @staticmethod
    def _format_candidate_summary(
        candidates: tuple[PageNumberCandidate, ...],
        page_height: float,
    ) -> str:
        return ", ".join(
            f"text={candidate.evidence.text!r} "
            f"normalized={candidate.evidence.normalized!r} "
            f"confidence={candidate.evidence.confidence:.4f} "
            f"center_y_ratio={candidate.center_y / page_height:.4f} "
            f"bbox={candidate.evidence.bbox!r}"
            for candidate in candidates
        )

    @staticmethod
    def _validated_number(
        key: str,
        value: object,
        *,
        minimum: float,
        maximum: float,
        minimum_inclusive: bool = False,
        maximum_inclusive: bool = True,
    ) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a number")
        value = float(value)
        minimum_valid = (
            value >= minimum if minimum_inclusive else value > minimum
        )
        maximum_valid = (
            value <= maximum if maximum_inclusive else value < maximum
        )
        if not math.isfinite(value) or not minimum_valid or not maximum_valid:
            minimum_operator = ">=" if minimum_inclusive else ">"
            maximum_operator = "<=" if maximum_inclusive else "<"
            raise ValueError(
                f"{key} must satisfy value {minimum_operator} {minimum} "
                f"and value {maximum_operator} {maximum}"
            )
        return value

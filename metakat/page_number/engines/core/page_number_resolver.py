from __future__ import annotations

import logging
import math
from enum import Enum
from typing import Iterable, Mapping

from metakat.page_number.engines.core.models import (
    PhysicalPageNumberEvidence,
)


logger = logging.getLogger(__name__)


class PageNumberSelectionMode(str, Enum):
    STANDARD = "standard"
    EDGE_ONLY = "edge_only"


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
        candidates: Iterable[PhysicalPageNumberEvidence],
        *,
        page_width: float | None,
        page_height: float | None,
        mode: PageNumberSelectionMode = PageNumberSelectionMode.STANDARD,
    ) -> PhysicalPageNumberEvidence | None:
        return self._select(
            tuple(candidates),
            page_width=page_width,
            page_height=page_height,
            mode=mode,
        )

    def _select(
        self,
        candidates: tuple[PhysicalPageNumberEvidence, ...],
        *,
        page_width: float | None,
        page_height: float | None,
        mode: PageNumberSelectionMode,
    ) -> PhysicalPageNumberEvidence | None:
        if not candidates:
            return None

        page_key = candidates[0].page_key
        if any(candidate.page_key != page_key for candidate in candidates[1:]):
            page_keys = tuple(
                dict.fromkeys(candidate.page_key for candidate in candidates)
            )
            raise ValueError(
                "Physical page-number selection requires all candidates "
                f"to belong to the same page; got page keys {page_keys!r}"
            )

        if (
            page_width is None
            or not math.isfinite(page_width)
            or page_width <= 0
        ):
            raise ValueError(
                "Physical page-number selection requires a finite positive "
                f"page width for page {page_key!r}; got {page_width!r}"
            )
        if (
            page_height is None
            or not math.isfinite(page_height)
            or page_height <= 0
        ):
            raise ValueError(
                "Physical page-number selection requires a finite positive "
                f"page height for page {page_key!r}; got {page_height!r}"
            )

        contained = tuple(
            candidate
            for candidate in candidates
            if self._is_inside_page(
                candidate,
                page_width=page_width,
                page_height=page_height,
            )
        )
        if not contained:
            logger.warning(
                "Page %s has %d valid PAGE_NUMBER candidate(s), but none "
                "has a finite positive-area bounding box contained within "
                "the page; leaving page number unresolved. Candidates: %s",
                page_key,
                len(candidates),
                self._format_candidate_summary(candidates, page_height),
            )
            return None
        if len(contained) != len(candidates):
            logger.warning(
                "Page %s discarded %d of %d valid PAGE_NUMBER candidate(s) "
                "whose bounding boxes were not contained within the page",
                page_key,
                len(candidates) - len(contained),
                len(candidates),
            )
        if len(contained) == 1 and mode is PageNumberSelectionMode.STANDARD:
            return contained[0]

        assert page_height is not None
        positioned: list[
            tuple[PhysicalPageNumberEvidence, float, float]
        ] = []
        for candidate in contained:
            center_y = candidate.bbox.y + candidate.bbox.height / 2
            normalized_center_y = center_y / page_height
            if not math.isfinite(normalized_center_y):
                continue
            edge_distance = min(normalized_center_y, 1.0 - normalized_center_y)
            if 0.0 <= edge_distance <= self.edge_band_ratio:
                edge_score = 1.0 - edge_distance / self.edge_band_ratio
                positioned.append((candidate, edge_distance, edge_score))

        summary = self._format_candidate_summary(contained, page_height)
        if not positioned:
            logger.warning(
                "Page %s has %d valid PAGE_NUMBER candidates but none in "
                "the top/bottom %.1f%% bands; leaving page number unresolved. "
                "Selection mode=%s. Candidates: %s",
                page_key,
                len(contained),
                self.edge_band_ratio * 100,
                mode.value,
                summary,
            )
            return None

        def selection_key(
            item: tuple[PhysicalPageNumberEvidence, float, float],
        ) -> tuple[float, float, float]:
            candidate, edge_distance, edge_score = item
            combined_score = (
                self.edge_score_weight * edge_score
                + (1.0 - self.edge_score_weight)
                * candidate.confidence
            )
            return (
                combined_score,
                candidate.confidence,
                -edge_distance,
            )

        selected, edge_distance, edge_score = max(
            positioned,
            key=selection_key,
        )
        combined_score = (
            self.edge_score_weight * edge_score
            + (1.0 - self.edge_score_weight)
            * selected.confidence
        )
        logger.info(
            "Page %s has %d valid PAGE_NUMBER candidate(s); selected %r "
            "from an edge band (mode=%s, confidence=%.4f, "
            "edge_distance=%.4f, combined_score=%.4f). Candidates: %s",
            page_key,
            len(contained),
            selected.output_text(),
            mode.value,
            selected.confidence,
            edge_distance,
            combined_score,
            summary,
        )
        return selected

    @staticmethod
    def _is_inside_page(
        candidate: PhysicalPageNumberEvidence,
        *,
        page_width: float,
        page_height: float,
    ) -> bool:
        bbox = candidate.bbox
        return (
            all(
                math.isfinite(value)
                for value in (bbox.x, bbox.y, bbox.width, bbox.height)
            )
            and bbox.width > 0
            and bbox.height > 0
            and bbox.x >= 0
            and bbox.y >= 0
            and bbox.x_max <= page_width
            and bbox.y_max <= page_height
        )

    @staticmethod
    def _format_candidate_summary(
        candidates: tuple[PhysicalPageNumberEvidence, ...],
        page_height: float,
    ) -> str:
        return ", ".join(
            f"text={candidate.text!r} "
            f"normalized={candidate.normalized!r} "
            f"confidence={candidate.confidence:.4f} "
            "center_y_ratio="
            f"{(candidate.bbox.y + candidate.bbox.height / 2) / page_height:.4f} "
            f"bbox={candidate.bbox!r}"
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

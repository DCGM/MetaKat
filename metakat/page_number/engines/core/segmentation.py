"""Infer numbering regimes before document-level page-number processing.

A regime boundary is evidence for separate processing, not proof of a separate
bibliographic document. In particular, Roman front matter may share a book
with an Arabic body. No recovered or supplied number participates in this fit.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import median
from typing import TYPE_CHECKING, Any

from metakat.page_number.engines.core.models import (
    PageNumberChunk,
    PhysicalPageNumberEvidence,
    valid_detected_page_number,
)

if TYPE_CHECKING:
    from metakat.page_number.engines.core.postprocessing import (
        PageNumberPostprocessInput,
    )

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageNumberSegmentationConfig:
    enabled: bool = True
    minimum_run_anchors: int = 3
    maximum_anchor_gap: int = 12
    minimum_offset_change: int = 4
    maximum_number_step: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("segmentation.enabled must be boolean")
        for name in (
            "minimum_run_anchors",
            "maximum_anchor_gap",
            "minimum_offset_change",
            "maximum_number_step",
        ):
            value = getattr(self, name)
            minimum = 3 if name == "minimum_run_anchors" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"Invalid segmentation.{name}")

    @classmethod
    def from_config(cls, raw: Mapping[str, Any]) -> PageNumberSegmentationConfig:
        if not isinstance(raw, Mapping):
            raise ValueError("postprocess.segmentation must be an object")
        if set(raw) - cls.__dataclass_fields__.keys():
            raise ValueError("Unknown postprocess.segmentation settings")
        return cls(**raw)


@dataclass(frozen=True)
class _Anchor:
    scan_index: int
    logical_index: int
    evidence: PhysicalPageNumberEvidence


class PageNumberChunker:
    def __init__(self, config: PageNumberSegmentationConfig | None = None):
        self.config = config or PageNumberSegmentationConfig()

    def split(
        self,
        pages: Sequence[PageNumberPostprocessInput],
        *,
        reason: str = "input_start",
    ) -> tuple[PageNumberChunk, ...]:
        """Split one explicitly bounded unit; preserve every scan in input order."""
        pages = tuple(pages)
        if not pages:
            return ()
        if len({page.page_key for page in pages}) != len(pages):
            raise ValueError("Numbering chunker requires unique page keys")
        if len({page.pagination.unit_key for page in pages}) != 1:
            raise ValueError("Numbering chunker expects one explicit unit at a time")
        if not self.config.enabled:
            return (self._chunk(pages, reason, False),)
        anchors = self._anchors(pages)
        runs: list[list[_Anchor]] = []
        current: list[_Anchor] = []
        for anchor in anchors:
            if current and not self._compatible(current[-1], anchor):
                if len(current) >= self.config.minimum_run_anchors:
                    runs.append(current)
                current = []
            current.append(anchor)
        if len(current) >= self.config.minimum_run_anchors:
            runs.append(current)

        # Known same-page segment relationships override inferred cuts.
        positions = {page.page_key: i for i, page in enumerate(pages)}
        linked_spans = [
            sorted((i, positions[page.pagination.segment_of]))
            for i, page in enumerate(pages)
            if page.pagination.segment_of in positions
        ]
        boundaries = []
        for left, right in zip(runs, runs[1:]):
            boundary_reason = self._boundary_reason(left, right)
            if boundary_reason is None:
                continue
            end_left = left[-1].scan_index + 1
            start_right = right[0].scan_index
            adjusted = self._respect_relationships(end_left, start_right, linked_spans)
            if adjusted is None:
                logger.debug(
                    "Ignoring inferred numbering boundary inside a known segment relationship"
                )
                continue
            end_left, start_right = adjusted
            boundaries.append((end_left, start_right, boundary_reason))
            logger.info(
                "Inferred numbering boundary: last_left=%s first_right=%s reason=%s anchors=%d/%d",
                pages[end_left - 1].page_key,
                pages[start_right].page_key,
                boundary_reason,
                len(left),
                len(right),
            )

        chunks = []
        start = 0
        next_reason = reason
        for end_left, start_right, boundary_reason in boundaries:
            chunks.append(self._chunk(pages[start:end_left], next_reason, True))
            if end_left < start_right:
                # No evidence says whether intervening scans are the first
                # document's tail or the second one's front matter. Do not use
                # either position distribution or NDK edge numbering on them.
                chunks.append(
                    self._chunk(
                        pages[end_left:start_right],
                        "uncertain_boundary",
                        True,
                        ambiguous=True,
                    )
                )
            start = start_right
            next_reason = boundary_reason
        chunks.append(self._chunk(pages[start:], next_reason, bool(boundaries)))
        return tuple(chunks)

    @staticmethod
    def _respect_relationships(end_left, start_right, linked_spans):
        # A known child in an otherwise ambiguous gap belongs with its parent.
        # Expand that side's scope before leaving the remaining gap unresolved.
        for _ in range(len(linked_spans) + 1):
            previous = end_left, start_right
            for a, b in linked_spans:
                if a < end_left <= b:
                    if b >= start_right:
                        return None
                    end_left = b + 1
                if a < start_right <= b:
                    if a < end_left:
                        return None
                    start_right = a
            if previous == (end_left, start_right):
                return previous
        return None

    def _anchors(self, pages):
        anchors = []
        logical_index = 0
        for i, page in enumerate(pages):
            if page.pagination.excluded or page.pagination.segment_of is not None:
                continue
            evidence = page.selected
            if evidence is not None and self._valid(evidence, page):
                anchors.append(_Anchor(i, logical_index, evidence))
            logical_index += 1
        return anchors

    @staticmethod
    def _valid(evidence, page):
        return (
            valid_detected_page_number(
                evidence,
                page_key=page.page_key,
                dimensions=page.dimensions,
            )
            and evidence.value > 0
        )

    def _compatible(self, left: _Anchor, right: _Anchor) -> bool:
        distance = right.logical_index - left.logical_index
        return (
            left.evidence.numeral_system == right.evidence.numeral_system
            and distance <= self.config.maximum_anchor_gap
            and 0
            < right.evidence.value - left.evidence.value
            <= self.config.maximum_number_step * distance
        )

    def _boundary_reason(self, left, right):
        a, b = left[-1], right[0]
        if a.evidence.numeral_system != b.evidence.numeral_system:
            return "numeral_system_change"
        difference = b.evidence.value - a.evidence.value
        if difference <= 0:
            return "numbering_restart"
        slopes = [
            (q.evidence.value - p.evidence.value) / (q.logical_index - p.logical_index)
            for run in (left, right)
            for p, q in zip(run, run[1:])
        ]
        expected = median(slopes) * (b.logical_index - a.logical_index)
        if difference - expected >= self.config.minimum_offset_change:
            return "numbering_offset_change"
        return None

    @staticmethod
    def _chunk(pages, reason, inferred, *, ambiguous=False):
        return PageNumberChunk(
            tuple(page.page_key for page in pages),
            reason,
            inferred=inferred,
            ambiguous=ambiguous,
            unit_key=pages[0].pagination.unit_key,
        )

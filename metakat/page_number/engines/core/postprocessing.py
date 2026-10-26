"""Document-level page-number reasoning, independent of detection and OCR backends.

Coordinates use each page's own dimensions. Recovery is a callback, so ALTO,
image-crop OCR, or a future detector can supply evidence through the same API.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import groupby
from typing import Any

from metakat.common.models import BoundingBox, PageDimensions
from metakat.page_number.engines.core.ndk_pagination import (
    NdkPageContext,
    NdkPaginationConfig,
    NdkPaginator,
)
from metakat.page_number.engines.core.models import (
    PageNumberCoreResult,
    PageNumberDecision,
    PageNumberNumeralSystem,
    PhysicalPageNumberEvidence,
    format_page_number,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageNumberPostprocessInput:
    page_key: str
    dimensions: PageDimensions
    candidates: tuple[PhysicalPageNumberEvidence, ...] = ()
    # A detector's initial selection seeds the position distribution. All other
    # candidates remain available for reconsideration.
    selected: PhysicalPageNumberEvidence | None = None
    pagination: NdkPageContext = field(default_factory=NdkPageContext)


Recovery = Callable[
    [PageNumberPostprocessInput, tuple[BoundingBox, ...]],
    Sequence[PhysicalPageNumberEvidence],
]


@dataclass(frozen=True)
class PageNumberPostprocessConfig:
    enabled: bool = True
    position_bandwidth_x: float = 0.04
    position_bandwidth_y: float = 0.025
    minimum_support: int = 3
    position_window: int = 40
    sequence_window: int = 8
    drop_threshold: float = 0.1
    ndk: NdkPaginationConfig = field(default_factory=NdkPaginationConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.ndk, NdkPaginationConfig):
            raise ValueError("postprocess.ndk must be an NdkPaginationConfig")
        if not isinstance(self.enabled, bool):
            raise ValueError("postprocess.enabled must be boolean")
        for name in ("position_bandwidth_x", "position_bandwidth_y"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 < value < 0.5
            ):
                raise ValueError(f"postprocess.{name} must be between 0 and 0.5")
        for name in ("minimum_support", "position_window", "sequence_window"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (3 if name == "minimum_support" else 1)
            ):
                raise ValueError(f"Invalid postprocess.{name}")
        if (
            isinstance(self.drop_threshold, bool)
            or not isinstance(self.drop_threshold, (int, float))
            or not math.isfinite(self.drop_threshold)
            or not 0 <= self.drop_threshold <= 1
        ):
            raise ValueError("postprocess.drop_threshold must be in [0, 1]")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> PageNumberPostprocessConfig:
        raw = config.get("postprocess", {})
        if not isinstance(raw, Mapping):
            raise ValueError("postprocess must be an object")
        unknown = set(raw) - cls.__dataclass_fields__.keys()
        if unknown:
            raise ValueError(f"Unknown postprocess settings: {sorted(unknown)}")
        values = dict(raw)
        if "ndk" in values:
            values["ndk"] = NdkPaginationConfig.from_config(values["ndk"])
        return cls(**values)


class _PositionDistribution:
    """A local kernel mixture with independent modes and optional scan parity.

    One seed per page prevents pages with many detections dominating the fit.
    A page never supplies its own positional support. The distribution is fit
    once, before recovery, so recovered candidates cannot reinforce themselves.
    """

    def __init__(
        self,
        pages: Sequence[PageNumberPostprocessInput],
        config: PageNumberPostprocessConfig,
    ) -> None:
        self.pages = pages
        self.config = config
        self._modes = {}
        self._peaks = {}
        self.seeds = {
            i: page.selected
            for i, page in enumerate(pages)
            if page.selected is not None and _valid(page.selected, page)
        }

    def center(
        self, evidence: PhysicalPageNumberEvidence, index: int
    ) -> tuple[float, float]:
        bbox = evidence.bbox
        assert bbox is not None
        dimensions = self.pages[index].dimensions
        return (
            (bbox.x + bbox.width / 2) / dimensions.width,
            (bbox.y + bbox.height / 2) / dimensions.height,
        )

    def distance(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        return ((a[0] - b[0]) / self.config.position_bandwidth_x) ** 2 + (
            (a[1] - b[1]) / self.config.position_bandwidth_y
        ) ** 2

    def modes(
        self, index: int
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        if index in self._modes:
            return self._modes[index]
        seeds = [
            (j, self.center(e, j))
            for j, e in self.seeds.items()
            if j != index and abs(j - index) <= self.config.position_window
        ]
        # Prefer parity-specific modes when there is enough evidence. This
        # keeps an outer-corner folio on the wrong side from appearing typical.
        same_parity = [(j, p) for j, p in seeds if j % 2 == index % 2]
        for group in (same_parity, seeds):
            modes = [
                p
                for _, p in group
                if sum(self.distance(p, q) <= 2 for _, q in group)
                >= self.config.minimum_support
            ]
            if modes:
                stable = [
                    p
                    for _, p in group
                    if any(self.distance(p, mode) <= 2 for mode in modes)
                ]
                self._modes[index] = modes, stable
                return self._modes[index]
        self._modes[index] = [], []
        return self._modes[index]

    def support(self, evidence: PhysicalPageNumberEvidence, index: int) -> float | None:
        modes, stable = self.modes(index)
        if not modes:
            return None
        center = self.center(evidence, index)
        # Only stable modes contribute, excluding isolated outliers.
        density = sum(
            math.exp(-0.5 * self.distance(center, p))
            for p in stable
            if self.distance(center, p) <= 9
        )
        # Normalize by the peak, rather than by a fixed count. Otherwise many
        # seeds would make even positions in the distribution's tails certain.
        if index not in self._peaks:
            self._peaks[index] = max(
                sum(
                    math.exp(-0.5 * self.distance(mode, p))
                    for p in stable
                    if self.distance(mode, p) <= 9
                )
                for mode in modes
            )
        return min(1.0, density / self._peaks[index])

    def windows(self, index: int) -> tuple[BoundingBox, ...]:
        modes, _ = self.modes(index)
        dimensions = self.pages[index].dimensions
        result = []
        for x, y in modes:
            if any(self.distance((x, y), previous) < 1 for previous in result):
                continue
            result.append((x, y))
        boxes = []
        for x, y in result:
            left = max(0, x - 3 * self.config.position_bandwidth_x)
            top = max(0, y - 3 * self.config.position_bandwidth_y)
            right = min(1, x + 3 * self.config.position_bandwidth_x)
            bottom = min(1, y + 3 * self.config.position_bandwidth_y)
            boxes.append(
                BoundingBox(
                    left * dimensions.width,
                    top * dimensions.height,
                    (right - left) * dimensions.width,
                    (bottom - top) * dimensions.height,
                )
            )
        return tuple(boxes)


def _valid(
    evidence: PhysicalPageNumberEvidence, page: PageNumberPostprocessInput
) -> bool:
    if evidence.page_key != page.page_key:
        raise ValueError(f"Candidate page key does not match {page.page_key!r}")
    box = evidence.bbox
    return (
        not evidence.supplied
        and not evidence.suffix
        and evidence.value is not None
        and evidence.value >= 0
        and evidence.normalized is not None
        and evidence.numeral_system is not None
        and box is not None
        and all(math.isfinite(v) for v in (box.x, box.y, box.width, box.height))
        and box.width > 0
        and box.height > 0
        and box.x >= 0
        and box.y >= 0
        and box.x_max <= page.dimensions.width
        and box.y_max <= page.dimensions.height
    )


class PageNumberPostprocessor:
    def __init__(self, config: PageNumberPostprocessConfig | None = None) -> None:
        self.config = config or PageNumberPostprocessConfig()

    def process(
        self,
        pages: Sequence[PageNumberPostprocessInput],
        *,
        recover: Recovery | None = None,
    ) -> PageNumberCoreResult:
        pages = tuple(pages)
        keys = [page.page_key for page in pages]
        if len(keys) != len(set(keys)):
            raise ValueError("Page-number postprocessing requires unique page keys")
        if not self.config.enabled:
            return self._process_detections(pages, recover=recover)
        numbers = dict.fromkeys(keys)
        decisions = {key: PageNumberDecision("unresolved") for key in keys}
        for _, unit in groupby(pages, key=lambda page: page.pagination.unit_key):
            ordinary = []
            for page in unit:
                if page.pagination.excluded:
                    decisions[page.page_key] = PageNumberDecision("excluded")
                elif page.pagination.segment_of is None:
                    ordinary.append(page)
            detected = self._process_detections(ordinary, recover=recover)
            numbers.update(detected.page_numbers)
            decisions.update(detected.decisions)
        return NdkPaginator(self.config.ndk).process(
            pages, PageNumberCoreResult(numbers, decisions)
        )

    def _process_detections(
        self,
        pages: Sequence[PageNumberPostprocessInput],
        *,
        recover: Recovery | None = None,
    ) -> PageNumberCoreResult:
        pages = tuple(pages)
        keys = [p.page_key for p in pages]
        if len(keys) != len(set(keys)):
            raise ValueError("Page-number postprocessing requires unique page keys")
        distribution = _PositionDistribution(pages, self.config)
        pools = []
        selected = []
        for i, page in enumerate(pages):
            candidates = [e for e in page.candidates if _valid(e, page)]
            if (
                page.selected is not None
                and _valid(page.selected, page)
                and page.selected not in candidates
            ):
                candidates.append(page.selected)
            if not self.config.enabled:
                pools.append(candidates)
                selected.append(page.selected)
                continue
            windows = distribution.windows(i)
            # Search even pages with detections: a well-placed but incorrect
            # detection must not hide the actual printed number.
            if recover is not None and windows:
                candidates.extend(
                    e
                    for e in recover(page, windows)
                    if _valid(e, page) and e not in candidates
                )
            pools.append(candidates)
            selected.append(self._choose(candidates, page.selected, distribution, i))

        # Use an immutable snapshot. A correction never becomes an anchor for
        # another correction, preventing cascades through weak pages.
        anchors = tuple(selected)
        results = {}
        decisions = {}
        for i, (page, evidence) in enumerate(zip(pages, anchors)):
            if not self.config.enabled:
                results[page.page_key] = evidence
                continue
            expected = self._expected(i, anchors, distribution)
            if expected is not None:
                system, value = expected
                matching = [
                    e
                    for e in pools[i]
                    if e.numeral_system == system and e.value == value
                    if (support := distribution.support(e, i)) is None or support >= 0.6
                ]
                supported = self._choose(matching, None, distribution, i)
                if supported is not None:
                    evidence = supported
            if evidence is None:
                results[page.page_key] = None
                decisions[page.page_key] = PageNumberDecision("unresolved")
                continue
            original = evidence
            position = distribution.support(evidence, i)
            sequence_support = self._sequence_support(
                i, evidence, anchors, distribution
            )
            confidence = 0.5 if position is None else 0.25 + 0.73 * position
            action = "detected" if evidence in page.candidates else "recovered"
            agreement = (
                expected == (evidence.numeral_system, evidence.value)
                if expected is not None
                else None
            )
            if expected is not None and not agreement:
                if (
                    position is not None
                    and position >= 0.6
                    and evidence.numeral_system == expected[0]
                ):
                    normalized = format_page_number(expected[1], evidence)
                    if normalized is not None:
                        evidence = replace(
                            evidence, value=expected[1], normalized=normalized
                        )
                        confidence = min(0.85, confidence)
                        action = "corrected"
                    else:
                        confidence = min(0.35, confidence)
                else:
                    confidence = min(0.35, confidence)
            elif agreement:
                confidence = min(0.99, confidence + 0.08)
            elif sequence_support is not None and sequence_support < 0.25:
                # A stable running-header year can look spatially perfect but
                # cannot be a progressing folio. Retain it for review without
                # assigning the certainty of an attested number sequence.
                confidence = min(0.35, confidence)
            if position is not None and position < self.config.drop_threshold:
                evidence = None
                action = "dropped"
            results[page.page_key] = (
                replace(evidence, confidence=confidence)
                if evidence is not None
                else None
            )
            decisions[page.page_key] = PageNumberDecision(
                action,
                position,
                agreement,
                original.confidence,
                original.output_text(),
                sequence_support,
            )
            if action in ("corrected", "dropped", "recovered"):
                logger.info(
                    "Page number %s: page=%s original=%r result=%r support=%s",
                    action,
                    page.page_key,
                    original.output_text(),
                    evidence.output_text() if evidence else None,
                    position,
                )
        return PageNumberCoreResult(results, decisions)

    def _choose(
        self,
        candidates: Sequence[PhysicalPageNumberEvidence],
        initial: PhysicalPageNumberEvidence | None,
        distribution: _PositionDistribution,
        index: int,
    ) -> PhysicalPageNumberEvidence | None:
        if not candidates:
            return None

        def rank(e: PhysicalPageNumberEvidence) -> tuple[float, bool]:
            support = distribution.support(e, index)
            return (support if support is not None else 0.5, e == initial)

        return max(candidates, key=rank)

    def _expected(
        self,
        index: int,
        anchors: Sequence[PhysicalPageNumberEvidence | None],
        distribution: _PositionDistribution,
    ) -> tuple[PageNumberNumeralSystem, int] | None:
        # At least two independent anchors on each side, agreeing on the
        # physical-index offset. No extrapolation, no crossing a nearby reset.
        sides = []
        for direction in (-1, 1):
            side = []
            for distance in range(1, self.config.sequence_window + 1):
                j = index + direction * distance
                if not 0 <= j < len(anchors):
                    break
                evidence = anchors[j]
                if evidence is None:
                    continue
                support = distribution.support(evidence, j)
                if support is not None and support < 0.6:
                    continue
                side.append((evidence.numeral_system, evidence.value - j))
            sides.append(side)
        if any(len(side) < 2 for side in sides):
            return None
        nearest = sides[0][0]
        if sides[1][0] != nearest or any(Counter(side)[nearest] < 2 for side in sides):
            return None
        value = nearest[1] + index
        return (nearest[0], value) if value > 0 else None

    def _sequence_support(
        self,
        index: int,
        evidence: PhysicalPageNumberEvidence,
        anchors: Sequence[PhysicalPageNumberEvidence | None],
        distribution: _PositionDistribution,
    ) -> float | None:
        neighboring_offsets = []
        compatible_progressions = 0
        for j in range(
            max(0, index - self.config.sequence_window),
            min(len(anchors), index + self.config.sequence_window + 1),
        ):
            other = anchors[j]
            if (
                j == index
                or other is None
                or other.numeral_system != evidence.numeral_system
            ):
                continue
            support = distribution.support(other, j)
            if support is None or support >= 0.6:
                neighboring_offsets.append(other.value - j)
                # Extra unnumbered leaves can slow progression without making
                # the observed folios unreliable. Do not penalize that pattern
                # merely because it has no constant physical-index offset.
                direction = 1 if j > index else -1
                if 0 < (other.value - evidence.value) * direction <= abs(j - index):
                    compatible_progressions += 1
        if len(neighboring_offsets) < 4:
            return None
        return max(
            neighboring_offsets.count(evidence.value - index), compatible_progressions
        ) / len(neighboring_offsets)

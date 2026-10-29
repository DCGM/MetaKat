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
from statistics import median
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
    valid_detected_page_number,
)
from metakat.page_number.engines.core.segmentation import (
    PageNumberChunker,
    PageNumberSegmentationConfig,
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
    firm_support_count: int = 5
    firm_cluster_spread: float = 0.5
    maximum_firm_positions: int = 2
    additional_position_support_cap: float = 0.65
    position_window: int = 40
    sequence_window: int = 8
    drop_threshold: float = 0.1
    # Fill the numbers of the pages that have none (preliminary pass) and run the final pass
    # (see the page-number README); `ndk` sets up the pagination.
    ndk_pagination: bool = True
    ndk: NdkPaginationConfig = field(default_factory=NdkPaginationConfig)
    segmentation: PageNumberSegmentationConfig = field(
        default_factory=PageNumberSegmentationConfig
    )

    def __post_init__(self) -> None:
        if not isinstance(self.segmentation, PageNumberSegmentationConfig):
            raise ValueError(
                "postprocess.segmentation must be a PageNumberSegmentationConfig"
            )
        if not isinstance(self.ndk, NdkPaginationConfig):
            raise ValueError("postprocess.ndk must be an NdkPaginationConfig")
        if not isinstance(self.enabled, bool):
            raise ValueError("postprocess.enabled must be boolean")
        if not isinstance(self.ndk_pagination, bool):
            raise ValueError("postprocess.ndk_pagination must be boolean")
        for name in ("position_bandwidth_x", "position_bandwidth_y"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 < value < 0.5
            ):
                raise ValueError(f"postprocess.{name} must be between 0 and 0.5")
        for name in (
            "minimum_support",
            "firm_support_count",
            "position_window",
            "sequence_window",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value
                < (3 if name in ("minimum_support", "firm_support_count") else 1)
            ):
                raise ValueError(f"Invalid postprocess.{name}")
        if (
            isinstance(self.maximum_firm_positions, bool)
            or not isinstance(self.maximum_firm_positions, int)
            or not 1 <= self.maximum_firm_positions <= 2
        ):
            raise ValueError("postprocess.maximum_firm_positions must be 1 or 2")
        if (
            isinstance(self.firm_cluster_spread, bool)
            or not isinstance(self.firm_cluster_spread, (int, float))
            or not math.isfinite(self.firm_cluster_spread)
            or self.firm_cluster_spread <= 0
        ):
            raise ValueError(
                "postprocess.firm_cluster_spread must be finite and positive"
            )
        cap = self.additional_position_support_cap
        if (
            isinstance(cap, bool)
            or not isinstance(cap, (int, float))
            or not math.isfinite(cap)
            or not 0 <= cap < 1
        ):
            raise ValueError(
                "postprocess.additional_position_support_cap must be in [0, 1)"
            )
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
        if "segmentation" in values:
            values["segmentation"] = PageNumberSegmentationConfig.from_config(
                values["segmentation"]
            )
        return cls(**values)


@dataclass(frozen=True)
class _PositionCluster:
    indices: tuple[int, ...]
    center: tuple[float, float]
    spread: float


@dataclass(frozen=True)
class _PositionFit:
    center: tuple[float, float]
    strength: float


class _PositionDistribution:
    """Independent position clusters, with at most two firm locations per chunk.

    Cluster membership and the firm-location shortlist use initial detections
    only. Per-page fits exclude that page; nearby peers refine firm centers.
    Counts saturate at a threshold rather than competing with other locations.
    """

    def __init__(
        self,
        pages: Sequence[PageNumberPostprocessInput],
        config: PageNumberPostprocessConfig,
    ) -> None:
        self.pages = pages
        self.config = config
        self.seeds = {
            i: page.selected
            for i, page in enumerate(pages)
            if page.selected is not None and _valid(page.selected, page)
        }
        self.centers = {i: self.center(e, i) for i, e in self.seeds.items()}
        self.clusters = self._clusters()
        eligible = [
            j
            for j, cluster in enumerate(self.clusters)
            if len(cluster.indices)
            >= max(config.minimum_support, config.firm_support_count)
            and cluster.spread <= config.firm_cluster_spread
        ]
        # One shortlist for the whole chunk: parity and moving windows must
        # never independently promote a third physical location to firm status.
        eligible.sort(
            key=lambda j: (
                -len(self.clusters[j].indices)
                * math.exp(
                    -0.5 * (self.clusters[j].spread / config.firm_cluster_spread) ** 2
                ),
                self.clusters[j].spread,
                self.clusters[j].center,
            )
        )
        self.firm = frozenset(eligible[: config.maximum_firm_positions])
        self.parities = self._parities()
        self._fits: dict[int, tuple[_PositionFit, ...]] = {}

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

    def _fit(self, indices: Sequence[int]) -> _PositionCluster | None:
        if len(indices) < self.config.minimum_support:
            return None
        # Trim isolated tails around a coordinate-wise median, so one displaced
        # detection cannot spoil an otherwise tight, well-supported position.
        robust = (
            median(self.centers[i][0] for i in indices),
            median(self.centers[i][1] for i in indices),
        )
        core = tuple(i for i in indices if self.distance(self.centers[i], robust) <= 2)
        if len(core) < self.config.minimum_support:
            return None
        center = (
            sum(self.centers[i][0] for i in core) / len(core),
            sum(self.centers[i][1] for i in core) / len(core),
        )
        spread = math.sqrt(
            sum(self.distance(self.centers[i], center) for i in core) / len(core)
        )
        return _PositionCluster(core, center, spread)

    def _clusters(self) -> tuple[_PositionCluster, ...]:
        # Deterministic spatial grouping: at most two bandwidths from the
        # running center. Sorting geometry avoids dependence on scan order.
        groups: list[list[int]] = []
        centers: list[tuple[float, float]] = []
        for index in sorted(self.centers, key=lambda i: (self.centers[i], i)):
            point = self.centers[index]
            distances = [self.distance(point, center) for center in centers]
            closest = (
                min(range(len(distances)), key=distances.__getitem__)
                if distances
                else None
            )
            if closest is None or distances[closest] > 4:
                groups.append([index])
                centers.append(point)
            else:
                group = groups[closest]
                n = len(group)
                old = centers[closest]
                centers[closest] = (
                    (old[0] * n + point[0]) / (n + 1),
                    (old[1] * n + point[1]) / (n + 1),
                )
                group.append(index)
        # Merge adjacent representations of the same region before choosing
        # firm positions; a diffuse region must not consume two firm slots.
        changed = True
        while changed:
            changed = False
            for a in range(len(groups)):
                for b in range(a + 1, len(groups)):
                    if self.distance(centers[a], centers[b]) <= 4:
                        groups[a].extend(groups.pop(b))
                        centers.pop(b)
                        centers[a] = (
                            sum(self.centers[i][0] for i in groups[a]) / len(groups[a]),
                            sum(self.centers[i][1] for i in groups[a]) / len(groups[a]),
                        )
                        changed = True
                        break
                if changed:
                    break
        return tuple(
            cluster for group in groups if (cluster := self._fit(group)) is not None
        )

    def _parities(self) -> dict[int, int]:
        # Enforce alternation only if the two firm locations independently
        # show opposite, strong parity preferences. A detector missing every
        # other page at a single location is not evidence of two placements.
        if len(self.firm) != 2:
            return {}
        result = {}
        for j in self.firm:
            indices = self.clusters[j].indices
            if len(indices) < 2 * self.config.minimum_support:
                return {}
            even = sum(i % 2 == 0 for i in indices)
            majority = max(even, len(indices) - even)
            if majority / len(indices) < 0.8:
                return {}
            result[j] = 0 if even > len(indices) / 2 else 1
        return result if len(set(result.values())) == 2 else {}

    def fits(self, index: int) -> tuple[_PositionFit, ...]:
        if index in self._fits:
            return self._fits[index]
        fits = []
        for j, cluster in enumerate(self.clusters):
            peers = [i for i in cluster.indices if i != index]
            if j in self.parities:
                if self.parities[j] != index % 2:
                    continue
                peers = [i for i in peers if i % 2 == index % 2]
            local = self._fit(
                [i for i in peers if abs(i - index) <= self.config.position_window]
            )
            # Once a position is firmly established for this chunk, absence
            # of nearby detections must not erase that evidence. Nearby peers
            # refine its center; the chunk's other detections establish it.
            basis = self._fit(peers) if j in self.firm else local
            if basis is None:
                continue
            local = local or basis
            count_support = min(
                1.0, len(basis.indices) / self.config.firm_support_count
            )
            spread = max(basis.spread, local.spread)
            compactness = (
                min(1.0, (self.config.firm_cluster_spread / spread) ** 2)
                if spread
                else 1.0
            )
            strength = count_support * compactness
            if j not in self.firm:
                strength = min(strength, self.config.additional_position_support_cap)
            fits.append(_PositionFit(local.center, strength))
        self._fits[index] = tuple(fits)
        return self._fits[index]

    def support(self, evidence: PhysicalPageNumberEvidence, index: int) -> float | None:
        fits = self.fits(index)
        if not fits:
            return None
        center = self.center(evidence, index)
        return max(
            (
                fit.strength * math.exp(-0.5 * self.distance(center, fit.center))
                for fit in fits
                if self.distance(center, fit.center) <= 9
            ),
            default=0.0,
        )

    def windows(self, index: int) -> tuple[BoundingBox, ...]:
        dimensions = self.pages[index].dimensions
        boxes = []
        for fit in self.fits(index):
            x, y = fit.center
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
    return valid_detected_page_number(
        evidence,
        page_key=page.page_key,
        dimensions=page.dimensions,
    )


class PageNumberPostprocessor:
    def __init__(
        self,
        config: PageNumberPostprocessConfig | None = None,
        *,
        bracketed: bool = True,
        complete_document: bool = False,
    ) -> None:
        """bracketed and complete_document are the pass of the NDK pagination (see NdkPaginator)."""
        self.config = config or PageNumberPostprocessConfig()
        self.ndk_pass = {"bracketed": bracketed, "complete_document": complete_document}

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
        chunks = []
        chunker = PageNumberChunker(self.config.segmentation)
        for _, unit in groupby(pages, key=lambda page: page.pagination.unit_key):
            unit = tuple(unit)
            chunks.extend(
                chunker.split(
                    unit,
                    reason=(
                        "explicit_unit"
                        if unit[0].pagination.unit_key is not None
                        else "input_start"
                    ),
                )
            )
        by_key = {page.page_key: page for page in pages}
        for chunk_index, chunk in enumerate(chunks):
            chunk_pages = tuple(by_key[key] for key in chunk.page_keys)
            if chunk.ambiguous:
                for page in chunk_pages:
                    evidence = page.selected
                    if page.pagination.excluded:
                        decisions[page.page_key] = PageNumberDecision(
                            "excluded", chunk_index=chunk_index
                        )
                        continue
                    if evidence is not None and _valid(evidence, page):
                        numbers[page.page_key] = replace(evidence, confidence=0.35)
                    decisions[page.page_key] = PageNumberDecision(
                        "boundary_uncertain",
                        detector_confidence=evidence.confidence if evidence else None,
                        original_reading=evidence.output_text() if evidence else None,
                        chunk_index=chunk_index,
                    )
                continue
            ordinary = []
            for page in chunk_pages:
                if page.pagination.excluded:
                    decisions[page.page_key] = PageNumberDecision("excluded")
                elif page.pagination.segment_of is None:
                    ordinary.append(page)
            detected = self._process_detections(ordinary, recover=recover)
            numbers.update(detected.page_numbers)
            decisions.update(detected.decisions)
            decisions.update(
                {
                    key: replace(decisions[key], chunk_index=chunk_index)
                    for key in chunk.page_keys
                }
            )
        observed = PageNumberCoreResult(numbers, decisions, tuple(chunks))
        if not self.config.ndk_pagination:
            return observed
        return NdkPaginator(self.config.ndk, **self.ndk_pass).process(pages, observed)

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

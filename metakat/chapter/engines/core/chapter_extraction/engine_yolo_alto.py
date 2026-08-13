from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from PIL import Image
from text_geometry_aligner import AlignmentRegion

from metakat.chapter.engines.core.models import (
    ChapterPageInput,
    ChapterBase,
    TocBase,
    ChapterPageNumberEvidence,
)
from metakat.common.models import BoundingBox, DetectionEvidence
from metakat.chapter.engines.core.pipeline_utils import (
    load_chapter_label_mapping,
    load_engine_config,
    region_label,
)
from metakat.chapter.engines.core.chapter_page_number_parsers import (
    ArabicRomanChapterPageNumberParser,
)
from metakat.common.engines.engine_yolo_alto import EngineYOLOALTO
from metakat.schemas.base_objects import ChapterType

logger = logging.getLogger(__name__)

_SubtitleGeometryScore = tuple[float, float]
_SubtitleForUnitScore = tuple[float, float, float, float, float]


@dataclass(frozen=True)
class _ChapterCandidate:
    chapter_type: ChapterType
    page_key: str
    bbox: BoundingBox
    text: str | None
    confidence: float | None

    @property
    def is_construction_ready(self) -> bool:
        return bool(self.text) and self.confidence is not None


@dataclass
class _Unit:
    toc_page_key: str
    level: int | None = None
    title: _ChapterCandidate | None = None
    subtitle: _ChapterCandidate | None = None
    part_number: _ChapterCandidate | None = None
    page_number: _ChapterCandidate | None = None

    @property
    def ordering_candidate(self) -> _ChapterCandidate:
        if self.title is not None:
            return self.title
        if self.page_number is not None:
            return self.page_number
        raise ValueError("A TOC unit has no ordering candidate")


@dataclass(frozen=True)
class _PageNumberAxis:
    x: float
    members: tuple[_ChapterCandidate, ...]


@dataclass(frozen=True)
class _ColumnLayout:
    axes: tuple[_PageNumberAxis, ...]
    spread_tolerance: float


@dataclass
class _MutableEntry:
    toc_page_key: str
    title: DetectionEvidence | None
    level: int
    subtitle: DetectionEvidence | None = None
    part_number: DetectionEvidence | None = None
    page_number: ChapterPageNumberEvidence | None = None
    children: list[_MutableEntry] = field(default_factory=list)


class ChapterExtractionEngineYOLOALTO:
    """Extract one cross-page TOC hierarchy from YOLO-aligned ALTO text."""

    DEFAULT_LABELS: dict[ChapterType, str] = {
        ChapterType.LEVEL_1_TITLE: "kapitola",
        ChapterType.LEVEL_2_TITLE: "jiny nadpis",
        ChapterType.SUBTITLE: "podnadpis",
        ChapterType.PAGE_NUMBER: "cislo strany",
        ChapterType.PART_NUMBER: "jine cislo",
    }

    def __init__(self, engine_dir, *, alignment_engine=None):
        self.engine_dir, self.config = load_engine_config(engine_dir)
        self.labels = load_chapter_label_mapping(
            self.config,
            self.DEFAULT_LABELS,
        )
        self.multicolumn_axis_min_count = self._integer_config(
            "multicolumn_axis_min_count",
            2,
            minimum=2,
        )
        self.multicolumn_axis_min_page_number_detection_count = (
            self._integer_config(
                "multicolumn_axis_min_page_number_detection_count",
                3,
                minimum=2,
            )
        )
        self.multicolumn_axis_min_provisional_title_count = (
            self._integer_config(
                "multicolumn_axis_min_provisional_title_count",
                1,
                minimum=1,
            )
        )
        self.multicolumn_axis_spread_median_page_number_bbox_width_multiplier = (
            self._positive_number_config(
                "multicolumn_axis_spread_median_page_number_bbox_width_"
                "multiplier",
                0.5,
            )
        )
        self.multicolumn_axis_min_spread_page_width_fraction = (
            self._fraction_config(
                "multicolumn_axis_min_spread_page_width_fraction",
                0.005,
            )
        )
        self.multicolumn_axis_max_spread_page_width_fraction = (
            self._fraction_config(
                "multicolumn_axis_max_spread_page_width_fraction",
                0.02,
                minimum_exclusive=True,
            )
        )
        if (
            self.multicolumn_axis_min_spread_page_width_fraction
            > self.multicolumn_axis_max_spread_page_width_fraction
        ):
            raise ValueError(
                "multicolumn_axis_min_spread_page_width_fraction must not "
                "exceed multicolumn_axis_max_spread_page_width_fraction"
            )
        self.multicolumn_axis_min_separation_page_width_fraction = (
            self._fraction_config(
                "multicolumn_axis_min_separation_page_width_fraction",
                0.20,
                minimum_exclusive=True,
            )
        )
        self.multicolumn_axis_min_explained_page_number_fraction = (
            self._fraction_config(
                "multicolumn_axis_min_explained_page_number_fraction",
                0.75,
            )
        )
        self.multicolumn_axis_max_title_overlap_page_width_fraction = (
            self._fraction_config(
                "multicolumn_axis_max_title_overlap_page_width_fraction",
                0.03,
            )
        )
        self.subtitle_max_vertical_gap_height_multiplier = (
            self._positive_number_config(
                "subtitle_max_vertical_gap_height_multiplier",
                1.5,
            )
        )
        self.subtitle_max_vertical_overlap_height_fraction = (
            self._fraction_config(
                "subtitle_max_vertical_overlap_height_fraction",
                0.25,
            )
        )
        self.subtitle_min_horizontal_overlap_fraction = (
            self._fraction_config(
                "subtitle_min_horizontal_overlap_fraction",
                0.25,
            )
        )
        self.alignment_engine = alignment_engine or EngineYOLOALTO(
            self.engine_dir
        )

    def process(
        self,
        toc_pages: Sequence[ChapterPageInput],
    ) -> TocBase:
        ordered_pages = tuple(
            sorted(toc_pages, key=lambda page: page.position)
        )
        if not ordered_pages:
            logger.info("Chapter extraction received no selected TOC pages")
            return TocBase(())

        logger.info(
            "Extracting TOC hierarchy from %d page(s): pages=%s",
            len(ordered_pages),
            [page.page_key for page in ordered_pages],
        )

        document = self.alignment_engine.process(
            images=[str(page.image_path) for page in ordered_pages],
            alto_files=[str(page.alto_path) for page in ordered_pages],
        )
        alignments = {page.page_key: page for page in document.pages}
        missing = [
            page.page_key
            for page in ordered_pages
            if page.page_key not in alignments
        ]
        if missing:
            raise ValueError(
                "Chapter extraction alignment omitted page(s): "
                + ", ".join(missing)
            )

        units: list[_Unit] = []
        for source_page in ordered_pages:
            alignment = alignments[source_page.page_key]
            page_units = self._extract_page_units(
                alignment.regions,
                source_page,
            )
            units.extend(page_units)
            logger.info(
                "Chapter extraction page=%r: aligned_regions=%d, entries=%d, "
                "titled_entries=%d, titleless_entries=%d",
                source_page.page_key,
                len(alignment.regions),
                len(page_units),
                sum(unit.title is not None for unit in page_units),
                sum(unit.title is None for unit in page_units),
            )

        self._infer_titleless_entry_levels(units)
        roots: list[_MutableEntry] = []
        active_parents: dict[int, _MutableEntry] = {}
        for unit_index, unit in enumerate(units):
            level = unit.level or 1
            page_number = self._parse_page_number(
                unit,
                unit_index,
            )
            entry = _MutableEntry(
                toc_page_key=unit.toc_page_key,
                title=self._to_detection_evidence(unit.title),
                level=level,
                subtitle=self._to_detection_evidence(unit.subtitle),
                part_number=self._to_detection_evidence(unit.part_number),
                page_number=page_number,
            )
            logger.debug(
                "Extracted TOC entry %d: page=%r, level=%s, title=%r, "
                "subtitle=%r, part_number=%r, page_number=%r",
                unit_index,
                unit.toc_page_key,
                unit.level,
                None if unit.title is None else unit.title.text,
                None if unit.subtitle is None else unit.subtitle.text,
                None if unit.part_number is None else unit.part_number.text,
                None if page_number is None else page_number.text,
            )
            parent = next(
                (
                    active_parents[parent_level]
                    for parent_level in range(level - 1, 0, -1)
                    if parent_level in active_parents
                ),
                None,
            )
            if parent is None:
                roots.append(entry)
            else:
                parent.children.append(entry)
            active_parents[level] = entry
            for active_level in tuple(active_parents):
                if active_level > level:
                    del active_parents[active_level]

        result = TocBase(tuple(self._freeze(root) for root in roots))
        logger.info(
            "Chapter extraction produced %d total entry/entries, %d root(s), "
            "%d titleless entry/entries, maximum_level=%d",
            len(units),
            len(result.chapters),
            sum(unit.title is None for unit in units),
            max((unit.level or 1 for unit in units), default=0),
        )
        return result

    @classmethod
    def _parse_page_number(
        cls,
        unit: _Unit,
        unit_index: int,
    ) -> ChapterPageNumberEvidence | None:
        candidate = unit.page_number
        if candidate is None:
            return None
        evidence = cls._to_detection_evidence(candidate)
        assert evidence is not None
        page_number = ArabicRomanChapterPageNumberParser.create(evidence)
        if not page_number.normalized_items:
            logger.warning(
                "TOC page number was rejected by the parser; its original "
                "evidence is retained: entry=%d, toc_page=%r, title=%r, "
                "source_number=%r",
                unit_index,
                candidate.page_key,
                None if unit.title is None else unit.title.text,
                page_number.text,
            )
            return page_number
        normalized = page_number.normalized_text()
        if normalized != page_number.text:
            logger.debug(
                "Normalized TOC page number: entry=%d, toc_page=%r, "
                "source_number=%r, normalized_number=%r, kind=%s",
                unit_index,
                candidate.page_key,
                page_number.text,
                normalized,
                page_number.kind.value,
            )
        return page_number

    @staticmethod
    def _to_detection_evidence(
        candidate: _ChapterCandidate | None,
    ) -> DetectionEvidence | None:
        if candidate is None:
            return None
        if not candidate.is_construction_ready:
            raise ValueError(
                "Cannot expose a chapter candidate without text and "
                "confidence as DetectionEvidence"
            )
        assert candidate.text is not None
        assert candidate.confidence is not None
        return DetectionEvidence(
            text=candidate.text,
            confidence=candidate.confidence,
            bbox=candidate.bbox,
            page_key=candidate.page_key,
        )

    def _extract_page_units(
        self,
        regions: Sequence[AlignmentRegion],
        page: ChapterPageInput,
    ) -> tuple[_Unit, ...]:
        candidates = self._collect_candidates(
            regions,
            page.page_key,
        )
        titles = self._candidates_of_type(
            candidates,
            ChapterType.LEVEL_1_TITLE,
            ChapterType.LEVEL_2_TITLE,
        )
        page_numbers = self._candidates_of_type(
            candidates,
            ChapterType.PAGE_NUMBER,
        )
        page_width = self._resolve_page_width(page)
        layout = self._detect_column_layout(
            titles,
            page_numbers,
            page,
            page_width,
        )
        if layout is None:
            candidate_groups = (candidates,)
        else:
            candidate_groups = self._partition_candidates_by_column(
                candidates,
                page.page_key,
                layout,
            )

        units: list[_Unit] = []
        for group_index, candidate_group in enumerate(candidate_groups):
            construction_candidates = self._construction_candidates(
                candidate_group,
                page.page_key,
                group_index,
            )
            group_units = self._construct_units(
                construction_candidates,
            )
            units.extend(group_units)
            logger.debug(
                "Constructed TOC candidate group: page=%r, group=%d, "
                "candidates=%d, construction_candidates=%d, entries=%d",
                page.page_key,
                group_index,
                len(candidate_group),
                len(construction_candidates),
                len(group_units),
            )

        logger.debug(
            "Constructed TOC units: page=%r, ordering=%s, titles=%d, "
            "subtitles=%d, part_numbers=%d, page_numbers=%d, entries=%d, "
            "assigned_subtitles=%d, "
            "assigned_part_numbers=%d, assigned_page_numbers=%d, "
            "titleless_page_numbers=%d",
            page.page_key,
            "column-wise" if layout is not None else "top-to-bottom",
            len(titles),
            len(
                self._candidates_of_type(
                    candidates,
                    ChapterType.SUBTITLE,
                )
            ),
            len(
                self._candidates_of_type(
                    candidates,
                    ChapterType.PART_NUMBER,
                )
            ),
            len(page_numbers),
            len(units),
            sum(unit.subtitle is not None for unit in units),
            sum(unit.part_number is not None for unit in units),
            sum(
                unit.title is not None and unit.page_number is not None
                for unit in units
            ),
            sum(
                unit.title is None and unit.page_number is not None
                for unit in units
            ),
        )
        return tuple(units)

    def _collect_candidates(
        self,
        regions: Sequence[AlignmentRegion],
        page_key: str,
    ) -> tuple[_ChapterCandidate, ...]:
        candidates: list[_ChapterCandidate] = []
        for region in regions:
            if region.input_geometry is None:
                continue
            label = region_label(region)
            chapter_type = next(
                (
                    candidate_type
                    for candidate_type in (
                        ChapterType.LEVEL_1_TITLE,
                        ChapterType.LEVEL_2_TITLE,
                        ChapterType.SUBTITLE,
                        ChapterType.PART_NUMBER,
                        ChapterType.PAGE_NUMBER,
                    )
                    if label == self.labels[candidate_type]
                ),
                None,
            )
            if chapter_type is None:
                continue
            bounds = region.input_geometry.bounds
            bbox = BoundingBox(
                x=bounds.x,
                y=bounds.y,
                width=bounds.width,
                height=bounds.height,
            )
            text = None
            if region.matched and region.alto_text:
                stripped_text = region.alto_text.strip()
                if stripped_text:
                    text = stripped_text
            candidates.append(
                _ChapterCandidate(
                    chapter_type=chapter_type,
                    page_key=page_key,
                    bbox=bbox,
                    text=text,
                    confidence=region.input_geometry_confidence,
                )
            )

        return tuple(candidates)

    @staticmethod
    def _candidates_of_type(
        candidates: Sequence[_ChapterCandidate],
        *chapter_types: ChapterType,
    ) -> tuple[_ChapterCandidate, ...]:
        return tuple(
            candidate
            for candidate in candidates
            if candidate.chapter_type in chapter_types
        )

    @staticmethod
    def _construction_candidates(
        candidates: Sequence[_ChapterCandidate],
        page_key: str,
        group_index: int,
    ) -> tuple[_ChapterCandidate, ...]:
        ready = tuple(
            candidate
            for candidate in candidates
            if candidate.is_construction_ready
        )
        discarded = len(candidates) - len(ready)
        if discarded:
            logger.debug(
                "Discarded candidates without complete construction "
                "evidence before "
                "TOC-unit construction: page=%r, group=%d, count=%d",
                page_key,
                group_index,
                discarded,
            )
        return ready

    def _construct_units(
        self,
        candidates: Sequence[_ChapterCandidate],
    ) -> tuple[_Unit, ...]:
        subtitles = self._candidates_of_type(
            candidates,
            ChapterType.SUBTITLE,
        )
        units = self._construct_basic_units(candidates)
        self._assign_subtitles(units, subtitles)
        return tuple(units)

    def _construct_basic_units(
        self,
        candidates: Sequence[_ChapterCandidate],
    ) -> list[_Unit]:
        titles = self._candidates_of_type(
            candidates,
            ChapterType.LEVEL_1_TITLE,
            ChapterType.LEVEL_2_TITLE,
        )
        available_part_numbers = list(
            self._candidates_of_type(
                candidates,
                ChapterType.PART_NUMBER,
            )
        )
        available_page_numbers = list(
            self._candidates_of_type(
                candidates,
                ChapterType.PAGE_NUMBER,
            )
        )
        units: list[_Unit] = []
        for title in sorted(
            titles,
            key=self._candidate_vertical_key,
        ):
            part_number = self._take_number_for_title(
                title,
                available_part_numbers,
                side="left",
            )
            page_number = self._take_number_for_title(
                title,
                available_page_numbers,
                side="right",
            )
            units.append(
                _Unit(
                    toc_page_key=title.page_key,
                    level=self._title_level(title.chapter_type),
                    title=title,
                    part_number=part_number,
                    page_number=page_number,
                )
            )

        units.extend(
            _Unit(
                toc_page_key=candidate.page_key,
                page_number=candidate,
            )
            for candidate in available_page_numbers
        )
        units.sort(key=self._unit_vertical_key)
        return units

    def _assign_subtitles(
        self,
        units: Sequence[_Unit],
        subtitles: Sequence[_ChapterCandidate],
    ) -> None:
        available_subtitles = sorted(
            subtitles,
            key=self._candidate_vertical_key,
        )
        for unit in units:
            if unit.title is None:
                continue
            eligible: list[
                tuple[
                    _SubtitleForUnitScore,
                    int,
                    _ChapterCandidate,
                ]
            ] = []
            for subtitle_index, subtitle in enumerate(available_subtitles):
                geometry_score = self._subtitle_geometry_score(
                    unit.title,
                    subtitle,
                )
                if geometry_score is None:
                    continue
                vertical_distance, horizontal_overlap_fraction = (
                    geometry_score
                )
                eligible.append(
                    (
                        (
                            vertical_distance,
                            -(
                                subtitle.confidence
                                if subtitle.confidence is not None
                                else -1.0
                            ),
                            -horizontal_overlap_fraction,
                            -abs(
                                subtitle.bbox.width
                                * subtitle.bbox.height
                            ),
                            -abs(subtitle.bbox.width),
                        ),
                        subtitle_index,
                        subtitle,
                    )
                )
            if not eligible:
                continue
            _, subtitle_index, subtitle = min(
                eligible,
                key=lambda item: item[0],
            )
            unit.subtitle = subtitle
            del available_subtitles[subtitle_index]

    def _subtitle_geometry_score(
        self,
        title: _ChapterCandidate,
        subtitle: _ChapterCandidate,
    ) -> _SubtitleGeometryScore | None:
        if subtitle.bbox.y < title.bbox.y:
            return None

        vertical_gap = subtitle.bbox.y - title.bbox.y_max
        maximum_overlap = (
            min(abs(title.bbox.height), abs(subtitle.bbox.height))
            * self.subtitle_max_vertical_overlap_height_fraction
        )
        if vertical_gap < -maximum_overlap:
            return None
        maximum_gap = (
            max(abs(title.bbox.height), abs(subtitle.bbox.height))
            * self.subtitle_max_vertical_gap_height_multiplier
        )
        if vertical_gap > maximum_gap:
            return None

        overlap_width = max(
            0.0,
            min(title.bbox.x_max, subtitle.bbox.x_max)
            - max(title.bbox.x, subtitle.bbox.x),
        )
        smaller_width = min(
            abs(title.bbox.width),
            abs(subtitle.bbox.width),
        )
        if smaller_width <= 0:
            return None
        horizontal_overlap_fraction = overlap_width / smaller_width
        if (
            horizontal_overlap_fraction
            < self.subtitle_min_horizontal_overlap_fraction
        ):
            return None

        return abs(vertical_gap), horizontal_overlap_fraction

    @staticmethod
    def _title_level(chapter_type: ChapterType) -> int:
        if chapter_type is ChapterType.LEVEL_1_TITLE:
            return 1
        if chapter_type is ChapterType.LEVEL_2_TITLE:
            return 2
        raise ValueError(
            f"Cannot derive a title level from {chapter_type.value!r}"
        )

    def _detect_column_layout(
        self,
        titles: Sequence[_ChapterCandidate],
        page_numbers: Sequence[_ChapterCandidate],
        page: ChapterPageInput,
        page_width: float,
    ) -> _ColumnLayout | None:
        minimum_evidence = (
            self.multicolumn_axis_min_count
            * self.multicolumn_axis_min_page_number_detection_count
        )
        if len(page_numbers) < minimum_evidence:
            logger.debug(
                "Multi-column TOC processing rejected: page=%r, reason=%s, "
                "page_number_detections=%d, required=%d",
                page.page_key,
                "insufficient raw page-number evidence",
                len(page_numbers),
                minimum_evidence,
            )
            return None

        spread_tolerance = self._axis_spread_tolerance(
            page_numbers,
            page_width,
        )
        axes = self._candidate_page_number_axes(
            page_numbers,
            spread_tolerance,
        )
        provisional_title_columns = self._provisional_title_columns(
            titles,
            axes,
        )
        rejection_reason = self._multicolumn_rejection_reason(
            axes,
            page_numbers,
            page_width,
            spread_tolerance,
            provisional_title_columns,
        )
        if rejection_reason is not None:
            logger.info(
                "Multi-column TOC processing rejected: page=%r, reason=%s, "
                "spread_tolerance=%.3f, candidate_axes=%s",
                page.page_key,
                rejection_reason,
                spread_tolerance,
                self._axis_diagnostics(axes),
            )
            return None

        layout = _ColumnLayout(
            axes=axes,
            spread_tolerance=spread_tolerance,
        )
        logger.info(
            "Multi-column TOC processing accepted: page=%r, columns=%d, "
            "spread_tolerance=%.3f, axes=%s, provisional_title_counts=%s",
            page.page_key,
            len(axes),
            spread_tolerance,
            self._axis_diagnostics(axes),
            [
                self._unique_candidate_count(column)
                for column in provisional_title_columns
            ],
        )
        return layout

    def _partition_candidates_by_column(
        self,
        candidates: Sequence[_ChapterCandidate],
        page_key: str,
        layout: _ColumnLayout,
    ) -> tuple[tuple[_ChapterCandidate, ...], ...]:
        """Partition chapter candidates without constructing TOC units."""
        columns: tuple[list[_ChapterCandidate], ...] = tuple(
            [] for _ in layout.axes
        )
        discarded_part_number_count = 0
        discarded_page_number_count = 0

        for candidate in candidates:
            if candidate.chapter_type in {
                ChapterType.LEVEL_1_TITLE,
                ChapterType.LEVEL_2_TITLE,
                ChapterType.SUBTITLE,
            }:
                column_index = self._title_column_index(candidate, layout)
            elif candidate.chapter_type is ChapterType.PART_NUMBER:
                candidate_center = (
                    candidate.bbox.x + candidate.bbox.width / 2
                )
                column_index = self._first_axis_to_right(
                    candidate_center,
                    layout.axes,
                )
                if column_index is None:
                    discarded_part_number_count += 1
                    continue
            elif candidate.chapter_type is ChapterType.PAGE_NUMBER:
                column_index, distance = self._nearest_axis(
                    candidate.bbox.x_max,
                    layout.axes,
                )
                if distance > layout.spread_tolerance:
                    discarded_page_number_count += 1
                    continue
            else:
                raise ValueError(
                    "Unsupported chapter candidate type: "
                    f"{candidate.chapter_type.value!r}"
                )
            columns[column_index].append(candidate)

        if discarded_part_number_count:
            logger.debug(
                "Discarded PartNumber detections without an alignment axis "
                "to their right: page=%r, count=%d",
                page_key,
                discarded_part_number_count,
            )
        if discarded_page_number_count:
            logger.debug(
                "Discarded PageNumber detections outside the spread of "
                "their nearest alignment axis: page=%r, count=%d, "
                "axis_spread_tolerance=%.3f",
                page_key,
                discarded_page_number_count,
                layout.spread_tolerance,
            )

        frozen_columns = tuple(
            tuple(column)
            for column in columns
        )
        logger.debug(
            "Partitioned multi-column TOC candidates: page=%r, columns=%s",
            page_key,
            [
                {
                    "axis_x": layout.axes[column_index].x,
                    "titles": len(
                        self._candidates_of_type(
                            column,
                            ChapterType.LEVEL_1_TITLE,
                            ChapterType.LEVEL_2_TITLE,
                        )
                    ),
                    "subtitles": len(
                        self._candidates_of_type(
                            column,
                            ChapterType.SUBTITLE,
                        )
                    ),
                    "part_numbers": len(
                        self._candidates_of_type(
                            column,
                            ChapterType.PART_NUMBER,
                        )
                    ),
                    "page_numbers": len(
                        self._candidates_of_type(
                            column,
                            ChapterType.PAGE_NUMBER,
                        )
                    ),
                }
                for column_index, column in enumerate(frozen_columns)
            ],
        )
        return frozen_columns

    @staticmethod
    def _candidate_vertical_key(
        candidate: _ChapterCandidate,
    ) -> tuple[float, float]:
        return candidate.bbox.y, candidate.bbox.x

    @staticmethod
    def _unit_vertical_key(unit: _Unit) -> tuple[float, float]:
        candidate = unit.ordering_candidate
        return candidate.bbox.y, candidate.bbox.x

    @classmethod
    def _take_number_for_title(
        cls,
        title: _ChapterCandidate,
        available: list[_ChapterCandidate],
        *,
        side: str,
    ) -> _ChapterCandidate | None:
        eligible = [
            candidate
            for candidate in available
            if cls._is_eligible_for_title(
                candidate.bbox,
                title.bbox,
                side=side,
            )
        ]
        if not eligible:
            return None

        if side == "left":
            outside = [
                candidate
                for candidate in eligible
                if candidate.bbox.x_max <= title.bbox.x
            ]
            edge_distance = lambda candidate: abs(
                title.bbox.x - candidate.bbox.x_max
            )
        else:
            outside = [
                candidate
                for candidate in eligible
                if candidate.bbox.x >= title.bbox.x_max
            ]
            edge_distance = lambda candidate: abs(
                candidate.bbox.x - title.bbox.x_max
            )
        selection_pool = outside or eligible
        selected = min(
            selection_pool,
            key=lambda candidate: (
                edge_distance(candidate),
                -(
                    candidate.confidence
                    if candidate.confidence is not None
                    else -1.0
                ),
                -abs(candidate.bbox.width * candidate.bbox.height),
                -abs(candidate.bbox.width),
            ),
        )
        available.remove(selected)
        return selected

    @staticmethod
    def _is_eligible_for_title(
        number: BoundingBox,
        title: BoundingBox,
        *,
        side: str,
    ) -> bool:
        number_vertical_center = number.y + number.height / 2
        if not (
            title.y
            <= number_vertical_center
            <= title.y_max
        ):
            return False
        number_horizontal_center = number.x + number.width / 2
        title_horizontal_center = title.x + title.width / 2
        if side == "left":
            return number_horizontal_center < title_horizontal_center
        if side == "right":
            return number_horizontal_center > title_horizontal_center
        raise ValueError(f"Unsupported TOC-number side: {side!r}")

    def _candidate_page_number_axes(
        self,
        page_numbers: Sequence[_ChapterCandidate],
        spread_tolerance: float,
    ) -> tuple[_PageNumberAxis, ...]:
        ordered = sorted(
            page_numbers,
            key=lambda candidate: candidate.bbox.x_max,
        )
        clusters: list[list[_ChapterCandidate]] = []
        for candidate in ordered:
            x = candidate.bbox.x_max
            if not clusters:
                clusters.append([candidate])
                continue
            cluster_minimum = clusters[-1][0].bbox.x_max
            if x - cluster_minimum <= spread_tolerance:
                clusters[-1].append(candidate)
            else:
                clusters.append([candidate])

        axes = [
            _PageNumberAxis(
                x=self._median(
                    candidate.bbox.x_max
                    for candidate in cluster
                ),
                members=tuple(cluster),
            )
            for cluster in clusters
            if (
                len(cluster)
                >= self.multicolumn_axis_min_page_number_detection_count
            )
        ]
        return tuple(sorted(axes, key=lambda axis: axis.x))

    def _multicolumn_rejection_reason(
        self,
        axes: Sequence[_PageNumberAxis],
        page_numbers: Sequence[_ChapterCandidate],
        page_width: float,
        spread_tolerance: float,
        provisional_title_columns: Sequence[
            Sequence[_ChapterCandidate]
        ],
    ) -> str | None:
        if len(axes) < self.multicolumn_axis_min_count:
            return "fewer than the required page-number alignment axes"

        minimum_separation = (
            page_width
            * self.multicolumn_axis_min_separation_page_width_fraction
        )
        if any(
            right.x - left.x < minimum_separation
            for left, right in zip(axes, axes[1:])
        ):
            return "adjacent page-number alignment axes are too close"

        explained_count = sum(
            min(
                abs(page_number.bbox.x_max - axis.x)
                for axis in axes
            )
            <= spread_tolerance
            for page_number in page_numbers
        )
        explained_fraction = explained_count / len(page_numbers)
        if (
            explained_fraction
            < self.multicolumn_axis_min_explained_page_number_fraction
        ):
            return (
                "alignment axes explain too little page-number evidence "
                f"({explained_fraction:.3f})"
            )

        overlap_tolerance = (
            page_width
            * self.multicolumn_axis_max_title_overlap_page_width_fraction
        )
        for column_index, column_titles in enumerate(
            provisional_title_columns
        ):
            title_count = self._unique_candidate_count(column_titles)
            if (
                title_count
                < self.multicolumn_axis_min_provisional_title_count
            ):
                return (
                    f"page-number alignment axis {column_index} has "
                    f"{title_count} distinct vertically compatible "
                    "provisional title(s), fewer than the required "
                    f"{self.multicolumn_axis_min_provisional_title_count}"
                )
        for right_index in range(1, len(axes)):
            left_axis = axes[right_index - 1]
            right_title_x = self._median(
                title.bbox.x
                for title in provisional_title_columns[right_index]
            )
            if right_title_x < left_axis.x - overlap_tolerance:
                return "title areas assigned to adjacent axes overlap"
        return None

    def _provisional_title_columns(
        self,
        titles: Sequence[_ChapterCandidate],
        axes: Sequence[_PageNumberAxis],
    ) -> tuple[tuple[_ChapterCandidate, ...], ...]:
        columns: list[tuple[_ChapterCandidate, ...]] = []
        for axis in axes:
            matched_titles: list[_ChapterCandidate] = []
            for page_number in axis.members:
                eligible = [
                    title
                    for title in titles
                    if self._is_eligible_for_title(
                        page_number.bbox,
                        title.bbox,
                        side="right",
                    )
                ]
                if not eligible:
                    continue
                matched_titles.append(
                    min(
                        eligible,
                        key=lambda title: (
                            max(
                                0.0,
                                page_number.bbox.x
                                - title.bbox.x_max,
                            ),
                            -(
                                title.confidence
                                if title.confidence is not None
                                else -1.0
                            ),
                            title.bbox.y,
                            title.bbox.x,
                        ),
                    )
                )
            columns.append(tuple(matched_titles))
        return tuple(columns)

    @staticmethod
    def _title_column_index(
        title: _ChapterCandidate,
        layout: _ColumnLayout,
    ) -> int:
        axes_to_right = [
            index
            for index, axis in enumerate(layout.axes)
            if axis.x >= title.bbox.x_max
        ]
        if axes_to_right:
            return min(
                axes_to_right,
                key=lambda index: layout.axes[index].x - title.bbox.x_max,
            )
        return min(
            range(len(layout.axes)),
            key=lambda index: abs(
                layout.axes[index].x - title.bbox.x_max
            ),
        )

    @staticmethod
    def _first_axis_to_right(
        x: float,
        axes: Sequence[_PageNumberAxis],
    ) -> int | None:
        return next(
            (
                index
                for index, axis in enumerate(axes)
                if axis.x >= x
            ),
            None,
        )

    @staticmethod
    def _nearest_axis(
        x: float,
        axes: Sequence[_PageNumberAxis],
    ) -> tuple[int, float]:
        index = min(
            range(len(axes)),
            key=lambda axis_index: abs(axes[axis_index].x - x),
        )
        return index, abs(axes[index].x - x)

    def _axis_spread_tolerance(
        self,
        page_numbers: Sequence[_ChapterCandidate],
        page_width: float,
    ) -> float:
        median_width = self._median(
            abs(page_number.bbox.width)
            for page_number in page_numbers
        )
        return min(
            page_width
            * self.multicolumn_axis_max_spread_page_width_fraction,
            max(
                median_width
                * self.multicolumn_axis_spread_median_page_number_bbox_width_multiplier,
                page_width
                * self.multicolumn_axis_min_spread_page_width_fraction,
            ),
        )

    @staticmethod
    def _axis_diagnostics(
        axes: Sequence[_PageNumberAxis],
    ) -> list[dict[str, float | int]]:
        return [
            {
                "x": round(axis.x, 3),
                "support": len(axis.members),
                "spread": round(
                    max(
                        candidate.bbox.x_max
                        for candidate in axis.members
                    )
                    - min(
                        candidate.bbox.x_max
                        for candidate in axis.members
                    ),
                    3,
                ),
            }
            for axis in axes
        ]

    @staticmethod
    def _median(values) -> float:
        ordered = sorted(values)
        midpoint = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[midpoint]
        return (ordered[midpoint - 1] + ordered[midpoint]) / 2

    @staticmethod
    def _unique_candidate_count(
        candidates: Sequence[_ChapterCandidate],
    ) -> int:
        return len(set(candidates))

    @staticmethod
    def _resolve_page_width(page: ChapterPageInput) -> float:
        if page.image_dimensions is not None:
            return page.image_dimensions.width
        if page.alto_dimensions is not None:
            return page.alto_dimensions.width
        try:
            with Image.open(Path(page.image_path)) as image:
                width = float(image.width)
        except OSError as error:
            raise ValueError(
                f"Unable to read page width from image {page.image_path}"
            ) from error
        if not math.isfinite(width) or width <= 0:
            raise ValueError(
                f"Page image has invalid width {width:g}: "
                f"{page.image_path}"
            )
        return width

    def _integer_config(
        self,
        name: str,
        default: int,
        *,
        minimum: int,
    ) -> int:
        value = self.config.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
        return value

    def _positive_number_config(
        self,
        name: str,
        default: float,
    ) -> float:
        value = self.config.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric")
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and greater than zero")
        return value

    def _fraction_config(
        self,
        name: str,
        default: float,
        *,
        minimum_exclusive: bool = False,
    ) -> float:
        value = self.config.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric")
        value = float(value)
        minimum_valid = value > 0 if minimum_exclusive else value >= 0
        if not math.isfinite(value) or not minimum_valid or value > 1:
            boundary = "(0, 1]" if minimum_exclusive else "[0, 1]"
            raise ValueError(f"{name} must be in {boundary}")
        return value

    @staticmethod
    def _infer_titleless_entry_levels(units: list[_Unit]) -> None:
        preceding_titled_level = None
        for unit in units:
            if unit.title is not None:
                preceding_titled_level = unit.level
                continue
            unit.level = preceding_titled_level or 1

    @classmethod
    def _freeze(cls, entry: _MutableEntry) -> ChapterBase:
        return ChapterBase(
            toc_page_key=entry.toc_page_key,
            title=entry.title,
            subtitle=entry.subtitle,
            part_number=entry.part_number,
            page_number=entry.page_number,
            children=tuple(cls._freeze(child) for child in entry.children),
        )

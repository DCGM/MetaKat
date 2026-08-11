from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

from text_geometry_aligner import AlignmentRegion

from metakat.chapter.engines.core.models import (
    ChapterPageInput,
    ChapterBase,
    TocBase,
    TocPageNumber,
)
from metakat.common.models import DetectionEvidence
from metakat.chapter.engines.core.pipeline_utils import (
    load_chapter_label_mapping,
    load_engine_config,
    region_label,
    region_to_evidence,
)
from metakat.chapter.engines.core.toc_extraction.toc_page_number_parser import (
    ArabicRomanTocPageNumberParser,
)
from metakat.common.engines.engine_yolo_alto import EngineYOLOALTO
from metakat.schemas.base_objects import ChapterType

logger = logging.getLogger(__name__)


@dataclass
class _Unit:
    toc_page_key: str
    level: int | None = None
    title: DetectionEvidence | None = None
    part_number: DetectionEvidence | None = None
    page_number: DetectionEvidence | None = None


@dataclass
class _MutableEntry:
    toc_page_key: str
    title: DetectionEvidence | None
    level: int
    part_number: DetectionEvidence | None = None
    page_number: TocPageNumber | None = None
    children: list[_MutableEntry] = field(default_factory=list)


class TocExtractionEngineYOLOALTO:
    """Extract one cross-page TOC hierarchy from YOLO-aligned ALTO text."""

    DEFAULT_LABELS: dict[ChapterType, str] = {
        ChapterType.CHAPTER: "kapitola",
        ChapterType.SUBCHAPTER: "jiny nadpis",
        ChapterType.PAGE_NUMBER: "cislo strany",
        ChapterType.PART_NUMBER: "jine cislo",
    }

    def __init__(self, engine_dir, *, alignment_engine=None):
        self.engine_dir, self.config = load_engine_config(engine_dir)
        self.labels = load_chapter_label_mapping(
            self.config,
            self.DEFAULT_LABELS,
        )
        self.row_tolerance = float(self.config.get("row_tolerance", 20))
        self.overlap_threshold = float(
            self.config.get("overlap_threshold", 0.5)
        )
        if self.row_tolerance < 0:
            raise ValueError("row_tolerance must not be negative")
        if not 0 <= self.overlap_threshold <= 1:
            raise ValueError("overlap_threshold must be within [0, 1]")
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
            logger.info("TOC extraction received no selected TOC pages")
            return TocBase(())

        logger.info(
            "Extracting TOC hierarchy from %d page(s): pages=%s, "
            "row_tolerance=%.2f, overlap_threshold=%.3f",
            len(ordered_pages),
            [page.page_key for page in ordered_pages],
            self.row_tolerance,
            self.overlap_threshold,
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
                "TOC extraction alignment omitted page(s): "
                + ", ".join(missing)
            )

        units: list[_Unit] = []
        for source_page in ordered_pages:
            alignment = alignments[source_page.page_key]
            filtered_regions = self._filter_overlaps(alignment.regions)
            rows = self._rows(filtered_regions)
            page_units = self._group_units(rows, source_page.page_key)
            units.extend(page_units)
            logger.info(
                "TOC extraction page=%r: detected_regions=%d, "
                "regions_after_overlap_filter=%d, rows=%d, entries=%d",
                source_page.page_key,
                len(alignment.regions),
                len(filtered_regions),
                len(rows),
                len(page_units),
            )

        self._infer_titleless_entry_levels(units)
        for unit_index, unit in enumerate(units):
            self._parse_page_number(unit, unit_index)
            logger.debug(
                "Extracted TOC entry %d: page=%r, level=%s, title=%r, "
                "part_number=%r, page_number=%r",
                unit_index,
                unit.toc_page_key,
                unit.level,
                None if unit.title is None else unit.title.text,
                None if unit.part_number is None else unit.part_number.text,
                None if unit.page_number is None else unit.page_number.text,
            )
        roots: list[_MutableEntry] = []
        active_parents: dict[int, _MutableEntry] = {}
        for unit in units:
            level = unit.level or 1
            entry = _MutableEntry(
                toc_page_key=unit.toc_page_key,
                title=unit.title,
                level=level,
                part_number=unit.part_number,
                page_number=unit.page_number,
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
            "TOC extraction produced %d total entry/entries, %d root(s), "
            "%d titleless entry/entries, maximum_level=%d",
            len(units),
            len(result.chapters),
            sum(unit.title is None for unit in units),
            max((unit.level or 1 for unit in units), default=0),
        )
        return result

    @staticmethod
    def _parse_page_number(unit: _Unit, unit_index: int) -> None:
        if unit.page_number is None:
            return
        page_number = ArabicRomanTocPageNumberParser.create(unit.page_number)
        unit.page_number = page_number
        if not page_number.normalized_items:
            logger.warning(
                "TOC page number was rejected by the parser; its original "
                "evidence is retained: entry=%d, toc_page=%r, title=%r, "
                "source_number=%r",
                unit_index,
                unit.toc_page_key,
                None if unit.title is None else unit.title.text,
                page_number.text,
            )
            return
        normalized = page_number.normalized_text()
        if normalized != page_number.text:
            logger.debug(
                "Normalized TOC page number: entry=%d, toc_page=%r, "
                "source_number=%r, normalized_number=%r, kind=%s",
                unit_index,
                unit.toc_page_key,
                page_number.text,
                normalized,
                page_number.kind.value,
            )

    def _group_units(
        self,
        rows: Sequence[Sequence[AlignmentRegion]],
        page_key: str,
    ) -> tuple[_Unit, ...]:
        units: list[_Unit] = []
        for row in rows:
            unit = _Unit(toc_page_key=page_key)
            for region in row:
                evidence = region_to_evidence(region, page_key)
                if evidence is None:
                    continue
                label = region_label(region)
                if label == self.labels[ChapterType.PART_NUMBER]:
                    if unit.part_number is None:
                        unit.part_number = evidence
                elif label in {
                    self.labels[ChapterType.CHAPTER],
                    self.labels[ChapterType.SUBCHAPTER],
                }:
                    if unit.title is None:
                        unit.title = evidence
                        unit.level = (
                            2
                            if label == self.labels[ChapterType.SUBCHAPTER]
                            else 1
                        )
                elif label == self.labels[ChapterType.PAGE_NUMBER]:
                    if unit.page_number is None:
                        unit.page_number = evidence
            if unit.title is None and unit.page_number is None:
                continue
            units.append(unit)
        return tuple(units)

    @staticmethod
    def _infer_titleless_entry_levels(units: list[_Unit]) -> None:
        previous_levels: list[int | None] = []
        previous = None
        for unit in units:
            if unit.title is not None:
                previous = unit.level
            previous_levels.append(previous)

        next_levels: list[int | None] = [None] * len(units)
        following = None
        for index in range(len(units) - 1, -1, -1):
            unit = units[index]
            if unit.title is not None:
                following = unit.level
            next_levels[index] = following

        for index, unit in enumerate(units):
            if unit.title is not None:
                continue
            preceding = previous_levels[index]
            following = next_levels[index]
            if preceding is not None and preceding == following:
                unit.level = preceding
            elif preceding is not None:
                unit.level = preceding
            else:
                unit.level = 1

    def _rows(
        self,
        regions: Sequence[AlignmentRegion],
    ) -> tuple[tuple[AlignmentRegion, ...], ...]:
        with_geometry = [
            region for region in regions if region.input_geometry is not None
        ]
        if not with_geometry:
            return ()
        by_y = sorted(
            with_geometry,
            key=lambda region: region.input_geometry.bounds.y,
        )
        rows: list[tuple[AlignmentRegion, ...]] = []
        row = [by_y[0]]
        for region in by_y[1:]:
            if (
                abs(
                    region.input_geometry.bounds.y
                    - row[-1].input_geometry.bounds.y
                )
                < self.row_tolerance
            ):
                row.append(region)
            else:
                rows.append(
                    tuple(
                        sorted(
                            row,
                            key=lambda item: item.input_geometry.bounds.x,
                        )
                    )
                )
                row = [region]
        rows.append(
            tuple(
                sorted(
                    row,
                    key=lambda item: item.input_geometry.bounds.x,
                )
            )
        )
        return tuple(rows)

    def _filter_overlaps(
        self,
        regions: Sequence[AlignmentRegion],
    ) -> tuple[AlignmentRegion, ...]:
        ordered = sorted(
            (region for region in regions if region.input_geometry is not None),
            key=lambda region: region.input_geometry_confidence or 0.0,
            reverse=True,
        )
        kept: list[AlignmentRegion] = []
        for region in ordered:
            if all(
                self._intersection_over_union(region, previous)
                <= self.overlap_threshold
                for previous in kept
            ):
                kept.append(region)
        return tuple(kept)

    @staticmethod
    def _intersection_over_union(
        first: AlignmentRegion,
        second: AlignmentRegion,
    ) -> float:
        a = first.input_geometry.bounds
        b = second.input_geometry.bounds
        x1, y1 = max(a.x, b.x), max(a.y, b.y)
        x2, y2 = min(a.x_max, b.x_max), min(a.y_max, b.y_max)
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = a.width * a.height + b.width * b.height - intersection
        return intersection / union if union > 0 else 0.0

    @classmethod
    def _freeze(cls, entry: _MutableEntry) -> ChapterBase:
        return ChapterBase(
            toc_page_key=entry.toc_page_key,
            title=entry.title,
            part_number=entry.part_number,
            page_number=entry.page_number,
            children=tuple(cls._freeze(child) for child in entry.children),
        )

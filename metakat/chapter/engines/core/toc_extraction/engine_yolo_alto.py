from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

from text_geometry_aligner import AlignmentRegion

from metakat.chapter.engines.core.toc_extraction.models import (
    ReferenceToc,
    ReferenceTocEntry,
)
from metakat.chapter.engines.core.toc_page_analysis.models import (
    ChapterPageInput,
    DetectionEvidence,
)
from metakat.chapter.engines.core.pipeline_utils import (
    load_engine_config,
    region_label,
    region_to_evidence,
)
from metakat.common.engines.engine_yolo_alto import EngineYOLOALTO

logger = logging.getLogger(__name__)


@dataclass
class _Unit:
    toc_page_key: str
    level: int | None = None
    title: DetectionEvidence | None = None
    part_number: DetectionEvidence | None = None
    page_number: DetectionEvidence | None = None
    anchor_only: bool = False


@dataclass
class _MutableEntry:
    toc_page_key: str
    title: DetectionEvidence | None
    level: int
    part_number: DetectionEvidence | None = None
    page_number: DetectionEvidence | None = None
    anchor_only: bool = False
    children: list[_MutableEntry] = field(default_factory=list)


class TocExtractionEngineYOLOALTO:
    """Extract one cross-page TOC hierarchy from YOLO-aligned ALTO text."""

    DEFAULT_LABELS = {
        "title_level_1": "kapitola",
        "title_level_2": "jiny nadpis",
        "page_number": "cislo strany",
        "part_number": "jine cislo",
    }

    def __init__(self, engine_dir, *, alignment_engine=None):
        self.engine_dir, self.config = load_engine_config(engine_dir)
        self.labels = {**self.DEFAULT_LABELS, **self.config.get("labels", {})}
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
    ) -> ReferenceToc:
        ordered_pages = tuple(
            sorted(toc_pages, key=lambda page: page.position)
        )
        if not ordered_pages:
            return ReferenceToc(())

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
            rows = self._rows(
                self._filter_overlaps(alignment.regions)
            )
            units.extend(self._group_units(rows, source_page.page_key))

        self._infer_anchor_levels(units)
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
                anchor_only=unit.anchor_only,
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

        result = ReferenceToc(tuple(self._freeze(root) for root in roots))
        logger.info(
            "TOC extraction produced %d root entry/entries",
            len(result.roots),
        )
        return result

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
                if label == self.labels["part_number"]:
                    if unit.part_number is None:
                        unit.part_number = evidence
                elif label in {
                    self.labels["title_level_1"],
                    self.labels["title_level_2"],
                }:
                    if unit.title is None:
                        unit.title = evidence
                        unit.level = (
                            2
                            if label == self.labels["title_level_2"]
                            else 1
                        )
                elif label == self.labels["page_number"]:
                    if unit.page_number is None:
                        unit.page_number = evidence
            if unit.title is None and unit.page_number is None:
                continue
            unit.anchor_only = unit.title is None
            units.append(unit)
        return tuple(units)

    @staticmethod
    def _infer_anchor_levels(units: list[_Unit]) -> None:
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
            if not unit.anchor_only:
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
    def _freeze(cls, entry: _MutableEntry) -> ReferenceTocEntry:
        return ReferenceTocEntry(
            toc_page_key=entry.toc_page_key,
            title=entry.title,
            part_number=entry.part_number,
            page_number=entry.page_number,
            anchor_only=entry.anchor_only,
            children=tuple(cls._freeze(child) for child in entry.children),
        )

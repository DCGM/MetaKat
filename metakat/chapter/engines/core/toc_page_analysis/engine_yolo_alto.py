from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from text_geometry_aligner import ALTOReader, AlignmentPage

from metakat.chapter.engines.core.toc_page_analysis.models import (
    ChapterPageInput,
    DestinationChapterEvidence,
    DetectionEvidence,
    TocPageAnalysisResult,
)
from metakat.chapter.engines.core.pipeline_utils import (
    load_chapter_label_mapping,
    load_engine_config,
    normalize_text,
    region_label,
    region_to_evidence,
)
from metakat.common.engines.engine_yolo_alto import EngineYOLOALTO
from metakat.schemas.base_objects import ChapterType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TocCandidate:
    page: ChapterPageInput
    visual_score: int
    contains_keyword: bool


class TocPageAnalysisEngineYOLOALTO:
    """Select TOC pages and collect heading evidence from body pages."""

    DEFAULT_LABELS: dict[ChapterType, str] = {
        ChapterType.CHAPTER: "kapitola",
        ChapterType.SUBCHAPTER: "jiny nadpis",
        ChapterType.PAGE_NUMBER: "cislo strany",
        ChapterType.DESTINATION_CHAPTER: "nadpis v textu",
    }
    DEFAULT_TOC_KEYWORDS = (
        "obsah",
        "content",
        "contents",
        "table of contents",
        "содержание",
        "зміст",
        "inhalt",
        "sommaire",
    )

    def __init__(self, engine_dir, *, alignment_engine=None):
        self.engine_dir, self.config = load_engine_config(engine_dir)
        self.labels = load_chapter_label_mapping(
            self.config,
            self.DEFAULT_LABELS,
        )
        self.toc_keywords = tuple(
            normalize_text(keyword)
            for keyword in self.config.get(
                "toc_keywords",
                self.DEFAULT_TOC_KEYWORDS,
            )
        )
        self.toc_search_fraction = float(
            self.config.get("toc_search_fraction", 0.25)
        )
        self.keyword_top_fraction = float(
            self.config.get("keyword_top_fraction", 0.5)
        )
        if not 0 < self.toc_search_fraction <= 0.5:
            raise ValueError("toc_search_fraction must be within (0, 0.5]")
        if not 0 < self.keyword_top_fraction <= 1:
            raise ValueError("keyword_top_fraction must be within (0, 1]")
        self.alignment_engine = alignment_engine or EngineYOLOALTO(
            self.engine_dir
        )
        self.alto_reader = ALTOReader()

    def process(
        self,
        pages: Sequence[ChapterPageInput],
    ) -> TocPageAnalysisResult:
        ordered_pages = tuple(sorted(pages, key=lambda page: page.position))
        if not ordered_pages:
            logger.info("TOC page analysis received no pages")
            return TocPageAnalysisResult((), ())

        logger.info(
            "Analyzing %d page(s) for TOC candidates: search_fraction=%.3f, "
            "keyword_top_fraction=%.3f",
            len(ordered_pages),
            self.toc_search_fraction,
            self.keyword_top_fraction,
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
                "Page-analysis alignment omitted page(s): "
                + ", ".join(missing)
            )

        candidates: list[_TocCandidate] = []
        page_numbers = {}
        total_pages = len(ordered_pages)

        for ordinal, page in enumerate(ordered_pages, start=1):
            alignment = alignments[page.page_key]
            counts = Counter(region_label(region) for region in alignment.regions)
            primary_count = counts[self.labels[ChapterType.CHAPTER]]
            secondary_count = counts[self.labels[ChapterType.SUBCHAPTER]]
            page_number_count = counts[self.labels[ChapterType.PAGE_NUMBER]]
            toc_candidate = (
                primary_count >= 3 and page_number_count >= 3
            ) or (
                primary_count + secondary_count >= 3
                and page_number_count >= 2
            )
            in_toc_search_area = (
                ordinal < total_pages * self.toc_search_fraction
                or ordinal > total_pages * (1 - self.toc_search_fraction)
            )
            is_toc = toc_candidate and in_toc_search_area
            page_number = self._page_number_evidence(alignment)
            if page_number is not None:
                page_numbers[page.page_key] = page_number

            logger.debug(
                "TOC page candidate check: page=%r, position=%d, "
                "primary_headings=%d, secondary_headings=%d, "
                "page_numbers=%d, visual_candidate=%s, "
                "in_search_area=%s, accepted_candidate=%s, "
                "physical_page_number=%r",
                page.page_key,
                page.position,
                primary_count,
                secondary_count,
                page_number_count,
                toc_candidate,
                in_toc_search_area,
                is_toc,
                None if page_number is None else page_number.text,
            )

            if is_toc:
                candidates.append(
                    _TocCandidate(
                        page=page,
                        visual_score=(
                            primary_count
                            + secondary_count
                            + page_number_count
                        ),
                        contains_keyword=self._contains_toc_keyword(page),
                    )
                )

        selected_group = self._best_group(candidates)
        toc_pages = tuple(candidate.page for candidate in selected_group)
        toc_page_keys = {page.page_key for page in toc_pages}
        destination_chapters = tuple(
            evidence
            for page in ordered_pages
            if page.page_key not in toc_page_keys
            for evidence in self._destination_evidence(
                alignments[page.page_key]
            )
        )
        page_numbers = {
            page_key: evidence
            for page_key, evidence in page_numbers.items()
            if page_key not in toc_page_keys
        }
        if selected_group:
            logger.info(
                "Selected consecutive TOC block: pages=%s, positions=%s, "
                "contains_keyword=%s, visual_score=%d, candidates=%d",
                [candidate.page.page_key for candidate in selected_group],
                [candidate.page.position for candidate in selected_group],
                any(candidate.contains_keyword for candidate in selected_group),
                sum(candidate.visual_score for candidate in selected_group),
                len(candidates),
            )
        else:
            logger.warning(
                "No TOC page block selected from %d analyzed page(s)",
                len(ordered_pages),
            )
        logger.info(
            "Page analysis selected %d TOC page(s), %d destination "
            "heading(s), and %d physical page number(s)",
            len(toc_pages),
            len(destination_chapters),
            len(page_numbers),
        )
        return TocPageAnalysisResult(
            toc_pages=toc_pages,
            destination_chapters=destination_chapters,
            page_numbers=page_numbers,
        )

    def _page_number_evidence(
        self,
        page: AlignmentPage,
    ) -> DetectionEvidence | None:
        page_numbers = [
            evidence
            for region in page.regions
            if region_label(region) == self.labels[ChapterType.PAGE_NUMBER]
            if (evidence := region_to_evidence(region, page.page_key)) is not None
        ]
        page_number = max(
            page_numbers,
            key=lambda evidence: evidence.confidence,
            default=None,
        )
        return page_number

    def _destination_evidence(
        self,
        page: AlignmentPage,
    ) -> list[DestinationChapterEvidence]:
        return [
            DestinationChapterEvidence(title=title)
            for region in page.regions
            if region_label(region)
            == self.labels[ChapterType.DESTINATION_CHAPTER]
            if (title := region_to_evidence(region, page.page_key)) is not None
        ]

    def _contains_toc_keyword(self, page: ChapterPageInput) -> bool:
        alto = self.alto_reader.read(page.alto_path)
        height = alto.height
        if height is None and alto.words:
            height = max(word.bbox.y_max for word in alto.words)
        cutoff = None if height is None else height * self.keyword_top_fraction
        text = " ".join(
            word.text
            for word in alto.words
            if cutoff is None or word.bbox.y <= cutoff
        )
        normalized = normalize_text(text)
        return any(keyword in normalized for keyword in self.toc_keywords)

    @staticmethod
    def _best_group(
        candidates: Sequence[_TocCandidate],
    ) -> tuple[_TocCandidate, ...]:
        if not candidates:
            return ()
        groups: list[list[_TocCandidate]] = [[candidates[0]]]
        for candidate in candidates[1:]:
            if candidate.page.position == groups[-1][-1].page.position + 1:
                groups[-1].append(candidate)
            else:
                groups.append([candidate])

        trimmed_groups: list[list[_TocCandidate]] = []
        for group_index, group in enumerate(groups):
            keyword_index = next(
                (
                    index
                    for index, candidate in enumerate(group)
                    if candidate.contains_keyword
                ),
                None,
            )
            trimmed = group if keyword_index is None else group[keyword_index:]
            trimmed_groups.append(trimmed)
            logger.debug(
                "TOC candidate block %d: original_pages=%s, selected_suffix=%s, "
                "keyword_index=%s, contains_keyword=%s, visual_score=%d",
                group_index,
                [candidate.page.page_key for candidate in group],
                [candidate.page.page_key for candidate in trimmed],
                keyword_index,
                any(candidate.contains_keyword for candidate in trimmed),
                sum(candidate.visual_score for candidate in trimmed),
            )

        best = max(
            trimmed_groups,
            key=lambda group: (
                any(candidate.contains_keyword for candidate in group),
                sum(candidate.visual_score for candidate in group),
            ),
        )
        return tuple(best)

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
    load_engine_config,
    normalize_text,
    region_label,
    region_to_evidence,
)
from metakat.common.engines.engine_yolo_alto import EngineYOLOALTO

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TocCandidate:
    page: ChapterPageInput
    visual_score: int
    contains_keyword: bool


class TocPageAnalysisEngineYOLOALTO:
    """Select TOC pages and collect heading evidence from body pages."""

    DEFAULT_LABELS = {
        "toc_title": "kapitola",
        "toc_secondary_title": "jiny nadpis",
        "page_number": "cislo strany",
        "destination_title": "nadpis v textu",
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
        self.labels = {**self.DEFAULT_LABELS, **self.config.get("labels", {})}
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
            return TocPageAnalysisResult((), ())

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
        destination_chapters: list[DestinationChapterEvidence] = []
        page_numbers = {}
        toc_candidate_keys: set[str] = set()
        total_pages = len(ordered_pages)

        for ordinal, page in enumerate(ordered_pages, start=1):
            alignment = alignments[page.page_key]
            counts = Counter(region_label(region) for region in alignment.regions)
            primary_count = counts[self.labels["toc_title"]]
            secondary_count = counts[self.labels["toc_secondary_title"]]
            page_number_count = counts[self.labels["page_number"]]
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

            if is_toc:
                toc_candidate_keys.add(page.page_key)
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
                continue

            destination_chapters.extend(
                self._destination_evidence(alignment)
            )

        toc_pages = tuple(candidate.page for candidate in self._best_group(candidates))
        page_numbers = {
            page_key: evidence
            for page_key, evidence in page_numbers.items()
            if page_key not in toc_candidate_keys
        }
        logger.info(
            "Page analysis selected %d TOC page(s), %d destination "
            "heading(s), and %d physical page number(s)",
            len(toc_pages),
            len(destination_chapters),
            len(page_numbers),
        )
        return TocPageAnalysisResult(
            toc_pages=toc_pages,
            destination_chapters=tuple(destination_chapters),
            page_numbers=page_numbers,
        )

    def _page_number_evidence(
        self,
        page: AlignmentPage,
    ) -> DetectionEvidence | None:
        page_numbers = [
            evidence
            for region in page.regions
            if region_label(region) == self.labels["page_number"]
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
            if region_label(region) == self.labels["destination_title"]
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
        for group in groups:
            keyword_index = next(
                (
                    index
                    for index, candidate in enumerate(group)
                    if candidate.contains_keyword
                ),
                None,
            )
            trimmed_groups.append(
                group if keyword_index is None else group[keyword_index:]
            )

        best = max(
            trimmed_groups,
            key=lambda group: (
                any(candidate.contains_keyword for candidate in group),
                sum(candidate.visual_score for candidate in group),
            ),
        )
        return tuple(best)

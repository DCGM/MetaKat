from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass
from statistics import median
from typing import Sequence

from PIL import Image
from text_geometry_aligner import ALTOReader, AlignmentPage

from metakat.chapter.engines.core.models import ChapterPageInput
from metakat.common.models import DetectionEvidence
from metakat.chapter.engines.core.toc_page_analysis.models import (
    DestinationChapterEvidence,
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
from metakat.page_number.engines.core.page_number_resolver import (
    PageNumberSelectionMode,
    PhysicalPageNumberResolver,
)
from metakat.page_number.engines.core.models import (
    PhysicalPageNumberEvidence,
)
from metakat.schemas.base_objects import ChapterType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TocCandidate:
    page: ChapterPageInput
    visual_score: int
    contains_keyword: bool


@dataclass(frozen=True)
class _TocCandidateWindows:
    qualifying_window_count: int
    title_count: int
    page_number_count: int
    toc_area_top: float
    toc_area_bottom: float
    topmost_detection_bottom: float

    @property
    def visual_score(self) -> int:
        return self.title_count + self.page_number_count


class TocPageAnalysisEngineYOLOALTO:
    """Select TOC pages and collect title evidence from body pages."""

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
        self.toc_candidate_min_title_count = self.config.get(
            "toc_candidate_min_title_count",
            2,
        )
        self.toc_candidate_min_page_number_count = self.config.get(
            "toc_candidate_min_page_number_count",
            2,
        )
        self.toc_candidate_window_height_multiplier = float(
            self.config.get(
                "toc_candidate_window_height_multiplier",
                10.0,
            )
        )
        self.toc_candidate_min_window_height_fraction = float(
            self.config.get(
                "toc_candidate_min_window_height_fraction",
                0.2,
            )
        )
        self.toc_candidate_max_window_height_fraction = float(
            self.config.get(
                "toc_candidate_max_window_height_fraction",
                0.5,
            )
        )
        if not 0 < self.toc_search_fraction <= 0.5:
            raise ValueError("toc_search_fraction must be within (0, 0.5]")
        for option_name, option_value in (
            (
                "toc_candidate_min_title_count",
                self.toc_candidate_min_title_count,
            ),
            (
                "toc_candidate_min_page_number_count",
                self.toc_candidate_min_page_number_count,
            ),
        ):
            if (
                isinstance(option_value, bool)
                or not isinstance(option_value, int)
                or option_value < 1
            ):
                raise ValueError(f"{option_name} must be a positive integer")
        if (
            not math.isfinite(
                self.toc_candidate_window_height_multiplier
            )
            or self.toc_candidate_window_height_multiplier <= 0
        ):
            raise ValueError(
                "toc_candidate_window_height_multiplier must be positive"
            )
        if (
            not math.isfinite(
                self.toc_candidate_min_window_height_fraction
            )
            or not 0
            < self.toc_candidate_min_window_height_fraction
            <= 1
        ):
            raise ValueError(
                "toc_candidate_min_window_height_fraction must be within "
                "(0, 1]"
            )
        if (
            not math.isfinite(
                self.toc_candidate_max_window_height_fraction
            )
            or not 0
            < self.toc_candidate_max_window_height_fraction
            <= 1
        ):
            raise ValueError(
                "toc_candidate_max_window_height_fraction must be within "
                "(0, 1]"
            )
        if (
            self.toc_candidate_min_window_height_fraction
            > self.toc_candidate_max_window_height_fraction
        ):
            raise ValueError(
                "toc_candidate_min_window_height_fraction must not exceed "
                "toc_candidate_max_window_height_fraction"
            )
        self.alignment_engine = alignment_engine or EngineYOLOALTO(
            self.engine_dir
        )
        self.page_number_resolver = (
            PhysicalPageNumberResolver.from_config(self.config)
        )
        self.alto_reader = ALTOReader()

    def process(
        self,
        pages: Sequence[ChapterPageInput],
    ) -> TocPageAnalysisResult:
        ordered_pages = tuple(sorted(pages, key=lambda page: page.position))
        if not ordered_pages:
            logger.info("TOC page analysis received no pages")
            return TocPageAnalysisResult((), (), ())

        logger.info(
            "Analyzing %d page(s) for TOC candidates: search_fraction=%.3f, "
            "minimum_titles=%d, minimum_page_numbers=%d, "
            "window_height_multiplier=%.3f, "
            "minimum_window_height_fraction=%.3f, "
            "maximum_window_height_fraction=%.3f",
            len(ordered_pages),
            self.toc_search_fraction,
            self.toc_candidate_min_title_count,
            self.toc_candidate_min_page_number_count,
            self.toc_candidate_window_height_multiplier,
            self.toc_candidate_min_window_height_fraction,
            self.toc_candidate_max_window_height_fraction,
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
        total_pages = len(ordered_pages)

        for page in ordered_pages:
            alignment = alignments[page.page_key]
            distance_from_start = page.position
            distance_from_end = total_pages - page.position - 1
            in_toc_search_area = (
                distance_from_start
                < total_pages * self.toc_search_fraction
                or distance_from_end
                < total_pages * self.toc_search_fraction
            )
            primary_count = None
            secondary_count = None
            page_number_count = None
            candidate_windows = None
            if in_toc_search_area:
                counts = Counter(
                    region_label(region) for region in alignment.regions
                )
                primary_count = counts[self.labels[ChapterType.CHAPTER]]
                secondary_count = counts[self.labels[ChapterType.SUBCHAPTER]]
                page_number_count = counts[
                    self.labels[ChapterType.PAGE_NUMBER]
                ]
                candidate_windows = self._find_candidate_windows(
                    alignment,
                    page,
                )
            toc_candidate = candidate_windows is not None
            is_toc = toc_candidate and in_toc_search_area
            logger.debug(
                "TOC page candidate check: page=%r, position=%d, "
                "in_search_area=%s, primary_titles=%s, "
                "secondary_titles=%s, page_numbers=%s, "
                "visual_candidate=%s, "
                "candidate_windows=%s, "
                "accepted_candidate=%s",
                page.page_key,
                page.position,
                in_toc_search_area,
                primary_count,
                secondary_count,
                page_number_count,
                toc_candidate,
                None
                if candidate_windows is None
                else {
                    "qualifying_windows": (
                        candidate_windows.qualifying_window_count
                    ),
                    "cumulative_titles": candidate_windows.title_count,
                    "cumulative_page_numbers": (
                        candidate_windows.page_number_count
                    ),
                    "cumulative_visual_score": (
                        candidate_windows.visual_score
                    ),
                },
                is_toc,
            )

            if is_toc:
                assert candidate_windows is not None
                candidates.append(
                    _TocCandidate(
                        page=page,
                        visual_score=candidate_windows.visual_score,
                        contains_keyword=self._contains_toc_keyword(
                            page,
                            toc_area_top=(
                                candidate_windows.toc_area_top
                            ),
                            topmost_detection_bottom=(
                                candidate_windows.topmost_detection_bottom
                            ),
                        ),
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
        destination_page_numbers = tuple(
            evidence
            for page in ordered_pages
            if page.page_key not in toc_page_keys
            if (
                evidence := self._page_number_evidence(
                    alignments[page.page_key],
                    page,
                    mode=PageNumberSelectionMode.STANDARD,
                )
            )
            is not None
        )
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
            "title(s), and %d destination page number(s)",
            len(toc_pages),
            len(destination_chapters),
            len(destination_page_numbers),
        )
        return TocPageAnalysisResult(
            toc_pages=toc_pages,
            destination_chapters=destination_chapters,
            destination_page_numbers=destination_page_numbers,
        )

    def _find_candidate_windows(
        self,
        alignment: AlignmentPage,
        page: ChapterPageInput,
    ) -> _TocCandidateWindows | None:
        title_labels = {
            self.labels[ChapterType.CHAPTER],
            self.labels[ChapterType.SUBCHAPTER],
        }
        page_number_label = self.labels[ChapterType.PAGE_NUMBER]
        relevant_regions = [
            region
            for region in alignment.regions
            if region.input_geometry is not None
            and (
                region_label(region) in title_labels
                or region_label(region) == page_number_label
            )
        ]
        if not relevant_regions:
            return None

        region_heights = [
            region.input_geometry.bounds.height
            for region in relevant_regions
        ]
        median_region_height = median(region_heights)
        page_height = self._page_height(page)
        window_height = min(
            max(
                median_region_height
                * self.toc_candidate_window_height_multiplier,
                page_height
                * self.toc_candidate_min_window_height_fraction,
            ),
            page_height
            * self.toc_candidate_max_window_height_fraction,
        )

        positioned_regions = sorted(
            (
                (
                    region.input_geometry.bounds.y
                    + region.input_geometry.bounds.height / 2,
                    region_label(region),
                    index,
                )
                for index, region in enumerate(relevant_regions)
            ),
            key=lambda item: item[0],
        )
        qualifying_window_count = 0
        covered_region_indices: set[int] = set()
        right = 0
        for left, (y_start, _, _) in enumerate(positioned_regions):
            right = max(right, left)
            while (
                right < len(positioned_regions)
                and positioned_regions[right][0] - y_start <= window_height
            ):
                right += 1
            window_regions = positioned_regions[left:right]
            title_count = sum(
                label in title_labels for _, label, _ in window_regions
            )
            page_number_count = sum(
                label == page_number_label
                for _, label, _ in window_regions
            )
            if (
                title_count >= self.toc_candidate_min_title_count
                and page_number_count
                >= self.toc_candidate_min_page_number_count
            ):
                qualifying_window_count += 1
                covered_region_indices.update(
                    index for _, _, index in window_regions
                )

        if not qualifying_window_count:
            logger.debug(
                "No TOC-like region window: page=%r, relevant_regions=%d, "
                "median_region_height=%.3f, page_height=%s, "
                "window_height=%.3f",
                page.page_key,
                len(relevant_regions),
                median_region_height,
                page_height,
                window_height,
            )
            return None

        covered_title_count = sum(
            region_label(relevant_regions[index]) in title_labels
            for index in covered_region_indices
        )
        covered_page_number_count = sum(
            region_label(relevant_regions[index]) == page_number_label
            for index in covered_region_indices
        )
        covered_bounds = [
            relevant_regions[index].input_geometry.bounds
            for index in covered_region_indices
        ]
        topmost_bounds = min(
            covered_bounds,
            key=lambda bounds: (bounds.y, bounds.y_max),
        )
        result = _TocCandidateWindows(
            qualifying_window_count=qualifying_window_count,
            title_count=covered_title_count,
            page_number_count=covered_page_number_count,
            toc_area_top=min(bounds.y for bounds in covered_bounds),
            toc_area_bottom=max(bounds.y_max for bounds in covered_bounds),
            topmost_detection_bottom=topmost_bounds.y_max,
        )
        logger.debug(
            "Qualified TOC-like region windows: page=%r, "
            "qualifying_windows=%d, cumulative_titles=%d, "
            "cumulative_page_numbers=%d, cumulative_visual_score=%d, "
            "toc_area_top=%.3f, toc_area_bottom=%.3f, "
            "topmost_detection_bottom=%.3f, "
            "median_region_height=%.3f, page_height=%s, "
            "window_height=%.3f",
            page.page_key,
            result.qualifying_window_count,
            result.title_count,
            result.page_number_count,
            result.visual_score,
            result.toc_area_top,
            result.toc_area_bottom,
            result.topmost_detection_bottom,
            median_region_height,
            page_height,
            window_height,
        )
        return result

    @staticmethod
    def _page_height(page: ChapterPageInput) -> float:
        if page.image_dimensions is not None:
            return page.image_dimensions.height
        if page.alto_dimensions is not None:
            return page.alto_dimensions.height
        try:
            with Image.open(page.image_path) as image:
                height = image.height
        except OSError as error:
            raise ValueError(
                f"Unable to read page height from image {page.image_path}"
            ) from error
        if height <= 0:
            raise ValueError(
                f"Page image has invalid height {height}: {page.image_path}"
            )
        return float(height)

    def _page_number_evidence(
        self,
        alignment: AlignmentPage,
        page: ChapterPageInput,
        *,
        mode: PageNumberSelectionMode,
    ) -> PhysicalPageNumberEvidence | None:
        regions = tuple(
            region
            for region in alignment.regions
            if region_label(region)
            == self.labels[ChapterType.PAGE_NUMBER]
        )
        if not regions:
            return None
        page_height = (
            self._page_height(page)
            if (
                mode is PageNumberSelectionMode.EDGE_ONLY
                or len(regions) > 1
            )
            else alignment.alto_height
        )
        resolution = self.page_number_resolver.resolve(
            alignment,
            regions,
            mode=mode,
            page_height=page_height,
        )
        return resolution.selected_evidence

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

    def _contains_toc_keyword(
        self,
        page: ChapterPageInput,
        *,
        toc_area_top: float,
        topmost_detection_bottom: float,
    ) -> bool:
        alto = self.alto_reader.read(page.alto_path)
        lines: dict[tuple[object, ...], list] = {}
        for word in alto.words:
            line_key: tuple[object, ...] = (
                ("line", word.block_index, word.line_index)
                if word.line_index is not None
                else ("word", word.index)
            )
            lines.setdefault(line_key, []).append(word)

        valid_occurrences: list[
            tuple[float, str, float, float, float, float]
        ] = []
        for words in lines.values():
            normalized = normalize_text(
                " ".join(word.text for word in words)
            )
            line_top = min(word.bbox.y for word in words)
            line_left = min(word.bbox.x for word in words)
            line_right = max(word.bbox.x_max for word in words)
            line_bottom = max(word.bbox.y_max for word in words)
            for keyword in self.toc_keywords:
                if (
                    keyword in normalized
                    and line_top <= topmost_detection_bottom
                ):
                    valid_occurrences.append(
                        (
                            abs(toc_area_top - line_top),
                            keyword,
                            line_left,
                            line_top,
                            line_right - line_left,
                            line_bottom - line_top,
                        )
                    )

        if not valid_occurrences:
            logger.debug(
                "No valid TOC keyword occurrence: page=%r, "
                "toc_area_top=%.3f, topmost_detection_bottom=%.3f",
                page.page_key,
                toc_area_top,
                topmost_detection_bottom,
            )
            return False

        for (
            distance,
            keyword,
            line_x,
            line_y,
            line_width,
            line_height,
        ) in sorted(valid_occurrences, key=lambda occurrence: occurrence[3]):
            logger.debug(
                "Valid TOC keyword occurrence: page=%r, keyword=%r, "
                "line_bbox=(x=%.3f, y=%.3f, width=%.3f, height=%.3f), "
                "toc_area_top=%.3f, distance=%.3f",
                page.page_key,
                keyword,
                line_x,
                line_y,
                line_width,
                line_height,
                toc_area_top,
                distance,
            )
        return True

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

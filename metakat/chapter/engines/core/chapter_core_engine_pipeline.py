from __future__ import annotations

import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from metakat.chapter.engines.core.chapter_core_engine import ChapterCoreEngine
from metakat.chapter.engines.core.models import (
    ChapterPageInput,
    ChapterResult,
    TocResult,
)
from metakat.chapter.engines.core.pipeline_utils import load_engine_config
from metakat.common.models import PageDimensions
from metakat.chapter.engines.core.toc_alignment import (
    TocAlignmentEngine,
    TocAlignmentEngineFuzzy,
)
from metakat.chapter.engines.core.toc_extraction import (
    TocExtractionEngine,
    TocExtractionEngineYOLOALTO,
)
from metakat.chapter.engines.core.toc_page_analysis import (
    DestinationChapterEvidence,
    TocPageAnalysisEngine,
    TocPageAnalysisEngineYOLOALTO,
)
from metakat.page_number.engines.core.models import (
    PhysicalPageNumberEvidence,
)

logger = logging.getLogger(__name__)


class ChapterPipelineCoreEngine(ChapterCoreEngine):
    """Compose independently replaceable chapter-processing stages."""

    def __init__(
        self,
        core_engine_dir,
        *,
        page_analysis_engine: TocPageAnalysisEngine | None = None,
        toc_extraction_engine: TocExtractionEngine | None = None,
        toc_alignment_engine: TocAlignmentEngine | None = None,
    ):
        super().__init__(core_engine_dir)
        stage_paths = self.config.get("stages", {})
        self.page_analysis_engine = page_analysis_engine or self._load_stage(
            "toc_page_analysis",
            stage_paths,
        )
        self.toc_extraction_engine = toc_extraction_engine or self._load_stage(
            "toc_extraction",
            stage_paths,
        )
        self.toc_alignment_engine = toc_alignment_engine or self._load_stage(
            "toc_alignment",
            stage_paths,
        )

    def process(
        self,
        images: Sequence[str],
        alto_files: Sequence[str],
        page_numbers: Sequence[PhysicalPageNumberEvidence] | None = None,
        image_dimensions: Sequence[PageDimensions | None] | None = None,
        alto_dimensions: Sequence[PageDimensions | None] | None = None,
    ) -> TocResult:
        pages = self._pair_pages(
            images,
            alto_files,
            image_dimensions,
            alto_dimensions,
        )
        external_destination_page_numbers = (
            self._external_destination_page_numbers(page_numbers)
        )
        external_page_number_count = (
            0
            if external_destination_page_numbers is None
            else len(external_destination_page_numbers)
        )
        logger.info(
            "Starting chapter pipeline: pages=%d, range=%r..%r, "
            "page_number_source=%s, available_page_numbers=%d",
            len(pages),
            None if not pages else pages[0].page_key,
            None if not pages else pages[-1].page_key,
            "external" if page_numbers is not None else "page analysis",
            external_page_number_count,
        )

        stage_started = time.perf_counter()
        logger.info("Starting TOC page analysis stage")
        analysis = self.page_analysis_engine.process(pages)
        logger.info(
            "Completed TOC page analysis stage in %.3f s: "
            "toc_pages=%d, destination_titles=%s, "
            "destination_page_numbers=%s",
            time.perf_counter() - stage_started,
            len(analysis.toc_pages),
            _evidence_capability_status(analysis.destination_chapters),
            _evidence_capability_status(
                analysis.destination_page_numbers
            ),
        )
        _validate_toc_pages(analysis.toc_pages, pages)
        effective_destination_page_numbers = (
            analysis.destination_page_numbers
            if external_destination_page_numbers is None
            else external_destination_page_numbers
        )
        if (
            analysis.destination_chapters is None
            and effective_destination_page_numbers is None
        ):
            raise ValueError(
                "The three-stage chapter pipeline requires stage 1 or "
                "external input to implement destination-title or "
                "destination-page-number evidence"
            )
        if not analysis.toc_pages:
            logger.warning(
                "No TOC pages were selected during page analysis; "
                "stopping chapter pipeline before TOC extraction and "
                "alignment"
            )
            return TocResult(chapters=())

        stage_started = time.perf_counter()
        logger.info(
            "Starting TOC extraction stage for pages=%s",
            [page.page_key for page in analysis.toc_pages],
        )
        reference_toc = self.toc_extraction_engine.process(
            analysis.toc_pages
        )
        logger.info(
            "Completed TOC extraction stage in %.3f s: roots=%d",
            time.perf_counter() - stage_started,
            len(reference_toc.chapters),
        )
        toc_page_keys = {
            page.page_key for page in analysis.toc_pages
        }
        destination_chapters = _filter_destination_chapters(
            analysis.destination_chapters,
            pages,
            toc_page_keys,
        )
        destination_page_numbers = _filter_destination_page_numbers(
            effective_destination_page_numbers,
            pages,
            toc_page_keys,
        )
        logger.info(
            "Destination page-number evidence for TOC alignment: "
            "source=%s, status=%s",
            "internally detected"
            if external_destination_page_numbers is None
            else "externally supplied",
            _evidence_capability_status(destination_page_numbers),
        )

        logger.info(
            "Passing all %d document page(s) and %d selected TOC page(s) "
            "to TOC alignment; destination_titles=%s, "
            "destination_page_numbers=%s",
            len(pages),
            len(toc_page_keys),
            _evidence_capability_status(destination_chapters),
            _evidence_capability_status(destination_page_numbers),
        )

        stage_started = time.perf_counter()
        logger.info("Starting TOC alignment stage")
        aligned_toc = self.toc_alignment_engine.process(
            pages=pages,
            toc_pages=analysis.toc_pages,
            reference_toc=reference_toc,
            destination_chapters=destination_chapters,
            destination_page_numbers=destination_page_numbers,
        )
        logger.info(
            "Completed TOC alignment stage in %.3f s: root_chapters=%d, "
            "total_chapters=%d",
            time.perf_counter() - stage_started,
            len(aligned_toc.chapters),
            _count_resolved_chapters(aligned_toc.chapters),
        )
        return self._prune_titleless_chapters(aligned_toc)

    @staticmethod
    def _prune_titleless_chapters(toc: TocResult) -> TocResult:
        pruned_count = 0

        def prune(
            chapters: Sequence[ChapterResult],
        ) -> tuple[ChapterResult, ...]:
            nonlocal pruned_count
            retained: list[ChapterResult] = []
            for chapter in chapters:
                children = prune(chapter.children)
                if (
                    chapter.title is None
                    and chapter.title_destination_page is None
                ):
                    pruned_count += 1
                    retained.extend(children)
                    continue
                if children == chapter.children:
                    retained.append(chapter)
                else:
                    retained.append(replace(chapter, children=children))
            return tuple(retained)

        chapters = prune(toc.chapters)
        if pruned_count == 0:
            return toc
        logger.info(
            "Pruned %d titleless chapter entry/entries from the chapter "
            "pipeline result and promoted their retained children",
            pruned_count,
        )
        return TocResult(chapters=chapters)

    def _load_stage(self, stage_name: str, stage_paths: dict):
        configured_path = stage_paths.get(
            stage_name,
            self.config.get(f"{stage_name}_engine"),
        )
        if not configured_path:
            raise ValueError(
                f"Missing stages.{stage_name} in pipeline engine config"
            )
        stage_dir = Path(configured_path)
        if not stage_dir.is_absolute():
            stage_dir = self.engine_dir / stage_dir
        _, stage_config = load_engine_config(stage_dir)
        implementation = stage_config.get("name")
        registries = {
            "toc_page_analysis": {
                "toc_page_analysis_engine_yolo_alto": (
                    TocPageAnalysisEngineYOLOALTO
                ),
            },
            "toc_extraction": {
                "toc_extraction_engine_yolo_alto": (
                    TocExtractionEngineYOLOALTO
                ),
            },
            "toc_alignment": {
                "toc_alignment_engine_fuzzy": TocAlignmentEngineFuzzy,
            },
        }
        engine_class = registries[stage_name].get(implementation)
        if engine_class is None:
            raise ValueError(
                f"Unknown {stage_name} engine: {implementation!r}"
            )
        return engine_class(stage_dir)

    @staticmethod
    def _pair_pages(
        images: Sequence[str],
        alto_files: Sequence[str],
        image_dimensions: Sequence[PageDimensions | None] | None = None,
        alto_dimensions: Sequence[PageDimensions | None] | None = None,
    ) -> tuple[ChapterPageInput, ...]:
        image_paths = tuple(Path(path) for path in images)
        alto_paths = tuple(Path(path) for path in alto_files)
        supplied_image_dimensions = (
            None if image_dimensions is None else tuple(image_dimensions)
        )
        supplied_alto_dimensions = (
            None if alto_dimensions is None else tuple(alto_dimensions)
        )
        for option_name, dimensions in (
            ("Image-dimension", supplied_image_dimensions),
            ("ALTO-dimension", supplied_alto_dimensions),
        ):
            if dimensions is not None and len(dimensions) != len(image_paths):
                raise ValueError(
                    f"{option_name} inputs must have the same length as "
                    f"images: {len(dimensions)} != {len(image_paths)}"
                )
            if dimensions is not None:
                for position, item in enumerate(dimensions):
                    if item is not None and not isinstance(
                        item,
                        PageDimensions,
                    ):
                        raise TypeError(
                            f"{option_name} inputs must contain only "
                            "PageDimensions or None; item "
                            f"{position} is {type(item).__name__}"
                        )
        for path in (*image_paths, *alto_paths):
            if not path.is_file():
                raise FileNotFoundError(f"Chapter pipeline input not found: {path}")

        images_by_stem = _unique_by_stem(image_paths, "image")
        altos_by_stem = _unique_by_stem(alto_paths, "ALTO")
        missing_altos = images_by_stem.keys() - altos_by_stem.keys()
        missing_images = altos_by_stem.keys() - images_by_stem.keys()
        if missing_altos or missing_images:
            details = []
            if missing_altos:
                details.append(
                    "missing ALTO for " + ", ".join(sorted(missing_altos))
                )
            if missing_images:
                details.append(
                    "missing image for " + ", ".join(sorted(missing_images))
                )
            raise ValueError("Cannot pair chapter inputs: " + "; ".join(details))

        return tuple(
            ChapterPageInput(
                page_key=image_path.stem,
                position=position,
                image_path=image_path,
                alto_path=altos_by_stem[image_path.stem],
                image_dimensions=(
                    None
                    if supplied_image_dimensions is None
                    else supplied_image_dimensions[position]
                ),
                alto_dimensions=(
                    None
                    if supplied_alto_dimensions is None
                    else supplied_alto_dimensions[position]
                ),
            )
            for position, image_path in enumerate(image_paths)
        )

    @staticmethod
    def _external_destination_page_numbers(
        page_numbers: Sequence[PhysicalPageNumberEvidence] | None,
    ) -> tuple[PhysicalPageNumberEvidence, ...] | None:
        if page_numbers is None:
            return None
        supplied_page_numbers = tuple(page_numbers)
        for position, page_number in enumerate(supplied_page_numbers):
            if not isinstance(page_number, PhysicalPageNumberEvidence):
                raise TypeError(
                    "Page-number inputs must contain only "
                    "PhysicalPageNumberEvidence; "
                    f"item {position} is {type(page_number).__name__}"
                )
        return supplied_page_numbers


def _unique_by_stem(
    paths: Sequence[Path],
    kind: str,
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in paths:
        if path.stem in result:
            raise ValueError(
                f"{kind} inputs must have unique stems: "
                f"{result[path.stem]} and {path}"
            )
        result[path.stem] = path
    return result


def _count_resolved_chapters(chapters) -> int:
    return sum(
        1 + _count_resolved_chapters(chapter.children)
        for chapter in chapters
    )


def _evidence_capability_status(evidence: Sequence | None) -> str:
    if evidence is None:
        return "unavailable"
    if not evidence:
        return "implemented-empty"
    return f"implemented(count={len(evidence)})"


def _validate_toc_pages(
    toc_pages: Sequence[ChapterPageInput],
    pages: Sequence[ChapterPageInput],
) -> None:
    known_page_keys = {page.page_key for page in pages}
    seen_page_keys: set[str] = set()
    for page in toc_pages:
        if page.page_key not in known_page_keys:
            raise ValueError(
                "TOC page analysis returned an unknown page_key: "
                f"{page.page_key!r}"
            )
        if page.page_key in seen_page_keys:
            raise ValueError(
                "TOC page analysis returned a duplicate page_key: "
                f"{page.page_key!r}"
            )
        seen_page_keys.add(page.page_key)


def _filter_destination_chapters(
    evidence: Sequence[DestinationChapterEvidence] | None,
    pages: Sequence[ChapterPageInput],
    toc_page_keys: set[str],
) -> tuple[DestinationChapterEvidence, ...] | None:
    if evidence is None:
        return None
    known_page_keys = {page.page_key for page in pages}
    filtered = []
    for item in evidence:
        page_key = item.title.page_key
        if page_key not in known_page_keys:
            raise ValueError(
                "Destination title evidence refers to an unknown page_key: "
                f"{page_key!r}"
            )
        if page_key not in toc_page_keys:
            filtered.append(item)
    return tuple(filtered)


def _filter_destination_page_numbers(
    evidence: Sequence[PhysicalPageNumberEvidence] | None,
    pages: Sequence[ChapterPageInput],
    toc_page_keys: set[str],
) -> tuple[PhysicalPageNumberEvidence, ...] | None:
    if evidence is None:
        return None
    known_page_keys = {page.page_key for page in pages}
    seen_page_keys: set[str] = set()
    filtered = []
    for item in evidence:
        if item.page_key not in known_page_keys:
            raise ValueError(
                "Destination page-number evidence refers to an unknown "
                f"page_key: {item.page_key!r}"
            )
        if item.page_key in seen_page_keys:
            raise ValueError(
                "Destination page-number evidence contains duplicate "
                f"page_key: {item.page_key!r}"
            )
        seen_page_keys.add(item.page_key)
        if item.page_key not in toc_page_keys:
            filtered.append(item)
    return tuple(filtered)

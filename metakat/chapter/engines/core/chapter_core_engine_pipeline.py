from __future__ import annotations

import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from metakat.chapter.engines.core.chapter_core_engine import ChapterCoreEngine
from metakat.chapter.engines.core.pipeline_utils import load_engine_config
from metakat.chapter.engines.core.toc_alignment import (
    TocAlignmentEngine,
    TocAlignmentEngineFuzzy,
)
from metakat.chapter.engines.core.toc_alignment.models import (
    ChapterCoreResult,
)
from metakat.chapter.engines.core.toc_extraction import (
    TocExtractionEngine,
    TocExtractionEngineYOLOALTO,
)
from metakat.chapter.engines.core.toc_page_analysis import (
    ChapterPageInput,
    TocPageAnalysisEngine,
    TocPageAnalysisEngineYOLOALTO,
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
        page_numbers: Sequence[str | None] | None = None,
    ) -> ChapterCoreResult:
        pages = self._pair_pages(images, alto_files, page_numbers)
        external_page_number_count = sum(
            page.page_number is not None for page in pages
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
            "toc_pages=%d, destination_headings=%d, detected_page_numbers=%d",
            time.perf_counter() - stage_started,
            len(analysis.toc_pages),
            len(analysis.destination_chapters),
            len(analysis.page_numbers),
        )

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
            len(reference_toc.roots),
        )
        if page_numbers is None:
            pages = tuple(
                replace(
                    page,
                    page_number=(
                        evidence.text
                        if (
                            evidence := analysis.page_numbers.get(
                                page.page_key
                            )
                        )
                        is not None
                        else None
                    ),
                )
                for page in pages
            )
            logger.info(
                "Using %d physical page number(s) detected during page "
                "analysis for TOC alignment",
                sum(page.page_number is not None for page in pages),
            )

        stage_started = time.perf_counter()
        logger.info("Starting TOC alignment stage")
        result = self.toc_alignment_engine.process(
            pages=pages,
            reference_toc=reference_toc,
            destination_chapters=analysis.destination_chapters,
        )
        logger.info(
            "Completed TOC alignment stage in %.3f s: root_chapters=%d, "
            "total_chapters=%d",
            time.perf_counter() - stage_started,
            len(result.chapters),
            _count_resolved_chapters(result.chapters),
        )
        return result

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
        page_numbers: Sequence[str | None] | None = None,
    ) -> tuple[ChapterPageInput, ...]:
        image_paths = tuple(Path(path) for path in images)
        alto_paths = tuple(Path(path) for path in alto_files)
        supplied_page_numbers = (
            None if page_numbers is None else tuple(page_numbers)
        )
        if (
            supplied_page_numbers is not None
            and len(supplied_page_numbers) != len(image_paths)
        ):
            raise ValueError(
                "Page-number inputs must have the same length as images: "
                f"{len(supplied_page_numbers)} != {len(image_paths)}"
            )
        if supplied_page_numbers is not None:
            for position, page_number in enumerate(supplied_page_numbers):
                if page_number is not None and not isinstance(page_number, str):
                    raise TypeError(
                        "Page-number inputs must contain only strings or "
                        f"None; item {position} is {type(page_number).__name__}"
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
                page_number=(
                    None
                    if supplied_page_numbers is None
                    else supplied_page_numbers[position]
                ),
            )
            for position, image_path in enumerate(image_paths)
        )


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

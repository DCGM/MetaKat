from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
from uuid import UUID, uuid4

from metakat.common.aux.document_groups import (
    LowestDocumentGroup,
    lowest_document_groups,
)
from metakat.chapter.engines.bind.chapter_bind_engine import ChapterBindEngine
from metakat.chapter.engines.core.models import (
    ChapterResult,
    TocResult,
)
from metakat.common.models import (
    BoundingBox,
    DetectionEvidence,
    PageDimensions,
)
from metakat.page_number.engines.core.models import (
    PhysicalPageNumberEvidence,
)
from metakat.page_number.engines.core.page_number_parsers import (
    DecoratedPageNumberParser,
)
from metakat.schemas.base_objects import (
    DocumentType,
    MetakatChapter,
    MetakatElement,
    MetakatIO,
    MetakatPage,
    ProarcIO,
)

logger = logging.getLogger(__name__)


@dataclass
class _BoundChapter:
    chapter: MetakatChapter
    depth: int
    container_id: UUID


class ChapterBindEngineBase(ChapterBindEngine):
    def __init__(self, config, core_config):
        super().__init__(config, core_config)

    def process(
        self,
        batch_dir: str,
        metakat_io: MetakatIO,
        proarc_io: ProarcIO = None,
    ) -> MetakatIO:
        metakat_io = copy.deepcopy(metakat_io)
        image_mapping = metakat_io.page_to_image_mapping or {}
        alto_mapping = metakat_io.page_to_alto_mapping or {}
        if metakat_io.detection_to_bbox is None:
            metakat_io.detection_to_bbox = {}
        if metakat_io.detection_to_page_mapping is None:
            metakat_io.detection_to_page_mapping = {}

        groups = self._document_groups(metakat_io)
        logger.info(
            "Starting chapter binding for %d lowest-level document(s)",
            len(groups),
        )
        for group in groups:
            processable_pages = [
                page
                for page in group.pages
                if page.id in image_mapping and page.id in alto_mapping
            ]
            skipped = len(group.pages) - len(processable_pages)
            if skipped:
                logger.warning(
                    "Skipping %d page(s) without both image and ALTO "
                    "mappings in %s %s",
                    skipped,
                    group.container.type,
                    group.container.id,
                )
            if not processable_pages:
                logger.warning(
                    "Skipping chapter processing for %s %s because it has "
                    "no pages with both image and ALTO mappings",
                    group.container.type,
                    group.container.id,
                )
                continue

            images = [
                os.path.join(batch_dir, image_mapping[page.id])
                for page in processable_pages
            ]
            alto_files = [
                os.path.join(batch_dir, alto_mapping[page.id])
                for page in processable_pages
            ]
            logger.info(
                "Processing %d page(s) from %s %s with chapter core engine: "
                "batch_index_range=%s..%s, page_index_range=%s..%s",
                len(processable_pages),
                group.container.type,
                group.container.id,
                processable_pages[0].batch_index,
                processable_pages[-1].batch_index,
                next(
                    (
                        page.pageIndex
                        for page in processable_pages
                        if page.pageIndex is not None
                    ),
                    None,
                ),
                next(
                    (
                        page.pageIndex
                        for page in reversed(processable_pages)
                        if page.pageIndex is not None
                    ),
                    None,
                ),
            )
            page_by_key = self._page_by_image_key(
                processable_pages,
                image_mapping,
            )
            page_key_by_id = {
                page.id: page_key for page_key, page in page_by_key.items()
            }
            existing_page_numbers = tuple(
                evidence
                for page in processable_pages
                if (
                    evidence := self._physical_page_number_from_metakat(
                        page,
                        page_key_by_id[page.id],
                        metakat_io,
                    )
                )
                is not None
            )
            logger.info(
                "Chapter core input for %s %s contains %d externally "
                "supplied page number(s)",
                group.container.type,
                group.container.id,
                len(existing_page_numbers),
            )
            image_dimensions = tuple(
                None
                if page.imageDim is None
                else PageDimensions(
                    width=page.imageDim.width,
                    height=page.imageDim.height,
                )
                for page in processable_pages
            )
            alto_dimensions = tuple(
                None
                if page.altoDim is None
                else PageDimensions(
                    width=page.altoDim.width,
                    height=page.altoDim.height,
                )
                for page in processable_pages
            )
            core_options = {
                "page_numbers": (
                    existing_page_numbers if existing_page_numbers else None
                ),
            }
            if any(value is not None for value in image_dimensions):
                core_options["image_dimensions"] = image_dimensions
            if any(value is not None for value in alto_dimensions):
                core_options["alto_dimensions"] = alto_dimensions
            core_result = self.core_engine.process(
                images,
                alto_files,
                **core_options,
            )
            flat_result = _flatten_resolved_chapters(core_result.chapters)
            logger.info(
                "Chapter core result for %s %s: roots=%d, chapters=%d, "
                "resolved_starts=%d, unresolved_starts=%d, explicit_ends=%d",
                group.container.type,
                group.container.id,
                len(core_result.chapters),
                len(flat_result),
                sum(chapter.page_start_key is not None for chapter in flat_result),
                sum(chapter.page_start_key is None for chapter in flat_result),
                sum(chapter.page_end_key is not None for chapter in flat_result),
            )
            new_elements, bbox_by_id, page_by_detection = (
                self.extract_metakat_elements_from_pipeline(
                    core_result,
                    page_by_key,
                    group.pages,
                    container_id=group.container.id,
                )
            )
            logger.info(
                "Chapter pipeline returned %d chapter(s) for %s %s",
                sum(
                    element.type == DocumentType.CHAPTER.value
                    for element in new_elements
                ),
                group.container.type,
                group.container.id,
            )
            self._insert_elements_after_container(
                metakat_io.elements,
                container_id=group.container.id,
                new_elements=new_elements,
            )
            metakat_io.detection_to_bbox.update(bbox_by_id)
            metakat_io.detection_to_page_mapping.update(page_by_detection)
        return metakat_io

    @staticmethod
    def _insert_elements_after_container(
        elements: list[MetakatElement],
        *,
        container_id: UUID,
        new_elements: list[MetakatElement],
    ) -> None:
        if not new_elements:
            return
        try:
            container_index = next(
                index
                for index, element in enumerate(elements)
                if element.id == container_id
            )
        except StopIteration as error:
            raise ValueError(
                "Cannot insert chapters after missing document container: "
                f"{container_id}"
            ) from error
        insertion_index = container_index + 1
        elements[insertion_index:insertion_index] = new_elements

    @staticmethod
    def _physical_page_number_from_metakat(
        page: MetakatPage,
        page_key: str,
        metakat_io: MetakatIO,
    ) -> PhysicalPageNumberEvidence | None:
        if page.pageNumber is None:
            return None
        text, confidence, detection_id = page.pageNumber
        bbox = (metakat_io.detection_to_bbox or {}).get(detection_id)
        if bbox is None:
            logger.warning(
                "Page %r has page-number evidence %s without geometry; "
                "omitting it from chapter-core input",
                page_key,
                detection_id,
            )
            return None
        return DecoratedPageNumberParser.create(
            page_key=page_key,
            text=text,
            confidence=confidence,
            bbox=BoundingBox(*bbox),
        )

    @staticmethod
    def _document_groups(
        metakat_io: MetakatIO,
    ) -> list[LowestDocumentGroup]:
        groups = lowest_document_groups(metakat_io, log=logger)
        for group in groups:
            if not group.synthetic:
                continue
            metakat_io.elements.append(group.container)
            for page in group.pages:
                page.parent_id = group.container.id
            logger.warning(
                "Assigned %d parentless page(s) to dummy monograph %s",
                len(group.pages),
                group.container.id,
            )
        return groups

    @staticmethod
    def _page_by_image_key(
        pages: List[MetakatPage],
        image_mapping: dict,
    ) -> dict[str, MetakatPage]:
        result: dict[str, MetakatPage] = {}
        page_by_id = {page.id: page for page in pages}
        for page_id, image_filename in image_mapping.items():
            if page_id not in page_by_id:
                continue
            page_key = Path(image_filename).stem
            if page_key in result:
                raise ValueError(
                    f"Page image mappings must have unique stems: {page_key}"
                )
            result[page_key] = page_by_id[page_id]
        return result

    def extract_metakat_elements_from_pipeline(
        self,
        result: TocResult,
        page_by_key: dict[str, MetakatPage],
        pages: List[MetakatPage],
        *,
        container_id: UUID,
    ) -> Tuple[List[MetakatElement], dict, dict]:
        elements: list[MetakatElement] = []
        bbox_by_id: dict[UUID, tuple[float, float, float, float]] = {}
        page_by_detection: dict[UUID, UUID] = {}
        records: list[_BoundChapter] = []

        def bind_evidence(
            evidence: DetectionEvidence | None,
            *,
            output_text: str | None = None,
        ) -> tuple[str, float, UUID] | None:
            if evidence is None:
                return None
            source_page = page_by_key.get(evidence.page_key)
            if source_page is None:
                raise ValueError(
                    f"Detection evidence refers to unknown page key: "
                    f"{evidence.page_key}"
                )
            detection_id = uuid4()
            bbox_by_id[detection_id] = (
                evidence.bbox.x,
                evidence.bbox.y,
                evidence.bbox.width,
                evidence.bbox.height,
            )
            page_by_detection[detection_id] = source_page.id
            return (
                evidence.text if output_text is None else output_text,
                evidence.confidence,
                detection_id,
            )

        def bind_chapter(
            resolved: ChapterResult,
            *,
            depth: int,
            parent_chapter_id: UUID | None,
        ) -> None:
            chapter_label = self._resolved_chapter_label(resolved)
            toc_page = page_by_key.get(resolved.toc_page_key)
            if toc_page is None:
                raise ValueError(
                    "Resolved chapter refers to unknown TOC page key: "
                    f"{resolved.toc_page_key}"
                )
            if toc_page.pageIndex is None:
                logger.warning(
                    "TOC page %r has no pageIndex for chapter %r",
                    resolved.toc_page_key,
                    chapter_label,
                )
            start_page = page_by_key.get(resolved.page_start_key)
            if resolved.page_start_key is not None and start_page is None:
                logger.warning(
                    "Chapter %r refers to unknown start page key %r",
                    chapter_label,
                    resolved.page_start_key,
                )
            elif start_page is not None and start_page.pageIndex is None:
                logger.warning(
                    "Start page %r has no pageIndex for chapter %r",
                    resolved.page_start_key,
                    chapter_label,
                )
            end_page = page_by_key.get(resolved.page_end_key)
            if resolved.page_end_key is not None and end_page is None:
                logger.warning(
                    "Chapter %r refers to unknown end page key %r",
                    chapter_label,
                    resolved.page_end_key,
                )
            elif end_page is not None and end_page.pageIndex is None:
                logger.warning(
                    "End page %r has no pageIndex for chapter %r",
                    resolved.page_end_key,
                    chapter_label,
                )

            if parent_chapter_id is not None:
                parent_id = parent_chapter_id
            else:
                parent_id = container_id

            chapter = MetakatChapter(
                id=uuid4(),
                parent_id=parent_id,
                pageIndexToc=toc_page.pageIndex,
                pageIndexStart=(
                    None if start_page is None else start_page.pageIndex
                ),
                pageIndexEnd=(
                    None if end_page is None else end_page.pageIndex
                ),
                title=bind_evidence(resolved.title),
                subTitle=bind_evidence(resolved.subtitle),
                partNumber=bind_evidence(resolved.part_number),
                pageNumber=bind_evidence(
                    resolved.page_number,
                    output_text=(
                        None
                        if resolved.page_number is None
                        else resolved.page_number.output_text()
                    ),
                ),
                title_destination_page=bind_evidence(
                    resolved.title_destination_page
                ),
            )
            logger.debug(
                "Binding chapter depth=%d, label=%r, toc_page=%r, "
                "start_page=%r, end_page=%r, pageIndexToc=%s, "
                "pageIndexStart=%s, pageIndexEnd=%s",
                depth,
                chapter_label,
                resolved.toc_page_key,
                resolved.page_start_key,
                resolved.page_end_key,
                chapter.pageIndexToc,
                chapter.pageIndexStart,
                chapter.pageIndexEnd,
            )
            elements.append(chapter)
            records.append(
                _BoundChapter(
                    chapter=chapter,
                    depth=depth,
                    container_id=container_id,
                )
            )
            for child in resolved.children:
                bind_chapter(
                    child,
                    depth=depth + 1,
                    parent_chapter_id=chapter.id,
                )

        for root in result.chapters:
            bind_chapter(
                root,
                depth=0,
                parent_chapter_id=None,
            )
        self._fill_missing_ends(records, pages)
        return elements, bbox_by_id, page_by_detection

    @staticmethod
    def _resolved_chapter_label(resolved: ChapterResult) -> str:
        for evidence in (
            resolved.title,
            resolved.subtitle,
            resolved.title_destination_page,
            resolved.page_number,
        ):
            if evidence is not None:
                return evidence.text
        return "<untitled chapter>"

    @staticmethod
    def _fill_missing_ends(
        records: List[_BoundChapter],
        pages: List[MetakatPage],
    ) -> None:
        all_page_indices = [
            page.pageIndex for page in pages if page.pageIndex is not None
        ]
        for index, record in enumerate(records):
            chapter = record.chapter
            if chapter.pageIndexEnd is not None or chapter.pageIndexStart is None:
                continue
            next_start = next(
                (
                    candidate.chapter.pageIndexStart
                    for candidate in records[index + 1:]
                    if candidate.container_id == record.container_id
                    and candidate.depth <= record.depth
                    and candidate.chapter.pageIndexStart is not None
                ),
                None,
            )
            if next_start is not None:
                chapter.pageIndexEnd = max(
                    chapter.pageIndexStart,
                    next_start - 1,
                )
                logger.debug(
                    "Inferred chapter %s end pageIndex=%s from following "
                    "chapter start=%s",
                    chapter.id,
                    chapter.pageIndexEnd,
                    next_start,
                )
                continue
            if all_page_indices:
                chapter.pageIndexEnd = max(
                    chapter.pageIndexStart,
                    max(all_page_indices),
                )
                logger.debug(
                    "Inferred final chapter %s end pageIndex=%s from "
                    "document end",
                    chapter.id,
                    chapter.pageIndexEnd,
                )


def _flatten_resolved_chapters(
    chapters: Tuple[ChapterResult, ...],
) -> tuple[ChapterResult, ...]:
    return tuple(
        chapter
        for root in chapters
        for chapter in (
            root,
            *_flatten_resolved_chapters(root.children),
        )
    )

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import List, Tuple
from uuid import UUID, uuid4

from metakat.common.aux.document_groups import (
    LowestDocumentGroup,
    lowest_document_groups,
)
from metakat.chapter.engines.bind.chapter_bind_engine import ChapterBindEngine
from metakat.chapter.engines.core.models import (
    ChapterPageNumberEvidence,
    ChapterPageNumberKind,
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
    GroupType,
    MetakatChapter,
    MetakatElement,
    MetakatGroup,
    MetakatIO,
    MetakatPage,
    ProarcIO,
    Value,
)

logger = logging.getLogger(__name__)


class ChapterBindEngineBase(ChapterBindEngine):
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
                "resolved_starts=%d, unresolved_starts=%d, resolved_ends=%d",
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
        detection_id = page.pageNumber.id
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
            text=page.pageNumber.text,
            confidence=page.pageNumber.confidence,
            bbox=BoundingBox(*bbox),
        )

    @staticmethod
    def _document_groups(
        metakat_io: MetakatIO,
    ) -> list[LowestDocumentGroup]:
        # The biblio stage attaches every page to an issue or volume; this
        # binder only reads those units and never creates one. Pages without
        # a unit can only come from an input MetakatIO, and are left out.
        groups = []
        for group in lowest_document_groups(metakat_io, log=logger):
            if group.synthetic:
                logger.warning(
                    "Skipping %d page(s) that belong to no issue or volume",
                    len(group.pages),
                )
                continue
            groups.append(group)
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
        *,
        container_id: UUID,
    ) -> Tuple[List[MetakatElement], dict, dict]:
        elements: list[MetakatElement] = []
        bbox_by_id: dict[UUID, tuple[float, float, float, float]] = {}
        page_by_detection: dict[UUID, UUID] = {}

        def page_index_entries(
            page: MetakatPage | None,
        ) -> list[tuple[int, UUID]] | None:
            # One scan run, so one entry, carrying its own id.
            if page is None or page.pageIndex is None:
                return None
            return [(page.pageIndex, uuid4())]

        def new_value(evidence: DetectionEvidence, text: str) -> Value:
            # Every value gets its own id; values read from one region share
            # its geometry and page.
            source_page = page_by_key.get(evidence.page_key)
            if source_page is None:
                raise ValueError(
                    f"Detection evidence refers to unknown page key: "
                    f"{evidence.page_key}"
                )
            value_id = uuid4()
            bbox_by_id[value_id] = (
                evidence.bbox.x,
                evidence.bbox.y,
                evidence.bbox.width,
                evidence.bbox.height,
            )
            page_by_detection[value_id] = source_page.id
            return Value(text=text, confidence=evidence.confidence, id=value_id)

        def bind_evidence(
            evidence: DetectionEvidence | None,
        ) -> list[Value] | None:
            # A one-element list: the schema's fields are lists, and the
            # chapter core yields at most one reading per field.
            if evidence is None:
                return None
            return [new_value(evidence, evidence.text)]

        def bind_page_numbers(
            evidence: ChapterPageNumberEvidence | None,
        ) -> tuple[list[Value] | None, list[Value] | None]:
            # The TOC entry's printed page reference, split by its parsed
            # shape: a single page is a start, a range a start and an end, a
            # list one start per page. An unparsed reference keeps its raw
            # text as the start.
            if evidence is None:
                return None, None
            if evidence.kind is ChapterPageNumberKind.RANGE:
                return (
                    [new_value(evidence, evidence.normalized_start())],
                    [new_value(evidence, evidence.normalized_end())],
                )
            if evidence.kind is ChapterPageNumberKind.LIST:
                return (
                    [new_value(evidence, text) for text, _, _ in evidence.normalized_items],
                    None,
                )
            if evidence.kind is ChapterPageNumberKind.SINGLE:
                return [new_value(evidence, evidence.normalized_start())], None
            return [new_value(evidence, evidence.output_text())], None

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

            # ChapterResult mirrors MetakatChapter: an unsuffixed field was
            # read on the destination page, a `_toc_page` one in the TOC entry.
            page_index_start = page_index_entries(start_page)
            page_index_end = page_index_entries(end_page)
            title = bind_evidence(resolved.title)
            title_toc_page = bind_evidence(resolved.title_toc_page)
            subtitle_toc_page = bind_evidence(resolved.subtitle_toc_page)
            part_number_toc_page = bind_evidence(resolved.part_number_toc_page)
            page_number_start, page_number_end = bind_page_numbers(
                resolved.page_number_toc_page
            )
            chapter = MetakatChapter(
                id=uuid4(),
                parent_id=parent_id,
                pageIndexStart=page_index_start,
                pageIndexEnd=page_index_end,
                title=title,
                pageIndexTocPage=toc_page.pageIndex,
                titleTocPage=title_toc_page,
                subTitleTocPage=subtitle_toc_page,
                partNumberTocPage=part_number_toc_page,
                pageNumberStartTocPage=page_number_start,
                pageNumberEndTocPage=page_number_end,
                groups=self._chapter_groups(
                    titles=(title, title_toc_page, subtitle_toc_page, part_number_toc_page),
                    page_index_start=page_index_start,
                    page_index_end=page_index_end,
                    page_number_start=page_number_start,
                    page_number_end=page_number_end,
                ),
            )
            logger.debug(
                "Binding chapter depth=%d, label=%r, toc_page=%r, "
                "start_page=%r, end_page=%r, pageIndexTocPage=%s, "
                "pageIndexStart=%s, pageIndexEnd=%s",
                depth,
                chapter_label,
                resolved.toc_page_key,
                resolved.page_start_key,
                resolved.page_end_key,
                chapter.pageIndexTocPage,
                chapter.pageIndexStart,
                chapter.pageIndexEnd,
            )
            elements.append(chapter)
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
        return elements, bbox_by_id, page_by_detection

    @staticmethod
    def _chapter_groups(
        *,
        titles: tuple[list[Value] | None, ...],
        page_index_start: list[tuple[int, UUID]] | None,
        page_index_end: list[tuple[int, UUID]] | None,
        page_number_start: list[Value] | None,
        page_number_end: list[Value] | None,
    ) -> list[MetakatGroup] | None:
        # Only what this binder knows belongs together. All title readings
        # describe the one chapter. The resolved scan start and end are its
        # one run, and a single or range TOC reference is the number the core
        # aligned that run to; a list is left out, since which of its pages
        # the run belongs to is not known here. A group of one pairs nothing.
        groups = []
        title_members = [value.id for values in titles for value in (values or [])]
        if len(title_members) > 1:
            groups.append(MetakatGroup(type=GroupType.TITLE_INFO, members=title_members))
        run_members = [entry_id for _, entry_id in (page_index_start or [])]
        run_members += [entry_id for _, entry_id in (page_index_end or [])]
        if page_number_start and len(page_number_start) == 1:
            run_members += [page_number_start[0].id]
            run_members += [value.id for value in (page_number_end or [])]
        if len(run_members) > 1:
            groups.append(MetakatGroup(type=GroupType.PAGE_RANGE, members=run_members))
        return groups or None

    @staticmethod
    def _resolved_chapter_label(resolved: ChapterResult) -> str:
        for evidence in (
            resolved.title_toc_page,
            resolved.subtitle_toc_page,
            resolved.title,
            resolved.page_number_toc_page,
        ):
            if evidence is not None:
                return evidence.text
        return "<untitled chapter>"


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

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import List
from uuid import uuid4

from metakat.page_number.engines.bind.page_number_bind_engine import (
    PageNumberBindEngine,
)
from metakat.page_number.engines.core.models import PageNumberCoreResult
from metakat.page_number.engines.core.ndk_pagination import NdkPageContext
from metakat.schemas.base_objects import (
    DocumentType,
    MetakatAltoRefs,
    MetakatIO,
    MetakatPage,
    ProarcIO,
    Value,
    is_annotated,
)


logger = logging.getLogger(__name__)


def _text(value) -> str | None:
    """The word of a classified value (page type, side), as the plain string the NDK context takes."""
    text = getattr(value, "text", None)
    return getattr(text, "value", text)


class PageNumberBindEngineBase(PageNumberBindEngine):
    """Bind page-number core results to MetaKat without resolving candidates."""

    def process(
        self,
        batch_dir: str,
        metakat_io: MetakatIO,
        proarc_io: ProarcIO = None,
    ) -> MetakatIO:
        return self._run(batch_dir, metakat_io, final=False)

    def process_final(
        self,
        batch_dir: str,
        metakat_io: MetakatIO,
        proarc_io: ProarcIO = None,
    ) -> MetakatIO:
        """Number the pages again, now by documents: this result replaces the preliminary one.

        The scan starts and ends with a document, and the page-type stage has found where the documents
        begin, so the labels at their edges (leading `[1]`, `[1a]`, inserted leaves) can be supplied too."""
        if not self.core_engine.has_final_pass:
            return metakat_io
        return self._run(batch_dir, metakat_io, final=True)

    def _run(self, batch_dir: str, metakat_io: MetakatIO, *, final: bool) -> MetakatIO:
        metakat_io = copy.deepcopy(metakat_io)
        pages = sorted(
            (
                element
                for element in metakat_io.elements
                if element.type == DocumentType.PAGE.value
            ),
            key=lambda page: page.batch_index,
        )
        if not pages or all(is_annotated(page.pageNumber) for page in pages):
            logger.info("Skipping page-number core: no pages need a prediction")
            return metakat_io
        image_mapping = metakat_io.page_to_image_mapping or {}
        alto_mapping = metakat_io.page_to_alto_mapping or {}
        processable_pages = [
            page
            for page in pages
            if page.id in image_mapping and page.id in alto_mapping
        ]
        skipped = len(pages) - len(processable_pages)
        if skipped:
            logger.warning(
                "Skipping %d page(s) without both image and ALTO mappings",
                skipped,
            )

        if not any(not is_annotated(page.pageNumber) for page in processable_pages):
            logger.info("Skipping page-number core: no processable page needs a prediction")
            return metakat_io

        images = [
            os.path.join(batch_dir, image_mapping[page.id])
            for page in processable_pages
        ]
        alto_files = [
            os.path.join(batch_dir, alto_mapping[page.id])
            for page in processable_pages
        ]
        logger.info(
            "Processing %d images with page number core engine",
            len(images),
        )
        if final:
            core_result = self.core_engine.process(
                images,
                alto_files,
                page_contexts=self._document_contexts(processable_pages, image_mapping),
                final=True,
            )
        else:
            core_result = self.core_engine.process(images, alto_files)
        logger.info(
            "Page number core engine returned %d resolved page number(s)",
            sum(evidence is not None for evidence in core_result.page_numbers.values()),
        )

        page_by_key = self._page_by_image_key(pages, image_mapping)
        self.bind_core_result(core_result, page_by_key, metakat_io, replace_existing=final)
        return metakat_io

    @staticmethod
    def _document_contexts(
        pages: List[MetakatPage],
        image_mapping: dict,
    ) -> dict[str, NdkPageContext] | None:
        """What the page-type stage found, for every page: its document (one unit per document, from the
        document starts, annotated or predicted), its page type and its side.

        The first page of a scan always starts a document and has no start prediction of its own. Without
        the starts of the other pages the scan is one unit."""
        starts = all(page.startsDocument is not None for page in pages[1:])
        unit = -1
        contexts = {}
        for i, page in enumerate(pages):
            unit += i == 0 or (starts and bool(page.startsDocument.val))
            contexts[Path(image_mapping[page.id]).stem] = NdkPageContext(
                unit_key=f"document-{unit}" if starts else None,
                page_type=_text(page.pageType),
                page_type_confidence=getattr(page.pageType, "confidence", None),
                side=_text(page.side),
            )
        return contexts

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

    @staticmethod
    def _forget(page: MetakatPage, metakat_io: MetakatIO) -> None:
        if page.pageNumber is None:
            return
        for mapping in (
            metakat_io.detection_to_bbox,
            metakat_io.detection_to_page_mapping,
            metakat_io.detection_to_alto,
        ):
            if mapping is not None:
                mapping.pop(page.pageNumber.id, None)
        page.pageNumber = None

    @staticmethod
    def bind_core_result(
        core_result: PageNumberCoreResult,
        page_by_key: dict[str, MetakatPage],
        metakat_io: MetakatIO,
        replace_existing: bool = False,
    ) -> None:
        """replace_existing: the result is authoritative, so it also replaces a number of an earlier pass
        and removes it when the core found none. Annotated numbers are never touched."""
        if metakat_io.detection_to_bbox is None:
            metakat_io.detection_to_bbox = {}
        if metakat_io.detection_to_page_mapping is None:
            metakat_io.detection_to_page_mapping = {}

        for page_key, evidence in core_result.page_numbers.items():
            metakat_page = page_by_key.get(page_key)
            if metakat_page is None:
                raise ValueError(
                    f"Page-number core returned an unknown page key: {page_key}"
                )
            if is_annotated(metakat_page.pageNumber):
                logger.debug("Keeping annotated page number on page %s", page_key)
                continue
            if replace_existing:
                PageNumberBindEngineBase._forget(metakat_page, metakat_io)
            if evidence is None:
                continue
            if (
                metakat_page.pageNumber is not None
                and metakat_page.pageNumber.confidence >= evidence.confidence
            ):
                logger.debug(
                    "Keeping existing page number on page %s because its "
                    "confidence is not lower than the core result",
                    page_key,
                )
                continue

            detection_id = uuid4()
            metakat_page.pageNumber = Value(
                text=evidence.output_text(),
                confidence=evidence.confidence,
                id=detection_id,
            )
            bbox = evidence.bbox
            if bbox is not None:
                metakat_io.detection_to_bbox[detection_id] = (
                    bbox.x,
                    bbox.y,
                    bbox.width,
                    bbox.height,
                )
            metakat_io.detection_to_page_mapping[detection_id] = (
                metakat_page.id
            )
            if not evidence.alto.is_empty():
                if metakat_io.detection_to_alto is None:
                    metakat_io.detection_to_alto = {}
                metakat_io.detection_to_alto[detection_id] = MetakatAltoRefs(
                    blocks=list(evidence.alto.blocks),
                    lines=list(evidence.alto.lines),
                    words=list(evidence.alto.words),
                )

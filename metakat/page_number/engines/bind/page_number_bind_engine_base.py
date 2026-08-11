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
from metakat.schemas.base_objects import (
    DocumentType,
    MetakatIO,
    MetakatPage,
    ProarcIO,
)


logger = logging.getLogger(__name__)


class PageNumberBindEngineBase(PageNumberBindEngine):
    """Bind page-number core results to MetaKat without resolving candidates."""

    def process(
        self,
        batch_dir: str,
        metakat_io: MetakatIO,
        proarc_io: ProarcIO = None,
    ) -> MetakatIO:
        metakat_io = copy.deepcopy(metakat_io)
        pages = sorted(
            (
                element
                for element in metakat_io.elements
                if element.type == DocumentType.PAGE.value
            ),
            key=lambda page: page.batch_index,
        )
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
        core_result = self.core_engine.process(images, alto_files)
        logger.info(
            "Page number core engine returned %d resolved page number(s)",
            len(core_result.page_numbers),
        )

        page_by_key = self._page_by_image_key(pages, image_mapping)
        self.bind_core_result(core_result, page_by_key, metakat_io)
        return metakat_io

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
    def bind_core_result(
        core_result: PageNumberCoreResult,
        page_by_key: dict[str, MetakatPage],
        metakat_io: MetakatIO,
    ) -> None:
        if metakat_io.detection_to_bbox is None:
            metakat_io.detection_to_bbox = {}
        if metakat_io.detection_to_page_mapping is None:
            metakat_io.detection_to_page_mapping = {}

        for page_key, evidence in core_result.page_numbers.items():
            metakat_page = page_by_key.get(page_key)
            if metakat_page is None:
                raise ValueError(
                    "Page-number core returned an unknown page key: "
                    f"{page_key}"
                )
            if (
                metakat_page.pageNumber is not None
                and metakat_page.pageNumber[1] >= evidence.confidence
            ):
                logger.debug(
                    "Keeping existing page number on page %s because its "
                    "confidence is not lower than the core result",
                    page_key,
                )
                continue

            detection_id = uuid4()
            metakat_page.pageNumber = (
                evidence.output_text(),
                evidence.confidence,
                detection_id,
            )
            bbox = evidence.bbox
            metakat_io.detection_to_bbox[detection_id] = (
                bbox.x,
                bbox.y,
                bbox.width,
                bbox.height,
            )
            metakat_io.detection_to_page_mapping[detection_id] = (
                metakat_page.id
            )

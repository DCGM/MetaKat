import copy
import logging
import os
from pathlib import Path

from typing import List, Tuple
from uuid import uuid4

from text_geometry_aligner import AlignmentPage

from metakat.chapter.engines.bind.chapter_bind_engine import ChapterBindEngine
from metakat.chapter.engines.bind.chapter_parser import parse_page_number

from metakat.schemas.base_objects import MetakatIO, ProarcIO, DocumentType, MetakatPage, ChapterType, \
     MetakatElement

logger = logging.getLogger(__name__)


class ChapterBindEngineBase(ChapterBindEngine):
    def __init__(self, bind_engine_dir: str, core_engine_dir: str):
        super().__init__(bind_engine_dir, core_engine_dir)

    def process(self, batch_dir: str, metakat_io: MetakatIO, proarc_io: ProarcIO = None) -> MetakatIO:
        metakat_io = copy.deepcopy(metakat_io)
        pages = [el for el in metakat_io.elements if el.type == DocumentType.PAGE.value]
        pages = sorted(pages, key=lambda x: x.batch_index)

        images = [os.path.join(batch_dir, metakat_io.page_to_image_mapping[page.id]) for page in pages if
                  page.id in metakat_io.page_to_image_mapping]
        alto_files = [os.path.join(batch_dir, metakat_io.page_to_alto_mapping[page.id]) for page in pages if
                      page.id in metakat_io.page_to_alto_mapping]

        logger.info(f"Processing {len(images)} images with chapter core engine")
        alignment_pages = self.core_engine.process(images, alto_files)
        logger.info(f"Chapter core engine returned "
                    f"{sum(page.matched_count for page in alignment_pages)} "
                    f"detections")

        metakat_page_id_to_metakat_page = {page.id: page for page in metakat_io.elements if page.type == DocumentType.PAGE.value}
        alignment_page_key_to_metakat_page = {
            Path(image_filename).stem: metakat_page_id_to_metakat_page[page_id]
            for page_id, image_filename in metakat_io.page_to_image_mapping.items()
        }

        logger.info(f"Adding page numbers to MetaKatPage elements")
        metakat_elements, detection_id_to_detection_bbox, detection_id_to_page_id = self.extract_metakat_elements_from_alignment(
            alignment_pages, alignment_page_key_to_metakat_page)
        metakat_io.elements = metakat_elements + metakat_io.elements
        return metakat_io

    def extract_metakat_elements_from_alignment(
        self,
        alignment_pages: List[AlignmentPage],
        alignment_page_key_to_metakat_page: dict,
    ) -> Tuple[List[MetakatElement], dict, dict]:
        elements = []
        detection_id_to_detection_bbox = {}
        detection_id_to_page_id = {}
        for alignment_page in alignment_pages:
            metakat_page = alignment_page_key_to_metakat_page[
                alignment_page.page_key
            ]
            page_elements, page_id_to_detection_bbox = self.get_metakat_elements_from_page(
                alignment_page,
                metakat_page,
            )
            elements.extend(page_elements)
            detection_id_to_detection_bbox.update(page_id_to_detection_bbox)
            for detection_id, bbox in page_id_to_detection_bbox.items():
                detection_id_to_page_id[detection_id] = metakat_page.id
        return elements, detection_id_to_detection_bbox, detection_id_to_page_id

    def get_metakat_elements_from_page(
        self,
        alignment_page: AlignmentPage,
        metakat_page: MetakatPage,
    ) -> Tuple[List[MetakatElement], dict]:
        elements = []
        detection_id_to_detection_bbox = {}
        for region in alignment_page.regions:
            if not region.matched:
                continue
            if (
                region.category_id is None
                or region.input_geometry is None
                or region.input_geometry_confidence is None
                or region.alto_text is None
            ):
                logger.warning(
                    "Matched region %s on page %s is missing YOLO metadata; "
                    "skipping detection",
                    region.region_id,
                    alignment_page.page_key,
                )
                continue

            class_id = str(region.category_id)
            bbox = region.input_geometry.bounds
            detection_bbox = (
                bbox.x,
                bbox.y,
                bbox.width,
                bbox.height,
            )
            detection_id = uuid4()
            detection_text = region.alto_text
            detection_confidence = region.input_geometry_confidence

            if class_id not in self.core_engine.id2label:
                logger.warning(
                    "CLASS_ID %s (label=%r, label_export=%r) not in "
                    "id2label, skipping - val: %s, conf: %s, bbox: %s, "
                    "page_key: %s",
                    class_id,
                    region.label,
                    region.label_export,
                    detection_text,
                    detection_confidence,
                    detection_bbox,
                    alignment_page.page_key,
                )
                continue
            chapter_type = ChapterType(self.core_engine.id2label[class_id])

            if chapter_type == ChapterType.PAGE_NUMBER:
                detection_text_parsed = parse_page_number(detection_text)
                if not detection_text_parsed:
                    logger.warning(f"Invalid PAGE_NUMBER, skipping - "
                                   f"val: {detection_text}, "
                                   f"conf: {detection_confidence}, "
                                   f"bbox: {detection_bbox}, "
                                   f"page_key: {alignment_page.page_key}")
                    continue
                if metakat_page.pageNumber is None or metakat_page.pageNumber[1] < detection_confidence:
                    metakat_page.pageNumber = (detection_text_parsed, detection_confidence, detection_id)
            else:
                continue

            detection_id_to_detection_bbox[detection_id] = detection_bbox


        return elements, detection_id_to_detection_bbox











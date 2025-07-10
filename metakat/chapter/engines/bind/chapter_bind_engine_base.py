import copy
import logging
import os

from typing import List, Tuple
from uuid import uuid4

from chapter.engines.bind.chapter_bind_engine import ChapterBindEngine
from chapter.engines.bind.chapter_parser import parse_page_number
from detector_wrapper.parsers.pero_ocr import ALTOMatchedPage

from schemas.base_objects import MetakatIO, ProarcIO, DocumentType, MetakatPage, ChapterType, \
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
        matched_pages = self.core_engine.process(images, alto_files)
        logger.info(f"Chapter core engine returned "
                    f"{sum([len(p.matched_detections) for p in matched_pages if p.matched_detections is not None])} "
                    f"detections")

        metakat_page_id_to_metakat_page = {page.id: page for page in metakat_io.elements if page.type == DocumentType.PAGE.value}
        matched_page_file_name_to_metakat_page = {v: metakat_page_id_to_metakat_page[k] for k, v in metakat_io.page_to_image_mapping.items()}

        logger.info(f"Adding page numbers to MetaKatPage elements")
        metakat_elements, detection_id_to_detection_bbox, detection_id_to_page_id = self.extract_metakat_elements_from_detections(
            matched_pages, matched_page_file_name_to_metakat_page)
        metakat_io.elements = metakat_elements + metakat_io.elements
        return metakat_io

    def extract_metakat_elements_from_detections(self,
        matched_pages: List[ALTOMatchedPage],
        matched_page_file_name_to_metakat_page: dict,
    ) -> Tuple[List[MetakatElement], dict, dict]:
        elements = []
        detection_id_to_detection_bbox = {}
        detection_id_to_page_id = {}
        for matched_page in matched_pages:
            metakat_page = matched_page_file_name_to_metakat_page[matched_page.detector_parser_page.image_filename]
            page_elements, page_id_to_detection_bbox = self.get_metakat_elements_from_page(matched_page, metakat_page)
            elements.extend(page_elements)
            detection_id_to_detection_bbox.update(page_id_to_detection_bbox)
            for detection_id, bbox in page_id_to_detection_bbox.items():
                detection_id_to_page_id[detection_id] = metakat_page.id
        return elements, detection_id_to_detection_bbox, detection_id_to_page_id

    def get_metakat_elements_from_page(self, matched_page: ALTOMatchedPage, metakat_page: MetakatPage) -> Tuple[List[MetakatElement], dict]:
        elements = []
        detection_id_to_detection_bbox = {}
        for matched_detection in matched_page.matched_detections:
            class_id = matched_detection.get_class_id()
            detection_bbox = (matched_detection.detector_parser_annotated_bounding_box.x,
                              matched_detection.detector_parser_annotated_bounding_box.y,
                              matched_detection.detector_parser_annotated_bounding_box.width,
                              matched_detection.detector_parser_annotated_bounding_box.height)
            detection_id = uuid4()
            detection_text = matched_detection.get_text()
            detection_confidence = matched_detection.get_confidence()

            if class_id not in self.core_engine.id2label:
                logger.warning(f"CLASS_ID {class_id} ({matched_detection.get_class()}) not in id2label, skipping - "
                               f"val: {detection_text}, "
                               f"conf: {detection_confidence}, "
                               f"bbox: {detection_bbox}, "
                               f"file_name: {matched_page.detector_parser_page.image_filename}")
                continue
            chapter_type = ChapterType(self.core_engine.id2label[class_id])

            if chapter_type == ChapterType.PAGE_NUMBER:
                detection_text_parsed = parse_page_number(detection_text)
                if not detection_text_parsed:
                    logger.warning(f"Invalid PAGE_NUMBER, skipping - "
                                   f"val: {detection_text}, "
                                   f"conf: {detection_confidence}, "
                                   f"bbox: {detection_bbox}, "
                                   f"file_name: {matched_page.detector_parser_page.image_filename}")
                    continue
                if metakat_page.pageNumber is None or metakat_page.pageNumber[1] < detection_confidence:
                    metakat_page.pageNumber = (detection_text_parsed, detection_confidence, detection_id)
            else:
                continue

            detection_id_to_detection_bbox[detection_id] = detection_bbox


        return elements, detection_id_to_detection_bbox













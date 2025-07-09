import logging
import os

from typing import List, Tuple, Union
from uuid import uuid4, UUID

from biblio.engines.core.biblio_engine_core import BiblioEngineCore
from biblio.engines.bind.bilbio_engine_bind import BiblioEngineBind
from detector_wrapper.parsers.pero_ocr import ALTOMatchedPage

from schemas.base_objects import MetakatIO, ProarcIO, DocumentType, MetakatPage, PageType, BiblioType, \
    MetakatVolume, MetakatIssue, MetakatElement

logger = logging.getLogger(__name__)


class BiblioEngineBindSimple(BiblioEngineBind):
    def __init__(self, biblio_engine_core: BiblioEngineCore):
        super().__init__(biblio_engine_core)

    def process(self, batch_dir: str, metakat_io: MetakatIO, proarc_io: ProarcIO = None) -> Tuple[MetakatIO]:
        pages = [el for el in metakat_io.elements if el.type == DocumentType.PAGE]
        pages = sorted(pages, key=lambda x: x.batch_index)
        title_pages = self.filter_title_pages(pages, 20)

        images = [os.path.join(batch_dir, metakat_io.page_to_image_mapping[page.id]) for page in title_pages if
                  page.id in metakat_io.page_to_image_mapping]
        alto_files = [os.path.join(batch_dir, metakat_io.page_to_alto_mapping[page.id]) for page in title_pages if
                      page.id in metakat_io.page_to_alto_mapping]

        matched_pages = self.biblio_engine_core.process(images, alto_files)

        matched_page_file_name_to_metakat_page_id = {v: k for k, v in metakat_io.page_to_image_mapping.items()}

        metakat_elements, detection_id_to_detection_bbox, detection_id_to_page_id = self.get_metakat_elements(
            matched_pages, matched_page_file_name_to_metakat_page_id)
        metakat_elements = self.process_metakat_elements(metakat_elements)
        metakat_io.elements.extend(metakat_elements)


    def process_metakat_elements(self, metakat_elements: List[MetakatElement]) -> List[MetakatElement]:
        return metakat_elements

    def get_metakat_elements(self, matched_pages: List[ALTOMatchedPage],
                             matched_page_file_name_to_metakat_page_id: dict) -> Tuple[List[MetakatElement],
                                                                                 dict, dict]:
        elements = []
        detection_id_to_detection_bbox = {}
        detection_id_to_page_id = {}
        for matched_page in matched_pages:
            page_elements, page_id_to_detection_bbox = self.get_metakat_elements_from_page(matched_page)
            elements.extend(page_elements)
            detection_id_to_detection_bbox.update(page_id_to_detection_bbox)
            metakat_page_id = matched_page_file_name_to_metakat_page_id[matched_page.detector_parser_page.image_filename]
            for detection_id, bbox in page_id_to_detection_bbox.items():
                detection_id_to_page_id[detection_id] = metakat_page_id
        return elements, detection_id_to_detection_bbox, detection_id_to_page_id

    def get_metakat_elements_from_page(self, matched_page: ALTOMatchedPage) -> Tuple[List[MetakatElement], dict]:
        elements = []
        detection_id_to_detection_bbox = {}
        metakat_volume = MetakatVolume(id=uuid4(), periodical=False)
        metakat_issue = MetakatIssue(id=uuid4())
        for matched_detection in matched_page.matched_detections:
            class_id = matched_detection.get_class_id()
            if class_id not in self.biblio_engine_core.id2label:
                logger.warning(
                    f"Id {class_id} ({matched_detection.get_class()}) not found in id2label mapping (read from metakat_engine_config.json), skipping detection")
                continue
            biblio_type = BiblioType(self.biblio_engine_core.id2label[class_id])

            detection_bbox = (matched_detection.detector_parser_annotated_bounding_box.x,
                              matched_detection.detector_parser_annotated_bounding_box.y,
                              matched_detection.detector_parser_annotated_bounding_box.width,
                              matched_detection.detector_parser_annotated_bounding_box.height)
            detection_id = uuid4()
            detection = (matched_detection.get_text(),
                         matched_detection.get_confidence(),
                         detection_id)

            if biblio_type == BiblioType.PERIODICAL_VOLUME_PART_NUMBER:
                metakat_volume.periodical = True
                metakat_volume.partNumber = detection

            elif biblio_type == BiblioType.PERIODICAL_VOLUME_DATE_ISSUED:
                metakat_volume.periodical = True
                metakat_volume.dateIssued = detection

            elif biblio_type == BiblioType.PERIODICAL_ISSUE_PART_NUMBER:
                metakat_issue.partNumber = detection

            elif biblio_type == BiblioType.PERIODICAL_ISSUE_DATE_ISSUED:
                metakat_issue.dateIssued = detection

            elif biblio_type == BiblioType.REDAKTOR:
                if metakat_issue.redaktor is None:
                    metakat_issue.redaktor = [detection]
                else:
                    metakat_issue.redaktor.append(detection)

            elif biblio_type == BiblioType.TITLE:
                metakat_volume.title = detection
                metakat_issue.title = detection

            elif biblio_type == BiblioType.SUBTITLE:
                metakat_volume.subTitle = detection
                metakat_issue.subTitle = detection

            elif biblio_type == BiblioType.PUBLISHER:
                if metakat_volume.publisher is None:
                    metakat_volume.publisher = [detection]
                else:
                    metakat_volume.publisher.append(detection)
                if metakat_issue.publisher is None:
                    metakat_issue.publisher = [detection]
                else:
                    metakat_issue.publisher.append(detection)

            elif biblio_type == BiblioType.PLACE_TERM:
                metakat_volume.placeTerm = detection
                metakat_issue.placeTerm = detection

            elif biblio_type == BiblioType.MANUFACTURE_PUBLISHER:
                if metakat_volume.manufacturePublisher is None:
                    metakat_volume.manufacturePublisher = [detection]
                else:
                    metakat_volume.manufacturePublisher.append(detection)
                if metakat_issue.manufacturePublisher is None:
                    metakat_issue.manufacturePublisher = [detection]
                else:
                    metakat_issue.manufacturePublisher.append(detection)

            elif biblio_type == BiblioType.MANUFACTURE_PLACE_TERM:
                if metakat_volume.manufacturePlaceTerm is None:
                    metakat_volume.manufacturePlaceTerm = [detection]
                else:
                    metakat_volume.manufacturePlaceTerm.append(detection)
                if metakat_issue.manufacturePlaceTerm is None:
                    metakat_issue.manufacturePlaceTerm = [detection]
                else:
                    metakat_issue.manufacturePlaceTerm.append(detection)

            elif biblio_type == BiblioType.PART_NAME:
                metakat_volume.partName = detection

            elif biblio_type == BiblioType.SERIES_NAME:
                if metakat_volume.seriesName is None:
                    metakat_volume.seriesName = [detection]
                else:
                    metakat_volume.seriesName.append(detection)

            elif biblio_type == BiblioType.SERIES_NUMBER:
                if metakat_volume.seriesNumber is None:
                    metakat_volume.seriesNumber = [detection]
                else:
                    metakat_volume.seriesNumber.append(detection)

            elif biblio_type == BiblioType.EDITION:
                metakat_volume.edition = detection

            elif biblio_type == BiblioType.DATE_ISSUED and metakat_volume.dateIssued is None:
                metakat_volume.dateIssued = detection

            elif biblio_type == BiblioType.AUTHOR:
                if metakat_volume.author is None:
                    metakat_volume.author = [detection]
                else:
                    metakat_volume.author.append(detection)

            elif biblio_type == BiblioType.ILLUSTRATOR:
                if metakat_volume.illustrator is None:
                    metakat_volume.illustrator = [detection]
                else:
                    metakat_volume.illustrator.append(detection)

            elif biblio_type == BiblioType.TRANSLATOR:
                if metakat_volume.translator is None:
                    metakat_volume.translator = [detection]
                else:
                    metakat_volume.translator.append(detection)

            elif biblio_type == BiblioType.EDITOR:
                if metakat_volume.editor is None:
                    metakat_volume.editor = [detection]
                else:
                    metakat_volume.editor.append(detection)

            else:
                continue

            detection_id_to_detection_bbox[detection_id] = detection_bbox


        if metakat_volume.title is not None:
            elements.append(metakat_volume)
            if metakat_issue.title is not None and (metakat_issue.partNumber is not None or
                                                    metakat_issue.dateIssued is not None):
                metakat_issue.parent_id = metakat_volume.id
                elements.append(metakat_issue)

        return elements, detection_id_to_detection_bbox


    def filter_title_pages(self, pages: List[MetakatPage], min_distance: int) -> List[MetakatPage]:
        # Sort pages by batch_index (already done in your code)
        pages = sorted(pages, key=lambda x: x.batch_index)

        # Select only pages that are predicted as title pages with confidence
        candidates = [
            page for page in pages
            if page.pageType and page.pageType[0] == PageType.TITLE_PAGE
        ]

        retained = []
        i = 0
        while i < len(candidates):
            current = candidates[i]
            group = [current]

            # Compare with following candidates to check if they are within N pages
            j = i + 1
            while j < len(candidates) and (candidates[j].batch_index - current.batch_index) < min_distance:
                group.append(candidates[j])
                j += 1

            # Keep only the one with the highest confidence from the group
            best = max(group, key=lambda p: p.pageType[1])
            retained.append(best)

            # Skip all the grouped elements
            i = j

        return retained





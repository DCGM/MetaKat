import copy
import logging
import os

from typing import List, Tuple, Union, Optional
from uuid import uuid4, UUID

from biblio.engines.core.biblio_core_engine import BiblioCoreEngine
from biblio.engines.bind.bilbio_bind_engine import BiblioBindEngine
from detector_wrapper.parsers.pero_ocr import ALTOMatchedPage

from schemas.base_objects import MetakatIO, ProarcIO, DocumentType, MetakatPage, PageType, BiblioType, \
    MetakatVolume, MetakatIssue, MetakatElement, MetakatTitle

logger = logging.getLogger(__name__)


class BiblioBindEngineBase(BiblioBindEngine):
    def __init__(self, bind_engine_dir: str, core_engine_dir: str):
        super().__init__(bind_engine_dir, core_engine_dir)

    def process(self, batch_dir: str, metakat_io: MetakatIO, proarc_io: ProarcIO = None) -> MetakatIO:
        metakat_io = copy.deepcopy(metakat_io)
        pages = [el for el in metakat_io.elements if el.type == DocumentType.PAGE]
        pages = sorted(pages, key=lambda x: x.batch_index)
        logger.info(f"Getting title pages from {len(pages)} pages")
        title_pages = self.filter_title_pages(pages, 20)
        logger.info(f"Found {len(title_pages)} title pages")

        images = [os.path.join(batch_dir, metakat_io.page_to_image_mapping[page.id]) for page in title_pages if
                  page.id in metakat_io.page_to_image_mapping]
        alto_files = [os.path.join(batch_dir, metakat_io.page_to_alto_mapping[page.id]) for page in title_pages if
                      page.id in metakat_io.page_to_alto_mapping]

        logger.info(f"Processing {len(images)} images with biblio core engine")
        matched_pages = self.core_engine.process(images, alto_files)
        logger.info(f"Biblio core engine returned "
                    f"{sum([len(p.matched_detections) for p in matched_pages if p.matched_detections is not None])} "
                    f"detections")

        matched_page_file_name_to_metakat_page_id = {v: k for k, v in metakat_io.page_to_image_mapping.items()}

        logger.info(f"Creating MetaKatVolume and MetaKatIssue elements from detections")
        metakat_elements, detection_id_to_detection_bbox, detection_id_to_page_id = self.extract_metakat_elements_from_detections(
            matched_pages, matched_page_file_name_to_metakat_page_id)
        logger.info(f"Created {len(metakat_elements)} MetaKatVolume and MetaKatIssue elements from detections")
        logger.info(f"Creating MetaKatTitle element, and filtering MetaKatVolume elements")
        metakat_elements = self.finalize_metakat_elements(metakat_elements)
        logger.info(f"Adding {len(metakat_elements)} MetaKat elements to MetaKatIO")
        metakat_io.elements = metakat_elements + metakat_io.elements
        return metakat_io

    def finalize_metakat_elements(self, metakat_elements: List[MetakatElement]) -> List[MetakatElement]:
        elements = self.finalize_metakat_periodical_volumes(metakat_elements)
        title_element = self.get_metakat_title_element(elements)
        if title_element is not None:
            elements = [title_element] + elements
        return elements

    # Create the MetakatTitle element from the list of MetakatElements
    # Extract the title and subtitle from the MetakatVolume element that has the most confident title detection
    def get_metakat_title_element(self, elements: List[MetakatElement]) -> Optional[MetakatTitle]:
        title_element = None
        for element in elements:
            if isinstance(element, MetakatVolume):
                if element.title and (title_element is None or element.title[1] > title_element.title[1]):
                    title_element = element

        if title_element is not None:
            return MetakatTitle(
                id=uuid4(),
                periodical=title_element.periodical,
                title=title_element.title,
                subTitle=title_element.subTitle
            )
        return None

    def finalize_metakat_periodical_volumes(self, metakat_elements: List[MetakatElement]) -> List[MetakatElement]:
        periodical_volume_bags = []
        periodical_volumes = [el for el in metakat_elements if isinstance(el, MetakatVolume) and el.periodical]

        # First add volumes that have both partNumber and dateIssued
        for periodical_volume in periodical_volumes:
            if periodical_volume.partNumber is not None and periodical_volume.dateIssued is not None:
                added = False
                for pb in periodical_volume_bags:
                    if pb.add_volume(periodical_volume):
                        added = True
                        break
                if not added:
                    periodical_volume_bags.append(PeriodicalMetakatVolumeBag(periodical_volume))

        # Then add volumes that have either partNumber or dateIssued, but not both
        for periodical_volume in periodical_volumes:
            if ((periodical_volume.partNumber is not None and periodical_volume.dateIssued is None) or
                (periodical_volume.partNumber is None and periodical_volume.dateIssued is not None)):
                added = False
                for pb in periodical_volume_bags:
                    if pb.add_volume(periodical_volume):
                        added = True
                        break
                if not added:
                    periodical_volume_bags.append(PeriodicalMetakatVolumeBag(periodical_volume))

        volume_id_to_root_volume_id = {}
        for pb in periodical_volume_bags:
            root_volume = pb.root_volume
            for volume in pb.volumes:
                volume_id_to_root_volume_id[volume.id] = root_volume.id

        issues = [el for el in metakat_elements if isinstance(el, MetakatIssue)]
        for issues in issues:
            if issues.parent_id in volume_id_to_root_volume_id:
                issues.parent_id = volume_id_to_root_volume_id[issues.parent_id]

        volumes = [pb.root_volume for pb in periodical_volume_bags]

        elements = volumes + issues
        for el in metakat_elements:
            if not (isinstance(el, MetakatVolume) and el.periodical) and not isinstance(el, MetakatIssue):
                elements.append(el)

        return elements

    def extract_metakat_elements_from_detections(self,
        matched_pages: List[ALTOMatchedPage],
        matched_page_file_name_to_metakat_page_id: dict
    ) -> Tuple[List[MetakatElement], dict, dict]:
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
            if class_id not in self.core_engine.id2label:
                logger.warning(
                    f"Id {class_id} ({matched_detection.get_class()}) not found in id2label mapping (read from metakat_engine_config.json), skipping detection")
                continue
            biblio_type = BiblioType(self.core_engine.id2label[class_id])

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
                if metakat_volume.partNumber is None or metakat_volume.partNumber[1] < detection[1]:
                    metakat_volume.partNumber = detection

            elif biblio_type == BiblioType.PERIODICAL_VOLUME_DATE_ISSUED:
                metakat_volume.periodical = True
                if metakat_volume.dateIssued is None or metakat_volume.dateIssued[1] < detection[1]:
                    metakat_volume.dateIssued = detection

            elif biblio_type == BiblioType.PERIODICAL_ISSUE_PART_NUMBER:
                if metakat_issue.partNumber is None or metakat_issue.partNumber[1] < detection[1]:
                    metakat_issue.partNumber = detection

            elif biblio_type == BiblioType.PERIODICAL_ISSUE_DATE_ISSUED:
                if metakat_issue.dateIssued is None or metakat_issue.dateIssued[1] < detection[1]:
                    metakat_issue.dateIssued = detection

            elif biblio_type == BiblioType.REDAKTOR:
                if metakat_issue.redaktor is None:
                    metakat_issue.redaktor = [detection]
                else:
                    metakat_issue.redaktor.append(detection)

            elif biblio_type == BiblioType.TITLE:
                if metakat_volume.title is None  or metakat_volume.title[1] < detection[1]:
                    metakat_volume.title = detection
                if metakat_issue.title is None or metakat_issue.title[1] < detection[1]:
                    metakat_issue.title = detection

            elif biblio_type == BiblioType.SUBTITLE:
                if metakat_volume.subTitle is None or metakat_volume.subTitle[1] < detection[1]:
                    metakat_volume.subTitle = detection
                if metakat_issue.subTitle is None or metakat_issue.subTitle[1] < detection[1]:
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
                if metakat_volume.placeTerm is None or metakat_volume.placeTerm[1] < detection[1]:
                    metakat_volume.placeTerm = detection
                if metakat_issue.placeTerm is None or metakat_issue.placeTerm[1] < detection[1]:
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
                if metakat_volume.partName is None or metakat_volume.partName[1] < detection[1]:
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
                if metakat_volume.edition is None or metakat_volume.edition[1] < detection[1]:
                    metakat_volume.edition = detection

            elif biblio_type == BiblioType.DATE_ISSUED and metakat_volume.dateIssued is None:
                if metakat_volume.dateIssued is None or metakat_volume.dateIssued[1] < detection[1]:
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


class PeriodicalMetakatVolumeBag:
    def __init__(self, volume: MetakatVolume):
        if not volume.periodical:
            raise ValueError("Volume must be a periodical volume")
        self.root_volume = volume
        self.volumes = []

    def add_volume(self, volume: MetakatVolume):
        if not volume.periodical:
            return False
        if volume.partNumber is None and volume.dateIssued is None:
            return False
        if self.root_volume.partNumber != volume.partNumber and self.root_volume.dateIssued != volume.dateIssued:
            return False


        if self.root_volume.partNumber is not None and self.root_volume.dateIssued is not None:
            # Added volume has both partNumber and dateIssued
            if volume.partNumber is not None and volume.dateIssued is not None:
                if volume.partNumber[1] + volume.dateIssued[1] > self.root_volume.partNumber[1] + self.root_volume.dateIssued[1]:
                    self.volumes.append(self.root_volume)
                    self.root_volume = volume
                else:
                    self.volumes.append(volume)
                return True
            # Added volume has only partNumber
            elif volume.partNumber == self.root_volume.partNumber:
                self.volumes.append(volume)
                return True
            # Added volume has only dateIssued
            elif volume.dateIssued == self.root_volume.dateIssued:
                self.volumes.append(volume)
                return True

        elif self.root_volume.partNumber is not None and self.root_volume.dateIssued is None and \
            volume.partNumber is not None and volume.dateIssued is None and \
            self.root_volume.partNumber == volume.partNumber:
            if volume.partNumber[1] > self.root_volume.partNumber[1]:
                self.volumes.append(self.root_volume)
                self.root_volume = volume
            else:
                self.volumes.append(volume)
            return True

        elif self.root_volume.partNumber is None and self.root_volume.dateIssued is not None and \
            volume.partNumber is None and volume.dateIssued is not None and \
            self.root_volume.dateIssued == volume.dateIssued:
            if volume.dateIssued[1] > self.root_volume.dateIssued[1]:
                self.volumes.append(self.root_volume)
                self.root_volume = volume
            else:
                self.volumes.append(volume)
            return True
        return False







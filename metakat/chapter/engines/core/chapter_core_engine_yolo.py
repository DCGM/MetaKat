import logging

from typing import List

from chapter.engines.core.chapter_core_engine import ChapterCoreEngine
from detector_wrapper.parsers.pero_ocr import ALTOMatchedPage

from metakat.common.engines.engine_yolo_alto import EngineYOLOALTO


logger = logging.getLogger(__name__)

class ChapterCoreEngineYOLO(ChapterCoreEngine):
    def __init__(self, core_engine_dir,
                 yolo_batch_size=32,
                 yolo_confidence_threshold=0.25,
                 yolo_image_size=640,
                 min_alto_word_area_in_detection_to_match=0.65):
        super().__init__(core_engine_dir=core_engine_dir)
        self.engine_yolo_alto = EngineYOLOALTO(
            engine_dir=core_engine_dir,
            yolo_batch_size=yolo_batch_size,
            yolo_confidence_threshold=yolo_confidence_threshold,
            yolo_image_size=yolo_image_size,
            min_alto_word_area_in_detection_to_match=min_alto_word_area_in_detection_to_match
        )

    def process(self, images: List[str], alto_files: List[str]) -> List[ALTOMatchedPage]:
        return self.engine_yolo_alto.process(
            images=images,
            alto_files=alto_files
        ).matched_pages





import logging

from typing import List

from text_geometry_aligner import AlignmentPage

from metakat.biblio.engines.core.biblio_core_engine import BiblioCoreEngine


from metakat.common.engines.engine_yolo_alto import EngineYOLOALTO

logger = logging.getLogger(__name__)

class BiblioCoreEngineYOLO(BiblioCoreEngine):
    def __init__(self, core_engine_dir,
                 yolo_batch_size=32,
                 yolo_confidence_threshold=0.25,
                 yolo_image_size=640,
                 minimum_overlap_coverage=0.65):
        super().__init__(core_engine_dir=core_engine_dir)
        self.engine_yolo_alto = EngineYOLOALTO(
            engine_dir=core_engine_dir,
            yolo_batch_size=yolo_batch_size,
            yolo_confidence_threshold=yolo_confidence_threshold,
            yolo_image_size=yolo_image_size,
            minimum_overlap_coverage=minimum_overlap_coverage,
        )

    def process(
        self,
        images: List[str],
        alto_files: List[str],
    ) -> List[AlignmentPage]:
        return self.engine_yolo_alto.process(
            images=images,
            alto_files=alto_files
        ).pages




import copy
import logging
import os

from metakat.page_type.engines.bind.page_type_bind_engine import PageTypeBindEngine
from metakat.page_type.engines.core.page_type_core_engine import PageTypeInput, PageTypePrediction
from metakat.schemas.base_objects import MetakatIO, ProarcIO, MetakatPage

logger = logging.getLogger(__name__)


class PageTypeBindEngineBase(PageTypeBindEngine):
    def process(self, batch_dir: str, metakat_io: MetakatIO,
                proarc_io: ProarcIO = None) -> MetakatIO:
        result = copy.deepcopy(metakat_io)
        pages = sorted((p for p in result.elements if isinstance(p, MetakatPage)),
                       key=lambda page: page.batch_index)
        if len({p.batch_index for p in pages}) != len(pages):
            raise ValueError("Page batch_index values must be unique for scan order")
        if len({p.id for p in pages}) != len(pages):
            raise ValueError("Page IDs must be unique")
        if not pages:
            return result

        def path(mapping, page_id):
            relative = (mapping or {}).get(page_id)
            return os.path.join(batch_dir, relative) if relative else None

        inputs = [PageTypeInput(
            image_path=path(result.page_to_image_mapping, p.id),
            ocr_path=(path(result.page_to_alto_mapping, p.id)
                      or path(result.page_to_xml_mapping, p.id)),
            page_number=p.pageNumber.text if p.pageNumber is not None else None) for p in pages]
        logger.info("Processing %d pages in scan order", len(inputs))
        predictions = self.core_engine.process(inputs)
        if len(predictions) != len(pages):
            raise ValueError("Page-type core must return exactly one result per input page")
        # Validate every output before attaching any predictions.
        predictions = [PageTypePrediction.model_validate(p) for p in predictions]
        for page, prediction in zip(pages, predictions, strict=True):
            page.pageType = prediction.pageType
            if prediction.side is not None:
                page.side = prediction.side
            if prediction.startsDocument is not None:
                page.startsDocument = prediction.startsDocument
        return result

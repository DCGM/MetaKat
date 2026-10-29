import copy
import logging
import os

from metakat.page_type.engines.bind.page_type_bind_engine import PageTypeBindEngine
from metakat.page_type.engines.core.page_type_core_engine import PageTypeInput, PageTypePrediction
from metakat.schemas.base_objects import MetakatIO, ProarcIO, MetakatPage, is_annotated

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
        annotated_starts = [is_annotated(page.startsDocument) for page in pages]
        if any(annotated_starts) and not all(annotated_starts):
            raise ValueError('Annotated document starts must be provided for every page')
        starts = [page.startsDocument.val for page in pages] if all(annotated_starts) else None
        if not pages or all(is_annotated(page.pageType) for page in pages):
            return result

        def path(mapping, page_id):
            relative = (mapping or {}).get(page_id)
            return os.path.join(batch_dir, relative) if relative else None

        inputs = [PageTypeInput(
            image_path=path(result.page_to_image_mapping, p.id),
            ocr_path=(path(result.page_to_alto_mapping, p.id)
                      or path(result.page_to_xml_mapping, p.id)),
            page_number=p.pageNumber.text if p.pageNumber is not None else None,
            document_start=starts[i] if starts is not None else None,
            document_end=(starts[i + 1] if i + 1 < len(pages) else True)
                         if starts is not None else None) for i, p in enumerate(pages)]
        logger.info("Processing %d pages in scan order", len(inputs))
        predictions = self.core_engine.process(inputs)
        if len(predictions) != len(pages):
            raise ValueError("Page-type core must return exactly one result per input page")
        # Validate every output before attaching any predictions.
        predictions = [PageTypePrediction.model_validate(p) for p in predictions]
        for page, prediction in zip(pages, predictions, strict=True):
            if not is_annotated(page.pageType):
                page.pageType = prediction.pageType
            if prediction.side is not None:
                page.side = prediction.side
            if prediction.startsDocument is not None and not is_annotated(page.startsDocument):
                page.startsDocument = prediction.startsDocument
        # The first page of a scan starts its first document by definition;
        # the core has no prediction for it, having no page before it to
        # compare with. Setting it here, with full confidence, spares every
        # later stage the implicit start.
        if not is_annotated(pages[0].startsDocument):
            pages[0].startsDocument = (True, 1.0)
        return result

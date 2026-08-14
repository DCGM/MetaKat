import copy
import logging
import os
from collections import OrderedDict

from natsort import natsorted

from metakat.page_type.engines.bind.page_type_bind_engine import PageTypeBindEngine
from metakat.schemas.base_objects import MetakatIO, ProarcIO, DocumentType

logger = logging.getLogger(__name__)


class PageTypeBindEngineBase(PageTypeBindEngine):
    def __init__(self, config, core_config):
        super().__init__(config, core_config)

    def process(self, batch_dir: str, metakat_io: MetakatIO, proarc_io: ProarcIO = None) -> MetakatIO:
        metakat_io=copy.deepcopy(metakat_io)
        image_path_to_page_id = OrderedDict(natsorted(
            (os.path.join(batch_dir, y), x) for x, y in metakat_io.page_to_image_mapping.items())
        )

        page_id_to_metakat_page = {page.id: page for page in metakat_io.elements if page.type == DocumentType.PAGE.value}
        logger.info(f"Processing {len(image_path_to_page_id)} MetaKatPage elements with page type core engine")
        predictions = self.core_engine.process(list(image_path_to_page_id.keys()))
        logger.info(f"Page type core engine returned {len(predictions)} predictions")

        logger.info(f"Adding page types to MetaKatPage elements")
        added_page_types = 0
        for image_path, probs in predictions.items():
            page_id = image_path_to_page_id[image_path]
            metakat_page = page_id_to_metakat_page[page_id]
            class_id = probs.index(max(probs))
            page_type = self.core_engine.page_type_by_class_id.get(class_id)
            if page_type is None:
                logger.warning(
                    "Class ID %d not found in the ViT checkpoint label "
                    "mapping; skipping detection",
                    class_id,
                )
                continue
            prob = max(probs)
            metakat_page.pageType = (page_type, prob)
            added_page_types += 1
        logger.info(f"Added page types to {added_page_types} MetaKatPage elements")

        return metakat_io

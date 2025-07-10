import copy
import logging
import os
from typing import Dict, List, Tuple
from collections import OrderedDict

from natsort import natsorted

from page_type.engines.bind.page_type_bind_engine import PageTypeBindEngine
from schemas.base_objects import MetakatIO, ProarcIO, PageType, MetakatPage

logger = logging.getLogger(__name__)


class PageTypeBindEngineBase(PageTypeBindEngine):
    def __init__(self, bind_engine_dir, core_engine_dir):
        super().__init__(bind_engine_dir, core_engine_dir)

    def process(self, batch_dir: str, metakat_io: MetakatIO, proarc_io: ProarcIO = None) -> MetakatIO:
        metakat_io=copy.deepcopy(metakat_io)
        image_path_to_page_id = OrderedDict(natsorted(
            (os.path.join(batch_dir, y), x) for x, y in metakat_io.page_to_image_mapping.items())
        )

        page_id_to_metakat_page = {page.id: page for page in metakat_io.elements if isinstance(page, MetakatPage)}
        logger.info(f"Processing {len(image_path_to_page_id)} MetaKatPage elements with page type core engine")
        predictions = self.core_engine.process(list(image_path_to_page_id.keys()))
        logger.info(f"Page type core engine returned {len(predictions)} predictions")

        logger.info(f"Adding page types to MetaKatPage elements")
        added_page_types = 0
        for i, (image_path, probs) in enumerate(predictions.items()):
            page_id = image_path_to_page_id[image_path]
            metakat_page = page_id_to_metakat_page[page_id]
            class_id = str(probs.index(max(probs)))
            if class_id not in self.core_engine.id2label:
                logger.warning(
                    f"Id {class_id} not found in id2label mapping (read from metakat_engine_config.json), skipping detection")
                continue
            page_type = self.core_engine.id2label[class_id]
            page_type = PageType(page_type)
            prob = max(probs)
            metakat_page.pageType = (page_type, prob)
            added_page_types += 1
        logger.info(f"Added page types to {added_page_types} MetaKatPage elements")

        return metakat_io




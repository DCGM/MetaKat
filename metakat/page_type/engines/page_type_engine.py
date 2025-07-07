import copy
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from natsort import natsorted
from collections import OrderedDict

from metakat.schemas.base_objects import MetakatIO, ProarcIO, MetakatPage, PageType

import logging


logger = logging.getLogger(__name__)


class PageTypeEngine(ABC):
    @abstractmethod
    def process(self, images: List[str]) -> Tuple[Dict[str, List[float]], Dict[int, str]]:
        pass

    def process_for_metakat(self, batch_dir: str, metakat_io: MetakatIO, proarc_io: ProarcIO = None) -> Tuple[MetakatIO, Dict[str, List[float]], Dict[int, str]]:
        metakat_io = copy.deepcopy(metakat_io)
        image_path_to_page_id = OrderedDict(natsorted(
            (os.path.join(batch_dir, y), x) for x, y in metakat_io.page_to_image_mapping.items())
        )
        page_id_to_metakat_page = {page.id: page for page in metakat_io.elements if isinstance(page, MetakatPage)}
        predictions, id2label = self.process(list(image_path_to_page_id.keys()))
        for image_path, probs in predictions.items():
            page_id = image_path_to_page_id[image_path]
            metakat_page = page_id_to_metakat_page[page_id]
            page_type = id2label[probs.index(max(probs))]
            if page_type not in PageType:
                logger.warning(f"Page type engine returned unknown page type: {page_type} for image: {image_path}, fix engine to return one of {list(PageType)}, skipping")
                continue
            page_type = PageType(page_type)
            prob = max(probs)
            metakat_page.pageType = (page_type, prob)
        return metakat_io, predictions, id2label

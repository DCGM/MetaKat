import os
from abc import ABC, abstractmethod
from typing import List, Tuple

from biblio.engines.core.biblio_engine_core import BiblioEngineCore
from detector_wrapper.parsers.pero_ocr import ALTOMatchedPage
from metakat.schemas.base_objects import MetakatIO, ProarcIO, MetakatPage

import logging

from schemas.base_objects import DocumentType, PageType

logger = logging.getLogger(__name__)


class BiblioEngineBind(ABC):
    def __init__(self, biblio_engine_core: BiblioEngineCore):
        self.biblio_engine_core = biblio_engine_core

    @abstractmethod
    def process(self, batch_dir: str, metakat_io: MetakatIO, proarc_io: ProarcIO = None) -> Tuple[MetakatIO]:
        pass
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from metakat.schemas.base_objects import MetakatIO, ProarcIO

import logging

from metakat.page_type.engines.core.definitions import load_page_type_core_engine
from metakat.engine_config import require_config_mapping, require_engine_name

logger = logging.getLogger(__name__)


class PageTypeBindEngine(ABC):
    def __init__(
        self,
        config: Mapping[str, Any],
        core_config: Mapping[str, Any],
    ):
        self.config = require_config_mapping(config, "Page-type bind config")
        self.name = require_engine_name(self.config, "Page-type bind config")
        self.core_engine = load_page_type_core_engine(core_config)

        logger.info(f"Loaded page type bind engine: {self.name}")

    @abstractmethod
    def process(self, batch_dir: str, metakat_io: MetakatIO, proarc_io: ProarcIO = None) -> MetakatIO:
        pass

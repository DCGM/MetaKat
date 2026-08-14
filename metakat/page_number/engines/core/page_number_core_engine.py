import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any
from typing import List

from metakat.page_number.engines.core.models import PageNumberCoreResult
from metakat.engine_config import require_config_mapping, require_engine_name


logger = logging.getLogger(__name__)


class PageNumberCoreEngine(ABC):
    def __init__(self, config: Mapping[str, Any]):
        self.config = require_config_mapping(config, "Page-number core config")
        self.name = require_engine_name(self.config, "Page-number core config")
        logger.info("Loaded page number core engine: %s", self.name)

    @abstractmethod
    def process(
        self,
        images: List[str],
        alto_files: List[str],
    ) -> PageNumberCoreResult:
        pass

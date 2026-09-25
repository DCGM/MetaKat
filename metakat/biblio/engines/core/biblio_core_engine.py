import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, List

from metakat.biblio.engines.core.models import BiblioCoreResult
from metakat.engine_config import require_config_mapping, require_engine_name

logger = logging.getLogger(__name__)


class BiblioCoreEngine(ABC):
    def __init__(self, config: Mapping[str, Any]):
        self.config = require_config_mapping(config, "Biblio core config")
        self.name = require_engine_name(self.config, "Biblio core config")
        logger.info("Loaded biblio core engine: %s", self.name)

    @abstractmethod
    def process(
        self,
        images: List[str],
        alto_files: List[str],
    ) -> BiblioCoreResult:
        pass

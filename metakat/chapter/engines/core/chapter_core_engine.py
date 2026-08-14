import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Sequence

from metakat.chapter.engines.core.models import TocResult
from metakat.common.models import PageDimensions
from metakat.page_number.engines.core.models import (
    PhysicalPageNumberEvidence,
)
from metakat.engine_config import require_config_mapping, require_engine_name

logger = logging.getLogger(__name__)


class ChapterCoreEngine(ABC):
    def __init__(self, config: Mapping[str, Any]):
        self.config = require_config_mapping(config, "Chapter core config")
        self.name = require_engine_name(self.config, "Chapter core config")
        logger.info(
            "Chapter core engine config: \n%s",
            json.dumps(self.config, indent=4),
        )
        logger.info("Loaded chapter core engine: %s", self.name)

    @abstractmethod
    def process(
        self,
        images: Sequence[str],
        alto_files: Sequence[str],
        page_numbers: Sequence[PhysicalPageNumberEvidence] | None = None,
        image_dimensions: Sequence[PageDimensions | None] | None = None,
        alto_dimensions: Sequence[PageDimensions | None] | None = None,
    ) -> TocResult:
        pass

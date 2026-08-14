from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any
from metakat.chapter.engines.core.definitions import load_chapter_core_engine
from metakat.engine_config import require_config_mapping, require_engine_name

from metakat.schemas.base_objects import MetakatIO, ProarcIO

import logging


logger = logging.getLogger(__name__)


class ChapterBindEngine(ABC):
    def __init__(
        self,
        config: Mapping[str, Any],
        core_config: Mapping[str, Any],
    ):
        self.config = require_config_mapping(config, "Chapter bind config")
        self.name = require_engine_name(self.config, "Chapter bind config")
        self.core_engine = load_chapter_core_engine(core_config)

        logger.info(f"Loaded chapter bind engine: {self.name}")

    @abstractmethod
    def process(self, batch_dir: str, metakat_io: MetakatIO, proarc_io: ProarcIO = None) -> MetakatIO:
        pass

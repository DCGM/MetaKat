import json
import logging
import os
from abc import ABC, abstractmethod
from typing import List

from metakat.page_number.engines.core.models import PageNumberCoreResult


logger = logging.getLogger(__name__)


class PageNumberCoreEngine(ABC):
    def __init__(self, core_engine_dir: str):
        logger.info(
            "Loading page number core engine from: %s",
            core_engine_dir,
        )
        self.engine_dir = core_engine_dir
        config_path = os.path.join(
            core_engine_dir,
            "metakat_engine_config.json",
        )
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Page number core engine config not found at {config_path}"
            )
        with open(config_path, "r", encoding="utf-8") as source:
            self.config = json.load(source)

        self.name = self.config["name"]
        logger.info("Loaded page number core engine: %s", self.name)

    @abstractmethod
    def process(
        self,
        images: List[str],
        alto_files: List[str],
    ) -> PageNumberCoreResult:
        pass

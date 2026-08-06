import json
import logging
import os
from abc import ABC, abstractmethod

from metakat.page_number.engines.core.definitions import (
    load_page_number_core_engine,
)
from metakat.schemas.base_objects import MetakatIO, ProarcIO


logger = logging.getLogger(__name__)


class PageNumberBindEngine(ABC):
    def __init__(self, bind_engine_dir: str, core_engine_dir: str):
        logger.info(
            "Loading page number bind engine from: %s",
            bind_engine_dir,
        )
        self.engine_dir = bind_engine_dir
        config_path = os.path.join(
            bind_engine_dir,
            "metakat_engine_config.json",
        )
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Page number bind engine config not found at {config_path}"
            )
        with open(config_path, "r", encoding="utf-8") as source:
            self.config = json.load(source)

        self.name = self.config["name"]
        self.core_engine = load_page_number_core_engine(core_engine_dir)
        logger.info("Loaded page number bind engine: %s", self.name)

    @abstractmethod
    def process(
        self,
        batch_dir: str,
        metakat_io: MetakatIO,
        proarc_io: ProarcIO = None,
    ) -> MetakatIO:
        pass

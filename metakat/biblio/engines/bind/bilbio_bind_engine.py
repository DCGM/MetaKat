import json
import os
from abc import ABC, abstractmethod
from typing import Tuple
from biblio.engines.core.definitions import load_biblio_core_engine

from metakat.schemas.base_objects import MetakatIO, ProarcIO

import logging


logger = logging.getLogger(__name__)


class BiblioBindEngine(ABC):
    def __init__(self, bind_engine_dir: str, core_engine_dir: str):
        logger.info(f"Loading biblio bind engine from: {bind_engine_dir}")
        self.engine_dir = bind_engine_dir
        config_path = os.path.join(bind_engine_dir, "metakat_engine_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Biblio bind engine config not found at {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        logger.info(f"Biblio bind engine config {config_path}: \n{json.dumps(self.config, indent=4)}")

        self.name = self.config['name']

        self.core_engine = load_biblio_core_engine(core_engine_dir)

        logger.info(f"Loaded biblio bind engine: {self.name}")

    @abstractmethod
    def process(self, batch_dir: str, metakat_io: MetakatIO, proarc_io: ProarcIO = None) -> MetakatIO:
        pass
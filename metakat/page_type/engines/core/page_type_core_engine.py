import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from metakat.schemas.base_objects import PageType

import logging


logger = logging.getLogger(__name__)


class PageTypeCoreEngine(ABC):
    def __init__(self, core_engine_dir: str):
        logger.info(f"Loading page type core engine from: {core_engine_dir}")
        self.engine_dir = core_engine_dir
        config_path = os.path.join(core_engine_dir, "metakat_engine_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Page type core engine config not found at {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        logger.info(f"Page type core engine config {config_path}: \n{json.dumps(self.config, indent=4)}")

        self.name = self.config['name']
        self.id2label = self.config['id2label']

        if not isinstance(self.id2label, dict):
            raise ValueError(f"Invalid id2label format in config: {self.id2label}")
        if not self.id2label:
            raise ValueError("id2label cannot be empty in config")

        for my_id, label in self.id2label.items():
            try:
                PageType(label)
            except ValueError:
                raise ValueError(f"Invalid PageType label in config: '{my_id}: {label}'")

        logger.info(f"Loaded page type core engine: {self.name}")
        logger.info(f"{len(self.id2label)}")

    @abstractmethod
    def process(self, images: List[str]) -> Dict[str, List[float]]:
        pass
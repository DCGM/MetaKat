import json
import os
import logging
from abc import ABC, abstractmethod
from typing import List

from detector_wrapper.parsers.pero_ocr import ALTOMatchedPage
from metakat.schemas.base_objects import ChapterType

logger = logging.getLogger(__name__)


class ChapterCoreEngine(ABC):
    def __init__(self, core_engine_dir: str):
        logger.info(f"Loading chapter core engine from: {core_engine_dir}")
        self.engine_dir = core_engine_dir
        config_path = os.path.join(core_engine_dir, "metakat_engine_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Chapter core engine config not found at {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        logger.info(f"Chapter core engine config: \n{json.dumps(self.config, indent=4)}")

        self.name = self.config['name']
        self.id2label = self.config['id2label']

        if not isinstance(self.id2label, dict):
            raise ValueError(f"Invalid id2label format in config: {self.id2label}")
        if not self.id2label:
            raise ValueError("id2label cannot be empty in config")

        for my_id, label in self.id2label.items():
            try:
                ChapterType(label)
            except ValueError:
                raise ValueError(f"Invalid ChapterType label in config: '{my_id}: {label}'")

        logger.info(f"Loaded chapter core engine: {self.name}")
        logger.info(f"{len(self.id2label)}")

    @abstractmethod
    def process(self, images: List[str], alto_files: List[str]) -> List[ALTOMatchedPage]:
        pass

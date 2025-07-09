import json
import os
import logging
from abc import ABC, abstractmethod
from typing import List

from detector_wrapper.parsers.pero_ocr import ALTOMatchedPage
from schemas.base_objects import BiblioType

logger = logging.getLogger(__name__)


class BiblioEngineCore(ABC):
    def __init__(self, engine_dir: str):
        self.engine_dir = engine_dir
        config_path = os.path.join(engine_dir, "metakat_engine_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Engine config not found at {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.name = self.config["name"]
        self.id2label = self.config["id2label"]

        logger.info(f"Loading biblio engine '{self.name}' from {engine_dir}")
        if not isinstance(self.id2label, dict):
            raise ValueError(f"Invalid id2label format in config: {self.id2label}")
        if not all(isinstance(k, int) and isinstance(v, str) for k, v in self.id2label.items()):
            raise ValueError(f"Invalid id2label format in config: {self.id2label}")
        if not self.id2label:
            raise ValueError("id2label cannot be empty in config")

        for my_id, label in self.id2label.items():
            try:
                BiblioType(label)
            except ValueError:
                raise ValueError(f"Invalid BiblioType label in config: '{my_id}: {label}'")

        logger.info(f"{len(self.id2label)}")

    @abstractmethod
    def process(self, images: List[str], alto_files: List[str]) -> List[ALTOMatchedPage]:
        pass

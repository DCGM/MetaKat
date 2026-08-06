import json
import logging
import os
from abc import ABC, abstractmethod
from typing import List

from text_geometry_aligner import AlignmentPage

from metakat.schemas.base_objects import PageNumberType


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
        self.id2label = self.config["id2label"]
        if not isinstance(self.id2label, dict) or not self.id2label:
            raise ValueError("id2label must be a non-empty object in config")
        for class_id, label in self.id2label.items():
            try:
                PageNumberType(label)
            except ValueError as error:
                raise ValueError(
                    "Invalid PageNumberType label in config: "
                    f"{class_id!r}: {label!r}"
                ) from error

        logger.info("Loaded page number core engine: %s", self.name)

    @abstractmethod
    def process(
        self,
        images: List[str],
        alto_files: List[str],
    ) -> List[AlignmentPage]:
        pass

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Mapping
from typing import Any, Dict, List

from metakat.schemas.base_objects import PageType
from metakat.engine_config import require_config_mapping, require_engine_name

import logging


logger = logging.getLogger(__name__)


class PageTypeCoreEngine(ABC):
    def __init__(self, config: Mapping[str, Any]):
        self.config = require_config_mapping(config, "Page-type core config")
        self.name = require_engine_name(self.config, "Page-type core config")
        if "id2label" in self.config:
            raise ValueError("id2label is not supported; use the labels mapping")
        configured_labels = self.config.get("labels", {})
        if not isinstance(configured_labels, dict):
            raise ValueError("labels must be an object")

        self.labels: dict[PageType, str] = {
            page_type: page_type.value for page_type in PageType
        }
        for raw_type, label in configured_labels.items():
            try:
                page_type = PageType(raw_type)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Unknown page-type label type: {raw_type!r}"
                ) from error
            if not isinstance(label, str) or not label.strip():
                raise ValueError(
                    f"Label for {page_type.value!r} must be a non-empty "
                    "string"
                )
            self.labels[page_type] = label

        duplicate_labels = {
            label
            for label, count in Counter(self.labels.values()).items()
            if count > 1
        }
        if duplicate_labels:
            raise ValueError(
                "Page-type model labels must be unique: "
                + ", ".join(sorted(duplicate_labels))
            )

        logger.info(f"Loaded page type core engine: {self.name}")
        logger.info("Loaded %d page-type label(s)", len(self.labels))

    @abstractmethod
    def process(self, images: List[str]) -> Dict[str, List[float]]:
        pass

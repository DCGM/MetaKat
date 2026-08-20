from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, List, TYPE_CHECKING

# Annotation-only import. text_geometry_aligner ships with the [inference]
# tier, but this module is reached by every `import metakat.process_batch`,
# including on installs that carry no engine. Keeping it off the runtime
# path lets those installs import the pipeline, so a missing engine
# dependency is reported by the engine preflight, naming the extra that
# supplies it, instead of failing here with a bare ImportError.
if TYPE_CHECKING:
    from text_geometry_aligner import AlignmentPage

from metakat.schemas.base_objects import BiblioType
from metakat.engine_config import require_config_mapping, require_engine_name

logger = logging.getLogger(__name__)


class BiblioCoreEngine(ABC):
    def __init__(self, config: Mapping[str, Any]):
        self.config = require_config_mapping(config, "Biblio core config")
        self.name = require_engine_name(self.config, "Biblio core config")
        if "id2label" in self.config:
            raise ValueError("id2label is not supported; use the labels mapping")
        configured_labels = self.config.get("labels")
        if not isinstance(configured_labels, dict) or not configured_labels:
            raise ValueError("labels must be a non-empty object")

        self.labels: dict[BiblioType, str] = {}
        for raw_type, label in configured_labels.items():
            try:
                biblio_type = BiblioType(raw_type)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Unknown bibliographic label type: {raw_type!r}"
                ) from error
            if not isinstance(label, str) or not label.strip():
                raise ValueError(
                    f"Label for {biblio_type.value!r} must be a non-empty "
                    "string"
                )
            if label in self.labels.values():
                raise ValueError(
                    f"Bibliographic model label is assigned more than once: "
                    f"{label!r}"
                )
            self.labels[biblio_type] = label

        self.biblio_type_by_label = {
            label: biblio_type
            for biblio_type, label in self.labels.items()
        }

        logger.info(f"Loaded biblio core engine: {self.name}")
        logger.info("Loaded %d bibliographic label(s)", len(self.labels))

    @abstractmethod
    def process(
        self,
        images: List[str],
        alto_files: List[str],
    ) -> List[AlignmentPage]:
        pass

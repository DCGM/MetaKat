import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

from metakat.chapter.engines.core.models import TocResult
from metakat.common.models import PageDimensions
from metakat.page_number.engines.core.models import (
    PhysicalPageNumberEvidence,
)

logger = logging.getLogger(__name__)


class ChapterCoreEngine(ABC):
    def __init__(self, core_engine_dir: str | Path):
        self.engine_dir = Path(core_engine_dir)
        logger.info("Loading chapter core engine from: %s", self.engine_dir)
        config_path = self.engine_dir / "metakat_engine_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Chapter core engine config not found at {config_path}"
            )
        with config_path.open("r", encoding="utf-8") as source:
            config: Any = json.load(source)
        if not isinstance(config, dict):
            raise ValueError(
                "Chapter core engine config must be a JSON object: "
                f"{config_path}"
            )
        name = config.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "Chapter core engine config must contain a non-empty "
                f"string 'name': {config_path}"
            )

        self.config: dict[str, Any] = config
        self.name = name
        logger.info(
            "Chapter core engine config: \n%s",
            json.dumps(self.config, indent=4),
        )
        logger.info("Loaded chapter core engine: %s", self.name)

    @abstractmethod
    def process(
        self,
        images: Sequence[str],
        alto_files: Sequence[str],
        page_numbers: Sequence[PhysicalPageNumberEvidence] | None = None,
        image_dimensions: Sequence[PageDimensions | None] | None = None,
        alto_dimensions: Sequence[PageDimensions | None] | None = None,
    ) -> TocResult:
        pass

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import List

from text_geometry_aligner import AlignmentPage

from metakat.page_number.engines.core.models import PageNumberCoreResult
from metakat.page_number.engines.core.page_number_resolver import (
    PhysicalPageNumberResolver,
)
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

        self.page_number_resolver = PhysicalPageNumberResolver.from_config(
            self.config
        )

        logger.info("Loaded page number core engine: %s", self.name)

    def process(
        self,
        images: List[str],
        alto_files: List[str],
    ) -> PageNumberCoreResult:
        alignment_pages = self._align(images, alto_files)
        page_numbers = {}
        seen_page_keys = set()
        for page in alignment_pages:
            if page.page_key in seen_page_keys:
                raise ValueError(
                    "Page-number core returned duplicate page key: "
                    f"{page.page_key}"
                )
            seen_page_keys.add(page.page_key)
            regions = []
            for region in page.regions:
                if region.category_id is None:
                    if region.matched:
                        logger.warning(
                            "Matched region %s on page %s has no category ID; "
                            "skipping detection",
                            region.region_id,
                            page.page_key,
                        )
                    continue
                class_id = str(region.category_id)
                if class_id not in self.id2label:
                    logger.warning(
                        "CLASS_ID %s (label=%r, label_export=%r) not in "
                        "id2label, skipping - page_key: %s",
                        class_id,
                        region.label,
                        region.label_export,
                        page.page_key,
                    )
                    continue
                if (
                    PageNumberType(self.id2label[class_id])
                    == PageNumberType.PAGE_NUMBER
                ):
                    regions.append(region)

            resolution = self.page_number_resolver.resolve(page, regions)
            selected_evidence = resolution.selected_evidence
            if selected_evidence is not None:
                page_numbers[page.page_key] = selected_evidence
        logger.info(
            "Page number core resolved %d page number(s) from %d page(s)",
            len(page_numbers),
            len(alignment_pages),
        )
        return PageNumberCoreResult(page_numbers=page_numbers)

    @abstractmethod
    def _align(
        self,
        images: List[str],
        alto_files: List[str],
    ) -> List[AlignmentPage]:
        pass

import logging
from typing import List

from text_geometry_aligner import AlignmentPage

from metakat.common.engines.engine_yolo_alto import EngineYOLOALTO
from metakat.page_number.engines.core.page_number_core_engine import (
    PageNumberCoreEngine,
)
from metakat.page_number.engines.core.models import PageNumberCoreResult
from metakat.page_number.engines.core.page_number_parsers import (
    DecoratedPageNumberParser,
)
from metakat.page_number.engines.core.page_number_resolver import (
    PhysicalPageNumberResolver,
)
from metakat.schemas.base_objects import PageNumberType


logger = logging.getLogger(__name__)


class PageNumberCoreEngineYOLO(PageNumberCoreEngine):
    DEFAULT_LABELS: dict[PageNumberType, str] = {
        PageNumberType.PAGE_NUMBER: "cislo strany",
    }

    def __init__(
        self,
        core_engine_dir: str,
        yolo_batch_size: int = 32,
        yolo_confidence_threshold: float = 0.25,
        yolo_image_size: int = 640,
        minimum_overlap_coverage: float = 0.65,
    ):
        super().__init__(core_engine_dir=core_engine_dir)
        self.labels = self._load_labels()
        self.page_number_parser = DecoratedPageNumberParser
        self.page_number_resolver = PhysicalPageNumberResolver.from_config(
            self.config
        )
        self.engine_yolo_alto = EngineYOLOALTO(
            engine_dir=self.engine_dir,
            yolo_batch_size=yolo_batch_size,
            yolo_confidence_threshold=yolo_confidence_threshold,
            yolo_image_size=yolo_image_size,
            minimum_overlap_coverage=minimum_overlap_coverage,
        )

    def process(
        self,
        images: List[str],
        alto_files: List[str],
    ) -> PageNumberCoreResult:
        alignment_pages = self._align(images, alto_files)
        page_numbers = {}
        seen_page_keys = set()
        page_number_label = self.labels[PageNumberType.PAGE_NUMBER]

        for page in alignment_pages:
            if page.page_key in seen_page_keys:
                raise ValueError(
                    "Page-number core returned duplicate page key: "
                    f"{page.page_key}"
                )
            seen_page_keys.add(page.page_key)

            candidates = tuple(
                evidence
                for region in page.regions
                if region.label_for_export == page_number_label
                and (
                    evidence := self.page_number_parser.parse_region(
                        page_key=page.page_key,
                        region=region,
                    )
                )
                is not None
            )
            selected = self.page_number_resolver.resolve(
                candidates,
                page_width=page.alto_width,
                page_height=page.alto_height,
            )
            if selected is not None:
                page_numbers[page.page_key] = selected

        logger.info(
            "Page number core resolved %d page number(s) from %d page(s)",
            len(page_numbers),
            len(alignment_pages),
        )
        return PageNumberCoreResult(page_numbers=page_numbers)

    def _align(
        self,
        images: List[str],
        alto_files: List[str],
    ) -> List[AlignmentPage]:
        return self.engine_yolo_alto.process(
            images=images,
            alto_files=alto_files,
        ).pages

    def _load_labels(self) -> dict[PageNumberType, str]:
        if "id2label" in self.config:
            raise ValueError(
                "id2label is not supported; use the labels mapping"
            )
        configured = self.config.get("labels", {})
        if not isinstance(configured, dict):
            raise ValueError("labels must be a JSON object")

        result = dict(self.DEFAULT_LABELS)
        for raw_type, label in configured.items():
            try:
                page_number_type = PageNumberType(raw_type)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Unknown page-number label type: {raw_type!r}"
                ) from error
            if page_number_type not in self.DEFAULT_LABELS:
                raise ValueError(
                    "Page-number label type "
                    f"{page_number_type.value!r} is not used by this engine"
                )
            if not isinstance(label, str) or not label.strip():
                raise ValueError(
                    f"Label for {page_number_type.value!r} must be a "
                    "non-empty string"
                )
            result[page_number_type] = label
        return result

import logging
from dataclasses import replace
from collections.abc import Mapping
from pathlib import Path
from typing import Any, List

from text_geometry_aligner import ALTOReader, AlignmentPage

from metakat.common.engines.engine_yolo_alto import EngineYOLOALTO
from metakat.common.models import PageDimensions
from metakat.page_number.engines.core.ndk_pagination import NdkPageContext
from metakat.page_number.engines.core.alto_recovery import AltoPageNumberRecovery
from metakat.page_number.engines.core.postprocessing import (
    PageNumberPostprocessor,
    PageNumberPostprocessConfig,
    PageNumberPostprocessInput,
)

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
        config: Mapping[str, Any],
        yolo_batch_size: int = 32,
        yolo_confidence_threshold: float = 0.25,
        yolo_image_size: int = 640,
        minimum_overlap_coverage: float = 0.65,
    ):
        super().__init__(config=config)
        self.labels = self._load_labels()
        self.page_number_parser = DecoratedPageNumberParser
        configured = PageNumberPostprocessConfig.from_config(self.config)
        # The scan starts and ends with a document, so the final pass asserts complete documents
        # (their borders come from the page-type stage); the preliminary pass fills plain numbers only.
        preliminary = replace(
            configured, ndk=replace(configured.ndk, wholly_unnumbered=False)
        )
        self.postprocessors = {
            False: PageNumberPostprocessor(
                preliminary, bracketed=False, complete_document=False
            ),
            True: PageNumberPostprocessor(
                configured, bracketed=True, complete_document=True
            ),
        }
        self._final_pass = configured.ndk.enabled
        self.page_number_resolver = PhysicalPageNumberResolver.from_config(self.config)
        self.engine_yolo_alto = EngineYOLOALTO(
            config=self.config,
            yolo_batch_size=yolo_batch_size,
            yolo_confidence_threshold=yolo_confidence_threshold,
            yolo_image_size=yolo_image_size,
            minimum_overlap_coverage=minimum_overlap_coverage,
        )

    @property
    def has_final_pass(self) -> bool:
        return self._final_pass

    def process(
        self,
        images: List[str],
        alto_files: List[str],
        *,
        page_contexts: Mapping[str, NdkPageContext] | None = None,
        final: bool = False,
    ) -> PageNumberCoreResult:
        if len(images) != len(alto_files):
            raise ValueError(
                "Page-number images and ALTO files must have equal lengths"
            )
        keys = [Path(image).stem for image in images]
        key_set = set(keys)
        contexts = page_contexts or {}
        if set(contexts) - key_set:
            raise ValueError("Page-number context contains unknown page keys")
        if any(
            not isinstance(context, NdkPageContext) for context in contexts.values()
        ):
            raise ValueError("Page-number contexts must be NdkPageContext objects")
        if len(keys) != len(key_set):
            raise ValueError("Page-number images must have unique stems")
        if any(Path(alto).stem != key for key, alto in zip(keys, alto_files)):
            raise ValueError("Corresponding image and ALTO stems must match")
        if not images:
            return PageNumberCoreResult()
        alignment_pages = self._align(images, alto_files)
        aligned_by_key = {}
        page_number_label = self.labels[PageNumberType.PAGE_NUMBER]
        for page in alignment_pages:
            if page.page_key in aligned_by_key:
                raise ValueError(
                    "Page-number core returned duplicate page key: " f"{page.page_key}"
                )
            if page.page_key not in key_set:
                raise ValueError(
                    f"Page-number core returned an unknown page key: {page.page_key}"
                )
            aligned_by_key[page.page_key] = page

        inputs = []
        for key, alto_path in zip(keys, alto_files):
            page = aligned_by_key.get(key)
            if page is None:
                # Some detectors omit empty pages; retain their physical place
                # in the sequence and still allow OCR recovery.
                alto = ALTOReader().read(alto_path)
                if alto.width is None or alto.height is None:
                    raise ValueError(
                        f"Page-number processing requires ALTO page dimensions: {key}"
                    )
                dimensions = PageDimensions(alto.width, alto.height)
                inputs.append(
                    PageNumberPostprocessInput(
                        key, dimensions, pagination=contexts.get(key, NdkPageContext())
                    )
                )
                continue
            if page.alto_width is None or page.alto_height is None:
                raise ValueError(
                    f"Page-number processing requires ALTO page dimensions: {key}"
                )
            dimensions = PageDimensions(page.alto_width, page.alto_height)

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
            inputs.append(
                PageNumberPostprocessInput(
                    key,
                    dimensions,
                    candidates,
                    selected,
                    pagination=contexts.get(key, NdkPageContext()),
                )
            )

        result = self.postprocessors[final].process(
            inputs, recover=AltoPageNumberRecovery(dict(zip(keys, alto_files)))
        )

        logger.info(
            "Page number core resolved %d page number(s) from %d page(s)",
            sum(evidence is not None for evidence in result.page_numbers.values()),
            len(inputs),
        )
        return result

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
            raise ValueError("id2label is not supported; use the labels mapping")
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

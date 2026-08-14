from __future__ import annotations

import unicodedata
import re
from typing import Any

from text_geometry_aligner import AlignmentRegion

from metakat.common.models import BoundingBox, DetectionEvidence
from metakat.schemas.base_objects import ChapterType


def load_chapter_label_mapping(
    config: dict[str, Any],
    defaults: dict[ChapterType, str],
) -> dict[ChapterType, str]:
    configured = config.get("labels", {})
    if not isinstance(configured, dict):
        raise ValueError("labels must be a JSON object")

    result = dict(defaults)
    for raw_type, label in configured.items():
        try:
            chapter_type = ChapterType(raw_type)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Unknown chapter label type: {raw_type!r}"
            ) from error
        if chapter_type not in defaults:
            raise ValueError(
                f"Chapter label type {chapter_type.value!r} is not used "
                "by this engine"
            )
        if not isinstance(label, str) or not label.strip():
            raise ValueError(
                f"Label for {chapter_type.value!r} must be a non-empty "
                "string"
            )
        result[chapter_type] = label
    return result


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower().strip())
    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )
    normalized = re.sub(r"[.\-_\xb7\u2022]{2,}", " ", normalized)
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip()


def region_to_evidence(
    region: AlignmentRegion,
    page_key: str,
) -> DetectionEvidence | None:
    if (
        not region.matched
        or region.input_geometry is None
        or region.input_geometry_confidence is None
        or not region.alto_text
    ):
        return None
    return DetectionEvidence(
        text=region.alto_text.strip(),
        confidence=region.input_geometry_confidence,
        bbox=BoundingBox(
            x=region.input_geometry.bounds.x,
            y=region.input_geometry.bounds.y,
            width=region.input_geometry.bounds.width,
            height=region.input_geometry.bounds.height,
        ),
        page_key=page_key,
    )


def region_label(region: AlignmentRegion) -> str:
    return region.label_for_export

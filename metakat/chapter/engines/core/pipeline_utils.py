from __future__ import annotations

import json
import unicodedata
import re
from pathlib import Path
from typing import Any

from text_geometry_aligner import AlignmentRegion

from metakat.chapter.engines.core.toc_page_analysis.models import (
    DetectionEvidence,
)


def load_engine_config(engine_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    directory = Path(engine_dir)
    config_path = directory / "metakat_engine_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Engine config not found at {config_path}")
    with config_path.open("r", encoding="utf-8") as source:
        config = json.load(source)
    return directory, config


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
        bbox=region.input_geometry.bounds,
        page_key=page_key,
    )


def region_label(region: AlignmentRegion) -> str:
    return region.label_for_export

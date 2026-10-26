from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Mapping

from metakat.common.models import DetectionEvidence, PageDimensions

PageNumberTextCase = Literal["lowercase", "uppercase"]


class PageNumberNumeralSystem(str, Enum):
    ARABIC = "arabic"
    ROMAN = "roman"


def apply_page_number_text_case(
    text: str,
    case: PageNumberTextCase | None,
) -> str:
    if case is None:
        return text
    if case == "lowercase":
        return text.lower()
    if case == "uppercase":
        return text.upper()
    raise ValueError(f"Unsupported page-number text case: {case!r}")


@dataclass(frozen=True)
class PhysicalPageNumberEvidence(DetectionEvidence):
    normalized: str | None
    value: int | None
    numeral_system: PageNumberNumeralSystem | None
    # Supplied cataloguing labels have no fabricated OCR text or geometry.
    supplied: bool = field(default=False, kw_only=True)
    suffix: str = field(default="", kw_only=True)
    bracketed: bool = field(default=False, kw_only=True)

    def normalized_text(
        self,
        *,
        case: PageNumberTextCase | None = None,
    ) -> str | None:
        if self.normalized is None:
            return None
        return apply_page_number_text_case(self.normalized, case)

    def output_text(
        self,
        *,
        case: PageNumberTextCase | None = None,
    ) -> str:
        normalized = self.normalized_text(case=case)
        if normalized is not None:
            label = normalized + self.suffix
            return f"[{label}]" if self.bracketed else label
        return apply_page_number_text_case(self.text, case)


@dataclass(frozen=True)
class PageNumberDecision:
    """Review information; confidence is a support score, not a calibrated probability."""

    action: str
    position_support: float | None = None
    sequence_agreement: bool | None = None
    detector_confidence: float | None = None
    original_reading: str | None = None
    sequence_support: float | None = None
    pagination_rule: str | None = None
    observation_action: str | None = None
    chunk_index: int | None = None


@dataclass(frozen=True)
class PageNumberChunk:
    """Ordered processing scope; an inferred scope need not be a whole book."""

    page_keys: tuple[str, ...]
    reason: str
    inferred: bool = False
    ambiguous: bool = False
    unit_key: str | None = None


@dataclass(frozen=True)
class PageNumberCoreResult:
    page_numbers: Mapping[str, PhysicalPageNumberEvidence | None] = field(
        default_factory=dict
    )
    decisions: Mapping[str, PageNumberDecision] = field(default_factory=dict)
    chunks: tuple[PageNumberChunk, ...] = ()


def format_page_number(value: int, template: PhysicalPageNumberEvidence) -> str | None:
    if template.normalized is None:
        return None
    if template.numeral_system == PageNumberNumeralSystem.ARABIC:
        token = str(value)
        if template.normalized.startswith("0"):
            token = token.zfill(len(template.normalized))
        return token
    if not 0 < value < 4000:
        return None
    result = ""
    for number, token in (
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ):
        count, value = divmod(value, number)
        result += token * count
    return result.lower() if template.normalized.islower() else result


def valid_detected_page_number(
    evidence: PhysicalPageNumberEvidence, *, page_key: str, dimensions: PageDimensions
) -> bool:
    """Shared eligibility for raw detections; supplied labels are not anchors."""
    if evidence.page_key != page_key:
        raise ValueError(f"Candidate page key does not match {page_key!r}")
    box = evidence.bbox
    return (
        not evidence.supplied
        and not evidence.suffix
        and evidence.value is not None
        and evidence.value >= 0
        and evidence.normalized is not None
        and evidence.numeral_system is not None
        and box is not None
        and all(math.isfinite(v) for v in (box.x, box.y, box.width, box.height))
        and box.width > 0
        and box.height > 0
        and box.x >= 0
        and box.y >= 0
        and box.x_max <= dimensions.width
        and box.y_max <= dimensions.height
    )

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Mapping

from metakat.common.models import DetectionEvidence


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
            return normalized
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


@dataclass(frozen=True)
class PageNumberCoreResult:
    page_numbers: Mapping[str, PhysicalPageNumberEvidence | None] = field(
        default_factory=dict
    )
    decisions: Mapping[str, PageNumberDecision] = field(default_factory=dict)

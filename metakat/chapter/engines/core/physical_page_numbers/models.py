from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

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

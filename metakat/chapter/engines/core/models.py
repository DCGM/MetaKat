from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path

from metakat.common.models import DetectionEvidence, PageDimensions
from metakat.page_number.engines.core.models import (
    PageNumberNumeralSystem,
    PageNumberTextCase,
    apply_page_number_text_case,
)


class TocPageNumberKind(str, enum.Enum):
    SINGLE = "single"
    RANGE = "range"
    LIST = "list"


NormalizedTocPageNumberItem = tuple[
    str,
    int,
    PageNumberNumeralSystem,
]


@dataclass(frozen=True)
class ChapterPageInput:
    page_key: str
    position: int
    image_path: Path
    alto_path: Path
    image_dimensions: PageDimensions | None = None
    alto_dimensions: PageDimensions | None = None


@dataclass(frozen=True)
class TocPageNumberEvidence(DetectionEvidence):
    kind: TocPageNumberKind | None
    normalized_items: tuple[NormalizedTocPageNumberItem, ...]

    def normalized_text(
        self,
        *,
        case: PageNumberTextCase | None = None,
    ) -> str | None:
        if self.kind is None or not self.normalized_items:
            return None
        separator = {
            TocPageNumberKind.SINGLE: "",
            TocPageNumberKind.RANGE: "-",
            TocPageNumberKind.LIST: ",",
        }[self.kind]
        return separator.join(
            apply_page_number_text_case(item_text, case)
            for item_text, _, _ in self.normalized_items
        )

    def normalized_start(
        self,
        *,
        case: PageNumberTextCase | None = None,
    ) -> str | None:
        if not self.normalized_items:
            return None
        return apply_page_number_text_case(
            self.normalized_items[0][0],
            case,
        )

    def normalized_end(
        self,
        *,
        case: PageNumberTextCase | None = None,
    ) -> str | None:
        if (
            self.kind is not TocPageNumberKind.RANGE
            or len(self.normalized_items) != 2
        ):
            return None
        return apply_page_number_text_case(
            self.normalized_items[1][0],
            case,
        )

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
class ChapterBase:
    toc_page_key: str
    title: DetectionEvidence | None
    part_number: DetectionEvidence | None = None
    page_number: TocPageNumberEvidence | None = None
    children: tuple[ChapterBase, ...] = ()


@dataclass(frozen=True)
class TocBase:
    chapters: tuple[ChapterBase, ...]


@dataclass(frozen=True)
class ChapterResult(ChapterBase):
    title_destination_page: DetectionEvidence | None = None
    page_start_key: str | None = None
    page_end_key: str | None = None
    children: tuple[ChapterResult, ...] = ()


@dataclass(frozen=True)
class TocResult:
    chapters: tuple[ChapterResult, ...]

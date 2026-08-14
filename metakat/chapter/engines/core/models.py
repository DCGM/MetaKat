from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path

from metakat.common.models import DetectionEvidence, PageDimensions
from metakat.page_number.engines.core.models import (
    PageNumberNumeralSystem,
    PageNumberTextCase,
    apply_page_number_text_case,
)


class ChapterPageNumberKind(str, enum.Enum):
    SINGLE = "single"
    RANGE = "range"
    LIST = "list"


NormalizedChapterPageNumberItem = tuple[
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
class ChapterPageNumberEvidence(DetectionEvidence):
    kind: ChapterPageNumberKind | None
    normalized_items: tuple[NormalizedChapterPageNumberItem, ...]

    def normalized_text(
        self,
        *,
        case: PageNumberTextCase | None = None,
    ) -> str | None:
        if self.kind is None or not self.normalized_items:
            return None
        separator = {
            ChapterPageNumberKind.SINGLE: "",
            ChapterPageNumberKind.RANGE: "-",
            ChapterPageNumberKind.LIST: ",",
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
            self.kind is not ChapterPageNumberKind.RANGE
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
    subtitle: DetectionEvidence | None = field(default=None, kw_only=True)
    part_number: DetectionEvidence | None = None
    page_number: ChapterPageNumberEvidence | None = None
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
    toc_monotonicity_score: float | None = None

    def __post_init__(self) -> None:
        score = self.toc_monotonicity_score
        if (
            score is not None
            and (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not 0 <= score <= 1
            )
        ):
            raise ValueError(
                "toc_monotonicity_score must be None or a number within [0, 1]"
            )

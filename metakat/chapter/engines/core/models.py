from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path

from metakat.common.models import AgentRole, DetectionEvidence, PageDimensions
from metakat.common.vlm.report import VLMCallReport
from metakat.schemas.base_objects import NameType
from metakat.chapter.engines.core.physical_page_numbers.models import (
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
    # The page type the page-type stage gave the page (a PageType value) and
    # its confidence; None when it gave none.
    page_type: str | None = None
    page_type_confidence: float | None = None


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
class ChapterAgent:
    """A person or body printed with a TOC entry: its name in its role (MetakatChapter's
    field of the role with the suffix TocPage), what it is when known, and its
    affiliations (affiliationTocPage)."""

    role: AgentRole
    name: DetectionEvidence
    type: NameType | None = None
    affiliations: tuple[DetectionEvidence, ...] = ()


# Field names follow the MetaKat schema (MetakatChapter): a `_toc_page` field
# was read in the table-of-contents entry, an unsuffixed one on the chapter's
# own destination page.
@dataclass(frozen=True)
class ChapterBase:
    toc_page_key: str
    title_toc_page: DetectionEvidence | None
    subtitle_toc_page: DetectionEvidence | None = field(default=None, kw_only=True)
    # The synopsis printed with the entry - what the chapter covers, often with
    # page numbers of its topics - not a subtitle (MetakatChapter.abstractTocPage).
    description_toc_page: DetectionEvidence | None = field(default=None, kw_only=True)
    # The persons and bodies printed with the entry, in printed order.
    agents_toc_page: tuple[ChapterAgent, ...] = field(default=(), kw_only=True)
    part_number_toc_page: DetectionEvidence | None = None
    page_number_toc_page: ChapterPageNumberEvidence | None = None
    children: tuple[ChapterBase, ...] = ()


@dataclass(frozen=True)
class TocBase:
    chapters: tuple[ChapterBase, ...]
    # One per VLM call, a failed one included, naming the pages it sent by
    # their keys; an engine without a VLM has none.
    reports: tuple[VLMCallReport, ...] = ()


@dataclass(frozen=True)
class ChapterResult(ChapterBase):
    title: DetectionEvidence | None = None
    page_start_key: str | None = None
    page_end_key: str | None = None
    children: tuple[ChapterResult, ...] = ()


@dataclass(frozen=True)
class TocResult:
    chapters: tuple[ChapterResult, ...]
    # The reports of the extraction stage's TocBase.
    reports: tuple[VLMCallReport, ...] = ()

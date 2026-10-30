from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path

from metakat.common.models import AgentRole, DetectionEvidence, PageDimensions
from metakat.common.vlm.report import VLMCallReport
from metakat.schemas.base_objects import ClassifiedValue, NameType, TocEntryKind
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
class ChapterPageRun:
    """One page or one range of a TOC entry's page reference, as the reader split it: the printed tokens,
    `end` None for a single page."""

    start: str
    end: str | None = None


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
    field of the role with the suffix TocPage), what it is when known, its terms
    of address (termsOfAddressTocPage) and its affiliations (affiliationTocPage)."""

    role: AgentRole
    name: DetectionEvidence
    type: NameType | None = None
    affiliations: tuple[DetectionEvidence, ...] = ()
    terms_of_address: tuple[DetectionEvidence, ...] = ()


@dataclass(frozen=True)
class ChapterTitleVersion:
    """A parallel version of a TOC entry's title in another language: its title,
    subtitle and synopsis, each in that language (titleTocPage, subTitleTocPage
    flagged translated, abstractTocPage, after the entry's main version)."""

    title: DetectionEvidence | None = None
    subtitle: DetectionEvidence | None = None
    description: DetectionEvidence | None = None


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
    # The phrase printed with them saying there are more than named ("a kol.").
    etal_toc_page: DetectionEvidence | None = field(default=None, kw_only=True)
    # The entry's title in other languages, after the version above.
    parallel_titles_toc_page: tuple[ChapterTitleVersion, ...] = field(default=(), kw_only=True)
    # Whether the title and subtitle above are a parallel version: false when
    # the reader says they are the main one, None when it cannot tell.
    title_translated_toc_page: bool | None = field(default=None, kw_only=True)
    part_number_toc_page: DetectionEvidence | None = None
    # The whole printed entry marker, its words and punctuation ("Kapitola 3."),
    # which the part number above is read from. Not a title: the normalization
    # makes the title of a chapter printed with only it.
    designation_toc_page: DetectionEvidence | None = field(default=None, kw_only=True)
    # The entry's page reference parsed into numbers, which the alignment
    # anchors the chapter by.
    page_number_toc_page: ChapterPageNumberEvidence | None = None
    # The page reference as printed, unparsed, when the reader gives it apart
    # from page_number_toc_page, whose text is otherwise the printed one.
    page_reference_toc_page: DetectionEvidence | None = field(default=None, kw_only=True)
    # The page reference split into its pages and ranges by the reader, in the
    # printed order; None when the reader does not split it, and the binder
    # splits page_number_toc_page by its parsed shape.
    page_runs_toc_page: tuple[ChapterPageRun, ...] | None = field(default=None, kw_only=True)
    # Whether the entry is a heading or an item, when the reader tells it.
    kind_toc_page: ClassifiedValue[TocEntryKind] | None = field(default=None, kw_only=True)
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

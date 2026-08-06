from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from metakat.chapter.engines.core.toc_extraction.models import ReferenceToc
from metakat.chapter.engines.core.toc_page_analysis.models import (
    ChapterPageInput,
    DestinationChapterEvidence,
    DetectionEvidence,
)


@dataclass(frozen=True)
class ResolvedChapter:
    toc_page_key: str
    title: DetectionEvidence | None
    part_number: DetectionEvidence | None = None
    page_number: DetectionEvidence | None = None
    title_destination_page: DetectionEvidence | None = None
    page_start_key: str | None = None
    page_end_key: str | None = None
    anchor_only: bool = False
    children: tuple[ResolvedChapter, ...] = ()


@dataclass(frozen=True)
class ChapterCoreResult:
    chapters: tuple[ResolvedChapter, ...]


class TocAlignmentEngine(Protocol):
    def process(
        self,
        *,
        pages: Sequence[ChapterPageInput],
        reference_toc: ReferenceToc,
        destination_chapters: Sequence[DestinationChapterEvidence],
    ) -> ChapterCoreResult:
        ...

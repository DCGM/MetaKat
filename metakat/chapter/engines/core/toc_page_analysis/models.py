from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from metakat.chapter.engines.core.models import ChapterPageInput
from metakat.common.models import DetectionEvidence
from metakat.page_number.engines.core.models import (
    PhysicalPageNumberEvidence,
)

@dataclass(frozen=True)
class DestinationChapterEvidence:
    title: DetectionEvidence


@dataclass(frozen=True)
class TocPageAnalysisResult:
    toc_pages: tuple[ChapterPageInput, ...]
    destination_chapters: tuple[
        DestinationChapterEvidence, ...
    ] | None = None
    destination_page_numbers: tuple[
        PhysicalPageNumberEvidence, ...
    ] | None = None


class TocPageAnalysisEngine(Protocol):
    def process(
        self,
        pages: Sequence[ChapterPageInput],
    ) -> TocPageAnalysisResult:
        ...

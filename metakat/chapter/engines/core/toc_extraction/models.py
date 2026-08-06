from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from metakat.chapter.engines.core.toc_page_analysis.models import (
    ChapterPageInput,
    DetectionEvidence,
)


@dataclass(frozen=True)
class ReferenceTocEntry:
    toc_page_key: str
    title: DetectionEvidence | None
    part_number: DetectionEvidence | None = None
    page_number: DetectionEvidence | None = None
    anchor_only: bool = False
    children: tuple[ReferenceTocEntry, ...] = ()


@dataclass(frozen=True)
class ReferenceToc:
    roots: tuple[ReferenceTocEntry, ...]


class TocExtractionEngine(Protocol):
    def process(
        self,
        toc_pages: Sequence[ChapterPageInput],
    ) -> ReferenceToc:
        ...

from __future__ import annotations

from typing import Protocol, Sequence

from metakat.chapter.engines.core.models import (
    ChapterPageInput,
    TocBase,
    TocResult,
)
from metakat.chapter.engines.core.toc_page_analysis.models import (
    DestinationChapterEvidence,
)
from metakat.page_number.engines.core.models import (
    PhysicalPageNumberEvidence,
)


class TocAlignmentEngine(Protocol):
    def process(
        self,
        *,
        pages: Sequence[ChapterPageInput],
        toc_pages: Sequence[ChapterPageInput],
        reference_toc: TocBase,
        destination_chapters: Sequence[DestinationChapterEvidence] | None,
        destination_page_numbers: (
            Sequence[PhysicalPageNumberEvidence] | None
        ),
    ) -> TocResult:
        ...

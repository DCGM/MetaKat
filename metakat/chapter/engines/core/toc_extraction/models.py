from __future__ import annotations

from typing import Protocol, Sequence

from metakat.chapter.engines.core.models import ChapterPageInput, TocBase


class TocExtractionEngine(Protocol):
    def process(
        self,
        toc_pages: Sequence[ChapterPageInput],
    ) -> TocBase:
        ...

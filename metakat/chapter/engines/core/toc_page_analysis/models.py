from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from text_geometry_aligner import BoundingBox


@dataclass(frozen=True)
class ChapterPageInput:
    page_key: str
    position: int
    image_path: Path
    alto_path: Path
    page_number: str | None = None


@dataclass(frozen=True)
class DetectionEvidence:
    text: str
    confidence: float
    bbox: BoundingBox
    page_key: str


@dataclass(frozen=True)
class DestinationChapterEvidence:
    title: DetectionEvidence


@dataclass(frozen=True)
class TocPageAnalysisResult:
    toc_pages: tuple[ChapterPageInput, ...]
    destination_chapters: tuple[DestinationChapterEvidence, ...]
    page_numbers: Mapping[str, DetectionEvidence] = field(
        default_factory=dict
    )


class TocPageAnalysisEngine(Protocol):
    def process(
        self,
        pages: Sequence[ChapterPageInput],
    ) -> TocPageAnalysisResult:
        ...

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable


__all__ = ["AltoRefs", "BoundingBox", "DetectionEvidence", "PageDimensions"]


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned MetaKat bounding box using top-left coordinates."""

    x: float
    y: float
    width: float
    height: float

    @property
    def x_max(self) -> float:
        return self.x + self.width

    @property
    def y_max(self) -> float:
        return self.y + self.height


@dataclass(frozen=True)
class PageDimensions:
    width: float
    height: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.width)
            or not math.isfinite(self.height)
            or self.width <= 0
            or self.height <= 0
        ):
            raise ValueError("Page dimensions must be finite and positive")


@dataclass(frozen=True)
class AltoRefs:
    """The ALTO elements a detection covers, by their ID attributes.

    IDs are only unique within one page's ALTO file, so they are read
    together with the evidence's page. Each level lists the IDs of the
    elements holding the detection's words, once each, in reading order,
    and only where the ALTO gives them an ID: producers differ, and most
    write IDs on blocks, many on lines, few on words.
    """

    blocks: tuple[str, ...] = ()
    lines: tuple[str, ...] = ()
    words: tuple[str, ...] = ()

    @classmethod
    def from_words(cls, words: Iterable | None) -> "AltoRefs":
        """From aligned words carrying alto_block_id, alto_line_id and alto_word_id."""
        words = list(words or ())

        def ids(attribute: str) -> tuple[str, ...]:
            return tuple(dict.fromkeys(
                value for word in words if (value := getattr(word, attribute, None))
            ))

        return cls(blocks=ids("alto_block_id"), lines=ids("alto_line_id"), words=ids("alto_word_id"))

    def is_empty(self) -> bool:
        return not (self.blocks or self.lines or self.words)


@dataclass(frozen=True)
class DetectionEvidence:
    text: str
    confidence: float
    bbox: BoundingBox
    page_key: str
    alto: AltoRefs = field(default=AltoRefs(), kw_only=True)

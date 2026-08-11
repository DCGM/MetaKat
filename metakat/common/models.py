from __future__ import annotations

import math
from dataclasses import dataclass


__all__ = ["BoundingBox", "DetectionEvidence", "PageDimensions"]


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
class DetectionEvidence:
    text: str
    confidence: float
    bbox: BoundingBox
    page_key: str

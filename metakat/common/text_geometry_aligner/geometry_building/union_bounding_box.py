"""Build a union bounding box from matched ALTO words."""

from __future__ import annotations

from typing import Sequence

from ..models import BoundingBox, OCRWord
from .base import GeometryBuilder


class UnionBoundingBoxGeometryBuilder(GeometryBuilder):
    """Return one rectangle covering all matched ALTO word boxes."""

    def build(self, words: Sequence[OCRWord]) -> BoundingBox:
        if not words:
            raise ValueError(
                "Cannot construct geometry from an empty word sequence"
            )

        x_min = min(word.bbox.x for word in words)
        y_min = min(word.bbox.y for word in words)
        x_max = max(word.bbox.x_max for word in words)
        y_max = max(word.bbox.y_max for word in words)
        return BoundingBox(
            x=x_min,
            y=y_min,
            width=x_max - x_min,
            height=y_max - y_min,
        )

"""Interface for constructing output geometry from matched ALTO words."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from ..models import OCRWord, OutputGeometry


class GeometryBuilder(ABC):
    """Convert matched OCR words into output geometry."""

    @abstractmethod
    def build(self, words: Sequence[OCRWord]) -> OutputGeometry:
        raise NotImplementedError

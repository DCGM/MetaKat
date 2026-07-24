"""Interface for constructing output text from matched ALTO words."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from ..models import OCRWord


class TextBuilder(ABC):
    """Convert matched OCR words into output text."""

    @abstractmethod
    def build(self, words: Sequence[OCRWord]) -> str | None:
        raise NotImplementedError

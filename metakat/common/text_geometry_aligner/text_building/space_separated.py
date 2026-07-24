"""Build output text by separating matched ALTO words with spaces."""

from __future__ import annotations

from typing import Sequence

from ..models import OCRWord
from .base import TextBuilder


class SpaceSeparatedTextBuilder(TextBuilder):
    """Join matched words in ALTO reading order using spaces."""

    def build(self, words: Sequence[OCRWord]) -> str | None:
        if not words:
            return None
        return " ".join(word.text for word in words)

"""The words of an aligned value on one page, with their box, as an aligned region is read.

A value printed across a page break keeps only its words on the page it
starts on, so the VLM engines rebuild its region from those words.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Sequence


def words_region(words: Sequence[Any]) -> SimpleNamespace:
    """Some words of an aligned region, with their box, as a region is read."""
    x, y = min(word.bbox.x for word in words), min(word.bbox.y for word in words)
    return SimpleNamespace(words=tuple(words), alto_geometry=SimpleNamespace(
        x=x, y=y, width=max(word.bbox.x_max for word in words) - x,
        height=max(word.bbox.y + word.bbox.height for word in words) - y))

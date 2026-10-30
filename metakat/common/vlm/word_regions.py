"""Some words of a value located in the ALTO, as the location of another value printed in them.

The text aligner gives a word to one value only. A value printed inside
another one's words - a chapter's part number in its designation ("3" of
"Kapitola 3"), an agent's affiliation in its description ("Komenského
university v Bratislavě" of "profesor Komenského university v Bratislavě") -
is therefore not aligned itself but takes its words from the located value.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any, Sequence


def bare(text: str) -> str:
    """The text without anything but letters and digits, case folded."""
    return re.sub(r"\W", "", text).casefold()


def words_run(words: Sequence[Any], text: str) -> tuple:
    """The first consecutive words that read `text`, punctuation and case aside; () when none do."""
    tokens = [token for token in map(bare, text.split()) if token]
    texts = [bare(word.text) for word in words]
    for start in range(len(words) - len(tokens) + 1):
        if tokens and texts[start:start + len(tokens)] == tokens:
            return tuple(words[start:start + len(tokens)])
    return ()


def words_region(words: Sequence[Any]) -> SimpleNamespace:
    """Some words of an aligned region, with their box, as a region is read."""
    x, y = min(word.bbox.x for word in words), min(word.bbox.y for word in words)
    return SimpleNamespace(words=tuple(words), alto_geometry=SimpleNamespace(
        x=x, y=y, width=max(word.bbox.x_max for word in words) - x,
        height=max(word.bbox.y + word.bbox.height for word in words) - y))

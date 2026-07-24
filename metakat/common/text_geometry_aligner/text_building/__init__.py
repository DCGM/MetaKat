"""Output-text building interfaces and implementations."""

from .base import TextBuilder
from .space_separated import SpaceSeparatedTextBuilder

__all__ = [
    "SpaceSeparatedTextBuilder",
    "TextBuilder",
]

"""In-memory JSON extraction and alignment-result merging."""

from .extractor import JSONValueExtractor
from .merger import JSONAlignmentMerger

__all__ = [
    "JSONAlignmentMerger",
    "JSONValueExtractor",
]

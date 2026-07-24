from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from .alto_processing import ALTOTextIndex
from .models import ALTOPage, JSONScalarValue
from .normalization import TextNormalizer


class AlignmentInputNormalizer:
    """Create normalized matching views without mutating source objects."""

    def __init__(self, normalizer: TextNormalizer):
        self.normalizer = normalizer

    def normalize_values(
        self,
        values: Sequence[JSONScalarValue],
    ) -> tuple[JSONScalarValue, ...]:
        return tuple(
            replace(
                value,
                normalized_text=self.normalizer.normalize(value.text),
            )
            for value in values
        )

    def build_alto_index(self, page: ALTOPage) -> ALTOTextIndex:
        return ALTOTextIndex(page, self.normalizer)

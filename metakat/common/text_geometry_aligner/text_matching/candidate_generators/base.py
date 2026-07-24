"""Base interface for text candidate generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from ...alto_processing import ALTOTextIndex
from ...models import AlignmentCandidate, JSONScalarValue


class CandidateGenerator(ABC):
    """Generate deterministic alignment candidates with unique IDs."""

    @abstractmethod
    def generate(
        self,
        values: Sequence[JSONScalarValue],
        alto_index: ALTOTextIndex,
    ) -> tuple[AlignmentCandidate, ...]:
        raise NotImplementedError

"""Base interface for text candidate selectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from ...models import AlignmentCandidate, JSONScalarValue


class CandidateSelector(ABC):
    """Select a globally compatible subset of alignment candidates."""

    @abstractmethod
    def select(
        self,
        candidates: Sequence[AlignmentCandidate],
        values: Sequence[JSONScalarValue],
    ) -> tuple[AlignmentCandidate, ...]:
        raise NotImplementedError

from __future__ import annotations

from typing import Sequence

from ...models import AlignmentCandidate, JSONScalarValue
from .base import CandidateSelector


class PassThroughCandidateSelector(CandidateSelector):
    """Return generated candidates unchanged."""

    def select(
        self,
        candidates: Sequence[AlignmentCandidate],
        values: Sequence[JSONScalarValue],
    ) -> tuple[AlignmentCandidate, ...]:
        return tuple(candidates)

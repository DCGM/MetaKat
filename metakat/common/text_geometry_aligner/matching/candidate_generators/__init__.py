"""Candidate-generation interfaces and implementations."""

from .anchored_fuzzy import (
    AnchoredFuzzyTextCandidateGenerator,
    FuzzyCandidateConfig,
)
from .base import CandidateGenerator
from .composite import CompositeCandidateGenerator
from .exact import ExactTextCandidateGenerator
from .ordered_alignment import (
    OrderedAlignmentCandidateConfig,
    OrderedAlignmentCandidateGenerator,
)

__all__ = [
    "AnchoredFuzzyTextCandidateGenerator",
    "CandidateGenerator",
    "CompositeCandidateGenerator",
    "ExactTextCandidateGenerator",
    "FuzzyCandidateConfig",
    "OrderedAlignmentCandidateConfig",
    "OrderedAlignmentCandidateGenerator",
]

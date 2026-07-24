"""Text candidate generation, global selection, and diagnostics."""

from .candidate_generators import (
    AnchoredFuzzyTextCandidateGenerator,
    CandidateGenerator,
    CompositeCandidateGenerator,
    ExactTextCandidateGenerator,
    FuzzyCandidateConfig,
    OrderedAlignmentCandidateConfig,
    OrderedAlignmentCandidateGenerator,
)
from .candidate_selectors import (
    CPSATCandidateSelector,
    CandidateSelector,
    PassThroughCandidateSelector,
)

__all__ = [
    "AnchoredFuzzyTextCandidateGenerator", "CandidateGenerator",
    "CompositeCandidateGenerator", "CPSATCandidateSelector",
    "CandidateSelector", "ExactTextCandidateGenerator",
    "FuzzyCandidateConfig", "OrderedAlignmentCandidateConfig",
    "OrderedAlignmentCandidateGenerator", "PassThroughCandidateSelector",
]

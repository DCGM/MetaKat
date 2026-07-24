"""Geometry-to-ALTO overlap calculation and word assignment."""

from .assignment import (
    AllOverThresholdWordAssigner,
    GeometryWordAssigner,
    GreatestCoverageWordAssigner,
    WordAssignmentStrategy,
    create_word_assigner,
)
from .overlap import (
    BoundingBoxOverlapCalculator,
    GeometryOverlapCalculator,
    ShapelyOverlapCalculator,
    WordCoverage,
    create_overlap_calculator,
)

__all__ = [
    "AllOverThresholdWordAssigner",
    "BoundingBoxOverlapCalculator",
    "GeometryOverlapCalculator",
    "GeometryWordAssigner",
    "GreatestCoverageWordAssigner",
    "ShapelyOverlapCalculator",
    "WordAssignmentStrategy",
    "WordCoverage",
    "create_overlap_calculator",
    "create_word_assigner",
]

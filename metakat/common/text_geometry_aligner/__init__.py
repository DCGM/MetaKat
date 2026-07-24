"""Bidirectional JSON text/geometry alignment against ALTO OCR."""

from .alto_io import ALTOReader
from .alto_processing import ALTOTextIndex
from .base_aligner import BaseAligner
from .geometry_building import (
    GeometryBuilder,
    OrthogonalPolygonGeometryBuilder,
    UnionBoundingBoxGeometryBuilder,
)
from .geometry_matching import (
    AllOverThresholdWordAssigner,
    BoundingBoxOverlapCalculator,
    GeometryOverlapCalculator,
    GeometryWordAssigner,
    GreatestCoverageWordAssigner,
    ShapelyOverlapCalculator,
    WordAssignmentStrategy,
    WordCoverage,
    create_overlap_calculator,
    create_word_assigner,
)
from .json_io import JSONReader, JSONWriter
from .json_processing import (
    JSONGeometryExtractor,
    JSONGeometryMerger,
    JSONTextExtractor,
    JSONTextMerger,
)
from .text_matching.candidate_generators import (
    AnchoredFuzzyTextCandidateGenerator,
    CandidateGenerator,
    CompositeCandidateGenerator,
    ExactTextCandidateGenerator,
    FuzzyCandidateConfig,
    OrderedAlignmentCandidateConfig,
    OrderedAlignmentCandidateGenerator,
)
from .text_matching.candidate_selectors import (
    CPSATCandidateSelector,
    CandidateSelector,
    PassThroughCandidateSelector,
)
from .models import (
    ALTOPage,
    CER_SCALE,
    SIMILARITY_SCALE,
    AlignmentCandidate,
    BoundingBox,
    GeometryAlignmentResult,
    GeometryWordAlignment,
    JSONGeometryRegion,
    JSONScalarValue,
    OCRWord,
    OCRWordSpan,
    OutputGeometry,
    OutputGeometryFormat,
    OutputTextSource,
    Point,
    Polygon,
    RenderAlignment,
    SelectedAlignment,
    TextAlignmentResult,
)
from .normalization import (
    DiacriticStrippingTextNormalizer,
    LowercaseTextNormalizer,
    PunctuationStrippingTextNormalizer,
    StrictTextNormalizer,
    TextNormalizationPipeline,
    TextNormalizer,
    UnicodeTextNormalizer,
    WhitespaceTextNormalizer,
)
from .preprocessing import AlignmentInputNormalizer
from .rendering import AlignmentRenderer, PillowAlignmentRenderer
from .text_building import SpaceSeparatedTextBuilder, TextBuilder


def __getattr__(name: str):
    if name == "TextAligner":
        from .text_aligner import TextAligner

        return TextAligner
    if name == "GeometryAligner":
        from .geometry_aligner import GeometryAligner

        return GeometryAligner
    raise AttributeError(name)


__all__ = [
    "ALTOPage", "ALTOReader", "ALTOTextIndex", "AlignmentCandidate",
    "AlignmentInputNormalizer", "AlignmentRenderer",
    "AllOverThresholdWordAssigner",
    "AnchoredFuzzyTextCandidateGenerator", "BaseAligner",
    "BoundingBox", "BoundingBoxOverlapCalculator", "CER_SCALE",
    "CPSATCandidateSelector", "CandidateGenerator", "CandidateSelector",
    "CompositeCandidateGenerator",
    "DiacriticStrippingTextNormalizer", "ExactTextCandidateGenerator",
    "FuzzyCandidateConfig", "GeometryAligner", "GeometryAlignmentResult",
    "GeometryBuilder", "GeometryOverlapCalculator",
    "GeometryWordAlignment",
    "GeometryWordAssigner", "GreatestCoverageWordAssigner",
    "JSONGeometryExtractor", "JSONGeometryMerger",
    "JSONGeometryRegion", "JSONReader", "JSONScalarValue",
    "JSONTextExtractor", "JSONTextMerger", "JSONWriter",
    "LowercaseTextNormalizer", "OCRWord",
    "OCRWordSpan", "OutputGeometry", "OutputGeometryFormat", "OutputTextSource",
    "OrderedAlignmentCandidateConfig", "OrderedAlignmentCandidateGenerator",
    "PillowAlignmentRenderer", "Point", "Polygon", "RenderAlignment",
    "PassThroughCandidateSelector",
    "OrthogonalPolygonGeometryBuilder",
    "PunctuationStrippingTextNormalizer",
    "SIMILARITY_SCALE", "SelectedAlignment", "ShapelyOverlapCalculator",
    "StrictTextNormalizer", "TextAligner", "TextAlignmentResult",
    "SpaceSeparatedTextBuilder", "TextBuilder",
    "TextNormalizationPipeline", "TextNormalizer",
    "UnicodeTextNormalizer", "UnionBoundingBoxGeometryBuilder",
    "WhitespaceTextNormalizer", "WordAssignmentStrategy", "WordCoverage",
    "create_overlap_calculator", "create_word_assigner",
]

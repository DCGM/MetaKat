"""Text-to-geometry alignment package."""

from .alto_io import ALTOReader
from .alto_processing import ALTOTextIndex
from .geometry import (
    GeometryBuilder,
    OrthogonalPolygonGeometryBuilder,
    UnionBoundingBoxGeometryBuilder,
    create_geometry_builder,
)
from .json_io import JSONReader, JSONWriter
from .json_processing import JSONAlignmentMerger, JSONValueExtractor
from .matching.candidate_generators import (
    AnchoredFuzzyTextCandidateGenerator,
    CandidateGenerator,
    CompositeCandidateGenerator,
    ExactTextCandidateGenerator,
    FuzzyCandidateConfig,
    OrderedAlignmentCandidateConfig,
    OrderedAlignmentCandidateGenerator,
)
from .matching.candidate_selectors import (
    CPSATCandidateSelector,
    CandidateSelector,
    PassThroughCandidateSelector,
)
from .models import (
    ALTOPage,
    CER_SCALE,
    SIMILARITY_SCALE,
    AlignmentCandidate,
    AlignmentDirection,
    BoundingBox,
    JSONScalarValue,
    OCRWord,
    OCRWordSpan,
    OutputGeometry,
    OutputGeometryFormat,
    OutputTextSource,
    PageAlignmentResult,
    Point,
    Polygon,
    SelectedAlignment,
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


def __getattr__(name: str):
    if name in {"TextGeometryAligner", "build_argument_parser", "main"}:
        from . import text_geometry_aligner

        return getattr(text_geometry_aligner, name)
    raise AttributeError(name)


__all__ = [
    "ALTOPage", "ALTOReader", "ALTOTextIndex", "AlignmentCandidate",
    "AlignmentDirection", "AlignmentInputNormalizer", "AlignmentRenderer",
    "AnchoredFuzzyTextCandidateGenerator", "BoundingBox", "CER_SCALE",
    "CPSATCandidateSelector", "CandidateGenerator", "CandidateSelector",
    "CompositeCandidateGenerator",
    "DiacriticStrippingTextNormalizer", "ExactTextCandidateGenerator",
    "FuzzyCandidateConfig", "GeometryBuilder",
    "JSONAlignmentMerger", "JSONReader", "JSONScalarValue",
    "JSONValueExtractor", "JSONWriter",
    "LowercaseTextNormalizer", "OCRWord",
    "OCRWordSpan", "OutputGeometry", "OutputGeometryFormat", "OutputTextSource",
    "OrderedAlignmentCandidateConfig", "OrderedAlignmentCandidateGenerator",
    "PageAlignmentResult", "PillowAlignmentRenderer", "Point", "Polygon",
    "PassThroughCandidateSelector",
    "OrthogonalPolygonGeometryBuilder",
    "PunctuationStrippingTextNormalizer",
    "SIMILARITY_SCALE", "SelectedAlignment", "StrictTextNormalizer",
    "TextGeometryAligner", "TextNormalizationPipeline", "TextNormalizer",
    "UnicodeTextNormalizer", "UnionBoundingBoxGeometryBuilder",
    "WhitespaceTextNormalizer", "build_argument_parser",
    "create_geometry_builder", "main",
]

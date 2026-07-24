from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

JSONPathPart = str | int
JSONPath = tuple[JSONPathPart, ...]
SIMILARITY_SCALE = 1_000_000
CER_SCALE = 1_000_000


class OutputTextSource(str, Enum):
    """Source text written to matched JSON values and rendered labels."""

    JSON = "json"
    ALTO = "alto"


class OutputGeometryFormat(str, Enum):
    """Supported geometry representations in aligned output."""

    BBOX = "bbox"
    POLYGON = "polygon"


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in the original ALTO coordinate system."""

    x: float
    y: float
    width: float
    height: float

    @property
    def x_max(self) -> float:
        return self.x + self.width

    @property
    def y_max(self) -> float:
        return self.y + self.height

    @property
    def bounds(self) -> BoundingBox:
        return self

    def to_json(self) -> dict[str, int | float]:
        return {
            "x": _clean_number(self.x),
            "y": _clean_number(self.y),
            "width": _clean_number(self.width),
            "height": _clean_number(self.height),
        }


Point = tuple[float, float]


@dataclass(frozen=True)
class Polygon:
    """Closed polygon in the original ALTO coordinate system."""

    points: tuple[Point, ...]

    def __post_init__(self) -> None:
        if len(self.points) < 4:
            raise ValueError(
                "A closed polygon requires at least three vertices"
            )
        if self.points[0] != self.points[-1]:
            raise ValueError(
                "Polygon points must be closed by repeating the first point"
            )

    @property
    def bounds(self) -> BoundingBox:
        vertices = self.points[:-1]
        x_min = min(point[0] for point in vertices)
        y_min = min(point[1] for point in vertices)
        x_max = max(point[0] for point in vertices)
        y_max = max(point[1] for point in vertices)
        return BoundingBox(
            x=x_min,
            y=y_min,
            width=x_max - x_min,
            height=y_max - y_min,
        )

    def to_json(self) -> list[list[int | float]]:
        return [
            [_clean_number(x), _clean_number(y)]
            for x, y in self.points
        ]


OutputGeometry = BoundingBox | Polygon


@dataclass(frozen=True)
class OCRWord:
    """One ALTO ``String`` element in document order."""

    index: int
    text: str
    bbox: BoundingBox
    line_index: Optional[int] = None
    block_index: Optional[int] = None
    element_id: Optional[str] = None


@dataclass(frozen=True)
class ALTOPage:
    """Parsed ALTO page."""

    source_path: Path
    words: tuple[OCRWord, ...]
    page_id: Optional[str] = None
    width: Optional[float] = None
    height: Optional[float] = None


@dataclass(frozen=True)
class OCRWordSpan:
    """Character interval occupied by an ALTO word in normalized page text."""

    word_index: int
    char_start: int
    char_end: int  # Exclusive.


@dataclass(frozen=True)
class JSONScalarValue:
    """A scalar JSON value and the path where its geometry will be written."""

    value_id: int
    path: JSONPath
    key: str
    original_value: str | int | float
    text: str
    normalized_text: str
    geometry_path: Optional[JSONPath] = None

    @property
    def query_length(self) -> int:
        """Normalized non-whitespace length used by optimization."""

        return sum(not character.isspace() for character in self.normalized_text)


@dataclass(frozen=True)
class AlignmentCandidate:
    """A possible alignment of one JSON value to a contiguous ALTO word range."""

    candidate_id: int
    value_id: int
    json_path: JSONPath
    start_word: int
    end_word: int  # Inclusive.
    start_char: int
    end_char: int  # Exclusive.
    query_text: str
    matched_text: str
    normalized_query_text: str
    normalized_matched_text: str
    exact: bool
    edit_distance: int
    cer_int: int
    similarity_int: int
    query_length: int
    quality_chars: int
    source: str

    @property
    def word_indexes(self) -> range:
        return range(self.start_word, self.end_word + 1)


@dataclass(frozen=True)
class SelectedAlignment:
    """Selected candidate together with its final geometry."""

    candidate: AlignmentCandidate
    geometry: OutputGeometry


@dataclass(frozen=True)
class RenderAlignment:
    """Direction-neutral data required to render one alignment."""

    alignment_id: int
    geometry: OutputGeometry
    text: str
    score: float


@dataclass
class TextAlignmentResult:
    """Text-to-geometry result and diagnostics for one JSON/ALTO pair."""

    output_data: Any
    values: tuple[JSONScalarValue, ...]
    candidates: tuple[AlignmentCandidate, ...]
    selected_alignments: tuple[SelectedAlignment, ...]
    unmatched_value_ids: tuple[int, ...]
    output_text_source: OutputTextSource = OutputTextSource.JSON
    output_geometry_format: OutputGeometryFormat = OutputGeometryFormat.BBOX
    ambiguous_value_ids: tuple[int, ...] = ()
    conflicted_value_ids: tuple[int, ...] = ()

    @property
    def matched_count(self) -> int:
        return len(self.selected_alignments)

    @property
    def unmatched_count(self) -> int:
        return len(self.unmatched_value_ids)

    def text_for_alignment(self, alignment: SelectedAlignment) -> str:
        if self.output_text_source is OutputTextSource.ALTO:
            return alignment.candidate.matched_text
        return alignment.candidate.query_text

    @property
    def render_alignments(self) -> tuple[RenderAlignment, ...]:
        return tuple(
            RenderAlignment(
                alignment_id=alignment.candidate.value_id,
                geometry=alignment.geometry,
                text=self.text_for_alignment(alignment),
                score=alignment.candidate.similarity_int / SIMILARITY_SCALE,
            )
            for alignment in self.selected_alignments
        )


@dataclass(frozen=True)
class JSONGeometryRegion:
    """One JSON geometry and the suffix-derived text destination it owns."""

    region_id: int
    geometry_path: JSONPath
    text_path: JSONPath
    geometry: OutputGeometry


@dataclass(frozen=True)
class GeometryWordAlignment:
    """ALTO words assigned to one input JSON geometry."""

    region: JSONGeometryRegion
    word_indexes: tuple[int, ...]
    word_coverages: tuple[float, ...]
    extracted_text: Optional[str]

    def __post_init__(self) -> None:
        if len(self.word_indexes) != len(self.word_coverages):
            raise ValueError(
                "word_indexes and word_coverages must have the same length"
            )
        if tuple(sorted(set(self.word_indexes))) != self.word_indexes:
            raise ValueError(
                "word_indexes must be unique and in ALTO document order"
            )
        if any(
            not 0.0 <= coverage <= 1.0
            for coverage in self.word_coverages
        ):
            raise ValueError("word_coverages must be within [0, 1]")
        if bool(self.word_indexes) != (self.extracted_text is not None):
            raise ValueError(
                "extracted_text must be present exactly when words are assigned"
            )

    @property
    def average_coverage(self) -> float:
        if not self.word_coverages:
            return 0.0
        return sum(self.word_coverages) / len(self.word_coverages)


@dataclass
class GeometryAlignmentResult:
    """Geometry-to-text result for one JSON/ALTO pair."""

    output_data: Any
    regions: tuple[JSONGeometryRegion, ...]
    alignments: tuple[GeometryWordAlignment, ...]

    @property
    def matched_count(self) -> int:
        return sum(
            alignment.extracted_text is not None
            for alignment in self.alignments
        )

    @property
    def unmatched_count(self) -> int:
        return len(self.alignments) - self.matched_count

    @property
    def unmatched_region_ids(self) -> tuple[int, ...]:
        return tuple(
            alignment.region.region_id
            for alignment in self.alignments
            if alignment.extracted_text is None
        )

    @property
    def render_alignments(self) -> tuple[RenderAlignment, ...]:
        return tuple(
            RenderAlignment(
                alignment_id=alignment.region.region_id,
                geometry=alignment.region.geometry,
                text=(
                    alignment.extracted_text
                    if alignment.extracted_text is not None
                    else "null"
                ),
                score=alignment.average_coverage,
            )
            for alignment in self.alignments
        )


def _clean_number(value: float) -> int | float:
    rounded = round(value)
    if math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-9):
        return int(rounded)
    return value

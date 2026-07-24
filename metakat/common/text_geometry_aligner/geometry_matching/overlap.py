from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from ..models import (
    BoundingBox,
    JSONGeometryRegion,
    OCRWord,
    Polygon,
)
from ..utils import _format_json_path


@dataclass(frozen=True)
class WordCoverage:
    """The fraction of one ALTO word covered by one JSON region."""

    region_id: int
    word_index: int
    coverage: float


class GeometryOverlapCalculator(ABC):
    """Calculate eligible region-to-word area coverage."""

    @abstractmethod
    def calculate(
        self,
        regions: Sequence[JSONGeometryRegion],
        words: Sequence[OCRWord],
        minimum_word_coverage: float,
    ) -> tuple[WordCoverage, ...]:
        raise NotImplementedError


class BoundingBoxOverlapCalculator(GeometryOverlapCalculator):
    """Dependency-free rectangle overlap calculator."""

    def calculate(
        self,
        regions: Sequence[JSONGeometryRegion],
        words: Sequence[OCRWord],
        minimum_word_coverage: float,
    ) -> tuple[WordCoverage, ...]:
        _validate_threshold(minimum_word_coverage)
        if any(isinstance(region.geometry, Polygon) for region in regions):
            raise RuntimeError(
                "Polygon geometry alignment requires Shapely. Install it "
                "with: python -m pip install Shapely"
            )

        coverages: list[WordCoverage] = []
        for region in regions:
            geometry = region.geometry
            if not isinstance(geometry, BoundingBox):
                continue
            for word in words:
                if word.bbox.width <= 0 or word.bbox.height <= 0:
                    continue
                word_area = word.bbox.width * word.bbox.height
                intersection_area = _bbox_intersection_area(
                    geometry,
                    word.bbox,
                )
                if intersection_area <= 0:
                    continue
                coverage = intersection_area / word_area
                if coverage + 1e-12 >= minimum_word_coverage:
                    coverages.append(
                        WordCoverage(
                            region_id=region.region_id,
                            word_index=word.index,
                            coverage=coverage,
                        )
                    )
        return tuple(coverages)


class ShapelyOverlapCalculator(GeometryOverlapCalculator):
    """Shapely-backed bbox and polygon overlap calculator."""

    def __init__(
        self,
        geometry_factory: Callable[..., Any] | None = None,
        polygon_class: Any = None,
    ):
        self.geometry_factory = (
            geometry_factory or _load_shapely_box_factory()
        )
        self.polygon_class = (
            polygon_class or _load_shapely_polygon_class()
        )

    def calculate(
        self,
        regions: Sequence[JSONGeometryRegion],
        words: Sequence[OCRWord],
        minimum_word_coverage: float,
    ) -> tuple[WordCoverage, ...]:
        _validate_threshold(minimum_word_coverage)
        region_shapes = {
            region.region_id: self._region_shape(region)
            for region in regions
        }
        word_shapes = {
            word.index: self.geometry_factory(
                word.bbox.x,
                word.bbox.y,
                word.bbox.x_max,
                word.bbox.y_max,
            )
            for word in words
            if word.bbox.width > 0 and word.bbox.height > 0
        }

        coverages: list[WordCoverage] = []
        for region in regions:
            region_shape = region_shapes[region.region_id]
            region_bounds = region.geometry.bounds
            for word in words:
                word_shape = word_shapes.get(word.index)
                if word_shape is None:
                    continue
                if _bbox_intersection_area(region_bounds, word.bbox) <= 0:
                    continue
                intersection_area = float(
                    region_shape.intersection(word_shape).area
                )
                if intersection_area <= 0:
                    continue
                word_area = word.bbox.width * word.bbox.height
                coverage = intersection_area / word_area
                if coverage + 1e-12 >= minimum_word_coverage:
                    coverages.append(
                        WordCoverage(
                            region_id=region.region_id,
                            word_index=word.index,
                            coverage=coverage,
                        )
                    )
        return tuple(coverages)

    def _region_shape(self, region: JSONGeometryRegion) -> Any:
        geometry = region.geometry
        if isinstance(geometry, BoundingBox):
            return self.geometry_factory(
                geometry.x,
                geometry.y,
                geometry.x_max,
                geometry.y_max,
            )

        shape = self.polygon_class(geometry.points[:-1])
        if shape.is_empty or shape.area <= 0 or not shape.is_valid:
            raise ValueError(
                "Invalid polygon at "
                f"{_format_json_path(region.geometry_path)}"
            )
        return shape


def create_overlap_calculator(
    regions: Sequence[JSONGeometryRegion],
) -> GeometryOverlapCalculator:
    try:
        box_factory = _load_shapely_box_factory()
    except ImportError as exc:
        if any(isinstance(region.geometry, Polygon) for region in regions):
            raise RuntimeError(
                "Polygon geometry alignment requires Shapely. Install it "
                "with: python -m pip install Shapely"
            ) from exc
        return BoundingBoxOverlapCalculator()
    return ShapelyOverlapCalculator(box_factory)


def _load_shapely_box_factory() -> Callable[..., Any]:
    from shapely.geometry import box

    return box


def _load_shapely_polygon_class() -> Any:
    from shapely.geometry import Polygon as ShapelyPolygon

    return ShapelyPolygon


def _bbox_intersection_area(
    first: BoundingBox,
    second: BoundingBox,
) -> float:
    width = min(first.x_max, second.x_max) - max(first.x, second.x)
    height = min(first.y_max, second.y_max) - max(first.y, second.y)
    if width <= 0 or height <= 0:
        return 0.0
    return width * height


def _validate_threshold(minimum_word_coverage: float) -> None:
    if not 0.0 <= minimum_word_coverage <= 1.0:
        raise ValueError("minimum_word_coverage must be within [0, 1]")

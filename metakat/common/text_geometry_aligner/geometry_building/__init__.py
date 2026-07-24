"""Output-geometry building interfaces and implementations."""

from .base import GeometryBuilder
from .orthogonal_polygon import OrthogonalPolygonGeometryBuilder
from .union_bounding_box import UnionBoundingBoxGeometryBuilder

__all__ = [
    "GeometryBuilder",
    "OrthogonalPolygonGeometryBuilder",
    "UnionBoundingBoxGeometryBuilder",
]

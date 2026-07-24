from __future__ import annotations

import math
from numbers import Real
from typing import Any

from ..models import (
    BoundingBox,
    JSONGeometryRegion,
    JSONPath,
    Point,
    Polygon,
)
from ..utils import _format_json_path


class JSONGeometryExtractor:
    """Extract suffix-selected geometries and their destination text paths."""

    def __init__(
        self,
        geometry_suffix: str = "_bbox",
        overwrite_existing_text: bool = False,
    ):
        if not geometry_suffix:
            raise ValueError("geometry_suffix must not be empty")
        self.geometry_suffix = geometry_suffix
        self.overwrite_existing_text = overwrite_existing_text

    def extract(self, data: Any) -> tuple[JSONGeometryRegion, ...]:
        if not isinstance(data, dict):
            raise TypeError("Geometry-alignment JSON root must be an object")

        regions: list[JSONGeometryRegion] = []

        def visit_document(node: Any, path: JSONPath) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if not isinstance(key, str):
                        continue
                    child_path = path + (key,)
                    if key.endswith(self.geometry_suffix):
                        destination_key = key[: -len(self.geometry_suffix)]
                        if not destination_key:
                            raise ValueError(
                                "Geometry suffix leaves an empty destination "
                                f"key at {_format_json_path(child_path)}"
                            )
                        visit_geometry(
                            value,
                            geometry_path=child_path,
                            text_path=path + (destination_key,),
                        )
                    else:
                        visit_document(value, child_path)
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    visit_document(value, path + (index,))

        def visit_geometry(
            node: Any,
            *,
            geometry_path: JSONPath,
            text_path: JSONPath,
        ) -> None:
            destination_exists, destination = _path_value(data, text_path)
            terminal_shape = isinstance(node, dict) or _is_point_sequence(
                node
            )
            if not self.overwrite_existing_text and destination_exists:
                if terminal_shape or not isinstance(destination, list):
                    return

            geometry = _parse_geometry_or_none(node, geometry_path)
            if geometry is not None:
                regions.append(
                    JSONGeometryRegion(
                        region_id=len(regions),
                        geometry_path=geometry_path,
                        text_path=text_path,
                        geometry=geometry,
                    )
                )
                return

            if node is None:
                return
            if not isinstance(node, list):
                raise ValueError(
                    "Expected a bbox, polygon, list, or null at "
                    f"{_format_json_path(geometry_path)}"
                )
            for index, value in enumerate(node):
                visit_geometry(
                    value,
                    geometry_path=geometry_path + (index,),
                    text_path=text_path + (index,),
                )

        visit_document(data, ())
        return tuple(regions)


def _parse_geometry_or_none(
    value: Any,
    path: JSONPath,
) -> BoundingBox | Polygon | None:
    if isinstance(value, dict):
        return _parse_bounding_box(value, path)
    if _is_point_sequence(value):
        return _parse_polygon(value, path)
    return None


def _parse_bounding_box(
    value: dict[Any, Any],
    path: JSONPath,
) -> BoundingBox:
    required_keys = {"x", "y", "width", "height"}
    if not required_keys.issubset(value):
        raise ValueError(
            f"Invalid bbox at {_format_json_path(path)}: expected keys "
            "x, y, width, and height"
        )
    x, y, width, height = (
        _finite_number(value[key], path, key)
        for key in ("x", "y", "width", "height")
    )
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid bbox at {_format_json_path(path)}: width and height "
            "must be positive"
        )
    return BoundingBox(x=x, y=y, width=width, height=height)


def _parse_polygon(value: list[Any], path: JSONPath) -> Polygon:
    points: list[Point] = []
    for index, coordinates in enumerate(value):
        x = _finite_number(coordinates[0], path + (index,), "x")
        y = _finite_number(coordinates[1], path + (index,), "y")
        points.append((x, y))
    try:
        polygon = Polygon(tuple(points))
    except ValueError as exc:
        raise ValueError(
            f"Invalid polygon at {_format_json_path(path)}: {exc}"
        ) from exc
    if polygon.bounds.width <= 0 or polygon.bounds.height <= 0:
        raise ValueError(
            f"Invalid polygon at {_format_json_path(path)}: area bounds "
            "must be positive"
        )
    return polygon


def _is_point_sequence(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) >= 4
        and all(
            isinstance(point, (list, tuple))
            and len(point) == 2
            and all(_is_number(coordinate) for coordinate in point)
            for point in value
        )
    )


def _finite_number(value: Any, path: JSONPath, label: str) -> float:
    if not _is_number(value):
        raise ValueError(
            f"Invalid {label} at {_format_json_path(path)}: expected a number"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(
            f"Invalid {label} at {_format_json_path(path)}: must be finite"
        )
    return number


def _is_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _path_value(root: Any, path: JSONPath) -> tuple[bool, Any]:
    node = root
    for component in path:
        if isinstance(component, str):
            if not isinstance(node, dict) or component not in node:
                return False, None
            node = node[component]
        elif isinstance(component, int):
            if (
                not isinstance(node, list)
                or not 0 <= component < len(node)
            ):
                return False, None
            node = node[component]
        else:
            return False, None
    return True, node

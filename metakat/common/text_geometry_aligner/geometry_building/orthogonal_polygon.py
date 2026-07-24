"""Build a tight orthogonal polygon from matched ALTO words."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from ..models import BoundingBox, OCRWord, Point, Polygon
from .base import GeometryBuilder
from .union_bounding_box import UnionBoundingBoxGeometryBuilder


class OrthogonalPolygonGeometryBuilder(GeometryBuilder):
    """Return an ALTO-aware polygon with horizontal and vertical edges.

    Words sharing an ALTO line are first consolidated into a tight line box;
    words without a line identifier retain their individual boxes.  Horizontal
    bands are then derived without relying on document order.  When two
    occupied bands are vertically separated, their horizontal overlap bridges
    the gap.  An edge moving inward therefore changes at the upper band's
    bottom, while an edge moving outward changes at the lower band's top.  This
    decision is independent for the left and right edges.
    """

    def build(self, words: Sequence[OCRWord]) -> Polygon:
        if not words:
            raise ValueError("Cannot construct geometry from an empty word sequence")

        source_boxes = _alto_line_or_word_boxes(words)
        horizontal_bands = _horizontal_bands(source_boxes)
        connected_bands = _connect_horizontal_bands(horizontal_bands)
        boundary = _outer_orthogonal_boundary(connected_bands)
        if len(boundary) < 3:
            return _union_bounding_polygon(words)

        polygon = Polygon(points=tuple((*boundary, boundary[0])))
        if not all(
            _polygon_contains_box(polygon, word.bbox)
            for word in words
        ):
            return _union_bounding_polygon(words)
        return polygon


def _bounding_box_corners(box: BoundingBox) -> tuple[Point, ...]:
    return (
        (box.x, box.y),
        (box.x_max, box.y),
        (box.x_max, box.y_max),
        (box.x, box.y_max),
    )


def _union_bounding_polygon(words: Sequence[OCRWord]) -> Polygon:
    corners = _bounding_box_corners(
        UnionBoundingBoxGeometryBuilder().build(words)
    )
    return Polygon(points=tuple((*corners, corners[0])))


def _alto_line_or_word_boxes(
    words: Sequence[OCRWord],
) -> list[BoundingBox]:
    """Use ALTO line envelopes, retaining individual boxes without a line ID."""

    line_groups: dict[tuple[int | None, int], list[BoundingBox]] = {}
    individual_boxes: list[BoundingBox] = []
    for word in words:
        if word.line_index is None:
            individual_boxes.append(word.bbox)
            continue
        line_key = (word.block_index, word.line_index)
        line_groups.setdefault(line_key, []).append(word.bbox)

    line_boxes = [
        _union_boxes(boxes)
        for boxes in line_groups.values()
    ]
    return [*line_boxes, *individual_boxes]


def _union_boxes(boxes: Sequence[BoundingBox]) -> BoundingBox:
    x_min = min(box.x for box in boxes)
    y_min = min(box.y for box in boxes)
    x_max = max(box.x_max for box in boxes)
    y_max = max(box.y_max for box in boxes)
    return BoundingBox(
        x=x_min,
        y=y_min,
        width=x_max - x_min,
        height=y_max - y_min,
    )


def _horizontal_bands(
    boxes: Sequence[BoundingBox],
) -> list[BoundingBox]:
    """Build order-independent horizontal envelopes from source boxes."""

    y_coordinates = sorted(
        {coordinate for box in boxes for coordinate in (box.y, box.y_max)}
    )
    bands: list[BoundingBox] = []
    for top, bottom in zip(y_coordinates, y_coordinates[1:]):
        if top == bottom:
            continue

        active_boxes = [
            box
            for box in boxes
            if box.y < bottom and box.y_max > top
        ]
        if not active_boxes:
            continue

        left = min(box.x for box in active_boxes)
        right = max(box.x_max for box in active_boxes)
        band = BoundingBox(
            x=left,
            y=top,
            width=right - left,
            height=bottom - top,
        )
        if (
            bands
            and bands[-1].x == band.x
            and bands[-1].x_max == band.x_max
            and bands[-1].y_max == band.y
        ):
            previous = bands[-1]
            bands[-1] = BoundingBox(
                x=previous.x,
                y=previous.y,
                width=previous.width,
                height=band.y_max - previous.y,
            )
        else:
            bands.append(band)
    return bands


def _connect_horizontal_bands(
    bands: Sequence[BoundingBox],
) -> list[BoundingBox]:
    """Bridge vertical gaps while keeping all occupied bands connected."""

    connected_boxes = list(bands)
    for upper_box, lower_box in zip(bands, bands[1:]):
        if upper_box.y_max >= lower_box.y:
            continue

        overlap_left = max(upper_box.x, lower_box.x)
        overlap_right = min(upper_box.x_max, lower_box.x_max)
        if overlap_left >= overlap_right:
            overlap_left = min(upper_box.x, lower_box.x)
            overlap_right = max(upper_box.x_max, lower_box.x_max)

        connected_boxes.append(
            BoundingBox(
                x=overlap_left,
                y=upper_box.y_max,
                width=overlap_right - overlap_left,
                height=lower_box.y - upper_box.y_max,
            )
        )
    return connected_boxes


def _outer_orthogonal_boundary(
    boxes: Sequence[BoundingBox],
) -> list[Point]:
    """Trace one outer boundary, or return no boundary for multiple components."""

    x_coordinates = sorted(
        {coordinate for box in boxes for coordinate in (box.x, box.x_max)}
    )
    y_coordinates = sorted(
        {coordinate for box in boxes for coordinate in (box.y, box.y_max)}
    )
    if len(x_coordinates) < 2 or len(y_coordinates) < 2:
        return []

    occupied: set[tuple[int, int]] = set()
    for y_index in range(len(y_coordinates) - 1):
        y_midpoint = (
            y_coordinates[y_index] + y_coordinates[y_index + 1]
        ) / 2
        for x_index in range(len(x_coordinates) - 1):
            x_midpoint = (
                x_coordinates[x_index] + x_coordinates[x_index + 1]
            ) / 2
            if any(
                box.x <= x_midpoint < box.x_max
                and box.y <= y_midpoint < box.y_max
                for box in boxes
            ):
                occupied.add((x_index, y_index))

    directed_edges: set[tuple[Point, Point]] = set()
    for x_index, y_index in occupied:
        left = x_coordinates[x_index]
        right = x_coordinates[x_index + 1]
        top = y_coordinates[y_index]
        bottom = y_coordinates[y_index + 1]

        if (x_index, y_index - 1) not in occupied:
            directed_edges.add(((left, top), (right, top)))
        if (x_index + 1, y_index) not in occupied:
            directed_edges.add(((right, top), (right, bottom)))
        if (x_index, y_index + 1) not in occupied:
            directed_edges.add(((right, bottom), (left, bottom)))
        if (x_index - 1, y_index) not in occupied:
            directed_edges.add(((left, bottom), (left, top)))

    loops = _trace_boundary_loops(directed_edges)
    if len(loops) != 1:
        return []
    return _rotate_to_top_left(
        _remove_collinear_vertices(
            loops[0]
        )
    )


def _trace_boundary_loops(
    directed_edges: set[tuple[Point, Point]],
) -> list[list[Point]]:
    outgoing_edges: dict[Point, list[Point]] = defaultdict(list)
    for start, end in directed_edges:
        outgoing_edges[start].append(end)
    for destinations in outgoing_edges.values():
        destinations.sort()

    remaining_edges = set(directed_edges)
    loops: list[list[Point]] = []
    while remaining_edges:
        start, first_end = min(remaining_edges)
        remaining_edges.remove((start, first_end))
        loop = [start]
        current = first_end

        while current != start:
            loop.append(current)
            next_points = [
                point
                for point in outgoing_edges[current]
                if (current, point) in remaining_edges
            ]
            if not next_points:
                break
            next_point = next_points[0]
            remaining_edges.remove((current, next_point))
            current = next_point

        if current == start:
            loops.append(loop)
    return loops


def _remove_collinear_vertices(points: Sequence[Point]) -> list[Point]:
    if len(points) < 3:
        return list(points)

    simplified: list[Point] = []
    for index, point in enumerate(points):
        previous = points[index - 1]
        following = points[(index + 1) % len(points)]
        if (
            previous[0] == point[0] == following[0]
            or previous[1] == point[1] == following[1]
        ):
            continue
        simplified.append(point)
    return simplified


def _rotate_to_top_left(points: Sequence[Point]) -> list[Point]:
    if not points:
        return []
    start_index = min(
        range(len(points)),
        key=lambda index: (points[index][1], points[index][0]),
    )
    return [*points[start_index:], *points[:start_index]]


def _polygon_contains_box(polygon: Polygon, box: BoundingBox) -> bool:
    return all(
        _polygon_contains_point(polygon, point)
        for point in _bounding_box_corners(box)
    )


def _polygon_contains_point(polygon: Polygon, point: Point) -> bool:
    x, y = point
    inside = False
    for start, end in zip(polygon.points, polygon.points[1:]):
        start_x, start_y = start
        end_x, end_y = end
        cross_product = (
            (x - start_x) * (end_y - start_y)
            - (y - start_y) * (end_x - start_x)
        )
        if (
            cross_product == 0
            and min(start_x, end_x) <= x <= max(start_x, end_x)
            and min(start_y, end_y) <= y <= max(start_y, end_y)
        ):
            return True

        if (start_y > y) != (end_y > y):
            intersection_x = (
                start_x
                + (y - start_y)
                * (end_x - start_x)
                / (end_y - start_y)
            )
            if x < intersection_x:
                inside = not inside
    return inside

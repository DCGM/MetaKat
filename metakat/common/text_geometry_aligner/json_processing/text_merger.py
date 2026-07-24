from __future__ import annotations

import copy
from typing import Any, MutableMapping, MutableSequence

from ..models import JSONPath
from ..utils import _format_json_path


class JSONTextMerger:
    """Write extracted ALTO text to suffix-derived JSON paths."""

    def __init__(
        self,
        geometry_suffix: str = "_bbox",
        overwrite_existing_text: bool = False,
    ):
        if not geometry_suffix:
            raise ValueError("geometry_suffix must not be empty")
        self.geometry_suffix = geometry_suffix
        self.overwrite_existing_text = overwrite_existing_text
        self._source_data: Any = None
        self._has_source_data = False

    def create_output(self, input_data: Any) -> Any:
        self._source_data = input_data
        self._has_source_data = True
        output_data = copy.deepcopy(input_data)
        self._prepare_missing_destinations(output_data)
        return output_data

    def set_text(
        self,
        output_data: Any,
        text_path: JSONPath,
        text: str | None,
    ) -> None:
        if not text_path:
            raise ValueError("Cannot replace the JSON root value")
        if not self._has_source_data:
            raise RuntimeError("create_output must be called before set_text")
        if (
            not self.overwrite_existing_text
            and _path_exists(self._source_data, text_path)
        ):
            return
        _set_or_create_path(
            output_data,
            text_path,
            text,
            overwrite=True,
        )

    def _prepare_missing_destinations(self, output_data: Any) -> None:
        def visit(node: Any) -> None:
            if isinstance(node, dict):
                for key, value in tuple(node.items()):
                    if (
                        isinstance(key, str)
                        and key.endswith(self.geometry_suffix)
                    ):
                        destination_key = key[: -len(self.geometry_suffix)]
                        if destination_key and destination_key not in node:
                            node[destination_key] = _empty_text_shape(value)
                    else:
                        visit(value)
            elif isinstance(node, list):
                for value in node:
                    visit(value)

        visit(output_data)


def _set_or_create_path(
    root: Any,
    path: JSONPath,
    value: Any,
    *,
    overwrite: bool,
) -> None:
    node = root
    for index, component in enumerate(path):
        is_last = index == len(path) - 1
        next_component = None if is_last else path[index + 1]

        if isinstance(component, str):
            if not isinstance(node, MutableMapping):
                raise TypeError(
                    f"Cannot create {_format_json_path(path)} through "
                    f"{type(node).__name__}"
                )
            if is_last:
                if component not in node or overwrite:
                    node[component] = value
                return
            if component not in node:
                node[component] = (
                    [] if isinstance(next_component, int) else {}
                )
            child = node[component]
        elif isinstance(component, int):
            if not isinstance(node, MutableSequence):
                raise TypeError(
                    f"Cannot create {_format_json_path(path)} through "
                    f"{type(node).__name__}"
                )
            while len(node) <= component:
                node.append(None)
            if is_last:
                if node[component] is None or overwrite:
                    node[component] = value
                return
            if node[component] is None:
                node[component] = (
                    [] if isinstance(next_component, int) else {}
                )
            child = node[component]
        else:
            raise TypeError(
                f"Invalid path component at {_format_json_path(path)}"
            )

        expected_type = (
            MutableSequence
            if isinstance(next_component, int)
            else MutableMapping
        )
        if not isinstance(child, expected_type):
            raise TypeError(
                f"Incompatible container while creating "
                f"{_format_json_path(path)}: found "
                f"{type(child).__name__}"
            )
        node = child


def _empty_text_shape(geometry_value: Any) -> Any:
    if isinstance(geometry_value, list):
        if _is_polygon_points(geometry_value):
            return None
        return [
            _empty_text_shape(item)
            for item in geometry_value
        ]
    return None


def _is_polygon_points(value: list[Any]) -> bool:
    return (
        len(value) >= 4
        and all(
            isinstance(point, (list, tuple))
            and len(point) == 2
            for point in value
        )
    )


def _path_exists(root: Any, path: JSONPath) -> bool:
    node = root
    for component in path:
        if isinstance(component, str):
            if not isinstance(node, dict) or component not in node:
                return False
            node = node[component]
        elif isinstance(component, int):
            if (
                not isinstance(node, list)
                or not 0 <= component < len(node)
            ):
                return False
            node = node[component]
        else:
            return False
    return True

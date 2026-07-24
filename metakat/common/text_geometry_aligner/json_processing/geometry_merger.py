from __future__ import annotations

import copy
import logging
from typing import Any, MutableMapping, MutableSequence, Sequence

from ..models import JSONPath, JSONScalarValue
from ..utils import _format_json_path, _resolve_json_path

logger = logging.getLogger(__name__)


class JSONGeometryMerger:
    """Merge aligned geometry into paths retained during text extraction."""

    def __init__(
        self,
        geometry_suffix: str = "_bbox",
        preserve_existing_geometry: bool = False,
    ):
        if not geometry_suffix:
            raise ValueError("geometry_suffix must not be empty")
        self.geometry_suffix = geometry_suffix
        self.preserve_existing_geometry = preserve_existing_geometry

    def create_output(
        self,
        input_data: Any,
        values: Sequence[JSONScalarValue],
    ) -> Any:
        """Copy the input and prepare parallel containers for list geometries."""

        output_data = copy.deepcopy(input_data)
        self._prepare_list_geometry_containers(output_data, values)
        return output_data

    def set_aligned_text(
        self,
        output_data: Any,
        value: JSONScalarValue,
        text: str,
    ) -> None:
        """Replace one scalar at its retained source path."""

        self._set_path_value(output_data, value.path, text)

    def set_geometry(
        self,
        output_data: Any,
        value: JSONScalarValue,
        geometry: Any | None,
    ) -> None:
        """Write one geometry at its retained parallel path."""

        target_path = value.geometry_path or self._default_geometry_path(
            value.path
        )
        parent = _resolve_json_path(output_data, target_path[:-1])
        component = target_path[-1]

        if isinstance(component, int) and isinstance(parent, MutableSequence):
            parent[component] = geometry
            return
        if not (
            isinstance(component, str)
            and isinstance(parent, MutableMapping)
        ):
            raise TypeError(
                f"Cannot set geometry at {_format_json_path(target_path)} on "
                f"{type(parent).__name__}"
            )
        if self.preserve_existing_geometry and component in parent:
            return
        if component in parent:
            logger.debug(
                "Overwriting existing geometry key %s",
                _format_json_path(target_path),
            )
        parent[component] = geometry

    def _prepare_list_geometry_containers(
        self,
        output_data: Any,
        values: Sequence[JSONScalarValue],
    ) -> None:
        prepared_paths: set[JSONPath] = set()
        for value in values:
            geometry_path = value.geometry_path
            if geometry_path is None or not isinstance(geometry_path[-1], int):
                continue

            trailing_indexes = 0
            for component in reversed(geometry_path):
                if not isinstance(component, int):
                    break
                trailing_indexes += 1

            geometry_container_path = geometry_path[:-trailing_indexes]
            if geometry_container_path in prepared_paths:
                continue
            prepared_paths.add(geometry_container_path)

            source_container_path = value.path[:-trailing_indexes]
            source_container = _resolve_json_path(
                output_data,
                source_container_path,
            )
            if not isinstance(source_container, list):
                raise TypeError(
                    f"Expected list at {_format_json_path(source_container_path)}, "
                    f"found {type(source_container).__name__}"
                )

            geometry_parent = _resolve_json_path(
                output_data,
                geometry_container_path[:-1],
            )
            geometry_key = geometry_container_path[-1]
            if not (
                isinstance(geometry_parent, MutableMapping)
                and isinstance(geometry_key, str)
            ):
                raise TypeError(
                    "List geometry must be owned by a JSON object at "
                    f"{_format_json_path(geometry_container_path)}"
                )
            geometry_parent[geometry_key] = self._empty_geometry_shape(
                source_container
            )

    @classmethod
    def _empty_geometry_shape(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [cls._empty_geometry_shape(item) for item in value]
        return None

    def _default_geometry_path(self, value_path: JSONPath) -> JSONPath:
        if not value_path or not isinstance(value_path[-1], str):
            raise ValueError(
                f"No geometry path is available for {_format_json_path(value_path)}"
            )
        return value_path[:-1] + (
            f"{value_path[-1]}{self.geometry_suffix}",
        )

    @staticmethod
    def _set_path_value(root: Any, path: JSONPath, value: Any) -> None:
        if not path:
            raise ValueError("Cannot replace the JSON root value")

        parent = _resolve_json_path(root, path[:-1])
        component = path[-1]
        if isinstance(component, str) and isinstance(parent, MutableMapping):
            parent[component] = value
            return
        if isinstance(component, int) and isinstance(parent, MutableSequence):
            parent[component] = value
            return
        raise TypeError(
            f"Cannot set {_format_json_path(path)} on "
            f"{type(parent).__name__}"
        )

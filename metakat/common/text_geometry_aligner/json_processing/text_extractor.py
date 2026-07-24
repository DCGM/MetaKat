from __future__ import annotations

import logging
from typing import Any, Sequence

from ..models import JSONPath, JSONScalarValue
from ..utils import _format_json_path

logger = logging.getLogger(__name__)


class JSONTextExtractor:
    """Find JSON text values with a representable output-geometry location."""

    def __init__(
        self,
        geometry_suffix: str = "_bbox",
        preserve_existing_geometry: bool = False,
        ignored_geometry_suffixes: Sequence[str] = ("_bbox", "_polygon"),
    ):
        if not geometry_suffix:
            raise ValueError("geometry_suffix must not be empty")
        if any(not suffix for suffix in ignored_geometry_suffixes):
            raise ValueError("ignored geometry suffixes must not be empty")
        self.geometry_suffix = geometry_suffix
        self.preserve_existing_geometry = preserve_existing_geometry
        self.ignored_geometry_suffixes = frozenset(
            (*ignored_geometry_suffixes, geometry_suffix)
        )

    def extract(self, data: Any) -> tuple[JSONScalarValue, ...]:
        values: list[JSONScalarValue] = []

        def append_value(
            value: str | int | float,
            path: JSONPath,
            key: str,
            geometry_path: JSONPath,
        ) -> None:
            values.append(
                JSONScalarValue(
                    value_id=len(values),
                    path=path,
                    key=key,
                    original_value=value,
                    text=str(value),
                    normalized_text="",
                    geometry_path=geometry_path,
                )
            )

        def visit(
            node: Any,
            path: JSONPath,
            list_geometry_path: JSONPath | None = None,
            list_key: str | None = None,
        ) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if not isinstance(key, str):
                        logger.debug("Skipping non-string JSON object key at %s: %r", path, key)
                        continue

                    child_path = path + (key,)
                    if any(
                        key.endswith(suffix)
                        for suffix in self.ignored_geometry_suffixes
                    ):
                        # Existing geometry is not an alignable text value.
                        continue

                    geometry_key = f"{key}{self.geometry_suffix}"
                    if (
                        self.preserve_existing_geometry
                        and geometry_key in node
                    ):
                        logger.debug("Preserving existing geometry at %s", _format_json_path(child_path))
                        continue

                    if _is_alignable_scalar(value):
                        append_value(
                            value=value,
                            path=child_path,
                            key=key,
                            geometry_path=path + (geometry_key,),
                        )
                    elif isinstance(value, list):
                        visit(
                            value,
                            child_path,
                            list_geometry_path=path + (geometry_key,),
                            list_key=key,
                        )
                    elif isinstance(value, dict):
                        visit(value, child_path)

            elif isinstance(node, list):
                for index, value in enumerate(node):
                    child_path = path + (index,)
                    child_geometry_path = (
                        None
                        if list_geometry_path is None
                        else list_geometry_path + (index,)
                    )
                    if isinstance(value, list):
                        visit(
                            value,
                            child_path,
                            list_geometry_path=child_geometry_path,
                            list_key=list_key,
                        )
                    elif isinstance(value, dict):
                        visit(value, child_path)
                    elif _is_alignable_scalar(value):
                        if child_geometry_path is None or list_key is None:
                            logger.debug(
                                "Skipping scalar list element at %s; its list has "
                                "no owning dictionary key",
                                _format_json_path(child_path),
                            )
                            continue
                        append_value(
                            value=value,
                            path=child_path,
                            key=list_key,
                            geometry_path=child_geometry_path,
                        )

        visit(data, ())
        return tuple(values)


def _is_alignable_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float)) and not isinstance(value, bool)

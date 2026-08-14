from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


Config = dict[str, Any]

_PATH_SUFFIXES = ("_path", "_dir")
_PATH_LIST_SUFFIXES = ("_paths", "_dirs")


def load_config_file(path: str | Path) -> Config:
    """Load a JSON or YAML configuration whose root is an object."""
    config_path = Path(path)
    suffix = config_path.suffix.lower()
    with config_path.open("r", encoding="utf-8") as source:
        if suffix == ".json":
            value = json.load(source)
        elif suffix in {".yaml", ".yml"}:
            value = yaml.safe_load(source)
        else:
            raise ValueError(
                "Engine configuration must use .json, .yaml, or .yml: "
                f"{config_path}"
            )
    config = _require_mapping(value, f"Configuration root in {config_path}")
    _validate_json_compatible(config, str(config_path))
    return copy.deepcopy(config)


def merge_configs(base: Mapping[str, Any], override: Mapping[str, Any]) -> Config:
    """Deep-merge mappings while replacing scalar values and whole lists."""
    result = copy.deepcopy(dict(base))
    for key, override_value in override.items():
        if key not in result:
            result[key] = copy.deepcopy(override_value)
            continue
        base_value = result[key]
        base_is_mapping = isinstance(base_value, Mapping)
        override_is_mapping = isinstance(override_value, Mapping)
        if base_is_mapping and override_is_mapping:
            result[key] = merge_configs(base_value, override_value)
        elif base_is_mapping != override_is_mapping:
            raise ValueError(
                f"Cannot override mapping and non-mapping values at {key!r}"
            )
        else:
            result[key] = copy.deepcopy(override_value)
    return result


def parse_assignment_value(value: str) -> Any:
    """Parse a command-line value as JSON, falling back to a string."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def apply_assignments(
    config: Mapping[str, Any],
    assignments: Sequence[Sequence[str]] | None,
) -> Config:
    """Apply ordered ``PATH VALUE`` assignments to a copied config."""
    result = copy.deepcopy(dict(config))
    for assignment in assignments or ():
        if len(assignment) != 2:
            raise ValueError("Each --set requires PATH and VALUE")
        raw_path, raw_value = assignment
        keys = raw_path.split(":")
        if any(not key for key in keys):
            raise ValueError(f"Invalid empty component in --set path: {raw_path!r}")
        cursor: Config = result
        for key in keys[:-1]:
            existing = cursor.get(key)
            if existing is None:
                child: Config = {}
                cursor[key] = child
                cursor = child
            elif isinstance(existing, dict):
                cursor = existing
            elif isinstance(existing, Mapping):
                child = copy.deepcopy(dict(existing))
                cursor[key] = child
                cursor = child
            else:
                raise ValueError(
                    f"Cannot descend through non-mapping --set value at {key!r}"
                )
        value = parse_assignment_value(raw_value)
        leaf = keys[-1]
        if leaf in cursor and (
            isinstance(cursor[leaf], Mapping) != isinstance(value, Mapping)
        ):
            raise ValueError(
                f"Cannot override mapping and non-mapping values at {raw_path!r}"
            )
        cursor[leaf] = value
    return result


def resolve_config_paths(
    config: Mapping[str, Any],
    base_dir: str | Path,
    *,
    require_within_base: bool = False,
) -> Config:
    """Resolve recognized path fields against one central base directory."""
    resolved_base = Path(base_dir).resolve()

    def resolve_path(value: Any, location: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise ValueError(f"{location} must be a non-empty path string")
        path = Path(value)
        if not path.is_absolute():
            path = resolved_base / path
        resolved = path.resolve()
        if require_within_base and not resolved.is_relative_to(resolved_base):
            raise ValueError(
                f"Resolved path escapes engine directory at {location}: {value}"
            )
        return str(resolved)

    def walk(value: Any, location: str, key: str | None = None) -> Any:
        if key is not None and key.endswith(_PATH_LIST_SUFFIXES):
            if value is None:
                return None
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise ValueError(f"{location} must be a list of paths")
            return [
                resolve_path(item, f"{location}[{index}]")
                for index, item in enumerate(value)
            ]
        if key is not None and key.endswith(_PATH_SUFFIXES):
            return resolve_path(value, location)
        if isinstance(value, Mapping):
            return {
                child_key: walk(
                    child_value,
                    f"{location}.{child_key}" if location else child_key,
                    child_key,
                )
                for child_key, child_value in value.items()
            }
        if isinstance(value, list):
            return [
                walk(item, f"{location}[{index}]")
                for index, item in enumerate(value)
            ]
        return copy.deepcopy(value)

    return walk(config, "")


def prepare_engine_config(
    base_config: Mapping[str, Any],
    *,
    override: Mapping[str, Any] | None = None,
    assignments: Sequence[Sequence[str]] | None = None,
    base_dir: str | Path,
    require_within_base: bool = False,
) -> Config:
    """Merge, assign, and finally resolve a complete pipeline config."""
    merged = merge_configs(base_config, override or {})
    assigned = apply_assignments(merged, assignments)
    return resolve_config_paths(
        assigned,
        base_dir,
        require_within_base=require_within_base,
    )


def require_config_mapping(value: Any, location: str) -> Config:
    """Validate and copy an engine configuration mapping."""
    config = _require_mapping(value, location)
    return copy.deepcopy(config)


def require_engine_name(config: Mapping[str, Any], location: str) -> str:
    name = config.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{location} must contain a non-empty string 'name'")
    return name


def _require_mapping(value: Any, location: str) -> Config:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{location} keys must be strings")
    return dict(value)


def _validate_json_compatible(value: Any, location: str) -> None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"Configuration keys must be strings at {location}")
        for key, child in value.items():
            _validate_json_compatible(child, f"{location}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_json_compatible(child, f"{location}[{index}]")
        return
    raise ValueError(
        f"Configuration value at {location} is not JSON-compatible: "
        f"{type(value).__name__}"
    )

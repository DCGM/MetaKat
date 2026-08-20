"""Deferred engine references.

An engine implementation module is imported only once that engine is actually
selected by the pipeline configuration, so a pipeline that never uses an engine
does not pay for the optional dependencies that engine pulls in.

Each registry entry records the third-party modules its implementation needs.
That lets the pipeline verify availability before any page is processed and
report a missing dependency together with both ways of supplying it, the extra
that provides it or a manual install at the pinned versions, instead of
surfacing a bare ModuleNotFoundError from an arbitrary import line.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from typing import Any, Optional

from metakat.engine_config import (
    Config,
    require_config_mapping,
    require_engine_name,
)


@dataclass(frozen=True)
class EngineEntry:
    """Names an engine implementation without importing it.

    Attributes:
        module: Module holding the engine class.
        attribute: Name of the engine class within that module.
        requires: Top-level third-party modules the implementation imports.
        extra: Package extra that provides `requires`, used in error messages.
    """

    module: str
    attribute: str
    requires: tuple[str, ...] = ()
    extra: Optional[str] = None

    def missing_requirements(self) -> tuple[str, ...]:
        """Return the required modules that are not importable."""
        return tuple(
            module_name
            for module_name in self.requires
            if find_spec(module_name) is None
        )

    def import_class(self) -> type:
        """Import the implementation module and return the engine class."""
        return getattr(import_module(self.module), self.attribute)


def _select(
    entries: Mapping[str, EngineEntry],
    config: Mapping[str, Any],
    location: str,
    label: str,
) -> tuple[EngineEntry, Config, str]:
    engine_config = require_config_mapping(config, location)
    name = require_engine_name(engine_config, location)
    entry = entries.get(name)
    if entry is None:
        raise ValueError(f"Unknown {label.lower()}: {name}")
    return entry, engine_config, name


def _require_available(entry: EngineEntry, label: str, name: str) -> None:
    missing = entry.missing_requirements()
    if not missing:
        return

    modules = ", ".join(f"'{module_name}'" for module_name in missing)
    dependency = "dependency" if len(missing) == 1 else "dependencies"
    message = f"{label} '{name}' requires the missing optional {dependency} {modules}."
    if entry.extra is not None:
        message += f' Install with: pip install -e ".[{entry.extra}]".'
    package = "package" if len(missing) == 1 else "packages"
    version = "version" if len(missing) == 1 else "versions"
    message += (
        f" The {package} can also be installed directly, in which case use the"
        f" {version} pinned in pyproject.toml: this check only verifies that a"
        " module is importable, not that its version is the one MetaKat expects."
    )
    raise ImportError(message)


def check_engine_requirements(
    entries: Mapping[str, EngineEntry],
    config: Mapping[str, Any],
    location: str,
    label: str,
) -> None:
    """Raise when the configured engine's optional dependencies are missing.

    Validates the engine name the same way loading does, so an unusable
    pipeline fails before any processing starts.
    """
    entry, _, name = _select(entries, config, location, label)
    _require_available(entry, label, name)


def resolve_engine_class(
    entries: Mapping[str, EngineEntry],
    config: Mapping[str, Any],
    location: str,
    label: str,
) -> tuple[type, Config]:
    """Return the configured engine class and its validated configuration."""
    entry, engine_config, name = _select(entries, config, location, label)
    _require_available(entry, label, name)
    return entry.import_class(), engine_config

from collections.abc import Mapping
from typing import Any

from metakat.common.engines.registry import (
    EngineEntry,
    check_engine_requirements,
    resolve_engine_class,
)

from metakat.page_type.engines.core.page_type_core_engine import PageTypeCoreEngine

_LOCATION = "Page type core config"
_LABEL = "Page type core engine"

page_type_core_engines = {
    'page_type_core_engine_vit': EngineEntry(
        module='metakat.page_type.engines.core.page_type_core_engine_vit',
        attribute='PageTypeCoreEngineViT',
        requires=('torch', 'transformers'),
        extra='inference',
    ),
}


def load_page_type_core_engine(config: Mapping[str, Any]) -> PageTypeCoreEngine:
    core_engine_class, engine_config = resolve_engine_class(
        page_type_core_engines,
        config,
        _LOCATION,
        _LABEL,
    )
    return core_engine_class(engine_config)


def check_page_type_core_engine(config: Mapping[str, Any]) -> None:
    """Verify the configured page-type core engine is available to load."""
    check_engine_requirements(
        page_type_core_engines,
        config,
        _LOCATION,
        _LABEL,
    )

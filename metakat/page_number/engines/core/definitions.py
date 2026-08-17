from collections.abc import Mapping
from typing import Any

from metakat.common.engines.registry import (
    EngineEntry,
    check_engine_requirements,
    resolve_engine_class,
)

from metakat.page_number.engines.core.page_number_core_engine import (
    PageNumberCoreEngine,
)

_LOCATION = "Page-number core config"
_LABEL = "Page number core engine"

page_number_core_engines = {
    "page_number_core_engine_yolo": EngineEntry(
        module="metakat.page_number.engines.core.page_number_core_engine_yolo",
        attribute="PageNumberCoreEngineYOLO",
        requires=("ultralytics",),
        extra="yolo",
    ),
}


def load_page_number_core_engine(
    config: Mapping[str, Any],
) -> PageNumberCoreEngine:
    core_engine_class, engine_config = resolve_engine_class(
        page_number_core_engines,
        config,
        _LOCATION,
        _LABEL,
    )
    return core_engine_class(engine_config)


def check_page_number_core_engine(config: Mapping[str, Any]) -> None:
    """Verify the configured page-number core engine is available to load."""
    check_engine_requirements(
        page_number_core_engines,
        config,
        _LOCATION,
        _LABEL,
    )

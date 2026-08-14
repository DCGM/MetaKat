from collections.abc import Mapping
from typing import Any

from metakat.engine_config import require_config_mapping, require_engine_name

from metakat.page_number.engines.core.page_number_core_engine import (
    PageNumberCoreEngine,
)
from metakat.page_number.engines.core.page_number_core_engine_yolo import (
    PageNumberCoreEngineYOLO,
)


page_number_core_engines = {
    "page_number_core_engine_yolo": PageNumberCoreEngineYOLO,
}


def load_page_number_core_engine(
    config: Mapping[str, Any],
) -> PageNumberCoreEngine:
    engine_config = require_config_mapping(config, "Page-number core config")
    name = require_engine_name(engine_config, "Page-number core config")
    core_engine_class = page_number_core_engines.get(name)
    if core_engine_class is None:
        raise ValueError(f"Unknown page number core engine: {name}")
    return core_engine_class(engine_config)

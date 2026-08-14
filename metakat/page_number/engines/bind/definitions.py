from collections.abc import Mapping
from typing import Any

from metakat.engine_config import require_config_mapping, require_engine_name

from metakat.page_number.engines.bind.page_number_bind_engine import (
    PageNumberBindEngine,
)
from metakat.page_number.engines.bind.page_number_bind_engine_base import (
    PageNumberBindEngineBase,
)


page_number_bind_engines = {
    "page_number_bind_engine_base": PageNumberBindEngineBase,
}


def load_page_number_bind_engine(
    bind_config: Mapping[str, Any],
    core_config: Mapping[str, Any],
) -> PageNumberBindEngine:
    engine_config = require_config_mapping(bind_config, "Page-number bind config")
    name = require_engine_name(engine_config, "Page-number bind config")
    bind_engine_class = page_number_bind_engines.get(name)
    if bind_engine_class is None:
        raise ValueError(f"Unknown page number bind engine: {name}")
    return bind_engine_class(engine_config, core_config)

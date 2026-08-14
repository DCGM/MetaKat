from collections.abc import Mapping
from typing import Any

from metakat.engine_config import require_config_mapping, require_engine_name

from metakat.page_type.engines.bind.page_type_bind_engine import PageTypeBindEngine
from metakat.page_type.engines.bind.page_type_bind_engine_base import PageTypeBindEngineBase

page_type_bind_engines = {
    'page_type_bind_engine_base': PageTypeBindEngineBase,
}

def load_page_type_bind_engine(
    bind_config: Mapping[str, Any],
    core_config: Mapping[str, Any],
) -> PageTypeBindEngine:
    engine_config = require_config_mapping(bind_config, "Page-type bind config")
    name = require_engine_name(engine_config, "Page-type bind config")
    bind_engine_class = page_type_bind_engines.get(name)
    if bind_engine_class is None:
        raise ValueError(f"Unknown page type bind engine: {name}")

    return bind_engine_class(engine_config, core_config)

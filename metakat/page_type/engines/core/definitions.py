from collections.abc import Mapping
from typing import Any

from metakat.engine_config import require_config_mapping, require_engine_name

from metakat.page_type.engines.core.page_type_core_engine import PageTypeCoreEngine
from metakat.page_type.engines.core.page_type_core_engine_vit import PageTypeCoreEngineViT

page_type_core_engines = {
    'page_type_core_engine_vit': PageTypeCoreEngineViT,
}


def load_page_type_core_engine(config: Mapping[str, Any]) -> PageTypeCoreEngine:
    engine_config = require_config_mapping(config, "Page-type core config")
    name = require_engine_name(engine_config, "Page-type core config")
    core_engine_class = page_type_core_engines.get(name)
    if core_engine_class is None:
        raise ValueError(f"Unknown page type core engine: {name}")

    return core_engine_class(engine_config)

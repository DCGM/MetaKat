import json
import os

from page_type.engines.core.page_type_core_engine import PageTypeCoreEngine
from page_type.engines.core.page_type_core_engine_vit import PageTypeCoreEngineViT

page_type_core_engines = {
    'page_type_core_engine_vit': PageTypeCoreEngineViT,
}


def load_page_type_core_engine(core_engine_dir: str) -> PageTypeCoreEngine:
    config_path = os.path.join(core_engine_dir, "metakat_engine_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Page type engine config not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    core_engine_class = page_type_core_engines.get(config['name'])
    if core_engine_class is None:
        raise ValueError(f"Unknown page type core engine: {config['name']}")

    return core_engine_class(core_engine_dir)
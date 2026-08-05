import os
import json

from metakat.page_type.engines.bind.page_type_bind_engine import PageTypeBindEngine
from metakat.page_type.engines.bind.page_type_bind_engine_base import PageTypeBindEngineBase

page_type_bind_engines = {
    'page_type_bind_engine_base': PageTypeBindEngineBase,
}

def load_page_type_bind_engine(bind_engine_dir: str, core_engine_dir: str) -> PageTypeBindEngine:
    config_path = os.path.join(bind_engine_dir, "metakat_engine_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Page type bind engine config not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    bind_engine_class = page_type_bind_engines.get(config['name'])
    if bind_engine_class is None:
        raise ValueError(f"Unknown page type bind engine: {config['name']}")

    return bind_engine_class(bind_engine_dir, core_engine_dir)
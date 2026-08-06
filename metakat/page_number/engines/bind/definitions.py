import json
import os

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
    bind_engine_dir: str,
    core_engine_dir: str,
) -> PageNumberBindEngine:
    config_path = os.path.join(
        bind_engine_dir,
        "metakat_engine_config.json",
    )
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Page number bind engine config not found at {config_path}"
        )
    with open(config_path, "r", encoding="utf-8") as source:
        config = json.load(source)

    bind_engine_class = page_number_bind_engines.get(config["name"])
    if bind_engine_class is None:
        raise ValueError(
            f"Unknown page number bind engine: {config['name']}"
        )
    return bind_engine_class(bind_engine_dir, core_engine_dir)

import json
import os

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
    core_engine_dir: str,
) -> PageNumberCoreEngine:
    config_path = os.path.join(
        core_engine_dir,
        "metakat_engine_config.json",
    )
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Page number core engine config not found at {config_path}"
        )
    with open(config_path, "r", encoding="utf-8") as source:
        config = json.load(source)

    core_engine_class = page_number_core_engines.get(config["name"])
    if core_engine_class is None:
        raise ValueError(
            f"Unknown page number core engine: {config['name']}"
        )
    return core_engine_class(core_engine_dir)

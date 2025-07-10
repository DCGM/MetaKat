import os
import json

from biblio.engines.core.biblio_core_engine import BiblioCoreEngine
from biblio.engines.core.biblio_core_engine_yolo import BiblioCoreEngineYOLO

biblio_core_engines = {
    'biblio_core_engine_yolo': BiblioCoreEngineYOLO
}

def load_biblio_core_engine(core_engine_dir: str) -> BiblioCoreEngine:
    config_path = os.path.join(core_engine_dir, "metakat_engine_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Biblio core engine config not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    core_engine_class = biblio_core_engines.get(config['name'])
    if core_engine_class is None:
        raise ValueError(f"Unknown biblio core engine: {config['name']}")

    return core_engine_class(core_engine_dir)
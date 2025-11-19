import os
import json

from metakat.biblio.engines.bind.bilbio_bind_engine import BiblioBindEngine
from metakat.biblio.engines.bind.biblio_bind_engine_base import BiblioBindEngineBase

biblio_bind_engines = {
    'biblio_bind_engine_base': BiblioBindEngineBase,
}

def load_biblio_bind_engine(bind_engine_dir: str, core_engine_dir: str) -> BiblioBindEngine:
    config_path = os.path.join(bind_engine_dir, "metakat_engine_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Biblio bind engine config not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    bind_engine_class = biblio_bind_engines.get(config['name'])
    if bind_engine_class is None:
        raise ValueError(f"Unknown biblio bind engine: {config['name']}")

    return bind_engine_class(bind_engine_dir, core_engine_dir)
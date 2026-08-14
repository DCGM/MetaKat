from collections.abc import Mapping
from typing import Any

from metakat.engine_config import require_config_mapping, require_engine_name

from metakat.biblio.engines.core.biblio_core_engine import BiblioCoreEngine
from metakat.biblio.engines.core.biblio_core_engine_yolo import BiblioCoreEngineYOLO

biblio_core_engines = {
    'biblio_core_engine_yolo': BiblioCoreEngineYOLO
}

def load_biblio_core_engine(config: Mapping[str, Any]) -> BiblioCoreEngine:
    engine_config = require_config_mapping(config, "Biblio core config")
    name = require_engine_name(engine_config, "Biblio core config")
    core_engine_class = biblio_core_engines.get(name)
    if core_engine_class is None:
        raise ValueError(f"Unknown biblio core engine: {name}")

    return core_engine_class(engine_config)

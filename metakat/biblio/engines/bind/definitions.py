from collections.abc import Mapping
from typing import Any

from metakat.engine_config import require_config_mapping, require_engine_name

from metakat.biblio.engines.bind.bilbio_bind_engine import BiblioBindEngine
from metakat.biblio.engines.bind.biblio_bind_engine_base import BiblioBindEngineBase

biblio_bind_engines = {
    'biblio_bind_engine_base': BiblioBindEngineBase,
}

def load_biblio_bind_engine(
    bind_config: Mapping[str, Any],
    core_config: Mapping[str, Any],
) -> BiblioBindEngine:
    engine_config = require_config_mapping(bind_config, "Biblio bind config")
    name = require_engine_name(engine_config, "Biblio bind config")
    bind_engine_class = biblio_bind_engines.get(name)
    if bind_engine_class is None:
        raise ValueError(f"Unknown biblio bind engine: {name}")

    return bind_engine_class(engine_config, core_config)

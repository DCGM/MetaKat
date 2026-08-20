from collections.abc import Mapping
from typing import Any

from metakat.common.engines.registry import (
    EngineEntry,
    check_engine_requirements,
    resolve_engine_class,
)

from metakat.biblio.engines.core.biblio_core_engine import BiblioCoreEngine

_LOCATION = "Biblio core config"
_LABEL = "Biblio core engine"

biblio_core_engines = {
    'biblio_core_engine_yolo': EngineEntry(
        module='metakat.biblio.engines.core.biblio_core_engine_yolo',
        attribute='BiblioCoreEngineYOLO',
        requires=('ultralytics', 'text_geometry_aligner'),
        extra='inference',
    ),
}

def load_biblio_core_engine(config: Mapping[str, Any]) -> BiblioCoreEngine:
    core_engine_class, engine_config = resolve_engine_class(
        biblio_core_engines,
        config,
        _LOCATION,
        _LABEL,
    )
    return core_engine_class(engine_config)


def check_biblio_core_engine(config: Mapping[str, Any]) -> None:
    """Verify the configured biblio core engine is available to load."""
    check_engine_requirements(
        biblio_core_engines,
        config,
        _LOCATION,
        _LABEL,
    )

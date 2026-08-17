from collections.abc import Mapping
from typing import Any

from metakat.engine_config import require_engine_name

from metakat.common.engines.registry import (
    EngineEntry,
    check_engine_requirements,
    resolve_engine_class,
)

from metakat.chapter.engines.core.chapter_core_engine import ChapterCoreEngine

_LOCATION = "Chapter core config"
_LABEL = "Chapter core engine"

# The pipeline engine itself needs nothing beyond the base dependencies; its
# requirements come from whichever stage implementations the config selects.
chapter_core_engines = {
    'chapter_core_engine_pipeline': EngineEntry(
        module='metakat.chapter.engines.core.chapter_core_engine_pipeline',
        attribute='ChapterPipelineCoreEngine',
    ),
}

def load_chapter_core_engine(config: Mapping[str, Any]) -> ChapterCoreEngine:
    core_engine_class, engine_config = resolve_engine_class(
        chapter_core_engines,
        config,
        _LOCATION,
        _LABEL,
    )
    return core_engine_class(engine_config)


def check_chapter_core_engine(config: Mapping[str, Any]) -> None:
    """Verify the configured chapter core engine and its stages are available."""
    check_engine_requirements(
        chapter_core_engines,
        config,
        _LOCATION,
        _LABEL,
    )

    if require_engine_name(config, _LOCATION) != 'chapter_core_engine_pipeline':
        return

    # Importing the pipeline module is cheap now that stages are deferred, and
    # it is the only place that knows which stage implementations exist.
    from metakat.chapter.engines.core.chapter_core_engine_pipeline import (
        check_chapter_pipeline_stages,
    )

    check_chapter_pipeline_stages(config)

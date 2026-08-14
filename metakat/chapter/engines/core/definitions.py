from collections.abc import Mapping
from typing import Any

from metakat.engine_config import require_config_mapping, require_engine_name

from metakat.chapter.engines.core.chapter_core_engine import ChapterCoreEngine
from metakat.chapter.engines.core.chapter_core_engine_pipeline import ChapterPipelineCoreEngine

chapter_core_engines = {
    'chapter_core_engine_pipeline': ChapterPipelineCoreEngine,
}

def load_chapter_core_engine(config: Mapping[str, Any]) -> ChapterCoreEngine:
    engine_config = require_config_mapping(config, "Chapter core config")
    name = require_engine_name(engine_config, "Chapter core config")
    core_engine_class = chapter_core_engines.get(name)
    if core_engine_class is None:
        raise ValueError(f"Unknown chapter core engine: {name}")

    return core_engine_class(engine_config)

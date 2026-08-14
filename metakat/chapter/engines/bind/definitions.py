from collections.abc import Mapping
from typing import Any

from metakat.engine_config import require_config_mapping, require_engine_name

from metakat.chapter.engines.bind.chapter_bind_engine import ChapterBindEngine
from metakat.chapter.engines.bind.chapter_bind_engine_base import ChapterBindEngineBase

chapter_bind_engines = {
    'chapter_bind_engine_base': ChapterBindEngineBase,
}

def load_chapter_bind_engine(
    bind_config: Mapping[str, Any],
    core_config: Mapping[str, Any],
) -> ChapterBindEngine:
    engine_config = require_config_mapping(bind_config, "Chapter bind config")
    name = require_engine_name(engine_config, "Chapter bind config")
    bind_engine_class = chapter_bind_engines.get(name)
    if bind_engine_class is None:
        raise ValueError(f"Unknown chapter bind engine: {name}")

    return bind_engine_class(engine_config, core_config)

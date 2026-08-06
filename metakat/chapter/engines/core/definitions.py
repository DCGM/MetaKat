import json
from pathlib import Path
from typing import Any

from metakat.chapter.engines.core.chapter_core_engine import ChapterCoreEngine
from metakat.chapter.engines.core.chapter_core_engine_pipeline import ChapterPipelineCoreEngine

chapter_core_engines = {
    'chapter_core_engine_pipeline': ChapterPipelineCoreEngine,
}

def load_chapter_core_engine(core_engine_dir: str) -> ChapterCoreEngine:
    engine_dir = Path(core_engine_dir)
    config_path = engine_dir / "metakat_engine_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Chapter core engine config not found at {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as source:
        config: Any = json.load(source)
    if not isinstance(config, dict):
        raise ValueError(
            f"Chapter core engine config must be a JSON object: {config_path}"
        )
    name = config.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            "Chapter core engine config must contain a non-empty string "
            f"'name': {config_path}"
        )

    core_engine_class = chapter_core_engines.get(name)
    if core_engine_class is None:
        raise ValueError(f"Unknown chapter core engine: {name}")

    return core_engine_class(engine_dir)

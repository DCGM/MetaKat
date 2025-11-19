import os
import json

from metakat.chapter.engines.core.chapter_core_engine import ChapterCoreEngine
from metakat.chapter.engines.core.chapter_core_engine_yolo import ChapterCoreEngineYOLO

chapter_core_engines = {
    'chapter_core_engine_yolo': ChapterCoreEngineYOLO
}

def load_chapter_core_engine(core_engine_dir: str) -> ChapterCoreEngine:
    config_path = os.path.join(core_engine_dir, "metakat_engine_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Chapter core engine config not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    core_engine_class = chapter_core_engines.get(config['name'])
    if core_engine_class is None:
        raise ValueError(f"Unknown chapter core engine: {config['name']}")

    return core_engine_class(core_engine_dir)
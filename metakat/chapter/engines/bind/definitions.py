import os
import json

from chapter.engines.bind.chapter_bind_engine import ChapterBindEngine
from chapter.engines.bind.chapter_bind_engine_base import ChapterBindEngineBase

chapter_bind_engines = {
    'chapter_bind_engine_base': ChapterBindEngineBase,
}

def load_chapter_bind_engine(bind_engine_dir: str, core_engine_dir: str) -> ChapterBindEngine:
    config_path = os.path.join(bind_engine_dir, "metakat_engine_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Chapter bind engine config not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    bind_engine_class = chapter_bind_engines.get(config['name'])
    if bind_engine_class is None:
        raise ValueError(f"Unknown chapter bind engine: {config['name']}")

    return bind_engine_class(bind_engine_dir, core_engine_dir)
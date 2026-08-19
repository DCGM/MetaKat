import json
from pathlib import Path
from typing import Union

from metakat.io_parsers.parser_mods import parse_mods
from metakat.schemas.base_objects import ProarcIO


def parse_proarc_json(data: dict) -> ProarcIO:
    """Validate a ProArc packageInfo.json dict into ProarcIO, with each object's
    parsed fields (see parser_mods.parse_mods) filled in from its MODS metadata.
    """
    package = ProarcIO.model_validate(data)
    for obj in package.objects:
        parsed = parse_mods(obj.metadata)
        for key, value in parsed.items():
            setattr(obj, key, value)
    return package


def parse_proarc_json_file(path: Union[str, Path]) -> ProarcIO:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_proarc_json(data)

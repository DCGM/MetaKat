from __future__ import annotations

import argparse
import json
import logging
from typing import Any

from .models import JSONPath


def _resolve_json_path(root: Any, path: JSONPath) -> Any:
    node = root
    for component in path:
        node = node[component]
    return node

def _format_json_path(path: JSONPath) -> str:
    if not path:
        return "$"

    output = "$"
    for component in path:
        if isinstance(component, int):
            output += f"[{component}]"
        elif component.isidentifier():
            output += f".{component}"
        else:
            output += f"[{json.dumps(component, ensure_ascii=False)}]"
    return output

def _parse_logging_level(value: str) -> int:
    if value.isdigit():
        return int(value)
    level = getattr(logging, value.upper(), None)
    if not isinstance(level, int):
        raise argparse.ArgumentTypeError(f"Invalid logging level: {value}")
    return level

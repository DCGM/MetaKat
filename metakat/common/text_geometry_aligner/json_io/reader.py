from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class JSONReader:
    """Read a UTF-8 JSON document from disk."""

    def read(self, input_path: str | os.PathLike[str]) -> Any:
        path = Path(input_path)
        with path.open("r", encoding="utf-8") as input_stream:
            return json.load(input_stream)

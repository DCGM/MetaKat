from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class JSONWriter:
    """Atomically write a UTF-8 JSON document to disk."""

    def write(
        self,
        data: Any,
        output_path: str | os.PathLike[str],
    ) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.tmp")
        try:
            with temporary_path.open("w", encoding="utf-8") as output_stream:
                json.dump(data, output_stream, ensure_ascii=False, indent=2)
                output_stream.write("\n")
            os.replace(temporary_path, path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

"""Fixtures shared across the whole test tree.

This file sits at the repository root rather than inside the package, so it is
an ancestor of every metakat/**/tests/ directory and is never installed as part
of the distribution.

Helpers that only one test module needs stay in that module. What lives here is
what several directories were each defining a private copy of.
"""

import json
from uuid import uuid4

import pytest
from PIL import Image

from metakat.schemas.base_objects import MetakatPage


@pytest.fixture
def metakat_page():
    """A minimal page, as the bind engines expect to receive one."""
    return MetakatPage(
        id=uuid4(),
        batch_id=uuid4(),
        batch_index=0,
    )


@pytest.fixture
def page_image():
    """Write a blank page image; several engines only need it to exist."""

    def _write(path, size=(100, 200)):
        Image.new("RGB", size, color="white").save(path)
        return path

    return _write


@pytest.fixture
def yolo_alignment_page():
    """A YOLO-derived page whose first PageNumber region carries matched text.

    The second region is geometry only, so binding has both a matched and an
    unmatched detection to choose between. Function scoped, so a test may mutate
    the regions it is handed.
    """
    from text_geometry_aligner import (
        AlignmentPage,
        AlignmentRegion,
        AlignmentWord,
        BoundingBox,
        InputFormat,
    )

    bbox = BoundingBox(10, 20, 30, 10)
    return AlignmentPage(
        page_key="scan.001",
        input_format=InputFormat.YOLO,
        regions=[
            AlignmentRegion(
                region_id=0,
                label="PageNumber",
                input_geometry=bbox,
                category_id=0,
                input_geometry_confidence=0.91,
                alto_text="12",
                words=[
                    AlignmentWord(
                        word_index=0,
                        text="12",
                        bbox=bbox,
                    )
                ],
            ),
            AlignmentRegion(
                region_id=1,
                label="PageNumber",
                input_geometry=BoundingBox(50, 50, 10, 10),
                category_id=0,
                input_geometry_confidence=0.99,
            ),
        ],
    )


@pytest.fixture
def write_engine_config():
    """Write the metakat_engine_config.json an engine directory is loaded from."""

    def _write(directory, data):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "metakat_engine_config.json").write_text(
            json.dumps(data),
            encoding="utf-8",
        )

    return _write


# Stands in for `vllm serve MODEL --port N ...`: answers /health and Chat
# Completions with the reply in $FAKE_VLLM_REPLY, records its arguments in
# $FAKE_VLLM_ARGS, and with $FAKE_VLLM_FAIL set dies while starting.
_FAKE_VLLM = """\
#!{python}
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

args = sys.argv[1:]
with open(os.environ["FAKE_VLLM_ARGS"], "w") as out:
    json.dump(args, out)
if os.environ.get("FAKE_VLLM_FAIL"):
    print("CUDA out of memory", flush=True)
    sys.exit(3)
port = int(args[args.index("--port") + 1])

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send({{}})

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self._send({{"id": "1", "object": "chat.completion", "created": 0, "model": "m",
                    "choices": [{{"index": 0, "finish_reason": "stop",
                                 "message": {{"role": "assistant", "content": os.environ["FAKE_VLLM_REPLY"]}}}}]}})

HTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""


@pytest.fixture
def fake_vllm(tmp_path, monkeypatch):
    """Make VLLM_EXECUTABLE a fake vLLM server; returns the file its arguments go to."""
    import sys

    executable = tmp_path / "vllm"
    executable.write_text(_FAKE_VLLM.format(python=sys.executable), encoding="utf-8")
    executable.chmod(0o755)
    args_path = tmp_path / "vllm_args.json"
    monkeypatch.setenv("FAKE_VLLM_ARGS", str(args_path))
    monkeypatch.setenv("FAKE_VLLM_REPLY", "{}")
    monkeypatch.setenv("VLLM_EXECUTABLE", str(executable))
    monkeypatch.setenv("VLLM_STARTUP_TIMEOUT", "30")
    return args_path


@pytest.fixture
def read_engine_config():
    """Read back what write_engine_config produced."""

    def _read(directory):
        return json.loads(
            (directory / "metakat_engine_config.json").read_text(encoding="utf-8")
        )

    return _read

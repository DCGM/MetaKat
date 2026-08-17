"""Fixtures for the page-number core engine tests.

A conftest module is imported before the test modules beside it, so priming
sys.modules with the ultralytics stub here covers every test in this directory
rather than each module repeating it.
"""

import importlib
import sys
import types
from unittest import mock

import pytest
from text_geometry_aligner import (
    AlignmentPage,
    AlignmentRegion,
    BoundingBox as AlignmentBoundingBox,
    InputFormat,
)


ultralytics_stub = types.ModuleType("ultralytics")
ultralytics_stub.__version__ = "test"
ultralytics_stub.YOLO = mock.Mock()
with mock.patch.dict(sys.modules, {"ultralytics": ultralytics_stub}):
    importlib.import_module("metakat.common.engines.engine_yolo")


@pytest.fixture
def alignment_page():
    """Build a page of page-number candidates as (text, confidence, y) triples."""

    def _build(
        candidates,
        *,
        alto_width=800,
        alto_height=1000,
        label="cislo strany",
        x=100,
        width=50,
        height=20,
    ):
        return AlignmentPage(
            page_key="page-1",
            input_format=InputFormat.YOLO,
            alto_width=alto_width,
            alto_height=alto_height,
            regions=[
                AlignmentRegion(
                    region_id=index,
                    label=label,
                    category_id=2,
                    input_geometry=AlignmentBoundingBox(
                        x,
                        y,
                        width,
                        height,
                    ),
                    input_geometry_confidence=confidence,
                    alto_text=text,
                    words=[],
                )
                for index, (text, confidence, y) in enumerate(candidates)
            ],
        )

    return _build

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


@pytest.fixture
def read_engine_config():
    """Read back what write_engine_config produced."""

    def _read(directory):
        return json.loads(
            (directory / "metakat_engine_config.json").read_text(encoding="utf-8")
        )

    return _read

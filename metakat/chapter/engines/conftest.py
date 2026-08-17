"""Fixtures shared by the chapter engine tests.

This sits at metakat/chapter/engines/ rather than inside any one tests directory
because pytest only collects conftest modules from *ancestor* directories: the
stage tests under core/chapter_*/tests/ and the binding tests under bind/tests/
are siblings, so a conftest inside either would be invisible to the other.

Builders used by a single test module stay in that module.
"""

from pathlib import Path

import pytest
from text_geometry_aligner import (
    AlignmentDocument,
    AlignmentMode,
    AlignmentPage,
    AlignmentRegion,
    AlignmentWord,
    BoundingBox as AlignmentBoundingBox,
    InputFormat,
)

from metakat.chapter.engines.core.chapter_page_number_parsers import (
    ArabicRomanChapterPageNumberParser,
)
from metakat.common.models import BoundingBox, DetectionEvidence
from metakat.page_number.engines.core.page_number_parsers import (
    DecoratedPageNumberParser,
)


@pytest.fixture
def evidence():
    """A detection with text, confidence and geometry on a named page."""

    def _build(
        text,
        page_key,
        x=10,
        y=10,
        confidence=0.9,
        width=100,
        height=20,
    ):
        return DetectionEvidence(
            text=text,
            confidence=confidence,
            bbox=BoundingBox(x, y, width, height),
            page_key=page_key,
        )

    return _build


@pytest.fixture
def toc_page_number_fields(evidence):
    """The page_number keyword arguments a TocBase or ChapterBase expects."""

    def _build(
        text,
        page_key,
        x=10,
        y=10,
        confidence=0.9,
        width=100,
        height=20,
    ):
        return {
            "page_number": ArabicRomanChapterPageNumberParser.create(
                evidence(
                    text,
                    page_key,
                    x=x,
                    y=y,
                    confidence=confidence,
                    width=width,
                    height=height,
                )
            ),
        }

    return _build


@pytest.fixture
def physical_page_number():
    """A resolved physical page number, as the page-number engine produces."""

    def _build(text, page_key, confidence=0.9):
        return DecoratedPageNumberParser.create(
            page_key=page_key,
            text=text,
            confidence=confidence,
            bbox=BoundingBox(10, 10, 100, 20),
        )

    return _build


@pytest.fixture
def region():
    """An aligner region carrying both geometry and matched text."""

    def _build(
        region_id,
        label,
        text,
        x,
        y,
        confidence=0.9,
        width=100,
        height=20,
    ):
        bbox = AlignmentBoundingBox(x, y, width, height)
        return AlignmentRegion(
            region_id=region_id,
            label=label,
            input_geometry=bbox,
            input_geometry_confidence=confidence,
            alto_text=text,
            words=[AlignmentWord(region_id, text, bbox)],
        )

    return _build


@pytest.fixture
def alignment_page():
    """A YOLO-format page wrapping the given regions."""

    def _build(page_key, regions):
        return AlignmentPage(
            page_key=page_key,
            input_format=InputFormat.YOLO,
            regions=regions,
        )

    return _build


@pytest.fixture
def fake_alignment_engine():
    """A stand-in aligner that returns canned pages and counts its calls."""

    class _FakeAlignmentEngine:
        def __init__(self, pages):
            self.pages = pages
            self.call_count = 0

        def process(self, images, alto_files):
            self.call_count += 1
            requested = {Path(image).stem for image in images}
            return AlignmentDocument(
                alignment_mode=AlignmentMode.GEOMETRY,
                pages=[page for page in self.pages if page.page_key in requested],
            )

    return _FakeAlignmentEngine

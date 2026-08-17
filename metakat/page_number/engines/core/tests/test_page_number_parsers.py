import pytest

from metakat.common.models import BoundingBox, DetectionEvidence
from metakat.page_number.engines.core.models import PageNumberNumeralSystem
from metakat.page_number.engines.core.page_number_parsers import (
    DecoratedPageNumberParser,
)


@pytest.mark.parametrize(
    "source,expected",
    (
        ("12", "12"),
        ("  0012  ", "0012"),
        ("- 12 -", "12"),
        ("— 12 —", "12"),
        ("[12].", "12"),
        ("• str. 12 •", "12"),
        ("1 2 3", "123"),
        ("１２３", "123"),
        ("١٢٣", "123"),
        ("XIV", "XIV"),
        ("— xiv —", "xiv"),
        ("str. IV.", "IV"),
        ("[Ⅻ]", "XII"),
    ),
)
def test_page_number_parser_tolerates_printed_decoration(source, expected):
    assert DecoratedPageNumberParser.parse(source) == expected


@pytest.mark.parametrize(
    "source",
    (
        "",
        "---",
        "page",
        "12-13",
        "1 of 10",
        "1/10",
        "IIII",
        "IC",
        "I / X",
    ),
)
def test_page_number_parser_rejects_missing_or_ambiguous_numbers(source):
    assert not DecoratedPageNumberParser.parse(source)


def test_page_number_evidence_retains_ocr_and_exposes_normalized_text():
    evidence = DecoratedPageNumberParser.create(
        page_key="page",
        text="— xiv —",
        confidence=0.8,
        bbox=BoundingBox(1, 2, 3, 4),
    )

    assert evidence.text == "— xiv —"
    assert isinstance(evidence, DetectionEvidence)
    assert isinstance(evidence.bbox, BoundingBox)
    assert evidence.normalized_text() == "xiv"
    assert evidence.normalized_text(case="uppercase") == "XIV"
    assert evidence.output_text() == "xiv"
    assert evidence.value == 14
    assert evidence.numeral_system == PageNumberNumeralSystem.ROMAN

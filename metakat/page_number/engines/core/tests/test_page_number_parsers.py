import pytest

from metakat.common.models import AltoRefs, BoundingBox, DetectionEvidence
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


def _region_with_words(*words):
    from text_geometry_aligner import AlignmentRegion, AlignmentWord, BoundingBox as AlignmentBoundingBox

    box = AlignmentBoundingBox(10, 20, 30, 40)
    return AlignmentRegion(
        region_id=0,
        label="cislo strany",
        category_id=0,
        input_geometry=box,
        input_geometry_confidence=0.9,
        alto_text="- 12 -",
        words=[
            AlignmentWord(word_index=index, text=text, bbox=box,
                          alto_block_id=block, alto_line_id=line, alto_word_id=word)
            for index, (text, block, line, word) in enumerate(words)
        ],
    )


def test_page_number_evidence_carries_the_alto_ids_of_its_words():
    evidence = DecoratedPageNumberParser.parse_region(
        page_key="page-1",
        region=_region_with_words(
            ("-", "TB1", "TL1", "S1"),
            ("12", "TB1", "TL1", "S2"),
            ("-", "TB1", "TL2", None),
        ),
    )

    # Each level once per ID, in reading order; a word without an ID adds none.
    assert evidence.alto == AltoRefs(blocks=("TB1",), lines=("TL1", "TL2"), words=("S1", "S2"))


def test_page_number_evidence_has_no_alto_ids_where_the_alto_has_none():
    evidence = DecoratedPageNumberParser.parse_region(
        page_key="page-1",
        region=_region_with_words(("12", None, None, None)),
    )

    assert evidence.alto.is_empty()

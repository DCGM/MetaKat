import logging
from pathlib import Path
from unittest import mock

import pytest

from metakat.chapter.engines.core.chapter_alignment.engine_fuzzy import (
    ChapterAlignmentEngineFuzzy,
    _toc_monotonicity_score,
    title_similarity,
)
from metakat.chapter.engines.core.chapter_page_analysis.models import (
    DestinationChapterEvidence,
)
from metakat.chapter.engines.core.chapter_page_number_parsers import (
    ArabicRomanChapterPageNumberParser,
)
from metakat.chapter.engines.core.models import (
    ChapterBase,
    ChapterPageInput,
    ChapterPageNumberKind,
    TocBase,
    TocResult,
)
from metakat.page_number.engines.core.models import PageNumberNumeralSystem

ALIGNMENT_LOGGER = (
    "metakat.chapter.engines.core.chapter_alignment.engine_fuzzy"
)
TITLE_SIMILARITY = f"{ALIGNMENT_LOGGER}.title_similarity"


def _pages(count):
    return tuple(
        ChapterPageInput(
            f"page-{position}",
            position,
            Path(f"page-{position}.jpg"),
            Path(f"page-{position}.xml"),
        )
        for position in range(count)
    )


def _log_output(caplog):
    return "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.name.startswith(ALIGNMENT_LOGGER)
    )


@pytest.fixture
def fuzzy_engine(tmp_path, write_engine_config, read_engine_config):
    """Build the fuzzy alignment engine from a configuration directory."""

    def _build(**config):
        write_engine_config(
            tmp_path,
            {"name": "chapter_alignment_engine_fuzzy", **config},
        )
        return ChapterAlignmentEngineFuzzy(read_engine_config(tmp_path))

    return _build


@pytest.fixture
def page_numbers(physical_page_number):
    """Physical page numbers keyed by the page position that carries them."""

    def _build(number_by_position):
        return tuple(
            physical_page_number(text, f"page-{position}")
            for position, text in number_by_position.items()
        )

    return _build


@pytest.mark.parametrize("invalid", (-1, 1.5, "2", True))
def test_maximum_destination_offset_must_be_non_negative_integer(
    fuzzy_engine,
    invalid,
):
    with pytest.raises(ValueError, match="must be a non-negative integer"):
        fuzzy_engine(
            maximum_destination_page_position_offset_from_expected=invalid
        )


@pytest.mark.parametrize(
    "invalid",
    (True, False, 0, 1, "enabled", None, [], {}),
)
def test_toc_monotonic_order_constraints_must_be_supported_mode(
    fuzzy_engine,
    invalid,
):
    with pytest.raises(
        ValueError,
        match="toc_monotonic_order_constraints must be one of",
    ):
        fuzzy_engine(toc_monotonic_order_constraints=invalid)


@pytest.mark.parametrize("invalid", (None, 0, 1, "yes", [], {}))
def test_use_anchors_must_be_boolean(fuzzy_engine, invalid):
    with pytest.raises(ValueError, match="use_anchors must be a boolean"):
        fuzzy_engine(use_anchors=invalid)


@pytest.mark.parametrize("invalid", (None, 0, 1, "yes", [], {}))
def test_infer_chapter_ends_must_be_boolean(fuzzy_engine, invalid):
    with pytest.raises(
        ValueError,
        match="infer_chapter_ends must be a boolean",
    ):
        fuzzy_engine(infer_chapter_ends=invalid)


@pytest.mark.parametrize("invalid", (-0.1, 1.1, "0.9", True, []))
def test_end_inference_score_must_be_null_or_within_unit_interval(
    fuzzy_engine,
    invalid,
):
    with pytest.raises(
        ValueError,
        match=(
            "minimum_toc_monotonicity_score_for_end_inference must be null "
            "or a number within"
        ),
    ):
        fuzzy_engine(
            minimum_toc_monotonicity_score_for_end_inference=invalid
        )


@pytest.mark.parametrize(
    "invalid",
    (0, -1, float("inf"), float("nan"), True, "60"),
)
def test_solver_time_limit_must_be_null_or_positive_finite_number(
    fuzzy_engine,
    invalid,
):
    with pytest.raises(
        ValueError,
        match=(
            "solver_time_limit_seconds must be null or a positive finite number"
        ),
    ):
        fuzzy_engine(solver_time_limit_seconds=invalid)


def test_solver_defaults_enable_anchors_without_a_time_limit(fuzzy_engine):
    engine = fuzzy_engine()

    assert engine.use_anchors
    assert engine.solver_time_limit_seconds is None


@pytest.mark.parametrize("invalid", (-0.1, 1.1, "0.9", True, None))
def test_minimum_toc_monotonicity_ratio_must_be_within_unit_interval(
    fuzzy_engine,
    invalid,
):
    with pytest.raises(
        ValueError,
        match="minimum_toc_number_monotonicity_ratio",
    ):
        fuzzy_engine(minimum_toc_number_monotonicity_ratio=invalid)


def test_auto_mode_uses_inclusive_monotonicity_threshold(fuzzy_engine):
    engine = fuzzy_engine()

    assert engine._resolve_toc_monotonic_order_constraints(0.9)
    assert not engine._resolve_toc_monotonic_order_constraints(0.899)
    assert engine._resolve_toc_monotonic_order_constraints(None)


def test_monotonicity_score_uses_longest_nondecreasing_subsequences(
    toc_page_number_fields,
):
    numbers = tuple(
        toc_page_number_fields(str(value), "toc")["page_number"]
        for value in (1, 2, 3, 4, 5, 7, 6, 8, 9, 10)
    )

    assert _toc_monotonicity_score(numbers) == 0.9


def test_monotonicity_score_separates_numeral_systems(toc_page_number_fields):
    numbers = tuple(
        toc_page_number_fields(text, "toc")["page_number"]
        for text in ("X", "XX", "1", "2")
    )

    assert _toc_monotonicity_score(numbers) == 1.0


def test_auto_mode_disables_constraints_for_nonmonotonic_toc(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    engine = fuzzy_engine()

    result = engine.process(
        pages=_pages(10),
        reference_toc=TocBase(
            (
                ChapterBase(
                    "toc",
                    evidence("Late", "toc"),
                    **toc_page_number_fields("20", "toc"),
                ),
                ChapterBase(
                    "toc",
                    evidence("Early", "toc"),
                    **toc_page_number_fields("10", "toc"),
                ),
            )
        ),
        destination_chapters=(
            DestinationChapterEvidence(evidence("Late", "page-8")),
            DestinationChapterEvidence(evidence("Early", "page-4")),
        ),
        destination_page_numbers=page_numbers({8: "20", 4: "10"}),
    )

    assert tuple(chapter.page_start_key for chapter in result.chapters) == (
        "page-8",
        "page-4",
    )


def test_yes_mode_forces_constraints_on_nonmonotonic_toc(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    engine = fuzzy_engine(toc_monotonic_order_constraints="yes")

    result = engine.process(
        pages=_pages(10),
        reference_toc=TocBase(
            (
                ChapterBase(
                    "toc",
                    evidence("Late", "toc"),
                    **toc_page_number_fields("20", "toc"),
                ),
                ChapterBase(
                    "toc",
                    evidence("Early", "toc"),
                    **toc_page_number_fields("10", "toc"),
                ),
            )
        ),
        destination_chapters=(
            DestinationChapterEvidence(evidence("Late", "page-8")),
            DestinationChapterEvidence(evidence("Early", "page-4")),
        ),
        destination_page_numbers=page_numbers({8: "20", 4: "10"}),
    )

    assert result.chapters[0].page_start_key == "page-8"
    assert result.chapters[1].page_start_key is None


def test_unordered_mode_retains_nonmonotonic_anchor_candidates(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    engine = fuzzy_engine(toc_monotonic_order_constraints="no")

    result = engine.process(
        pages=_pages(10),
        reference_toc=TocBase(
            (
                ChapterBase(
                    "toc",
                    evidence("Late", "toc"),
                    **toc_page_number_fields("10", "toc"),
                ),
                ChapterBase(
                    "toc",
                    evidence("Early", "toc"),
                    **toc_page_number_fields("20", "toc"),
                ),
            )
        ),
        destination_chapters=(
            DestinationChapterEvidence(evidence("Late", "page-8")),
            DestinationChapterEvidence(evidence("Early", "page-4")),
        ),
        destination_page_numbers=page_numbers({8: "10", 4: "20"}),
    )

    assert tuple(chapter.page_start_key for chapter in result.chapters) == (
        "page-8",
        "page-4",
    )


def test_unordered_exact_match_ignores_anchor_bounds(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    engine = fuzzy_engine(toc_monotonic_order_constraints="no")

    result = engine.process(
        pages=_pages(10),
        reference_toc=TocBase(
            (
                ChapterBase(
                    "toc",
                    evidence("Anchor", "toc"),
                    **toc_page_number_fields("10", "toc"),
                ),
                ChapterBase(
                    "toc",
                    evidence("Expected", "toc"),
                    **toc_page_number_fields("20", "toc"),
                ),
            )
        ),
        destination_chapters=(
            DestinationChapterEvidence(evidence("Anchor", "page-8")),
            DestinationChapterEvidence(evidence("Different", "page-4")),
        ),
        destination_page_numbers=page_numbers({8: "10", 4: "20"}),
    )

    assert result.chapters[0].page_start_key == "page-8"
    assert result.chapters[1].page_start_key == "page-4"
    assert result.chapters[1].title_destination_page is None


def test_unordered_many_to_one_does_not_require_title_order(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    engine = fuzzy_engine(toc_monotonic_order_constraints="no")

    result = engine.process(
        pages=_pages(7),
        reference_toc=TocBase(
            tuple(
                ChapterBase(
                    "toc",
                    evidence(title, "toc"),
                    **toc_page_number_fields("10", "toc"),
                )
                for title in ("A", "B")
            )
        ),
        destination_chapters=(
            DestinationChapterEvidence(evidence("B", "page-5", y=10)),
            DestinationChapterEvidence(evidence("A", "page-5", y=50)),
        ),
        destination_page_numbers=page_numbers({5: "10"}),
    )

    assert tuple(chapter.page_start_key for chapter in result.chapters) == (
        "page-5",
        "page-5",
    )
    assert tuple(
        chapter.title_destination_page.text for chapter in result.chapters
    ) == ("A", "B")


def test_unordered_many_to_one_canonicalizes_equal_title_pairings(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    engine = fuzzy_engine(toc_monotonic_order_constraints="no")

    result = engine.process(
        pages=_pages(7),
        reference_toc=TocBase(
            tuple(
                ChapterBase(
                    "toc",
                    evidence("Shared", "toc"),
                    **toc_page_number_fields("10", "toc"),
                )
                for _ in range(2)
            )
        ),
        destination_chapters=(
            DestinationChapterEvidence(evidence("Shared", "page-5", y=10)),
            DestinationChapterEvidence(evidence("Shared", "page-5", y=50)),
        ),
        destination_page_numbers=page_numbers({5: "10"}),
    )

    assert tuple(chapter.page_start_key for chapter in result.chapters) == (
        "page-5",
        "page-5",
    )
    assert tuple(
        chapter.title_destination_page.bbox.y for chapter in result.chapters
    ) == (10, 50)


def test_unordered_many_to_many_allows_decreasing_pages(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    engine = fuzzy_engine(toc_monotonic_order_constraints="no")

    result = engine.process(
        pages=_pages(10),
        reference_toc=TocBase(
            tuple(
                ChapterBase(
                    "toc",
                    evidence(title, "toc"),
                    **toc_page_number_fields("10", "toc"),
                )
                for title in ("A", "B")
            )
        ),
        destination_chapters=(
            DestinationChapterEvidence(evidence("A", "page-8")),
            DestinationChapterEvidence(evidence("B", "page-4")),
        ),
        destination_page_numbers=page_numbers({8: "10", 4: "10"}),
    )

    assert tuple(chapter.page_start_key for chapter in result.chapters) == (
        "page-8",
        "page-4",
    )


def test_unordered_title_fallback_is_global(fuzzy_engine, evidence):
    engine = fuzzy_engine(toc_monotonic_order_constraints="no")

    result = engine.process(
        pages=_pages(5),
        reference_toc=TocBase(
            (
                ChapterBase("toc", evidence("Meddle", "toc")),
                ChapterBase("toc", evidence("Middle", "toc")),
            )
        ),
        destination_chapters=(
            DestinationChapterEvidence(evidence("Middle", "page-3")),
        ),
        destination_page_numbers=(),
    )

    assert result.chapters[0].page_start_key is None
    assert result.chapters[1].page_start_key == "page-3"


def test_unordered_title_candidates_are_canonicalized(fuzzy_engine, evidence):
    engine = fuzzy_engine(toc_monotonic_order_constraints="no")

    result = engine.process(
        pages=_pages(5),
        reference_toc=TocBase(
            tuple(
                ChapterBase("toc", evidence("Shared", "toc"))
                for _ in range(2)
            )
        ),
        destination_chapters=(
            DestinationChapterEvidence(evidence("Shared", "page-2")),
            DestinationChapterEvidence(evidence("Shared", "page-3")),
        ),
        destination_page_numbers=(),
    )

    assert tuple(chapter.page_start_key for chapter in result.chapters) == (
        "page-2",
        "page-3",
    )


def test_unordered_range_end_ignores_following_anchor(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    engine = fuzzy_engine(toc_monotonic_order_constraints="no")

    result = engine.process(
        pages=_pages(14),
        reference_toc=TocBase(
            (
                ChapterBase(
                    "toc",
                    evidence("Range", "toc"),
                    **toc_page_number_fields("10-12", "toc"),
                ),
                ChapterBase(
                    "toc",
                    evidence("Earlier", "toc"),
                    **toc_page_number_fields("20", "toc"),
                ),
            )
        ),
        destination_chapters=(
            DestinationChapterEvidence(evidence("Range", "page-8")),
            DestinationChapterEvidence(evidence("Earlier", "page-4")),
        ),
        destination_page_numbers=page_numbers({8: "10", 12: "12", 4: "20"}),
    )

    assert result.chapters[0].page_start_key == "page-8"
    assert result.chapters[0].page_end_key == "page-12"


def test_chapter_page_number_parser_public_parse_io():
    parsed = ArabicRomanChapterPageNumberParser.parse("xiv–xvi")

    assert parsed == (
        ChapterPageNumberKind.RANGE,
        (
            ("xiv", 14, PageNumberNumeralSystem.ROMAN),
            ("xvi", 16, PageNumberNumeralSystem.ROMAN),
        ),
    )
    assert ArabicRomanChapterPageNumberParser.parse("12/45") is None


def test_chapter_page_number_parser_normalizes_extraction_values(evidence):
    # The two case lists stay inline loops rather than becoming parameters:
    # they are a small part of a long contract test, and parametrizing the
    # whole test would repeat every other assertion for each case.
    def parse(text):
        return ArabicRomanChapterPageNumberParser.create(evidence(text, "toc"))

    roman = parse("XIV")
    assert roman.normalized_items[0][1] == 14
    assert roman.normalized_end() is None
    assert roman.normalized_items[0][2] == PageNumberNumeralSystem.ROMAN
    assert roman.kind == ChapterPageNumberKind.SINGLE
    assert roman.normalized_text() == "XIV"
    assert roman.normalized_text(case="lowercase") == "xiv"

    arabic = parse("str. 004")
    assert arabic.normalized_items[0][1] == 4
    assert arabic.normalized_end() is None
    assert arabic.normalized_items[0][2] == PageNumberNumeralSystem.ARABIC
    assert arabic.normalized_text() == "4"
    assert parse("１２３").normalized_text() == "123"
    assert parse("١٢٣").normalized_text() == "123"

    arabic_range = parse("str. 23–24")
    assert tuple(item[1] for item in arabic_range.normalized_items) == (23, 24)
    assert arabic_range.kind == ChapterPageNumberKind.RANGE
    assert arabic_range.normalized_text() == "23-24"
    roman_range = parse("xii—xiv")
    assert tuple(item[1] for item in roman_range.normalized_items) == (12, 14)
    assert roman_range.normalized_text() == "xii-xiv"
    assert roman_range.normalized_text(case="uppercase") == "XII-XIV"

    page_list = parse("23, 27, 31")
    assert page_list.normalized_items[0][1] == 23
    assert page_list.normalized_end() is None
    assert page_list.kind == ChapterPageNumberKind.LIST
    assert page_list.normalized_text() == "23,27,31"
    mixed_list = parse("XII, 14")
    assert mixed_list.normalized_text() == "XII,14"

    for incomplete_range in ("45-", "45–"):
        parsed = parse(incomplete_range)
        assert parsed.kind == ChapterPageNumberKind.SINGLE
        assert parsed.normalized_text() == "45"

    assert parse("not a page").normalized_text() is None
    descending_arabic = parse("24-23")
    assert descending_arabic.kind == ChapterPageNumberKind.SINGLE
    assert descending_arabic.normalized_text() == "24"
    assert descending_arabic.normalized_start() == "24"
    assert descending_arabic.normalized_end() is None
    descending_roman = parse("XIV-XII")
    assert descending_roman.kind == ChapterPageNumberKind.SINGLE
    assert descending_roman.normalized_text() == "XIV"
    assert parse("XII-14").normalized_text() is None
    for rejected in (
        "-45",
        "–45",
        "—45",
        "−45",
        "+45",
        "str. -45",
        "0",
        "3. 45",
        "12/45",
        "12 45",
        "23-24-25",
    ):
        number = parse(rejected)
        assert number.normalized_text() is None
        assert number.output_text() == rejected


def test_title_similarity_looks_for_the_toc_entry_inside_the_heading():
    # title_similarity(candidate, reference): the TOC entry is the reference,
    # the destination heading the uncertain reading tested against it. A
    # heading carrying more than the TOC entry - a chapter number set into the
    # heading, a running head, a subtitle - still matches, because the shorter
    # reference is looked for inside it.
    assert title_similarity("1. ÚVOD", "Úvod") == 1.0
    assert title_similarity("Kapitola 1: Úvod", "Úvod") == 1.0
    assert title_similarity("ÚVOD", "Úvod") == 1.0


def test_title_similarity_rejects_a_one_character_heading():
    # Regression test: the shorter side used to be located inside the longer
    # one whichever it was, so a one-character heading detection scored a
    # perfect match against any TOC entry containing that character and could
    # satisfy the similarity threshold outright.
    for fragment in ("U", "o", "d"):
        assert title_similarity(fragment, "Úvod do problematiky") < 0.2


def test_a_chapter_number_only_in_the_toc_entry_costs_the_match():
    # The reverse direction gets no substring licence, so a number the TOC
    # entry carries and the heading does not now counts against the match.
    # 0.667 is below the 0.7 default threshold: this pairing used to align and
    # no longer does. Extraction keeps part numbers in ChapterBase.part_number
    # rather than in the title, so this is the leaked-number case rather than
    # the normal one.
    assert title_similarity("ÚVOD", "1. Úvod") == pytest.approx(2 / 3, abs=1e-3)


def test_alignment_preserves_raw_toc_evidence(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    engine = fuzzy_engine()
    reference = TocBase(
        chapters=(
            ChapterBase(
                toc_page_key="page-0",
                title=evidence("Introduction", "page-0"),
                subtitle=evidence("Background", "page-0", y=30),
                **toc_page_number_fields("XIV", "page-0"),
            ),
        )
    )
    destinations = (
        DestinationChapterEvidence(
            title=evidence("INTRODUCTION", "page-15")
        ),
    )

    result = engine.process(
        pages=_pages(16),
        reference_toc=reference,
        destination_chapters=destinations,
        destination_page_numbers=page_numbers({15: "XIV"}),
    )

    assert isinstance(result, TocResult)
    chapter = result.chapters[0]
    assert chapter.subtitle.text == "Background"
    assert chapter.page_number.text == "XIV"
    assert chapter.page_start_key == "page-15"
    assert chapter.title_destination_page.page_key == "page-15"


def test_alignment_preserves_normalized_toc_page_number_evidence(
    fuzzy_engine,
    evidence,
    page_numbers,
):
    engine = fuzzy_engine()
    source_evidence = evidence("str. 004", "toc", x=500, confidence=0.83)

    result = engine.process(
        pages=_pages(6),
        destination_page_numbers=page_numbers({4: "4"}),
        reference_toc=TocBase(
            (
                ChapterBase(
                    toc_page_key="toc",
                    title=evidence("Chapter", "toc"),
                    page_number=ArabicRomanChapterPageNumberParser.create(
                        source_evidence
                    ),
                ),
            )
        ),
        destination_chapters=(
            DestinationChapterEvidence(evidence("Chapter", "page-4")),
        ),
    )

    normalized = result.chapters[0].page_number
    assert normalized.text == "str. 004"
    assert normalized.output_text() == "4"
    assert normalized.confidence == source_evidence.confidence
    assert normalized.bbox == source_evidence.bbox
    assert normalized.page_key == source_evidence.page_key


def test_duplicate_physical_number_requires_a_title_match(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    engine = fuzzy_engine()
    reference = TocBase(
        (
            ChapterBase(
                toc_page_key="toc",
                title=evidence("Second chapter", "toc"),
                **toc_page_number_fields("10", "toc"),
            ),
        )
    )
    destinations = (
        DestinationChapterEvidence(evidence("First chapter", "page-4")),
        DestinationChapterEvidence(evidence("SECOND CHAPTER", "page-8")),
    )

    result = engine.process(
        pages=_pages(10),
        destination_page_numbers=page_numbers({4: "10", 8: "10"}),
        reference_toc=reference,
        destination_chapters=destinations,
    )

    assert result.chapters[0].page_start_key == "page-8"


def test_one_to_many_multiple_title_matches_do_not_create_anchor(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
    caplog,
):
    engine = fuzzy_engine()
    reference = TocBase(
        (
            ChapterBase(
                toc_page_key="toc",
                title=evidence("Repeated chapter", "toc"),
                **toc_page_number_fields("10", "toc"),
            ),
        )
    )

    with caplog.at_level(logging.WARNING, logger=ALIGNMENT_LOGGER):
        result = engine.process(
            pages=_pages(10),
            destination_page_numbers=page_numbers({4: "10", 8: "10"}),
            reference_toc=reference,
            destination_chapters=(
                DestinationChapterEvidence(
                    evidence("Repeated chapter", "page-4")
                ),
                DestinationChapterEvidence(
                    evidence("REPEATED CHAPTER", "page-8")
                ),
            ),
        )

    assert (
        "Anchor support is enabled, but no consistent page-number "
        "anchors were selected" in _log_output(caplog)
    )
    assert result.chapters[0].page_start_key == "page-4"
    assert result.chapters[0].title_destination_page.page_key == "page-4"


def test_one_to_many_ideal_position_precedes_better_title_match(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    engine = fuzzy_engine()
    reference = TocBase(
        tuple(
            ChapterBase(
                toc_page_key="toc",
                title=evidence(title, "toc"),
                **toc_page_number_fields(number, "toc"),
            )
            for title, number in (
                ("First", "10"),
                ("Middle", "15"),
                ("Last", "20"),
            )
        )
    )

    result = engine.process(
        pages=_pages(32),
        destination_page_numbers=page_numbers(
            {20: "10", 25: "15", 26: "15", 30: "20"}
        ),
        reference_toc=reference,
        destination_chapters=(
            DestinationChapterEvidence(evidence("First", "page-20")),
            DestinationChapterEvidence(evidence("Meddle", "page-25")),
            DestinationChapterEvidence(evidence("Middle", "page-26")),
            DestinationChapterEvidence(evidence("Last", "page-30")),
        ),
    )

    assert result.chapters[1].page_start_key == "page-25"
    assert result.chapters[1].title_destination_page.page_key == "page-25"


def test_exact_one_to_many_does_not_fall_through_to_off_number_title(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    engine = fuzzy_engine()

    result = engine.process(
        pages=_pages(5),
        destination_page_numbers=page_numbers({1: "10", 3: "10"}),
        reference_toc=TocBase(
            (
                ChapterBase(
                    "toc",
                    evidence("Expected", "toc"),
                    **toc_page_number_fields("10", "toc"),
                ),
            )
        ),
        destination_chapters=(
            DestinationChapterEvidence(evidence("Expected", "page-2")),
        ),
    )

    assert result.chapters[0].page_start_key is None
    assert result.chapters[0].title_destination_page is None


def test_many_to_many_number_group_does_not_create_anchors(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
    caplog,
):
    engine = fuzzy_engine()
    reference = TocBase(
        (
            ChapterBase(
                "toc",
                evidence("First", "toc"),
                **toc_page_number_fields("10", "toc"),
            ),
            ChapterBase(
                "toc",
                evidence("Second", "toc"),
                **toc_page_number_fields("10", "toc"),
            ),
        )
    )

    with caplog.at_level(logging.WARNING, logger=ALIGNMENT_LOGGER):
        result = engine.process(
            pages=_pages(10),
            destination_page_numbers=page_numbers({4: "10", 8: "10"}),
            reference_toc=reference,
            destination_chapters=(
                DestinationChapterEvidence(evidence("First", "page-4")),
                DestinationChapterEvidence(evidence("Second", "page-8")),
            ),
        )

    log_output = _log_output(caplog)
    assert "Skipping many-to-many number-anchor group" in log_output
    assert (
        "Anchor support is enabled, but no consistent page-number "
        "anchors were selected" in log_output
    )
    assert result.chapters[0].page_start_key == "page-4"
    assert result.chapters[1].page_start_key == "page-8"


def test_many_to_many_non_anchor_resolution_is_global_and_monotonic(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    engine = fuzzy_engine()
    reference = TocBase(
        (
            ChapterBase(
                "toc",
                evidence("entry-0", "toc"),
                **toc_page_number_fields("10", "toc"),
            ),
            ChapterBase(
                "toc",
                evidence("entry-1", "toc"),
                **toc_page_number_fields("10", "toc"),
            ),
        )
    )
    destinations = (
        DestinationChapterEvidence(evidence("destination-0", "page-4")),
        DestinationChapterEvidence(evidence("destination-1", "page-8")),
    )
    scores = {
        ("entry-0", "destination-0"): 0.80,
        ("entry-0", "destination-1"): 0.95,
        ("entry-1", "destination-0"): 0.90,
        ("entry-1", "destination-1"): 0.85,
    }

    with mock.patch(
        TITLE_SIMILARITY,
        # title_similarity is called (candidate, reference) - the destination
        # heading first, the TOC entry it is tested against second - while
        # these tables read naturally keyed (entry, destination).
        side_effect=lambda candidate, reference: scores[(reference, candidate)],
    ):
        result = engine.process(
            pages=_pages(10),
            destination_page_numbers=page_numbers({4: "10", 8: "10"}),
            reference_toc=reference,
            destination_chapters=destinations,
        )

    assert result.chapters[0].page_start_key == "page-4"
    assert result.chapters[1].page_start_key == "page-8"


def test_many_to_many_equal_assignments_use_canonical_order(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    engine = fuzzy_engine()
    reference = TocBase(
        tuple(
            ChapterBase(
                "toc",
                evidence("Chapter", "toc"),
                **toc_page_number_fields("10", "toc"),
            )
            for _ in range(2)
        )
    )
    destinations = tuple(
        DestinationChapterEvidence(
            evidence("Chapter", f"page-{position}", y=y)
        )
        for position in (4, 8)
        for y in (10, 50)
    )

    result = engine.process(
        pages=_pages(10),
        destination_page_numbers=page_numbers({4: "10", 8: "10"}),
        reference_toc=reference,
        destination_chapters=destinations,
    )

    assert tuple(chapter.page_start_key for chapter in result.chapters) == (
        "page-4",
        "page-4",
    )
    assert tuple(
        chapter.title_destination_page.bbox.y for chapter in result.chapters
    ) == (10, 50)


def test_many_to_one_non_anchors_all_resolve_without_title_matches(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    engine = fuzzy_engine()
    reference = TocBase(
        (
            ChapterBase(
                "toc",
                evidence("First", "toc"),
                **toc_page_number_fields("10", "toc"),
            ),
            ChapterBase(
                "toc",
                evidence("Second", "toc"),
                **toc_page_number_fields("10", "toc"),
            ),
        )
    )

    result = engine.process(
        pages=_pages(7),
        destination_page_numbers=page_numbers({5: "10"}),
        reference_toc=reference,
        destination_chapters=(
            DestinationChapterEvidence(evidence("Different", "page-5")),
        ),
    )

    assert result.chapters[0].page_start_key == "page-5"
    assert result.chapters[1].page_start_key == "page-5"
    assert result.chapters[0].title_destination_page is None
    assert result.chapters[1].title_destination_page is None


def test_ordered_assignment_maximizes_anchor_count_before_similarity(
    fuzzy_engine,
    evidence,
):
    engine = fuzzy_engine()
    entries = (
        (0, ChapterBase("toc", evidence("entry-0", "toc"))),
        (1, ChapterBase("toc", evidence("entry-1", "toc"))),
    )
    destinations = (
        DestinationChapterEvidence(evidence("destination-0", "page", y=10)),
        DestinationChapterEvidence(evidence("destination-1", "page", y=50)),
    )
    scores = {
        ("entry-0", "destination-0"): 0.90,
        ("entry-0", "destination-1"): 0.0,
        ("entry-1", "destination-0"): 0.95,
        ("entry-1", "destination-1"): 0.75,
    }

    with mock.patch(
        TITLE_SIMILARITY,
        # title_similarity is called (candidate, reference) - the destination
        # heading first, the TOC entry it is tested against second - while
        # these tables read naturally keyed (entry, destination).
        side_effect=lambda candidate, reference: scores[(reference, candidate)],
    ):
        selected = engine._assign_titles(
            entries,
            range(len(destinations)),
            destinations,
            enforce_toc_monotonic_order=True,
        )

    assert [
        (item["entry_index"], item["destination_index"]) for item in selected
    ] == [(0, 0), (1, 1)]


def test_ordered_assignment_follows_destination_y_order(fuzzy_engine, evidence):
    engine = fuzzy_engine()
    entries = (
        (0, ChapterBase("toc", evidence("entry-0", "toc"))),
        (1, ChapterBase("toc", evidence("entry-1", "toc"))),
    )
    destinations = (
        DestinationChapterEvidence(evidence("upper", "page", y=40)),
        DestinationChapterEvidence(evidence("middle", "page", y=60)),
        DestinationChapterEvidence(evidence("lower", "page", y=80)),
    )
    scores = {
        ("entry-0", "upper"): 0.80,
        ("entry-0", "middle"): 0.0,
        ("entry-0", "lower"): 0.99,
        ("entry-1", "upper"): 0.0,
        ("entry-1", "middle"): 0.90,
        ("entry-1", "lower"): 0.0,
    }

    with mock.patch(
        TITLE_SIMILARITY,
        # title_similarity is called (candidate, reference) - the destination
        # heading first, the TOC entry it is tested against second - while
        # these tables read naturally keyed (entry, destination).
        side_effect=lambda candidate, reference: scores[(reference, candidate)],
    ):
        selected = engine._assign_titles(
            entries,
            range(len(destinations)),
            destinations,
            enforce_toc_monotonic_order=True,
        )

    assert [item["destination_index"] for item in selected] == [0, 1]


def test_ordered_assignment_tie_prefers_earliest_destination_sequence(evidence):
    entries = (
        (0, ChapterBase("toc", evidence("entry-0", "toc"))),
        (1, ChapterBase("toc", evidence("entry-1", "toc"))),
    )
    destinations = tuple(
        DestinationChapterEvidence(
            evidence(f"destination-{index}", "page", y=10 + index * 20)
        )
        for index in range(3)
    )

    def assignments(destination_indices):
        return tuple(
            {
                "entry_index": entry_index,
                "destination_index": destination_index,
                "title_score": 0.8,
            }
            for entry_index, destination_index in enumerate(destination_indices)
        )

    first = assignments((0, 1))
    second = assignments((0, 2))
    third = assignments((1, 2))
    comparison_args = (entries, destinations)

    assert ChapterAlignmentEngineFuzzy._title_assignment_is_better(
        first,
        second,
        *comparison_args,
    )
    assert ChapterAlignmentEngineFuzzy._title_assignment_is_better(
        second,
        third,
        *comparison_args,
    )
    assert not ChapterAlignmentEngineFuzzy._title_assignment_is_better(
        third,
        first,
        *comparison_args,
    )


def test_anchor_confidence_sums_all_supporting_evidence(
    evidence,
    toc_page_number_fields,
    physical_page_number,
):
    entry = ChapterBase(
        "toc",
        evidence("Chapter", "toc", confidence=0.2),
        **toc_page_number_fields("10", "toc", confidence=0.1),
    )
    destinations = (
        DestinationChapterEvidence(
            evidence("Chapter", "page-3", confidence=0.3)
        ),
    )

    option = ChapterAlignmentEngineFuzzy._anchor_option(
        0,
        _pages(4)[3],
        0,
        1.0,
        entry,
        destinations,
        entry.page_number,
        physical_page_number("10", "page-3", confidence=0.4),
    )

    assert option["confidence"] == pytest.approx(1.0, abs=5e-8)


def test_unique_page_number_aligns_without_destination_titles(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    engine = fuzzy_engine()
    reference = TocBase(
        (
            ChapterBase(
                toc_page_key="toc",
                title=evidence("Chapter", "toc"),
                **toc_page_number_fields("10", "toc"),
            ),
        )
    )

    result = engine.process(
        pages=_pages(5),
        reference_toc=reference,
        destination_chapters=None,
        destination_page_numbers=page_numbers({3: "10"}),
    )

    chapter = result.chapters[0]
    assert chapter.page_start_key == "page-3"
    assert chapter.title_destination_page is None


# Subsumes the former test_unique_page_number_resolves_non_anchor_when_title_
# mismatches: that test had the same TOC entry, the same {3: "10"} page number
# and the same mismatching "Different chapter" destination on page-3, and
# asserted the same outcome. This adds a competing exact title match on page-4,
# so it exercises the same branch under strictly more pressure.
def test_unique_exact_number_precedes_off_number_title_match(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    engine = fuzzy_engine()

    result = engine.process(
        pages=_pages(6),
        reference_toc=TocBase(
            (
                ChapterBase(
                    toc_page_key="toc",
                    title=evidence("Expected chapter", "toc"),
                    **toc_page_number_fields("10", "toc"),
                ),
            )
        ),
        destination_chapters=(
            DestinationChapterEvidence(evidence("Different chapter", "page-3")),
            DestinationChapterEvidence(evidence("Expected chapter", "page-4")),
        ),
        destination_page_numbers=page_numbers({3: "10"}),
    )

    chapter = result.chapters[0]
    assert chapter.page_start_key == "page-3"
    assert chapter.title_destination_page is None


def test_exact_one_to_one_title_prefers_width_then_reading_order(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
    caplog,
):
    engine = fuzzy_engine()
    reference = TocBase(
        (
            ChapterBase(
                toc_page_key="toc",
                title=evidence("Chapter", "toc"),
                **toc_page_number_fields("10", "toc"),
            ),
        )
    )

    with caplog.at_level(logging.WARNING, logger=ALIGNMENT_LOGGER):
        result = engine.process(
            pages=_pages(5),
            reference_toc=reference,
            destination_chapters=(
                DestinationChapterEvidence(
                    evidence("Chapter", "page-3", y=10, width=100)
                ),
                DestinationChapterEvidence(
                    evidence("CHAPTER", "page-3", y=50, width=140)
                ),
                DestinationChapterEvidence(
                    evidence("Chapter", "page-3", y=30, width=140)
                ),
            ),
            destination_page_numbers=page_numbers({3: "10"}),
        )

    chapter = result.chapters[0]
    assert chapter.page_start_key == "page-3"
    assert chapter.title_destination_page.text == "Chapter"
    assert chapter.title_destination_page.bbox.width == 140
    assert chapter.title_destination_page.bbox.y == 30
    assert (
        "Anchor support is enabled, but no consistent page-number "
        "anchors were selected" in _log_output(caplog)
    )


def test_destination_page_numbers_must_reference_unique_input_pages(
    fuzzy_engine,
    physical_page_number,
):
    engine = fuzzy_engine()
    common = {
        "pages": _pages(2),
        "reference_toc": TocBase(()),
        "destination_chapters": (),
    }

    with pytest.raises(ValueError, match="duplicate page_key"):
        engine.process(
            **common,
            destination_page_numbers=(
                physical_page_number("1", "page-1"),
                physical_page_number("2", "page-1"),
            ),
        )
    with pytest.raises(ValueError, match="not available"):
        engine.process(
            **common,
            destination_page_numbers=(
                physical_page_number("1", "unknown"),
            ),
        )


def test_alignment_accepts_both_evidence_collections_empty(
    fuzzy_engine,
    evidence,
    caplog,
):
    engine = fuzzy_engine()
    reference = TocBase(
        (
            ChapterBase(
                toc_page_key="toc",
                title=evidence("Chapter", "toc"),
            ),
        )
    )

    with caplog.at_level(logging.WARNING, logger=ALIGNMENT_LOGGER):
        result = engine.process(
            pages=_pages(2),
            reference_toc=reference,
            destination_chapters=(),
            destination_page_numbers=(),
        )

    assert _log_output(caplog)
    assert result.chapters[0].page_start_key is None


def test_chapters_may_share_a_page_but_not_a_heading_detection(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    engine = fuzzy_engine()
    child = ChapterBase(
        toc_page_key="toc",
        title=evidence("Section heading", "toc"),
        **toc_page_number_fields("10", "toc"),
    )
    reference = TocBase(
        (
            ChapterBase(
                toc_page_key="toc",
                title=evidence("Volume heading", "toc"),
                **toc_page_number_fields("10", "toc"),
                children=(child,),
            ),
        )
    )
    destinations = (
        DestinationChapterEvidence(
            evidence("VOLUME HEADING", "page-5", y=10)
        ),
        DestinationChapterEvidence(
            evidence("SECTION HEADING", "page-5", y=50)
        ),
    )

    result = engine.process(
        pages=_pages(7),
        destination_page_numbers=page_numbers({5: "10"}),
        reference_toc=reference,
        destination_chapters=destinations,
    )

    root = result.chapters[0]
    assert root.page_start_key == "page-5"
    assert root.children[0].page_start_key == "page-5"
    assert root.title_destination_page.text == "VOLUME HEADING"
    assert root.children[0].title_destination_page.text == "SECTION HEADING"


def test_anchor_chain_prefers_title_supported_monotonic_solution(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
    caplog,
):
    engine = fuzzy_engine()
    reference = TocBase(
        tuple(
            ChapterBase(
                toc_page_key="toc",
                title=evidence(title, "toc"),
                **toc_page_number_fields(str(number), "toc"),
            )
            for number, title in (
                (1, "First"),
                (2, "Conflicting"),
                (3, "Third"),
            )
        )
    )

    with caplog.at_level(logging.WARNING, logger=ALIGNMENT_LOGGER):
        result = engine.process(
            pages=_pages(14),
            destination_page_numbers=page_numbers({5: "2", 10: "1", 12: "3"}),
            reference_toc=reference,
            destination_chapters=(
                DestinationChapterEvidence(evidence("First", "page-10")),
                DestinationChapterEvidence(evidence("Third", "page-12")),
            ),
        )

    assert result.chapters[0].page_start_key == "page-10"
    assert result.chapters[1].page_start_key is None
    assert result.chapters[2].page_start_key == "page-12"
    assert (
        "Failed to resolve non-anchor TOC entry by unified solver: entry=1"
        in _log_output(caplog)
    )


def test_mismatched_anchor_offsets_use_the_physical_interval(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
    caplog,
):
    engine = fuzzy_engine()
    middle = ChapterBase(
        toc_page_key="toc",
        title=evidence("Middle", "toc"),
        **toc_page_number_fields("15", "toc"),
    )
    reference = TocBase(
        (
            ChapterBase(
                toc_page_key="toc",
                title=evidence("First", "toc"),
                **toc_page_number_fields("10", "toc"),
                children=(middle,),
            ),
            ChapterBase(
                toc_page_key="toc",
                title=evidence("Last", "toc"),
                **toc_page_number_fields("20", "toc"),
            ),
        )
    )
    destinations = (
        DestinationChapterEvidence(evidence("First", "page-20")),
        DestinationChapterEvidence(evidence("Middle", "page-25", height=20)),
        DestinationChapterEvidence(evidence("Middle", "page-31", height=40)),
        DestinationChapterEvidence(evidence("Last", "page-32")),
        DestinationChapterEvidence(evidence("Middle", "page-33", height=60)),
    )

    with caplog.at_level(logging.INFO, logger=ALIGNMENT_LOGGER):
        result = engine.process(
            pages=_pages(34),
            destination_page_numbers=page_numbers({20: "10", 32: "20"}),
            reference_toc=reference,
            destination_chapters=destinations,
        )

    assert result.chapters[0].page_start_key == "page-20"
    assert result.chapters[0].children[0].page_start_key == "page-31"
    assert result.chapters[1].page_start_key == "page-32"
    log_output = _log_output(caplog)
    assert "Selected TOC anchor: entry=0" in log_output
    assert "Selected TOC anchor: entry=2" in log_output
    assert "Generating unified resolution candidates: entry=1" in log_output
    assert "physical_bounds=20..32" in log_output
    assert "expected_position=None" in log_output
    assert (
        "offset_mode=no compatible ideal offset; anchor bounds only"
        in log_output
    )
    assert (
        "Resolved non-anchor TOC entry by unified solver: entry=1" in log_output
    )
    assert "destination_page='page-31'" in log_output


def test_matching_anchor_offsets_keep_the_tolerance_constraint(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    engine = fuzzy_engine()
    middle = ChapterBase(
        toc_page_key="toc",
        title=evidence("Middle", "toc"),
        **toc_page_number_fields("15", "toc"),
    )
    reference = TocBase(
        (
            ChapterBase(
                toc_page_key="toc",
                title=evidence("First", "toc"),
                **toc_page_number_fields("10", "toc"),
                children=(middle,),
            ),
            ChapterBase(
                toc_page_key="toc",
                title=evidence("Last", "toc"),
                **toc_page_number_fields("20", "toc"),
            ),
        )
    )

    result = engine.process(
        pages=_pages(32),
        destination_page_numbers=page_numbers({20: "10", 30: "20"}),
        reference_toc=reference,
        destination_chapters=(
            DestinationChapterEvidence(evidence("First", "page-20")),
            DestinationChapterEvidence(
                evidence("Middle", "page-25", height=20)
            ),
            DestinationChapterEvidence(
                evidence("Middle", "page-29", height=40)
            ),
            DestinationChapterEvidence(evidence("Last", "page-30")),
        ),
    )

    assert result.chapters[0].children[0].page_start_key == "page-25"


def test_anchor_derived_position_resolves_without_title_match(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
    caplog,
):
    engine = fuzzy_engine()
    reference = TocBase(
        tuple(
            ChapterBase(
                toc_page_key="toc",
                title=evidence(title, "toc"),
                **toc_page_number_fields(number, "toc"),
            )
            for title, number in (
                ("First", "10"),
                ("Middle", "15"),
                ("Last", "20"),
            )
        )
    )

    with caplog.at_level(logging.INFO, logger=ALIGNMENT_LOGGER):
        result = engine.process(
            pages=_pages(32),
            destination_page_numbers=page_numbers({20: "10", 30: "20"}),
            reference_toc=reference,
            destination_chapters=(
                DestinationChapterEvidence(evidence("First", "page-20")),
                DestinationChapterEvidence(evidence("Last", "page-30")),
            ),
        )

    middle = result.chapters[1]
    assert middle.page_start_key == "page-25"
    assert middle.title_destination_page is None
    log_output = _log_output(caplog)
    assert "Generating unified resolution candidates: entry=1" in log_output
    assert (
        "Resolved non-anchor TOC entry by unified solver: entry=1" in log_output
    )
    assert "source=anchor_position" in log_output


def test_disabled_anchors_use_the_unified_solver_without_offsets(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
    caplog,
):
    engine = fuzzy_engine(use_anchors=False)
    reference = TocBase(
        tuple(
            ChapterBase(
                toc_page_key="toc",
                title=evidence(title, "toc"),
                **toc_page_number_fields(number, "toc"),
            )
            for title, number in (
                ("First", "10"),
                ("Middle", "15"),
                ("Last", "20"),
            )
        )
    )

    with caplog.at_level(logging.INFO, logger=ALIGNMENT_LOGGER):
        result = engine.process(
            pages=_pages(32),
            destination_page_numbers=page_numbers({20: "10", 30: "20"}),
            reference_toc=reference,
            destination_chapters=(
                DestinationChapterEvidence(evidence("First", "page-20")),
                DestinationChapterEvidence(evidence("Middle", "page-25")),
                DestinationChapterEvidence(evidence("Last", "page-30")),
            ),
        )

    assert tuple(chapter.page_start_key for chapter in result.chapters) == (
        "page-20",
        "page-25",
        "page-30",
    )
    log_output = _log_output(caplog)
    assert "using disabled mode" in log_output
    assert "fixed_anchors=0" in log_output
    assert (
        "entry=1, toc_page='toc', title='Middle', toc_number='15', "
        "exact_pages=0" in log_output
    )
    assert "expected_position=None" in log_output


def test_exact_number_resolution_precedes_total_resolution_count(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    engine = fuzzy_engine(
        use_anchors=False,
        toc_monotonic_order_constraints="yes",
    )

    result = engine.process(
        pages=_pages(12),
        destination_page_numbers=page_numbers({8: "10"}),
        reference_toc=TocBase(
            (
                ChapterBase(
                    "toc",
                    evidence("Exact", "toc"),
                    **toc_page_number_fields("10", "toc"),
                ),
                ChapterBase("toc", evidence("Second", "toc")),
                ChapterBase("toc", evidence("Third", "toc")),
            )
        ),
        destination_chapters=(
            DestinationChapterEvidence(evidence("Second", "page-2")),
            DestinationChapterEvidence(evidence("Third", "page-3")),
        ),
    )

    assert result.chapters[0].page_start_key == "page-8"
    assert result.chapters[1].page_start_key is None
    assert result.chapters[2].page_start_key is None


def test_distant_title_keeps_anchor_derived_position(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    engine = fuzzy_engine()
    reference = TocBase(
        tuple(
            ChapterBase(
                toc_page_key="toc",
                title=evidence(title, "toc"),
                **toc_page_number_fields(number, "toc"),
            )
            for title, number in (
                ("First", "10"),
                ("Middle", "15"),
                ("Last", "20"),
            )
        )
    )

    result = engine.process(
        pages=_pages(32),
        destination_page_numbers=page_numbers({20: "10", 30: "20"}),
        reference_toc=reference,
        destination_chapters=(
            DestinationChapterEvidence(evidence("First", "page-20")),
            DestinationChapterEvidence(evidence("Middle", "page-29")),
            DestinationChapterEvidence(evidence("Last", "page-30")),
        ),
    )

    middle = result.chapters[1]
    assert middle.page_start_key == "page-25"
    assert middle.title_destination_page is None


def test_title_within_tolerance_precedes_anchor_position_fallback(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    engine = fuzzy_engine()
    reference = TocBase(
        tuple(
            ChapterBase(
                toc_page_key="toc",
                title=evidence(title, "toc"),
                **toc_page_number_fields(number, "toc"),
            )
            for title, number in (
                ("First", "10"),
                ("Middle", "15"),
                ("Last", "20"),
            )
        )
    )

    result = engine.process(
        pages=_pages(32),
        destination_page_numbers=page_numbers({20: "10", 30: "20"}),
        reference_toc=reference,
        destination_chapters=(
            DestinationChapterEvidence(evidence("First", "page-20")),
            DestinationChapterEvidence(evidence("Middle", "page-26")),
            DestinationChapterEvidence(evidence("Last", "page-30")),
        ),
    )

    middle = result.chapters[1]
    assert middle.page_start_key == "page-26"
    assert middle.title_destination_page.page_key == "page-26"


def test_failed_title_fallback_logs_final_reason(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    caplog,
):
    engine = fuzzy_engine()

    with caplog.at_level(logging.INFO, logger=ALIGNMENT_LOGGER):
        result = engine.process(
            pages=_pages(4),
            destination_page_numbers=(),
            reference_toc=TocBase(
                (
                    ChapterBase(
                        toc_page_key="toc",
                        title=evidence("Missing", "toc"),
                        **toc_page_number_fields("10", "toc"),
                    ),
                )
            ),
            destination_chapters=(),
        )

    assert result.chapters[0].page_start_key is None
    log_output = _log_output(caplog)
    assert "Generating unified resolution candidates: entry=0" in log_output
    assert (
        "Failed to resolve non-anchor TOC entry by unified solver: entry=0"
        in log_output
    )
    assert "reason='no eligible candidate was generated'" in log_output
    assert "candidate_count=0" in log_output


def test_one_compatible_anchor_still_supplies_the_offset(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    engine = fuzzy_engine()
    reference = TocBase(
        (
            ChapterBase(
                toc_page_key="toc",
                title=evidence("Roman", "toc"),
                **toc_page_number_fields("X", "toc"),
            ),
            ChapterBase(
                toc_page_key="toc",
                title=evidence("Middle", "toc"),
                **toc_page_number_fields("15", "toc"),
            ),
            ChapterBase(
                toc_page_key="toc",
                title=evidence("Arabic", "toc"),
                **toc_page_number_fields("20", "toc"),
            ),
        )
    )

    result = engine.process(
        pages=_pages(32),
        destination_page_numbers=page_numbers({10: "X", 30: "20"}),
        reference_toc=reference,
        destination_chapters=(
            DestinationChapterEvidence(evidence("Roman", "page-10")),
            DestinationChapterEvidence(
                evidence("Middle", "page-20", height=40)
            ),
            DestinationChapterEvidence(
                evidence("Middle", "page-25", height=20)
            ),
            DestinationChapterEvidence(evidence("Arabic", "page-30")),
        ),
    )

    assert result.chapters[1].page_start_key == "page-25"


def test_no_compatible_offsets_use_a_complete_anchor_interval(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    engine = fuzzy_engine()
    reference = TocBase(
        tuple(
            ChapterBase(
                toc_page_key="toc",
                title=evidence(title, "toc"),
                **toc_page_number_fields(number, "toc"),
            )
            for title, number in (
                ("Roman start", "X"),
                ("Arabic middle", "15"),
                ("Roman end", "XX"),
            )
        )
    )

    result = engine.process(
        pages=_pages(32),
        destination_page_numbers=page_numbers({10: "X", 30: "XX"}),
        reference_toc=reference,
        destination_chapters=(
            DestinationChapterEvidence(evidence("Roman start", "page-10")),
            DestinationChapterEvidence(evidence("Arabic middle", "page-24")),
            DestinationChapterEvidence(evidence("Roman end", "page-30")),
        ),
    )

    assert result.chapters[1].page_start_key == "page-24"


def test_incompatible_preceding_anchor_supplies_a_one_sided_bound(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    engine = fuzzy_engine()
    reference = TocBase(
        (
            ChapterBase(
                toc_page_key="toc",
                title=evidence("Roman", "toc"),
                **toc_page_number_fields("X", "toc"),
            ),
            ChapterBase(
                toc_page_key="toc",
                title=evidence("Arabic", "toc"),
                **toc_page_number_fields("15", "toc"),
            ),
        )
    )

    result = engine.process(
        pages=_pages(24),
        destination_page_numbers=page_numbers({10: "X"}),
        reference_toc=reference,
        destination_chapters=(
            DestinationChapterEvidence(
                evidence("Arabic", "page-5", height=40)
            ),
            DestinationChapterEvidence(evidence("Roman", "page-10")),
            DestinationChapterEvidence(evidence("Arabic", "page-20")),
        ),
    )

    assert result.chapters[1].page_start_key == "page-20"


def test_incompatible_following_anchor_supplies_a_one_sided_bound(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    engine = fuzzy_engine()
    reference = TocBase(
        (
            ChapterBase(
                toc_page_key="toc",
                title=evidence("Arabic", "toc"),
                **toc_page_number_fields("15", "toc"),
            ),
            ChapterBase(
                toc_page_key="toc",
                title=evidence("Roman", "toc"),
                **toc_page_number_fields("XX", "toc"),
            ),
        )
    )

    result = engine.process(
        pages=_pages(28),
        destination_page_numbers=page_numbers({20: "XX"}),
        reference_toc=reference,
        destination_chapters=(
            DestinationChapterEvidence(
                evidence("Arabic", "page-10", height=20)
            ),
            DestinationChapterEvidence(evidence("Roman", "page-20")),
            DestinationChapterEvidence(
                evidence("Arabic", "page-25", height=40)
            ),
        ),
    )

    assert result.chapters[0].page_start_key == "page-10"


def test_range_resolves_explicit_end_page(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    engine = fuzzy_engine()
    reference = TocBase(
        (
            ChapterBase(
                toc_page_key="toc",
                title=evidence("Range chapter", "toc"),
                **toc_page_number_fields("10–12", "toc"),
            ),
        )
    )

    result = engine.process(
        pages=_pages(24),
        destination_page_numbers=page_numbers({20: "10", 22: "12"}),
        reference_toc=reference,
        destination_chapters=(
            DestinationChapterEvidence(evidence("Range chapter", "page-20")),
        ),
    )

    chapter = result.chapters[0]
    assert chapter.page_start_key == "page-20"
    assert chapter.page_end_key == "page-22"
    assert chapter.page_number.output_text() == "10-12"


def test_range_end_distance_tie_prefers_earlier_page_position(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    engine = fuzzy_engine()
    reference = TocBase(
        (
            ChapterBase(
                toc_page_key="toc",
                title=evidence("Range chapter", "toc"),
                **toc_page_number_fields("10–12", "toc"),
            ),
        )
    )

    result = engine.process(
        pages=_pages(24),
        destination_page_numbers=page_numbers({23: "12", 20: "10", 21: "12"}),
        reference_toc=reference,
        destination_chapters=(
            DestinationChapterEvidence(evidence("Range chapter", "page-20")),
        ),
    )

    assert result.chapters[0].page_end_key == "page-21"


def test_list_uses_first_number_for_anchor_and_preserves_full_list(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    engine = fuzzy_engine()

    result = engine.process(
        pages=_pages(22),
        destination_page_numbers=page_numbers({20: "10"}),
        reference_toc=TocBase(
            (
                ChapterBase(
                    toc_page_key="toc",
                    title=evidence("Listed chapter", "toc"),
                    **toc_page_number_fields("010, 12, 14", "toc"),
                ),
            )
        ),
        destination_chapters=(
            DestinationChapterEvidence(evidence("Listed chapter", "page-20")),
        ),
    )

    chapter = result.chapters[0]
    assert chapter.page_start_key == "page-20"
    assert chapter.page_end_key is None
    assert chapter.page_number.output_text() == "10,12,14"


def test_descending_range_uses_start_as_single_number_anchor(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    engine = fuzzy_engine()

    result = engine.process(
        pages=_pages(7),
        destination_page_numbers=page_numbers({5: "24"}),
        reference_toc=TocBase(
            (
                ChapterBase(
                    toc_page_key="toc",
                    title=evidence("Chapter", "toc"),
                    **toc_page_number_fields("24-23", "toc"),
                ),
            )
        ),
        destination_chapters=(
            DestinationChapterEvidence(evidence("Chapter", "page-5")),
        ),
    )

    chapter = result.chapters[0]
    assert chapter.page_start_key == "page-5"
    assert chapter.page_end_key is None
    assert chapter.page_number.output_text() == "24"

def test_end_inference_infers_from_the_following_entry_and_document_end(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    engine = fuzzy_engine()
    reference = TocBase(
        (
            ChapterBase(
                toc_page_key="toc",
                title=evidence("First", "toc"),
                **toc_page_number_fields("3", "toc"),
            ),
            ChapterBase(
                toc_page_key="toc",
                title=evidence("Second", "toc"),
                **toc_page_number_fields("6", "toc"),
            ),
        )
    )

    result = engine.process(
        pages=_pages(8),
        destination_page_numbers=page_numbers({3: "3", 6: "6"}),
        reference_toc=reference,
        destination_chapters=(
            DestinationChapterEvidence(evidence("First", "page-3")),
            DestinationChapterEvidence(evidence("Second", "page-6")),
        ),
    )

    first, second = result.chapters
    assert (first.page_start_key, first.page_end_key) == ("page-3", "page-5")
    assert (second.page_start_key, second.page_end_key) == ("page-6", "page-7")


def test_disabled_end_inference_leaves_implicit_ends_unresolved(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
    caplog,
):
    engine = fuzzy_engine(infer_chapter_ends=False)
    reference = TocBase(
        (
            ChapterBase(
                toc_page_key="toc",
                title=evidence("First", "toc"),
                **toc_page_number_fields("3", "toc"),
            ),
            ChapterBase(
                toc_page_key="toc",
                title=evidence("Second", "toc"),
                **toc_page_number_fields("6", "toc"),
            ),
        )
    )

    with caplog.at_level(logging.INFO, logger=ALIGNMENT_LOGGER):
        result = engine.process(
            pages=_pages(8),
            destination_page_numbers=page_numbers({3: "3", 6: "6"}),
            reference_toc=reference,
            destination_chapters=(
                DestinationChapterEvidence(evidence("First", "page-3")),
                DestinationChapterEvidence(evidence("Second", "page-6")),
            ),
        )

    assert all(chapter.page_end_key is None for chapter in result.chapters)
    assert "infer_chapter_ends is disabled" in _log_output(caplog)


def test_end_inference_requires_the_configured_monotonicity_score(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
    caplog,
):
    engine = fuzzy_engine()
    reference = TocBase(
        (
            ChapterBase(
                toc_page_key="toc",
                title=evidence("Late", "toc"),
                **toc_page_number_fields("20", "toc"),
            ),
            ChapterBase(
                toc_page_key="toc",
                title=evidence("Early", "toc"),
                **toc_page_number_fields("10", "toc"),
            ),
        )
    )

    with caplog.at_level(logging.INFO, logger=ALIGNMENT_LOGGER):
        result = engine.process(
            pages=_pages(10),
            destination_page_numbers=page_numbers({8: "20", 4: "10"}),
            reference_toc=reference,
            destination_chapters=(
                DestinationChapterEvidence(evidence("Late", "page-8")),
                DestinationChapterEvidence(evidence("Early", "page-4")),
            ),
        )

    assert all(chapter.page_end_key is None for chapter in result.chapters)
    assert "monotonicity score=0.5 is missing or below" in _log_output(caplog)


def test_null_end_inference_threshold_infers_without_a_usable_score(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    reference = TocBase(
        (
            ChapterBase(
                toc_page_key="toc",
                title=evidence("Only chapter", "toc"),
                **toc_page_number_fields("10", "toc"),
            ),
        )
    )
    arguments = {
        "pages": _pages(5),
        "destination_page_numbers": page_numbers({3: "10"}),
        "reference_toc": reference,
        "destination_chapters": (
            DestinationChapterEvidence(evidence("Only chapter", "page-3")),
        ),
    }

    # A single TOC number yields no comparable sequence and therefore no score.
    assert fuzzy_engine().process(**arguments).chapters[0].page_end_key is None

    engine = fuzzy_engine(
        minimum_toc_monotonicity_score_for_end_inference=None
    )

    assert engine.process(**arguments).chapters[0].page_end_key == "page-4"


def test_end_inference_stops_a_parent_at_a_following_entry_not_its_child(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    engine = fuzzy_engine()
    reference = TocBase(
        (
            ChapterBase(
                toc_page_key="toc",
                title=evidence("First", "toc"),
                **toc_page_number_fields("2", "toc"),
                children=(
                    ChapterBase(
                        toc_page_key="toc",
                        title=evidence("Sub", "toc"),
                        **toc_page_number_fields("4", "toc"),
                    ),
                ),
            ),
            ChapterBase(
                toc_page_key="toc",
                title=evidence("Second", "toc"),
                **toc_page_number_fields("7", "toc"),
            ),
        )
    )

    result = engine.process(
        pages=_pages(9),
        destination_page_numbers=page_numbers({2: "2", 4: "4", 7: "7"}),
        reference_toc=reference,
        destination_chapters=(
            DestinationChapterEvidence(evidence("First", "page-2")),
            DestinationChapterEvidence(evidence("Sub", "page-4")),
            DestinationChapterEvidence(evidence("Second", "page-7")),
        ),
    )

    first, second = result.chapters
    child = first.children[0]
    # The child has greater depth, so it cannot terminate its parent.
    assert first.page_end_key == "page-6"
    assert child.page_end_key == "page-6"
    assert second.page_end_key == "page-8"


def test_end_inference_never_overwrites_an_explicit_range_end(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    engine = fuzzy_engine()
    reference = TocBase(
        (
            ChapterBase(
                toc_page_key="toc",
                title=evidence("Range chapter", "toc"),
                **toc_page_number_fields("2–3", "toc"),
            ),
            ChapterBase(
                toc_page_key="toc",
                title=evidence("Later chapter", "toc"),
                **toc_page_number_fields("8", "toc"),
            ),
        )
    )

    result = engine.process(
        pages=_pages(10),
        destination_page_numbers=page_numbers({2: "2", 3: "3", 8: "8"}),
        reference_toc=reference,
        destination_chapters=(
            DestinationChapterEvidence(evidence("Range chapter", "page-2")),
            DestinationChapterEvidence(evidence("Later chapter", "page-8")),
        ),
    )

    assert result.chapters[0].page_end_key == "page-3"


def test_titleless_entry_terminates_the_chapter_before_it(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    """The wrapper prunes number-only entries, so only alignment can use them."""
    engine = fuzzy_engine()
    reference = TocBase(
        (
            ChapterBase(
                toc_page_key="toc",
                title=evidence("First", "toc"),
                **toc_page_number_fields("2", "toc"),
            ),
            ChapterBase(
                toc_page_key="toc",
                title=None,
                **toc_page_number_fields("5", "toc"),
            ),
            ChapterBase(
                toc_page_key="toc",
                title=evidence("Second", "toc"),
                **toc_page_number_fields("8", "toc"),
            ),
        )
    )

    result = engine.process(
        pages=_pages(10),
        destination_page_numbers=page_numbers({2: "2", 5: "5", 8: "8"}),
        reference_toc=reference,
        destination_chapters=(
            DestinationChapterEvidence(evidence("First", "page-2")),
            DestinationChapterEvidence(evidence("Second", "page-8")),
        ),
    )

    first, titleless, second = result.chapters
    assert titleless.title is None
    assert titleless.page_start_key == "page-5"
    # Without the number-only entry the first chapter would run to page-7.
    assert first.page_end_key == "page-4"
    assert second.page_end_key == "page-9"

def test_titleless_unique_number_resolves_without_destination_title(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    engine = fuzzy_engine()
    reference = TocBase(
        (
            ChapterBase(
                toc_page_key="toc",
                title=None,
                **toc_page_number_fields("10", "toc"),
            ),
        )
    )

    result = engine.process(
        pages=_pages(7),
        destination_page_numbers=page_numbers({5: "10"}),
        reference_toc=reference,
        destination_chapters=(
            DestinationChapterEvidence(evidence("Detected title", "page-5")),
        ),
    )

    assert len(result.chapters) == 1
    assert result.chapters[0].title is None
    assert result.chapters[0].title_destination_page is None
    assert result.chapters[0].page_start_key == "page-5"


def test_numberless_title_match_can_share_exact_number_page(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    engine = fuzzy_engine()
    reference = TocBase(
        (
            ChapterBase(
                toc_page_key="toc",
                title=None,
                **toc_page_number_fields("10", "toc"),
            ),
            ChapterBase(
                toc_page_key="toc",
                title=evidence("Detected title", "toc"),
            ),
        )
    )

    result = engine.process(
        pages=_pages(7),
        destination_page_numbers=page_numbers({5: "10"}),
        reference_toc=reference,
        destination_chapters=(
            DestinationChapterEvidence(evidence("Detected title", "page-5")),
        ),
    )

    titleless_anchor, titled_entry = result.chapters
    assert titleless_anchor.page_start_key == "page-5"
    assert titleless_anchor.title_destination_page is None
    assert titled_entry.page_start_key == "page-5"
    assert titled_entry.title_destination_page.text == "Detected title"


def test_titleless_entry_is_returned_for_wrapper_pruning(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    engine = fuzzy_engine()
    child = ChapterBase(
        toc_page_key="toc",
        title=evidence("Child", "toc"),
        **toc_page_number_fields("11", "toc"),
    )
    reference = TocBase(
        (
            ChapterBase(
                toc_page_key="toc",
                title=None,
                **toc_page_number_fields("10", "toc"),
                children=(child,),
            ),
        )
    )

    result = engine.process(
        pages=_pages(8),
        destination_page_numbers=page_numbers({5: "10", 6: "11"}),
        reference_toc=reference,
        destination_chapters=(
            DestinationChapterEvidence(evidence("Child", "page-6")),
        ),
    )

    assert len(result.chapters) == 1
    assert result.chapters[0].title is None
    assert result.chapters[0].title_destination_page is None
    assert result.chapters[0].page_start_key == "page-5"
    assert len(result.chapters[0].children) == 1
    assert result.chapters[0].children[0].title.text == "Child"
    assert result.chapters[0].children[0].page_start_key == "page-6"


def test_without_anchors_title_matching_uses_the_whole_document(
    fuzzy_engine,
    evidence,
    page_numbers,
):
    engine = fuzzy_engine()
    reference = TocBase(
        (
            ChapterBase(
                toc_page_key="toc",
                title=evidence("Chapter", "toc"),
            ),
        )
    )

    result = engine.process(
        pages=_pages(4),
        destination_page_numbers=page_numbers({1: "1", 3: "3"}),
        reference_toc=reference,
        destination_chapters=(
            DestinationChapterEvidence(
                evidence("Chapter", "page-1", height=20)
            ),
            DestinationChapterEvidence(
                evidence("Chapter", "page-3", height=40)
            ),
        ),
    )

    assert result.chapters[0].page_start_key == "page-3"
    assert result.chapters[0].title_destination_page.page_key == "page-3"


def test_title_fallback_globally_maximizes_monotonic_matches(
    fuzzy_engine,
    evidence,
):
    engine = fuzzy_engine()
    destinations = (
        DestinationChapterEvidence(evidence("destination-0", "page-2", y=10)),
        DestinationChapterEvidence(evidence("destination-1", "page-2", y=50)),
    )
    scores = {
        ("flexible", "destination-0"): 0.80,
        ("flexible", "destination-1"): 0.95,
        ("specific", "destination-0"): 0.0,
        ("specific", "destination-1"): 0.80,
    }

    with mock.patch(
        TITLE_SIMILARITY,
        # title_similarity is called (candidate, reference) - the destination
        # heading first, the TOC entry it is tested against second - while
        # these tables read naturally keyed (entry, destination).
        side_effect=lambda candidate, reference: scores[(reference, candidate)],
    ):
        result = engine.process(
            pages=_pages(4),
            destination_page_numbers=None,
            reference_toc=TocBase(
                (
                    ChapterBase("toc", evidence("flexible", "toc")),
                    ChapterBase("toc", evidence("specific", "toc")),
                )
            ),
            destination_chapters=destinations,
        )

    assert tuple(
        chapter.title_destination_page.text for chapter in result.chapters
    ) == ("destination-0", "destination-1")


def test_title_fallback_global_assignment_enforces_monotonicity(
    fuzzy_engine,
    evidence,
):
    engine = fuzzy_engine()
    destinations = (
        DestinationChapterEvidence(evidence("upper", "page-2", y=10)),
        DestinationChapterEvidence(evidence("lower", "page-2", y=50)),
    )
    scores = {
        ("first", "upper"): 0.0,
        ("first", "lower"): 0.90,
        ("second", "upper"): 0.80,
        ("second", "lower"): 0.0,
    }

    with mock.patch(
        TITLE_SIMILARITY,
        # title_similarity is called (candidate, reference) - the destination
        # heading first, the TOC entry it is tested against second - while
        # these tables read naturally keyed (entry, destination).
        side_effect=lambda candidate, reference: scores[(reference, candidate)],
    ):
        result = engine.process(
            pages=_pages(4),
            destination_page_numbers=None,
            reference_toc=TocBase(
                (
                    ChapterBase("toc", evidence("first", "toc")),
                    ChapterBase("toc", evidence("second", "toc")),
                )
            ),
            destination_chapters=destinations,
        )

    assert result.chapters[0].title_destination_page.text == "lower"
    assert result.chapters[1].title_destination_page is None


def test_unordered_title_fallback_does_not_enforce_monotonicity(
    fuzzy_engine,
    evidence,
):
    engine = fuzzy_engine(toc_monotonic_order_constraints="no")
    destinations = (
        DestinationChapterEvidence(evidence("upper", "page-2", y=10)),
        DestinationChapterEvidence(evidence("lower", "page-2", y=50)),
    )
    scores = {
        ("first", "upper"): 0.0,
        ("first", "lower"): 0.90,
        ("second", "upper"): 0.80,
        ("second", "lower"): 0.0,
    }

    with mock.patch(
        TITLE_SIMILARITY,
        # title_similarity is called (candidate, reference) - the destination
        # heading first, the TOC entry it is tested against second - while
        # these tables read naturally keyed (entry, destination).
        side_effect=lambda candidate, reference: scores[(reference, candidate)],
    ):
        result = engine.process(
            pages=_pages(4),
            destination_page_numbers=None,
            reference_toc=TocBase(
                (
                    ChapterBase("toc", evidence("first", "toc")),
                    ChapterBase("toc", evidence("second", "toc")),
                )
            ),
            destination_chapters=destinations,
        )

    assert tuple(
        chapter.title_destination_page.text for chapter in result.chapters
    ) == ("lower", "upper")


def test_title_fallback_uses_reading_order_as_final_tie_break(
    fuzzy_engine,
    evidence,
):
    engine = fuzzy_engine()
    reference = TocBase(
        (
            ChapterBase(
                toc_page_key="toc",
                title=evidence("Chapter", "toc"),
            ),
        )
    )

    result = engine.process(
        pages=_pages(4),
        destination_page_numbers=None,
        reference_toc=reference,
        destination_chapters=(
            DestinationChapterEvidence(evidence("Chapter", "page-2", y=50)),
            DestinationChapterEvidence(evidence("Chapter", "page-2", y=10)),
        ),
    )

    selected = result.chapters[0].title_destination_page
    assert selected.page_key == "page-2"
    assert selected.bbox.y == 10


def test_unparsable_toc_number_uses_title_fallback(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
):
    engine = fuzzy_engine()
    reference = TocBase(
        (
            ChapterBase(
                toc_page_key="toc",
                title=evidence("Chapter", "toc"),
                **toc_page_number_fields("unknown", "toc"),
            ),
        )
    )

    result = engine.process(
        pages=_pages(4),
        destination_page_numbers=None,
        reference_toc=reference,
        destination_chapters=(
            DestinationChapterEvidence(evidence("Chapter", "page-2")),
        ),
    )

    assert result.chapters[0].page_start_key == "page-2"
    assert result.chapters[0].page_number.text == "unknown"


def test_unified_title_solver_uses_canonical_toc_order(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
):
    engine = fuzzy_engine()

    result = engine.process(
        pages=_pages(4),
        destination_page_numbers=(),
        reference_toc=TocBase(
            (
                ChapterBase(
                    "toc",
                    evidence("Shared", "toc"),
                    **toc_page_number_fields("unknown", "toc"),
                ),
                ChapterBase(
                    "toc",
                    evidence("Shared", "toc"),
                    **toc_page_number_fields("10", "toc"),
                ),
            )
        ),
        destination_chapters=(
            DestinationChapterEvidence(evidence("Shared", "page-2")),
        ),
    )

    assert result.chapters[0].page_start_key == "page-2"
    assert result.chapters[1].page_start_key is None


def test_rejected_numeric_fragment_cannot_anchor_but_is_preserved(
    fuzzy_engine,
    evidence,
    toc_page_number_fields,
    page_numbers,
):
    engine = fuzzy_engine()
    reference = TocBase(
        (
            ChapterBase(
                toc_page_key="toc",
                title=evidence("Chapter", "toc"),
                **toc_page_number_fields("-45", "toc"),
            ),
        )
    )

    result = engine.process(
        pages=_pages(4),
        destination_page_numbers=page_numbers({1: "45"}),
        reference_toc=reference,
        destination_chapters=(
            DestinationChapterEvidence(
                evidence("Different heading", "page-1")
            ),
            DestinationChapterEvidence(evidence("Chapter", "page-2")),
        ),
    )

    chapter = result.chapters[0]
    assert chapter.page_start_key == "page-2"
    assert chapter.page_number.text == "-45"

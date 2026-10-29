from dataclasses import replace

import pytest

from metakat.common.models import AltoRefs, BoundingBox, PageDimensions
from metakat.page_number.engines.core.models import PageNumberCoreResult
from metakat.page_number.engines.core.ndk_pagination import (
    NdkPageContext,
    NdkPaginationConfig,
    NdkPaginator,
    suffix,
)
from metakat.page_number.engines.core.page_number_parsers import (
    DecoratedPageNumberParser,
)
from metakat.page_number.engines.core.postprocessing import (
    PageNumberPostprocessor,
    PageNumberPostprocessConfig,
    PageNumberPostprocessInput,
)


def inputs(values, contexts=None):
    contexts = contexts or {}
    pages = []
    for i, value in enumerate(values):
        evidence = (
            None
            if value is None
            else DecoratedPageNumberParser.create(
                page_key=f"p{i}",
                text=str(value),
                confidence=0.99,
                bbox=BoundingBox(480, 950, 40, 20),
            )
        )
        pages.append(
            PageNumberPostprocessInput(
                f"p{i}",
                PageDimensions(1000, 1000),
                (evidence,) if evidence else (),
                evidence,
                contexts.get(i, NdkPageContext()),
            )
        )
    return tuple(pages)


def edges(**settings):
    """The final pass: complete documents, so the edges are labelled too."""
    return NdkPaginator(NdkPaginationConfig(**settings), complete_document=True)


def paginate(values, *, config=None, contexts=None):
    pages = inputs(values, contexts)
    result = PageNumberCoreResult({p.page_key: p.selected for p in pages})
    paginator = config if isinstance(config, NdkPaginator) else NdkPaginator(config)
    return paginator.process(pages, result)


def labels(result):
    return [
        e.output_text() if e is not None else None for e in result.page_numbers.values()
    ]


def test_interior_missing_number_is_plain_and_has_no_false_ocr_provenance():
    result = paginate([10, 11, None, 13, 14])
    assert labels(result) == ["10", "11", "12", "13", "14"]
    supplied = result.page_numbers["p2"]
    assert supplied.supplied and not supplied.bracketed
    assert supplied.text == "" and supplied.bbox is None and supplied.alto == AltoRefs()
    assert supplied.confidence < 0.99
    assert result.decisions["p2"].pagination_rule == "interior_sequence"


def test_extra_leaves_are_bracketed_with_letter_suffixes():
    result = paginate([25, 26, None, None, 27, 28])
    assert labels(result) == ["25", "26", "[26a]", "[26b]", "27", "28"]


def test_mixed_gap_uses_one_of_the_ndk_permitted_conventions():
    result = paginate([15, 16, None, None, None, 18, 19])
    assert labels(result) == ["15", "16", "[16a]", "[16b]", "[17]", "18", "19"]
    assert result.page_numbers["p2"].confidence == 0.55


def test_roman_inserts_preserve_numerals_and_lowercase_suffix_under_case_conversion():
    result = paginate(["XVI", "XVII", None, None, "XVIII", "XIX"])
    assert labels(result) == ["XVI", "XVII", "[XVIIa]", "[XVIIb]", "XVIII", "XIX"]
    assert result.page_numbers["p2"].output_text(case="lowercase") == "[xviia]"
    assert result.page_numbers["p2"].output_text(case="uppercase") == "[XVIIa]"


@pytest.mark.parametrize(
    "values",
    [
        [10, 11, None, 1, 2],  # numbering reset
        ["iii", "iv", None, 1, 2],  # numeral-system transition
        [10, 11, None, 15, 16],  # absent scans cannot be repaired by renumbering
        [10, None, 12],  # too little anchor support
        [None, None, None],  # OCR failure is not proof the volume is unnumbered
        [None, None, 4, 5, 6],  # unknown leading document boundary
        [10, 11, 12, None, None],  # ambiguous trailing continuation vs inserts
    ],
)
def test_ambiguous_cases_remain_unresolved(values):
    result = paginate(values)
    assert all(
        result.page_numbers[f"p{i}"] is None
        for i, value in enumerate(values)
        if value is None
    )


def test_long_gaps_are_not_filled_by_default():
    result = paginate([1, 2] + [None] * 9 + [12, 13])
    assert result.page_numbers["p2"] is None


def test_confirmed_complete_leading_gap_is_bracketed():
    config = edges()
    assert labels(paginate([None, None, None, 4, 5], config=config)) == [
        "[1]",
        "[2]",
        "[3]",
        "4",
        "5",
    ]
    assert labels(paginate([None, None, 1, 2], config=config)) == [
        "[1a]",
        "[1b]",
        "1",
        "2",
    ]


def test_complete_document_does_not_automatically_resolve_ambiguous_tail():
    result = paginate([1, 2, None], config=edges())
    assert labels(result) == ["1", "2", None]


@pytest.mark.parametrize(
    "mode,expected",
    [("continue", ["1", "2", "[3]", "[4]"]), ("insert", ["1", "2", "[2a]", "[2b]"])],
)
def test_confirmed_tail_policies(mode, expected):
    config = edges(tail_mode=mode)
    assert labels(paginate([1, 2, None, None], config=config)) == expected


def test_confirmed_counting_context_resolves_tail_and_excludes_cover():
    contexts = {2: NdkPageContext(counted=True), 3: NdkPageContext(excluded=True)}
    config = edges()
    assert labels(paginate([1, 2, None, None], config=config, contexts=contexts)) == [
        "1",
        "2",
        "[3]",
        None,
    ]


def test_fully_unnumbered_volume_requires_explicit_confirmation():
    config = edges(profile="monograph", wholly_unnumbered=True)
    assert labels(paginate([None, None, None], config=config)) == ["[1]", "[2]", "[3]"]
    with pytest.raises(ValueError, match="conflicts"):
        paginate([1, None], config=config)
    with pytest.raises(ValueError, match="complete_document"):
        NdkPaginator(NdkPaginationConfig(profile="monograph", wholly_unnumbered=True))


def test_context_can_associate_an_insert_with_following_page():
    contexts = {
        2: NdkPageContext(insert_anchor="next"),
        3: NdkPageContext(insert_anchor="next"),
    }
    assert labels(paginate([25, 26, None, None, 27, 28], contexts=contexts))[2:4] == [
        "[27a]",
        "[27b]",
    ]


def test_context_does_not_allow_assigning_extra_leaves_to_counted_pages():
    result = paginate(
        [25, 26, None, 27, 28], contexts={2: NdkPageContext(counted=True)}
    )
    assert result.page_numbers["p2"] is None


def test_known_foldout_segments_share_parent_number_without_consuming_pagination():
    contexts = {
        2: NdkPageContext(segment_of="p1", segment_index=1),
        3: NdkPageContext(segment_of="p1", segment_index=2),
    }
    result = paginate([1, 2, None, None, 3, 4], contexts=contexts)
    assert labels(result) == ["1", "2", "[2a]", "[2b]", "3", "4"]
    assert result.decisions["p2"].pagination_rule == "foldout_segment"


def test_two_scans_are_not_assumed_to_be_segments_without_context():
    result = paginate([None, None])
    assert labels(result) == [None, None]


@pytest.mark.parametrize(
    "contexts",
    [
        {2: NdkPageContext(segment_of="missing", segment_index=1)},
        {2: NdkPageContext(segment_of="p2", segment_index=1)},
        {
            2: NdkPageContext(segment_of="p1", segment_index=1),
            3: NdkPageContext(segment_of="p1", segment_index=1),
        },
    ],
)
def test_invalid_segment_relationships_fail(contexts):
    with pytest.raises(ValueError, match="segment"):
        paginate([1, 2, None, None, 3, 4], contexts=contexts)


def test_unit_boundaries_stop_both_sequence_corrections_and_label_inference():
    contexts = {i: NdkPageContext(unit_key="a" if i < 4 else "b") for i in range(8)}
    pages = inputs([1, 2, None, None, 5, 6, 7, 8], contexts)
    result = PageNumberPostprocessor().process(pages)
    assert result.page_numbers["p2"] is None and result.page_numbers["p3"] is None


def test_full_postprocessor_supplies_after_dropping_outlier_and_keeps_diagnostics():
    pages = list(inputs([10, 11, 12, 13, 999, 15, 16, 17, 18]))
    false = replace(pages[4].selected, bbox=BoundingBox(200, 400, 40, 20))
    pages[4] = replace(pages[4], selected=false, candidates=(false,))
    result = PageNumberPostprocessor().process(pages)
    assert result.page_numbers["p4"].output_text() == "14"
    assert result.page_numbers["p4"].supplied and result.page_numbers["p4"].bbox is None
    assert result.decisions["p4"].observation_action == "dropped"
    assert result.decisions["p4"].original_reading == "999"


def test_ndk_stage_can_be_disabled_independently():
    pages = inputs([10, 11, None, 13, 14])
    config = PageNumberPostprocessConfig(ndk_pagination=False)
    result = PageNumberPostprocessor(config).process(pages)
    assert result.page_numbers["p2"] is None


def test_ascii_suffix_sequence_omits_digraph_and_diacritics():
    assert [suffix(i) for i in range(1, 10)] == list("abcdefghi")
    assert suffix(26) == "z" and suffix(27) == "aa"
    assert "ch" not in {suffix(i) for i in range(1, 100)}


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {"invalid": True},
        {"enabled": True},  # the switch is postprocess.ndk_pagination
        {"profile": "old_print"},
        {"max_gap_pages": 0},
        {"minimum_anchor_confidence": float("nan")},
        {"tail_mode": "guess"},
    ],
)
def test_invalid_ndk_configuration(raw):
    with pytest.raises(ValueError):
        PageNumberPostprocessConfig.from_config({"postprocess": {"ndk": raw}})


def test_insert_suffixes_continue_when_both_sides_refer_to_the_same_number():
    contexts = {
        2: NdkPageContext(insert_anchor="next"),
        4: NdkPageContext(insert_anchor="previous"),
    }
    result = paginate([24, 25, None, 26, None, 27, 28], contexts=contexts)
    assert labels(result)[2] == "[26a]"
    assert labels(result)[4] == "[26b]"


def test_leading_labels_do_not_collide_with_labels_after_page_one():
    contexts = {3: NdkPageContext(segment_of="p1", segment_index=1)}
    result = paginate(
        [None, 1, 2, None, 3, 4],
        config=edges(),
        contexts=contexts,
    )
    assert labels(result) == ["[1aa]", "1", "2", "[1a]", "3", "4"]


def test_identical_labels_in_other_units_do_not_change_leading_convention():
    contexts = {i: NdkPageContext(unit_key="a" if i < 5 else "b") for i in range(8)}
    contexts[3] = replace(contexts[3], segment_of="p0", segment_index=1)
    result = paginate(
        [1, 2, 3, None, 4, None, 1, 2],
        config=edges(),
        contexts=contexts,
    )
    assert result.page_numbers["p3"].output_text() == "[1a]"
    assert result.page_numbers["p5"].output_text() == "[1a]"


def test_known_excluded_scan_does_not_advance_number_inference():
    result = paginate(
        [10, 11, None, None, 13, 14], contexts={2: NdkPageContext(excluded=True)}
    )
    assert labels(result) == ["10", "11", None, "12", "13", "14"]


def test_end_counting_facts_override_profile_and_conflicting_tail_policy():
    config = edges(profile="periodical", tail_mode="continue")
    contexts = {2: NdkPageContext(counted=False)}
    assert labels(paginate([25, 26, None], config=config, contexts=contexts)) == [
        "25",
        "26",
        "[26a]",
    ]
    mixed = {2: NdkPageContext(counted=False), 3: NdkPageContext(counted=True)}
    assert labels(paginate([25, 26, None, None], config=config, contexts=mixed)) == [
        "25",
        "26",
        None,
        None,
    ]


def test_supplied_label_confidence_does_not_exceed_anchor_confidence():
    pages = inputs([10, 11, None, 13, 14])
    numbers = {
        p.page_key: replace(p.selected, confidence=0.65) if p.selected else None
        for p in pages
    }
    result = NdkPaginator().process(pages, PageNumberCoreResult(numbers))
    assert result.page_numbers["p2"].confidence == 0.65


def test_regular_unnumbered_reverse_sides_do_not_destroy_observed_folio_confidence():
    pages = inputs([1, None, 2, None, 3, None, 4, None, 5, None, 6])
    result = PageNumberPostprocessor().process(pages)
    for i in range(0, len(pages), 2):
        assert result.page_numbers[f"p{i}"].confidence >= 0.98
    assert result.page_numbers["p3"].output_text() == "[2a]"
    assert result.page_numbers["p5"].output_text() == "[3a]"
    assert result.page_numbers["p7"].output_text() == "[4a]"


@pytest.mark.parametrize("unnumbered", [False, True])
def test_archival_pagination_is_idempotent(unnumbered):
    pages = inputs([None, None, None] if unnumbered else [10, 11, None, 13, 14])
    config = NdkPaginationConfig(profile="monograph", wholly_unnumbered=unnumbered)
    paginator = NdkPaginator(config, complete_document=unnumbered)
    original = PageNumberCoreResult({p.page_key: p.selected for p in pages})
    once = paginator.process(pages, original)
    assert paginator.process(pages, once) == once


def test_without_brackets_only_plain_numbers_are_supplied():
    config = NdkPaginator(NdkPaginationConfig(tail_mode="insert"), bracketed=False, complete_document=True)
    assert labels(paginate([None, None, 3, 4, None, 6, 7, None, None, 10, 11, None], config=config)) == [
        None, None, "3", "4", "5", "6", "7", "8", "9", "10", "11", None]
    assert labels(paginate([25, 26, None, None, 27, 28], config=config))[2:4] == [None, None]
    assert labels(paginate([15, 16, None, None, None, 18, 19], config=config))[2:5] == [None, None, None]


def numbered_units(values, units):
    pages = [
        replace(p, pagination=NdkPageContext(unit_key=units[i])) for i, p in enumerate(inputs(values))
    ]
    from itertools import groupby
    from metakat.page_number.engines.core.segmentation import PageNumberChunker

    chunks = tuple(
        chunk
        for _, group in groupby(pages, key=lambda p: p.pagination.unit_key)
        for chunk in PageNumberChunker().split(tuple(group), reason="explicit_unit")
    )
    result = PageNumberCoreResult({p.page_key: p.selected for p in pages}, {}, chunks)
    return edges(tail_mode="insert").process(tuple(pages), result)


def test_each_unit_has_its_own_leading_and_trailing_edge():
    result = numbered_units(
        [None, None, 3, 4, 5, None, None, 1, 2, 3, None], ["a"] * 5 + ["b"] * 6
    )
    assert labels(result) == ["[1]", "[2]", "3", "4", "5", "[1a]", "[1b]", "1", "2", "3", "[3a]"]


def test_a_numbering_restart_inside_a_unit_is_not_a_unit_edge():
    values = [None, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, None]
    labels_ = labels(numbered_units(values, ["a"] * len(values)))
    assert labels_[0] == "[10]" and labels_[-1] == "[6a]"
    # nothing is supplied at the restart itself: both runs keep their own numbers
    assert labels_[6:8] == ["16", "1"]

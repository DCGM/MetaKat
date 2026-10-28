from dataclasses import replace
from unittest.mock import Mock

import pytest

from metakat.common.models import BoundingBox, PageDimensions
from metakat.page_number.engines.core.page_number_parsers import (
    DecoratedPageNumberParser,
)
from metakat.page_number.engines.core.postprocessing import (
    PageNumberPostprocessor,
    PageNumberPostprocessConfig,
    PageNumberPostprocessInput,
)


def page(index, number=None, *, x=480, y=950, scale=1, confidence=0.9):
    key = f"p{index}"
    evidence = (
        None
        if number is None
        else DecoratedPageNumberParser.create(
            page_key=key,
            text=str(number),
            confidence=confidence,
            bbox=BoundingBox(x * scale, y * scale, 40 * scale, 20 * scale),
        )
    )
    return PageNumberPostprocessInput(
        key,
        PageDimensions(1000 * scale, 1000 * scale),
        (evidence,) if evidence else (),
        evidence,
    )


def run(pages, **kwargs):
    return PageNumberPostprocessor().process(pages, **kwargs)


def test_recovery_and_position_confidence_ignore_detector_confidence():
    pages = [page(i, i + 10 if i != 3 else None, confidence=0.01) for i in range(9)]
    recovered = page(3, 13).selected
    callback = Mock(
        side_effect=lambda p, windows: [recovered] if p.page_key == "p3" else []
    )
    result = run(pages, recover=callback)
    assert list(result.page_numbers) == [p.page_key for p in pages]
    assert result.page_numbers["p3"].value == 13
    assert result.decisions["p3"].action == "recovered"
    assert result.page_numbers["p4"].confidence >= 0.98
    assert result.page_numbers["p3"].confidence >= 0.98
    assert callback.call_count == len(pages)


def test_normalized_coordinates_allow_different_resolutions():
    result = run([page(i, i + 10, scale=(i % 3 + 1)) for i in range(9)])
    assert all(e.confidence >= 0.98 for e in result.page_numbers.values())


def test_alternating_corners_and_multiple_vertical_modes():
    pages = [
        page(i, i + 10, x=50 if i % 2 else 900, y=30 if i % 2 else 950)
        for i in range(14)
    ]
    result = run(pages)
    assert all(e.confidence >= 0.98 for e in result.page_numbers.values())
    wrong_side = page(6, 16, x=50, y=30)
    pages[6] = wrong_side
    result = run(pages)
    assert result.page_numbers["p6"].supplied
    assert result.page_numbers["p6"].value == 16
    assert result.decisions["p6"].observation_action == "dropped"


def test_outlier_dropped_and_typical_alternative_replaces_it():
    pages = [page(i, i + 10) for i in range(10)]
    pages[5] = page(5, 999, x=300, y=400, confidence=1)
    assert run(pages).decisions["p5"].observation_action == "dropped"
    actual = page(5, 15, confidence=0.02).selected
    result = run(
        pages, recover=lambda p, windows: [actual] if p.page_key == "p5" else []
    )
    assert result.page_numbers["p5"].value == 15
    assert result.page_numbers["p5"].confidence >= 0.98


def test_obvious_sequence_error_corrected_without_altering_source_text():
    pages = [page(i, i + 10) for i in range(9)]
    pages[4] = page(4, 99)
    result = run(pages)
    corrected = result.page_numbers["p4"]
    assert corrected.value == 14
    assert corrected.normalized == "14"
    assert corrected.text == "99"
    assert corrected.bbox == pages[4].selected.bbox
    assert corrected.confidence <= 0.85
    assert result.decisions["p4"].action == "corrected"
    assert result.decisions["p4"].original_reading == "99"
    assert pages[4].selected.value == 99


def test_sequence_prefers_attested_alternative_over_inference():
    pages = [page(i, i + 10) for i in range(9)]
    wrong = page(4, 99).selected
    right = page(4, 14).selected
    pages[4] = replace(pages[4], selected=wrong, candidates=(wrong, right))
    result = run(pages)
    assert result.page_numbers["p4"].text == "14"
    assert result.decisions["p4"].action != "corrected"


@pytest.mark.parametrize(
    "numbers",
    [
        [1, 2, 3, 4, 1, 2, 3, 4, 5],  # restart
        [1, 2, 3, 4, None, 5, 6, 7, 8],  # unnumbered insertion
        [1, 2, 3, 4, 6, 7, 8, 9, 10],  # missing scan
        ["i", "ii", "iii", "iv", "1", "2", "3", "4", "5"],
    ],
)
def test_resets_insertions_missing_scans_and_numeral_transitions_preserved(numbers):
    pages = [page(i, number) for i, number in enumerate(numbers)]
    result = run(pages)
    for p, number in zip(pages, numbers):
        evidence = result.page_numbers[p.page_key]
        assert (evidence.output_text() if evidence else None) == (
            str(number) if number else "[4a]"
        )


def test_blank_page_gets_a_supplied_label_without_fabricated_ocr_or_geometry():
    pages = [page(i, i + 10 if i != 4 else None) for i in range(9)]
    evidence = run(pages).page_numbers["p4"]
    assert evidence.supplied and evidence.value == 14
    assert evidence.text == "" and evidence.bbox is None


def test_short_document_does_not_claim_certainty_or_drop_unique_layout():
    pages = [page(0, 10), page(1, 99, y=400), page(2)]
    result = run(pages)
    assert result.page_numbers["p1"].value == 99
    assert result.page_numbers["p1"].confidence == 0.5
    assert result.page_numbers["p2"] is None


def test_roman_sequence_correction_preserves_case():
    pages = [
        page(i, value)
        for i, value in enumerate(
            ["i", "ii", "iii", "iv", "xx", "vi", "vii", "viii", "ix"]
        )
    ]
    result = run(pages)
    assert result.page_numbers["p4"].output_text() == "v"
    assert result.page_numbers["p4"].text == "xx"


def test_disabled_preserves_original_confidence():
    pages = [page(0, 10), page(1)]
    recovery = Mock()
    result = PageNumberPostprocessor(
        PageNumberPostprocessConfig(enabled=False)
    ).process(pages, recover=recovery)
    assert result.page_numbers == {"p0": pages[0].selected, "p1": None}
    recovery.assert_not_called()


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {"unknown": 1},
        {"enabled": 1},
        {"minimum_support": 2},
        {"position_bandwidth_x": float("nan")},
        {"drop_threshold": -1},
        {"sequence_window": True},
    ],
)
def test_invalid_config(raw):
    with pytest.raises(ValueError):
        PageNumberPostprocessConfig.from_config({"postprocess": raw})


def test_duplicate_pages_and_foreign_candidates_rejected():
    with pytest.raises(ValueError, match="unique"):
        run([page(0), page(0)])
    with pytest.raises(ValueError, match="page key"):
        run([replace(page(0), candidates=(page(1, 10).selected,))])


def test_outside_and_nonfinite_geometry_not_selected():
    pages = [page(i, i + 10) for i in range(9)]
    pages[4] = page(4, 14, x=float("nan"))
    evidence = run(pages).page_numbers["p4"]
    assert evidence.supplied and evidence.value == 14
    assert evidence.text == "" and evidence.bbox is None
    pages[4] = page(4, 14, x=990)
    evidence = run(pages).page_numbers["p4"]
    assert evidence.supplied and evidence.value == 14
    assert evidence.text == "" and evidence.bbox is None


def test_empty_document():
    assert run([]).page_numbers == {}


def test_many_anchors_do_not_make_tail_positions_certain():
    pages = [page(i, i + 10) for i in range(40)]
    pages[20] = page(20, 30, y=900)  # two vertical bandwidths from the mode
    result = run(pages)
    assert 0.1 < result.decisions["p20"].position_support < 0.2
    assert result.page_numbers["p20"].confidence < 0.5


def test_local_distribution_supports_layout_changes():
    pages = [page(i, i + 10, y=30 if i < 50 else 950) for i in range(100)]
    result = run(pages)
    assert all(e is not None for e in result.page_numbers.values())
    assert result.page_numbers["p10"].confidence >= 0.98
    assert result.page_numbers["p90"].confidence >= 0.98


def test_recovery_cannot_create_a_self_reinforcing_position_model():
    pages = [page(i, i + 10) if i < 5 else page(i) for i in range(10)]

    def recover(p, windows):
        return [page(int(p.page_key[1:]), 99, x=200, y=400).selected]

    result = run(pages, recover=recover)
    assert all(result.page_numbers[f"p{i}"] is None for i in range(5, 10))


def test_an_isolated_error_at_end_is_not_extrapolated():
    pages = [page(i, i + 10) for i in range(9)]
    pages[-1] = page(8, 99)
    assert run(pages).page_numbers["p8"].value == 99


def test_expected_number_in_body_does_not_displace_correctable_folio():
    pages = [page(i, i + 10) for i in range(9)]
    wrong = page(4, 99).selected
    body = page(4, 14, y=400).selected
    pages[4] = replace(pages[4], selected=wrong, candidates=(wrong, body))
    result = run(pages)
    assert result.page_numbers["p4"].value == 14
    assert result.page_numbers["p4"].bbox == wrong.bbox
    assert result.decisions["p4"].action == "corrected"


def test_half_of_document_can_be_recovered_without_yolo_detections():
    pages = [page(i, i + 1 if i % 2 == 0 else None) for i in range(24)]
    result = run(
        pages,
        recover=lambda p, windows: [
            page(int(p.page_key[1:]), int(p.page_key[1:]) + 1).selected
        ],
    )
    assert [e.value for e in result.page_numbers.values()] == list(range(1, 25))
    assert sum(d.action == "recovered" for d in result.decisions.values()) == 12


def test_repeating_running_header_number_is_not_confident_as_a_folio():
    result = run([page(i, 1948) for i in range(12)])
    assert all(
        e.value == 1948 and e.confidence <= 0.35 for e in result.page_numbers.values()
    )
    assert all(d.sequence_support == 0 for d in result.decisions.values())

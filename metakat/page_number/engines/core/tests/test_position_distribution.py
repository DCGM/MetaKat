"""Review-confidence regressions for independently established number positions."""

from random import Random

import pytest

from metakat.common.models import BoundingBox, PageDimensions
from metakat.page_number.engines.core.ndk_pagination import NdkPaginationConfig
from metakat.page_number.engines.core.page_number_parsers import (
    DecoratedPageNumberParser,
)
from metakat.page_number.engines.core.postprocessing import (
    PageNumberPostprocessConfig,
    PageNumberPostprocessor,
    PageNumberPostprocessInput,
)

LOCATIONS = {"left": 50, "center": 480, "right": 900}


def make_page(i, location=None, *, displacement=0):
    key = f"p{i}"
    evidence = (
        None
        if location is None
        else DecoratedPageNumberParser.create(
            page_key=key,
            text=str(i + 1),
            confidence=0.01,
            bbox=BoundingBox(LOCATIONS[location] + displacement, 950, 40, 20),
        )
    )
    return PageNumberPostprocessInput(
        key, PageDimensions(1000, 1000), (evidence,) if evidence else (), evidence
    )


def process(pages, *, recover=None, **config):
    return PageNumberPostprocessor(
        PageNumberPostprocessConfig(
            ndk=NdkPaginationConfig(enabled=False),
            **config,
        )
    ).process(pages, recover=recover)


def shuffled_layout(left, right, center=0, missing=0):
    locations = (
        ["left"] * left + ["right"] * right + ["center"] * center + [None] * missing
    )
    Random(37).shuffle(locations)
    return locations


def test_thirty_left_and_forty_right_are_both_firm_despite_thirty_missing():
    locations = shuffled_layout(30, 40, missing=30)
    result = process([make_page(i, location) for i, location in enumerate(locations)])
    for i, location in enumerate(locations):
        if location is None:
            assert result.page_numbers[f"p{i}"] is None
        else:
            assert result.decisions[f"p{i}"].position_support == pytest.approx(1)
            assert result.page_numbers[f"p{i}"].confidence >= 0.98


def test_recovered_and_original_readings_at_either_firm_location_are_equally_supported():
    locations = shuffled_layout(30, 40, missing=30)
    pages = [make_page(i, location) for i, location in enumerate(locations)]
    targets = [i for i, location in enumerate(locations) if location is None][:2]
    recovered = {
        f"p{index}": make_page(index, location).selected
        for index, location in zip(targets, ("left", "right"))
    }
    result = process(
        pages,
        recover=lambda page, windows: (
            [recovered[page.page_key]] if page.page_key in recovered else []
        ),
    )
    for key in recovered:
        assert result.decisions[key].position_support == pytest.approx(1)
        assert result.decisions[key].action == "recovered"
        assert result.page_numbers[key].confidence >= 0.98


def test_frequent_right_cluster_does_not_reduce_left_cluster_support():
    positions = shuffled_layout(20, 20, missing=60)
    baseline = [make_page(i, location) for i, location in enumerate(positions)]
    more_right = [
        make_page(i, "right" if location is None else location)
        for i, location in enumerate(positions)
    ]
    before, after = process(baseline), process(more_right)
    for i, location in enumerate(positions):
        if location == "left":
            assert before.decisions[f"p{i}"].position_support == pytest.approx(1)
            assert after.decisions[f"p{i}"].position_support == pytest.approx(1)


def test_only_two_locations_can_be_firm_even_if_three_are_tightly_clustered():
    locations = shuffled_layout(30, 40, center=20)
    result = process([make_page(i, location) for i, location in enumerate(locations)])
    for i, location in enumerate(locations):
        decision = result.decisions[f"p{i}"]
        evidence = result.page_numbers[f"p{i}"]
        if location == "center":
            assert decision.position_support <= 0.65
            assert evidence.confidence <= 0.805
        else:
            assert decision.position_support == pytest.approx(1)
            assert evidence.confidence >= 0.98


def test_third_location_is_still_searched_but_recovery_cannot_make_it_firm():
    locations = shuffled_layout(30, 40, center=20, missing=5)
    pages = [make_page(i, location) for i, location in enumerate(locations)]
    target = next(i for i, location in enumerate(locations) if location is None)
    recovered = make_page(target, "center").selected
    searched = []

    def recover(page, windows):
        if page.page_key == recovered.page_key:
            searched.extend(windows)
            return [recovered]
        return []

    result = process(pages, recover=recover)
    assert any(box.x <= 500 <= box.x_max for box in searched)
    assert result.decisions[recovered.page_key].action == "recovered"
    assert result.decisions[recovered.page_key].position_support <= 0.65
    assert result.page_numbers[recovered.page_key].confidence <= 0.805


def test_firm_count_requires_independent_peers_and_can_be_configured():
    pages = [make_page(i, "center") for i in range(6)]
    default = process(pages)
    stricter = process(pages, firm_support_count=10)
    assert all(
        d.position_support == pytest.approx(1) for d in default.decisions.values()
    )
    assert all(d.position_support < 1 for d in stricter.decisions.values())
    small = process(pages[:4])
    assert all(
        d.position_support == pytest.approx(0.6) for d in small.decisions.values()
    )


def test_diffuse_cluster_remains_uncertain_even_when_numerous():
    displacements = [-36, -24, -12, 0, 12, 24, 36] * 10
    result = process(
        [make_page(i, "center", displacement=x) for i, x in enumerate(displacements)]
    )
    assert all(d.position_support < 1 for d in result.decisions.values())
    assert max(e.confidence for e in result.page_numbers.values()) < 0.9


def test_tighter_cluster_scores_higher_with_the_same_number_of_detections():
    narrow = [-4, -2, 0, 2, 4] * 4
    broad = [-36, -18, 0, 18, 36] * 4
    tight_result = process(
        [make_page(i, "center", displacement=x) for i, x in enumerate(narrow)]
    )
    broad_result = process(
        [make_page(i, "center", displacement=x) for i, x in enumerate(broad)]
    )
    # Compare detections at the center, so proximity itself does not explain
    # the difference: only their surrounding cluster dispersion changes.
    for i in range(2, 20, 5):
        assert (
            tight_result.decisions[f"p{i}"].position_support
            > broad_result.decisions[f"p{i}"].position_support
        )


def test_one_displaced_detection_does_not_spoil_a_firm_location():
    pages = [make_page(i, "center") for i in range(20)]
    pages[-1] = make_page(19, "center", displacement=80)
    result = process(pages)
    assert all(
        result.decisions[f"p{i}"].position_support == pytest.approx(1)
        for i in range(19)
    )
    assert result.decisions["p19"].position_support < 0.2


def test_firm_limit_applies_across_parity_not_separately_to_each_parity():
    # Even scans have a large left mode and a smaller center mode; odd scans
    # have the right mode. A separate two-mode allowance per parity would
    # incorrectly make all three certain.
    locations = [
        "right" if i % 2 else "center" if i % 6 == 0 else "left" for i in range(120)
    ]
    result = process([make_page(i, location) for i, location in enumerate(locations)])
    for i, location in enumerate(locations):
        if location == "center":
            assert result.decisions[f"p{i}"].position_support <= 0.65


def test_can_limit_firm_positions_to_one():
    locations = shuffled_layout(20, 40)
    result = process(
        [make_page(i, location) for i, location in enumerate(locations)],
        maximum_firm_positions=1,
    )
    for i, location in enumerate(locations):
        if location == "left":
            assert result.decisions[f"p{i}"].position_support <= 0.65


@pytest.mark.parametrize(
    "config",
    [
        {"maximum_firm_positions": 3},
        {"maximum_firm_positions": True},
        {"firm_support_count": 2},
        {"firm_support_count": True},
        {"firm_cluster_spread": 0},
        {"firm_cluster_spread": float("nan")},
        {"additional_position_support_cap": 1},
        {"additional_position_support_cap": -1},
    ],
)
def test_invalid_firm_position_config(config):
    with pytest.raises(ValueError):
        PageNumberPostprocessConfig(**config)


def test_firm_position_remains_supported_after_a_long_gap_without_detections():
    locations = ["left"] * 30 + ["right"] * 40 + [None] * 30
    pages = [make_page(i, location) for i, location in enumerate(locations)]
    recovered = make_page(99, "left").selected
    result = process(
        pages,
        recover=lambda page, windows: [recovered] if page.page_key == "p99" else [],
    )
    assert result.decisions["p99"].position_support == pytest.approx(1)
    assert result.page_numbers["p99"].confidence >= 0.98

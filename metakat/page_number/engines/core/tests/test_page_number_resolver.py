import logging

import pytest

from metakat.common.models import BoundingBox
from metakat.page_number.engines.core.page_number_parsers import (
    DecoratedPageNumberParser,
)
from metakat.page_number.engines.core.page_number_resolver import (
    PageNumberSelectionMode,
    PhysicalPageNumberResolver,
)

RESOLVER_LOGGER = "metakat.page_number.engines.core.page_number_resolver"


def _resolve(page, *, mode=PageNumberSelectionMode.STANDARD):
    candidates = tuple(
        evidence
        for region in page.regions
        if (
            evidence := DecoratedPageNumberParser.parse_region(
                page_key=page.page_key,
                region=region,
            )
        )
        is not None
    )
    return PhysicalPageNumberResolver().resolve(
        candidates,
        page_width=page.alto_width,
        page_height=page.alto_height,
        mode=mode,
    )


def test_single_interior_candidate_is_retained(alignment_page):
    selected = _resolve(alignment_page([("42", 0.7, 490)]))

    assert (selected.output_text(), selected.confidence) == ("42", 0.7)


def test_candidates_must_belong_to_the_same_page():
    candidates = tuple(
        DecoratedPageNumberParser.create(
            page_key=page_key,
            text=text,
            confidence=0.9,
            bbox=BoundingBox(100, 20, 50, 20),
        )
        for page_key, text in (("page-1", "1"), ("page-2", "2"))
    )

    with pytest.raises(
        ValueError,
        match="requires all candidates to belong to the same page",
    ):
        PhysicalPageNumberResolver().resolve(
            candidates,
            page_width=800,
            page_height=1000,
        )


@pytest.mark.parametrize(
    "edge,geometry",
    (
        ("left", {"x": -1}),
        ("top", {"y": -1}),
        ("right", {"x": 751}),
        ("bottom", {"y": 981}),
    ),
)
def test_candidate_bbox_must_be_fully_inside_page(
    alignment_page,
    caplog,
    edge,
    geometry,
):
    with caplog.at_level(logging.WARNING, logger=RESOLVER_LOGGER):
        selected = _resolve(
            alignment_page(
                [("42", 0.7, geometry.get("y", 20))],
                x=geometry.get("x", 100),
            )
        )

    assert [
        record for record in caplog.records if record.name.startswith(RESOLVER_LOGGER)
    ]
    assert selected is None


def test_missing_page_width_fails(alignment_page):
    with pytest.raises(ValueError, match="requires a finite positive page width"):
        _resolve(alignment_page([("42", 0.7, 20)], alto_width=None))


def test_multiple_candidates_prefer_edge_over_confident_interior(alignment_page):
    selected = _resolve(
        alignment_page(
            [
                ("12", 0.6, 20),
                ("99", 0.99, 490),
            ]
        )
    )

    assert (selected.output_text(), selected.confidence) == ("12", 0.6)


def test_multiple_interior_candidates_leave_number_unresolved(alignment_page, caplog):
    with caplog.at_level(logging.WARNING, logger=RESOLVER_LOGGER):
        selected = _resolve(
            alignment_page(
                [
                    ("12", 0.8, 300),
                    ("13", 0.9, 600),
                ]
            )
        )

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name.startswith(RESOLVER_LOGGER)
    ]
    assert messages
    assert selected is None
    assert "leaving page number unresolved" in " ".join(messages)


def test_multiple_edge_candidates_use_position_and_confidence_score(alignment_page):
    selected = _resolve(
        alignment_page(
            [
                ("12", 0.6, 10),
                ("13", 0.99, 920),
            ]
        )
    )

    assert (selected.output_text(), selected.confidence) == ("12", 0.6)


def test_missing_page_height_fails_when_selection_requires_geometry(alignment_page):
    with pytest.raises(ValueError, match="requires a finite positive page height"):
        _resolve(
            alignment_page(
                [
                    ("12", 0.6, 20),
                    ("13", 0.9, 490),
                ],
                alto_height=None,
            )
        )


def test_edge_only_missing_page_height_fails(alignment_page):
    with pytest.raises(ValueError, match="requires a finite positive page height"):
        _resolve(
            alignment_page([("42", 0.9, 20)], alto_height=None),
            mode=PageNumberSelectionMode.EDGE_ONLY,
        )


def test_edge_only_rejects_single_interior_candidate(alignment_page):
    selected = _resolve(
        alignment_page([("42", 0.9, 490)]),
        mode=PageNumberSelectionMode.EDGE_ONLY,
    )

    assert selected is None

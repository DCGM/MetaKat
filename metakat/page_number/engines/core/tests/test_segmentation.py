from dataclasses import replace

import pytest

from metakat.common.models import BoundingBox, PageDimensions
from metakat.page_number.engines.core.ndk_pagination import (
    NdkPageContext,
    NdkPaginationConfig,
)
from metakat.page_number.engines.core.page_number_parsers import (
    DecoratedPageNumberParser,
)
from metakat.page_number.engines.core.postprocessing import (
    PageNumberPostprocessor,
    PageNumberPostprocessConfig,
    PageNumberPostprocessInput,
)
from metakat.page_number.engines.core.segmentation import (
    PageNumberChunker,
    PageNumberSegmentationConfig,
)


def pages(values, *, x=480, y=950, offset=0):
    result = []
    for i, number in enumerate(values, offset):
        key = f"p{i}"
        e = (
            None
            if number is None
            else DecoratedPageNumberParser.create(
                page_key=key,
                text=str(number),
                confidence=0.01,
                bbox=BoundingBox(x, y, 40, 20),
            )
        )
        result.append(
            PageNumberPostprocessInput(
                key, PageDimensions(1000, 1000), (e,) if e else (), e
            )
        )
    return result


def boundaries(chunks):
    return [chunk.page_keys[0] for chunk in chunks]


def test_restart_isolates_position_distributions_including_smaller_second_book():
    first = pages(range(90, 102), x=900, y=950)
    second = pages(range(1, 7), x=50, y=20, offset=12)
    result = PageNumberPostprocessor().process(first + second)
    assert boundaries(result.chunks) == ["p0", "p12"]
    assert result.chunks[1].reason == "numbering_restart"
    assert [e.value for e in result.page_numbers.values()] == list(
        range(90, 102)
    ) + list(range(1, 7))
    assert all(
        not e.supplied and e.confidence >= 0.98 for e in result.page_numbers.values()
    )
    assert result.decisions["p12"].chunk_index == 1


def test_roman_to_arabic_is_a_processing_boundary_without_claiming_a_new_book():
    batch = pages(["i", "ii", "iii", "iv", "v", "vi"], y=20)
    batch += pages(range(1, 8), y=950, offset=6)
    result = PageNumberPostprocessor().process(batch)
    assert boundaries(result.chunks) == ["p0", "p6"]
    assert result.chunks[1].reason == "numeral_system_change"
    assert result.chunks[1].inferred
    assert all(e.confidence >= 0.98 for e in result.page_numbers.values())


def test_large_forward_change_is_split_but_a_single_missing_scan_is_not():
    assert boundaries(PageNumberChunker().split(pages([1, 2, 3, 40, 41, 42]))) == [
        "p0",
        "p3",
    ]
    chunks = PageNumberChunker().split(pages([1, 2, 3, 5, 6, 7]))
    assert len(chunks) == 1
    assert not chunks[0].inferred


@pytest.mark.parametrize("error", [99, 1, "iv"])
def test_single_bad_reading_does_not_create_a_document(error):
    batch = pages([10, 11, 12, 13, error, 15, 16, 17, 18])
    assert len(PageNumberChunker().split(batch)) == 1


def test_sequence_correction_still_operates_inside_unsplit_document():
    result = PageNumberPostprocessor().process(
        pages([10, 11, 12, 13, 99, 15, 16, 17, 18])
    )
    assert len(result.chunks) == 1
    assert result.page_numbers["p4"].value == 14
    assert result.decisions["p4"].action == "corrected"


@pytest.mark.parametrize(
    "values",
    [
        [1, 2, 3, None, None, 4, 5, 6],  # inserted leaves
        [1, None, 2, None, 3, None, 4, None, 5, None, 6],  # regular unnumbered reverses
        [1, 2, 3, None, 5, 6, 7],  # missed detection
        [2, 4, 6, 8, 10, 12],  # consistent two-number steps
        [1948] * 10,  # repeated header years provide no progressing runs
    ],
)
def test_insertions_gaps_and_repeated_headers_do_not_create_false_units(values):
    assert len(PageNumberChunker().split(pages(values))) == 1


def test_ambiguous_unnumbered_border_is_quarantined_even_with_ndk_edge_policy():
    batch = pages([90, 91, 92, 93, 94, 95, None, None, 1, 2, 3, 4, 5, 6])
    config = PageNumberPostprocessConfig(
        ndk=NdkPaginationConfig(complete_document=True, tail_mode="insert")
    )
    result = PageNumberPostprocessor(config).process(batch)
    assert boundaries(result.chunks) == ["p0", "p6", "p8"]
    assert result.chunks[1].ambiguous
    assert result.chunks[1].reason == "uncertain_boundary"
    for key in ("p6", "p7"):
        assert result.page_numbers[key] is None
        assert result.decisions[key].action == "boundary_uncertain"
    assert all(e is None or not e.supplied for e in result.page_numbers.values())


def test_unsupported_reading_between_regimes_is_kept_only_at_review_confidence():
    result = PageNumberPostprocessor().process(
        pages([90, 91, 92, 93, 94, 95, 99, None, 1, 2, 3, 4, 5, 6])
    )
    assert result.chunks[1].page_keys == ("p6", "p7")
    assert result.page_numbers["p6"].value == 99
    assert result.page_numbers["p6"].confidence == 0.35
    assert result.decisions["p6"].action == "boundary_uncertain"


def test_ndk_fills_only_within_its_chunk():
    batch = pages([10, 11, None, 13, 14, 15, 1, 2, None, 4, 5, 6])
    result = PageNumberPostprocessor().process(batch)
    assert boundaries(result.chunks) == ["p0", "p6"]
    assert result.page_numbers["p2"].output_text() == "12"
    assert result.page_numbers["p8"].output_text() == "3"
    assert result.page_numbers["p2"].supplied and result.page_numbers["p8"].supplied


def test_recovery_windows_use_only_the_corresponding_book_layout():
    batch = pages(range(90, 98), y=950)
    batch += pages([1, 2, None, 4, 5, 6, 7, 8], y=20, offset=8)
    recovery = pages([3], y=20, offset=10)[0].selected
    windows_seen = {}

    def recover(page, windows):
        windows_seen[page.page_key] = windows
        return [recovery] if page.page_key == "p10" else []

    result = PageNumberPostprocessor().process(batch, recover=recover)
    assert len(result.chunks) == 2
    assert all(
        box.y > 800
        for key, windows in windows_seen.items()
        if int(key[1:]) < 8
        for box in windows
    )
    assert all(
        box.y_max < 200
        for key, windows in windows_seen.items()
        if int(key[1:]) >= 8
        for box in windows
    )
    assert result.page_numbers["p10"].value == 3
    assert result.decisions["p10"].action == "recovered"


def test_explicit_unit_boundaries_are_kept_even_if_numbering_is_continuous():
    batch = pages(range(1, 13))
    batch = [
        replace(p, pagination=NdkPageContext(unit_key="a" if i < 6 else "b"))
        for i, p in enumerate(batch)
    ]
    result = PageNumberPostprocessor().process(batch)
    assert boundaries(result.chunks) == ["p0", "p6"]
    assert all(
        not chunk.inferred and chunk.reason == "explicit_unit"
        for chunk in result.chunks
    )
    assert [chunk.unit_key for chunk in result.chunks] == ["a", "b"]


def test_known_segment_in_border_is_attached_to_its_parent():
    batch = pages([90, 91, 92, 93, None, 1, 2, 3, 4])
    batch[4] = replace(
        batch[4], pagination=NdkPageContext(segment_of="p3", segment_index=1)
    )
    result = PageNumberPostprocessor().process(batch)
    assert boundaries(result.chunks) == ["p0", "p5"]
    assert result.page_numbers["p4"].output_text() == "[93a]"


def test_known_same_page_relationship_prevents_an_incompatible_cut():
    batch = pages([90, 91, 92, 93, 1, 2, 3, 4, None])
    batch[-1] = replace(
        batch[-1], pagination=NdkPageContext(segment_of="p3", segment_index=1)
    )
    assert len(PageNumberChunker().split(batch)) == 1


def test_sparse_anchors_still_support_numbering_restart():
    batch = pages([90, None, 92, None, 94, None, None, 1, None, 3, None, 5])
    chunks = PageNumberChunker().split(batch)
    assert boundaries(chunks) == ["p0", "p5", "p7"]
    assert chunks[1].ambiguous


def test_insufficient_evidence_does_not_assert_a_boundary():
    assert len(PageNumberChunker().split(pages([90, 91, 1, 2]))) == 1


def test_chunker_preserves_every_page_exactly_once():
    batch = pages([90, 91, 92, None, 1, 2, 3, None, "i", "ii", "iii"])
    chunks = PageNumberChunker().split(batch)
    assert [key for chunk in chunks for key in chunk.page_keys] == [
        p.page_key for p in batch
    ]


def test_segmentation_can_be_disabled_independently():
    config = PageNumberPostprocessConfig(
        segmentation=PageNumberSegmentationConfig(enabled=False)
    )
    result = PageNumberPostprocessor(config).process(pages([90, 91, 92, 1, 2, 3]))
    assert len(result.chunks) == 1
    assert not result.chunks[0].inferred


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {"unknown": True},
        {"enabled": 1},
        {"minimum_run_anchors": 2},
        {"maximum_anchor_gap": 0},
        {"minimum_offset_change": -1},
        {"maximum_number_step": True},
    ],
)
def test_invalid_segmentation_config(raw):
    with pytest.raises(ValueError):
        PageNumberPostprocessConfig.from_config({"postprocess": {"segmentation": raw}})


def test_empty_input_has_no_chunks():
    assert PageNumberChunker().split([]) == ()
    assert PageNumberPostprocessor().process([]).chunks == ()


def test_ndk_reuse_respects_existing_chunks_and_uncertain_boundary():
    from metakat.page_number.engines.core.ndk_pagination import NdkPaginator

    batch = pages([90, 91, 92, 93, 94, 95, None, None, 1, 2, 3, 4, 5, 6])
    result = PageNumberPostprocessor().process(batch)
    paginated = NdkPaginator(
        NdkPaginationConfig(complete_document=True, tail_mode="insert")
    ).process(tuple(batch), result)
    assert paginated == result


def test_explicit_unit_can_refine_into_numbering_regimes_without_losing_identity():
    batch = [
        replace(p, pagination=NdkPageContext(unit_key="known-volume"))
        for p in pages([10, 11, 12, 1, 2, 3])
    ]
    result = PageNumberPostprocessor().process(batch)
    assert len(result.chunks) == 2
    assert all(chunk.unit_key == "known-volume" for chunk in result.chunks)
    assert result.chunks[1].reason == "numbering_restart"


def test_invalid_geometry_and_supplied_numbers_cannot_anchor_a_boundary():
    batch = pages([90, 91, 92, 1, 2, 3])
    batch = [
        replace(p, selected=replace(p.selected, supplied=True)) if i >= 3 else p
        for i, p in enumerate(batch)
    ]
    assert len(PageNumberChunker().split(batch)) == 1
    batch = pages([90, 91, 92, 1, 2, 3])
    batch[4] = replace(
        batch[4], selected=replace(batch[4].selected, bbox=BoundingBox(-1, 20, 40, 20))
    )
    assert len(PageNumberChunker().split(batch)) == 1


def test_chunker_rejects_duplicate_keys():
    with pytest.raises(ValueError, match="unique"):
        PageNumberChunker().split(pages([1]) + pages([2]))

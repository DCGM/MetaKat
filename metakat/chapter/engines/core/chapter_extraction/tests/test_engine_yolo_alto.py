import logging
from pathlib import Path

import pytest
from text_geometry_aligner import (
    AlignmentRegion,
    BoundingBox as AlignmentBoundingBox,
)

from metakat.chapter.engines.core.chapter_extraction.engine_yolo_alto import (
    ChapterExtractionEngineYOLOALTO,
)
from metakat.chapter.engines.core.models import ChapterPageInput, TocBase
from metakat.common.models import BoundingBox, PageDimensions
from metakat.schemas.base_objects import ChapterType

EXTRACTION_LOGGER = (
    "metakat.chapter.engines.core.chapter_extraction.engine_yolo_alto"
)


def _geometry_only_region(
    region_id,
    label,
    x,
    y,
    confidence=0.9,
    width=100,
    height=20,
):
    """A detection with no matched text, used to prove columns need only geometry."""
    return AlignmentRegion(
        region_id=region_id,
        label=label,
        input_geometry=AlignmentBoundingBox(x, y, width, height),
        input_geometry_confidence=confidence,
    )


def _toc_page(page_key="toc", position=0, *, width=1000, height=1000):
    return ChapterPageInput(
        page_key,
        position,
        Path(f"{page_key}.jpg"),
        Path(f"{page_key}.xml"),
        image_dimensions=PageDimensions(width, height),
    )


def _log_output(caplog):
    return "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.name.startswith(EXTRACTION_LOGGER)
    )


@pytest.fixture
def extraction_engine(
    tmp_path,
    write_engine_config,
    read_engine_config,
    fake_alignment_engine,
):
    """Build the extraction engine over a config directory and fake aligner."""

    def _build(config=None, alignment_pages=()):
        write_engine_config(
            tmp_path,
            {
                "name": "chapter_extraction_engine_yolo_alto",
                **(config or {}),
            },
        )
        return ChapterExtractionEngineYOLOALTO(
            read_engine_config(tmp_path),
            alignment_engine=fake_alignment_engine(list(alignment_pages)),
        )

    return _build


def test_uses_same_chapter_type_keys_for_shared_labels(extraction_engine):
    engine = extraction_engine(
        {
            "labels": {
                "Level1Title": "primary",
                "Level2Title": "secondary",
                "Subtitle": "subtitle",
                "PageNumber": "page",
                "PartNumber": "part",
            },
        }
    )

    assert engine.labels == {
        ChapterType.LEVEL_1_TITLE: "primary",
        ChapterType.LEVEL_2_TITLE: "secondary",
        ChapterType.SUBTITLE: "subtitle",
        ChapterType.PAGE_NUMBER: "page",
        ChapterType.PART_NUMBER: "part",
    }


def test_assigns_subtitles_with_configured_geometry_guards(
    extraction_engine,
    alignment_page,
    region,
):
    engine = extraction_engine(
        {
            "subtitle_max_vertical_gap_height_multiplier": 1.5,
            "subtitle_max_vertical_overlap_height_fraction": 0.25,
            "subtitle_min_horizontal_overlap_fraction": 0.5,
        },
        [
            alignment_page(
                "toc",
                [
                    region(0, "kapitola", "Overlap", 100, 10, height=20),
                    region(1, "podnadpis", "Overlap subtitle", 120, 28),
                    region(2, "kapitola", "Too far", 100, 100, height=20),
                    region(3, "podnadpis", "Distant subtitle", 120, 160),
                    region(4, "kapitola", "No horizontal", 100, 220),
                    region(5, "podnadpis", "Marginal text", 400, 245),
                    region(6, "kapitola", "Assigned", 100, 300, width=200),
                    region(7, "podnadpis", "Assigned subtitle", 250, 330),
                ],
            )
        ],
    )

    result = engine.process((_toc_page(),))

    assert engine.subtitle_max_vertical_gap_height_multiplier == 1.5
    assert engine.subtitle_max_vertical_overlap_height_fraction == 0.25
    assert engine.subtitle_min_horizontal_overlap_fraction == 0.5
    chapters = {chapter.title.text: chapter for chapter in result.chapters}
    assert chapters["Overlap"].subtitle.text == "Overlap subtitle"
    assert chapters["Too far"].subtitle is None
    assert chapters["No horizontal"].subtitle is None
    assert chapters["Assigned"].subtitle.text == "Assigned subtitle"


def test_subtitles_are_partitioned_with_their_multicolumn_titles(
    extraction_engine,
    alignment_page,
    region,
):
    regions = []
    region_id = 0
    for prefix, title_x, number_x in (
        ("Left", 50, 430),
        ("Right", 550, 930),
    ):
        for position, y in enumerate((10, 60, 110), start=1):
            regions.extend(
                (
                    region(
                        region_id,
                        "kapitola",
                        f"{prefix} {position}",
                        title_x,
                        y,
                        width=200,
                    ),
                    region(
                        region_id + 1,
                        "cislo strany",
                        str(position),
                        number_x,
                        y,
                        width=20,
                    ),
                )
            )
            region_id += 2
        regions.append(
            region(
                region_id,
                "podnadpis",
                f"{prefix} subtitle",
                title_x + 10,
                32,
                width=180,
                height=12,
            )
        )
        region_id += 1
    engine = extraction_engine(alignment_pages=[alignment_page("toc", regions)])

    result = engine.process((_toc_page(),))

    chapters = {chapter.title.text: chapter for chapter in result.chapters}
    assert chapters["Left 1"].subtitle.text == "Left subtitle"
    assert chapters["Right 1"].subtitle.text == "Right subtitle"


def test_units_claim_best_available_subtitle_in_reading_order(
    extraction_engine,
    alignment_page,
    region,
):
    engine = extraction_engine(
        alignment_pages=[
            alignment_page(
                "toc",
                [
                    region(0, "kapitola", "Earlier", 100, 10),
                    region(1, "kapitola", "Later", 100, 20),
                    region(2, "podnadpis", "First subtitle", 110, 42),
                    region(3, "podnadpis", "Second subtitle", 110, 55),
                ],
            )
        ]
    )

    result = engine.process((_toc_page(),))

    chapters = {chapter.title.text: chapter for chapter in result.chapters}
    assert chapters["Earlier"].subtitle.text == "First subtitle"
    assert chapters["Later"].subtitle.text == "Second subtitle"


def test_equal_title_scores_retain_group_reading_order(
    extraction_engine,
    alignment_page,
    region,
):
    engine = extraction_engine(
        {"subtitle_max_vertical_gap_height_multiplier": 3.0},
        [
            alignment_page(
                "toc",
                [
                    region(
                        0,
                        "kapitola",
                        "First in reading order",
                        100,
                        10,
                        height=20,
                    ),
                    region(1, "kapitola", "Lower top edge", 100, 20, height=10),
                    region(2, "podnadpis", "Subtitle", 100, 50),
                ],
            )
        ],
    )

    result = engine.process((_toc_page(),))

    chapters = {chapter.title.text: chapter for chapter in result.chapters}
    assert chapters["First in reading order"].subtitle.text == "Subtitle"
    assert chapters["Lower top edge"].subtitle is None


def test_subtitle_confidence_precedes_horizontal_overlap(
    extraction_engine,
    alignment_page,
    region,
):
    engine = extraction_engine(
        alignment_pages=[
            alignment_page(
                "toc",
                [
                    region(0, "kapitola", "Title", 100, 10, width=200),
                    region(
                        1,
                        "podnadpis",
                        "Greater overlap",
                        100,
                        35,
                        confidence=0.7,
                        width=100,
                    ),
                    region(
                        2,
                        "podnadpis",
                        "Higher confidence",
                        250,
                        35,
                        confidence=0.9,
                        width=100,
                    ),
                ],
            )
        ]
    )

    result = engine.process((_toc_page(),))

    assert result.chapters[0].subtitle.text == "Higher confidence"


def test_subtitle_ties_prefer_area_then_width(
    extraction_engine,
    alignment_page,
    region,
):
    engine = extraction_engine(
        alignment_pages=[
            alignment_page(
                "toc",
                [
                    region(0, "kapitola", "Title", 100, 10, width=300),
                    region(
                        1,
                        "podnadpis",
                        "Wider but smaller area",
                        100,
                        35,
                        width=200,
                        height=8,
                    ),
                    region(
                        2,
                        "podnadpis",
                        "Narrower equal area",
                        100,
                        35,
                        width=100,
                        height=20,
                    ),
                    region(
                        3,
                        "podnadpis",
                        "Wider equal area",
                        100,
                        35,
                        width=200,
                        height=10,
                    ),
                ],
            )
        ]
    )

    result = engine.process((_toc_page(),))

    assert result.chapters[0].subtitle.text == "Wider equal area"


@pytest.mark.parametrize(
    "setting,value",
    (
        ("subtitle_max_vertical_gap_height_multiplier", 0),
        ("subtitle_max_vertical_overlap_height_fraction", 1.1),
        ("subtitle_min_horizontal_overlap_fraction", -0.1),
    ),
)
def test_rejects_invalid_subtitle_configuration(extraction_engine, setting, value):
    with pytest.raises(ValueError):
        extraction_engine({setting: value})


def test_rejects_labels_not_used_by_the_stage(extraction_engine):
    with pytest.raises(ValueError, match="not used by this engine"):
        extraction_engine({"labels": {"DestinationTitle": "destination"}})


def test_distinct_overlapping_roles_are_not_suppressed_after_alignment(
    extraction_engine,
    alignment_page,
    region,
):
    engine = extraction_engine(
        alignment_pages=[
            alignment_page(
                "toc",
                [
                    region(0, "kapitola", "Chapter", 100, 10),
                    region(1, "cislo strany", "12", 100, 10),
                ],
            )
        ]
    )

    result = engine.process((_toc_page(),))

    assert len(result.chapters) == 2
    assert result.chapters[0].title.text == "Chapter"
    assert result.chapters[0].page_number is None
    assert result.chapters[1].title is None
    assert result.chapters[1].page_number.output_text() == "12"


def test_title_bands_assign_nearest_numbers_once(
    extraction_engine,
    alignment_page,
    region,
):
    engine = extraction_engine(
        alignment_pages=[
            alignment_page(
                "toc",
                [
                    region(0, "kapitola", "First", 100, 10, width=200, height=30),
                    region(1, "kapitola", "Second", 100, 20, width=200, height=30),
                    region(2, "jine cislo", "remote", 0, 15, width=20),
                    region(3, "jine cislo", "1", 60, 15, width=20),
                    region(4, "cislo strany", "10", 340, 15, width=20),
                ],
            )
        ]
    )

    result = engine.process((_toc_page(),))

    assert len(result.chapters) == 2
    assert result.chapters[0].title.text == "First"
    assert result.chapters[0].part_number.text == "1"
    assert result.chapters[0].page_number.output_text() == "10"
    assert result.chapters[1].title.text == "Second"
    assert result.chapters[1].part_number.text == "remote"
    assert result.chapters[0].part_number != result.chapters[1].part_number
    assert result.chapters[1].page_number is None


def test_title_bands_prefer_outside_then_area_then_width(
    extraction_engine,
    alignment_page,
    region,
):
    engine = extraction_engine(
        alignment_pages=[
            alignment_page(
                "toc",
                [
                    region(0, "kapitola", "Chapter", 100, 10, width=200, height=30),
                    region(
                        1,
                        "jine cislo",
                        "outside wider smaller area",
                        60,
                        20,
                        width=30,
                        height=10,
                    ),
                    region(
                        2,
                        "jine cislo",
                        "outside greater area",
                        70,
                        15,
                        width=20,
                        height=20,
                    ),
                    region(3, "jine cislo", "overlapping", 90, 15, width=20),
                    region(4, "cislo strany", "11", 310, 10, width=20, height=30),
                    region(5, "cislo strany", "12", 310, 15, width=30),
                    region(6, "cislo strany", "13", 290, 15, width=20),
                ],
            )
        ]
    )

    result = engine.process((_toc_page(),))

    chapter = next(
        chapter for chapter in result.chapters if chapter.title is not None
    )
    assert chapter.part_number.text == "outside greater area"
    assert chapter.page_number.output_text() == "12"


def test_uses_column_order_for_supported_page_number_lines(
    extraction_engine,
    alignment_page,
    region,
    caplog,
):
    regions = []
    region_id = 0
    for prefix, title_x, number_x in (
        ("Left", 50, 430),
        ("Right", 550, 930),
        ("Third", 1050, 1430),
    ):
        for position, y in enumerate((10, 50, 90), start=1):
            regions.extend(
                (
                    region(
                        region_id,
                        "kapitola",
                        f"{prefix} {position}",
                        title_x,
                        y,
                        width=200,
                    ),
                    region(
                        region_id + 1,
                        "cislo strany",
                        str(position),
                        number_x,
                        y,
                        width=20,
                    ),
                )
            )
            region_id += 2
    regions.append(
        region(region_id, "cislo strany", "outlier", 700, 140, width=20)
    )
    engine = extraction_engine(alignment_pages=[alignment_page("toc", regions)])

    with caplog.at_level(logging.INFO, logger=EXTRACTION_LOGGER):
        result = engine.process((_toc_page(width=1500),))

    assert [chapter.title.text for chapter in result.chapters] == [
        "Left 1",
        "Left 2",
        "Left 3",
        "Right 1",
        "Right 2",
        "Right 3",
        "Third 1",
        "Third 2",
        "Third 3",
    ]
    assert "Multi-column TOC processing accepted" in _log_output(caplog)


def test_column_partition_prevents_cross_column_number_assignment(
    extraction_engine,
    alignment_page,
    region,
    caplog,
):
    regions = [
        region(0, "kapitola", "Left without page", 50, 10, width=200),
        region(1, "kapitola", "Right first", 550, 10, width=200),
        region(2, "cislo strany", "101", 930, 10, width=20),
        region(3, "jine cislo", "foreign part", 350, 10, width=20),
        region(4, "jine cislo", "outside all axes", 970, 10, width=20),
    ]
    region_id = 5
    for position, y in enumerate((50, 90, 130), start=1):
        regions.extend(
            (
                region(
                    region_id,
                    "kapitola",
                    f"Left {position}",
                    50,
                    y,
                    width=200,
                ),
                region(
                    region_id + 1,
                    "cislo strany",
                    str(position),
                    430,
                    y,
                    width=20,
                ),
                region(
                    region_id + 2,
                    "kapitola",
                    f"Right {position}",
                    550,
                    y,
                    width=200,
                ),
                region(
                    region_id + 3,
                    "cislo strany",
                    str(101 + position),
                    930,
                    y,
                    width=20,
                ),
            )
        )
        region_id += 4
    engine = extraction_engine(alignment_pages=[alignment_page("toc", regions)])

    with caplog.at_level(logging.DEBUG, logger=EXTRACTION_LOGGER):
        result = engine.process((_toc_page(),))

    chapters = {chapter.title.text: chapter for chapter in result.chapters}
    assert chapters["Left without page"].page_number is None
    assert chapters["Left without page"].part_number is None
    assert chapters["Right first"].page_number.output_text() == "101"
    assert chapters["Right first"].part_number is None
    assert (
        "Discarded PartNumber detections without an alignment axis to "
        "their right: page='toc', count=1" in _log_output(caplog)
    )


def test_geometry_only_page_numbers_can_establish_columns(
    extraction_engine,
    alignment_page,
    region,
):
    regions = []
    region_id = 0
    for prefix, title_x, number_x in (
        ("Left", 50, 430),
        ("Right", 550, 930),
    ):
        for position, y in enumerate((10, 50, 90), start=1):
            regions.extend(
                (
                    region(
                        region_id,
                        "kapitola",
                        f"{prefix} {position}",
                        title_x,
                        y,
                        width=200,
                    ),
                    _geometry_only_region(
                        region_id + 1,
                        "cislo strany",
                        number_x,
                        y,
                        width=20,
                    ),
                )
            )
            region_id += 2
    engine = extraction_engine(alignment_pages=[alignment_page("toc", regions)])

    result = engine.process((_toc_page(),))

    assert [chapter.title.text for chapter in result.chapters] == [
        "Left 1",
        "Left 2",
        "Left 3",
        "Right 1",
        "Right 2",
        "Right 3",
    ]
    assert all(chapter.page_number is None for chapter in result.chapters)


def test_raises_when_column_analysis_cannot_resolve_page_width(
    tmp_path,
    extraction_engine,
    alignment_page,
):
    regions = [
        _geometry_only_region(
            region_id,
            "cislo strany",
            400 if region_id < 3 else 800,
            region_id * 30,
            width=20,
        )
        for region_id in range(6)
    ]
    engine = extraction_engine(alignment_pages=[alignment_page("toc", regions)])
    missing_image = tmp_path / "toc.jpg"

    with pytest.raises(ValueError, match="Unable to read page width from image"):
        engine.process(
            (
                ChapterPageInput(
                    "toc",
                    0,
                    missing_image,
                    Path("toc.xml"),
                ),
            )
        )


def test_rejects_false_columns_when_title_areas_overlap(
    extraction_engine,
    alignment_page,
    region,
    caplog,
):
    regions = []
    for position, y in enumerate((10, 50, 90, 130, 170, 210)):
        number_x = 380 if position % 2 == 0 else 680
        regions.extend(
            (
                region(
                    position * 2,
                    "kapitola",
                    f"Entry {position}",
                    100,
                    y,
                    width=200,
                ),
                region(
                    position * 2 + 1,
                    "cislo strany",
                    str(position + 1),
                    number_x,
                    y,
                    width=20,
                ),
            )
        )
    engine = extraction_engine(alignment_pages=[alignment_page("toc", regions)])

    with caplog.at_level(logging.INFO, logger=EXTRACTION_LOGGER):
        result = engine.process((_toc_page(),))

    assert [chapter.title.text for chapter in result.chapters] == [
        f"Entry {position}" for position in range(6)
    ]
    assert "title areas assigned to adjacent axes overlap" in _log_output(caplog)


def test_hierarchy_continues_across_toc_pages_and_preserves_page_keys(
    extraction_engine,
    alignment_page,
    region,
    caplog,
):
    engine = extraction_engine(
        alignment_pages=[
            alignment_page(
                "toc-1",
                [
                    region(0, "kapitola", "First part", 100, 10),
                    region(1, "cislo strany", "1", 500, 10),
                ],
            ),
            alignment_page(
                "toc-2",
                [
                    region(0, "jiny nadpis", "Child", 100, 10),
                    region(1, "cislo strany", "2", 500, 10),
                ],
            ),
        ]
    )
    pages = (_toc_page("toc-1", 0), _toc_page("toc-2", 1))

    with caplog.at_level(logging.INFO, logger=EXTRACTION_LOGGER):
        result = engine.process(pages)

    assert isinstance(result, TocBase)
    assert len(result.chapters) == 1
    assert result.chapters[0].toc_page_key == "toc-1"
    assert result.chapters[0].page_number.text == "1"
    assert isinstance(result.chapters[0].title.bbox, BoundingBox)
    assert isinstance(result.chapters[0].page_number.bbox, BoundingBox)
    assert len(result.chapters[0].children) == 1
    assert result.chapters[0].children[0].toc_page_key == "toc-2"
    assert result.chapters[0].children[0].page_number.text == "2"
    log_output = _log_output(caplog)
    assert "Extracting TOC hierarchy from 2 page(s)" in log_output
    assert "Chapter extraction page='toc-1'" in log_output
    assert "Chapter extraction page='toc-2'" in log_output
    assert (
        "Chapter extraction produced 2 total entry/entries, 1 root(s)"
        in log_output
    )


def test_number_only_unit_inherits_preceding_titled_level(
    extraction_engine,
    alignment_page,
    region,
):
    engine = extraction_engine(
        alignment_pages=[
            alignment_page(
                "toc",
                [
                    region(0, "kapitola", "Part", 100, 10),
                    region(1, "cislo strany", "1", 500, 10),
                    region(2, "jiny nadpis", "First", 100, 50),
                    region(3, "cislo strany", "2", 500, 50),
                    region(4, "cislo strany", "str. 003", 500, 90),
                    region(5, "jiny nadpis", "Third", 100, 130),
                    region(6, "cislo strany", "4", 500, 130),
                ],
            )
        ]
    )

    result = engine.process((_toc_page(),))

    assert len(result.chapters) == 1
    children = result.chapters[0].children
    assert len(children) == 3
    assert children[1].page_number.text == "str. 003"
    assert children[1].page_number.normalized_start() == "3"
    assert children[1].title is None
    assert not hasattr(children[1], "anchor_only")


def test_number_only_unit_inherits_preceding_level_across_pages(
    extraction_engine,
    alignment_page,
    region,
):
    engine = extraction_engine(
        alignment_pages=[
            alignment_page(
                "toc-1",
                [
                    region(0, "kapitola", "Root", 100, 10),
                    region(1, "cislo strany", "1", 500, 10),
                    region(2, "jiny nadpis", "Child", 100, 50),
                    region(3, "cislo strany", "2", 500, 50),
                ],
            ),
            alignment_page(
                "toc-2",
                [
                    region(0, "cislo strany", "3", 500, 10),
                    region(1, "kapitola", "Next root", 100, 50),
                    region(2, "cislo strany", "4", 500, 50),
                ],
            ),
        ]
    )

    result = engine.process((_toc_page("toc-1", 0), _toc_page("toc-2", 1)))

    assert len(result.chapters) == 2
    inherited = result.chapters[0].children[1]
    assert inherited.title is None
    assert inherited.page_number.output_text() == "3"
    assert inherited.toc_page_key == "toc-2"
    assert result.chapters[1].title.text == "Next root"

import logging
from pathlib import Path
from unittest import mock

import pytest

from metakat.chapter.engines.core.chapter_page_analysis.engine_yolo_alto import (
    ChapterPageAnalysisEngineYOLOALTO,
    _TocCandidate,
)
from metakat.chapter.engines.core.models import ChapterPageInput
from metakat.common.models import PageDimensions
from metakat.schemas.base_objects import ChapterType

ANALYSIS_LOGGER = (
    "metakat.chapter.engines.core.chapter_page_analysis.engine_yolo_alto"
)


def _alto_xml(text):
    return f"""\
<alto xmlns="http://www.loc.gov/standards/alto/ns-v2#">
  <Layout><Page ID="page" WIDTH="1000" HEIGHT="1000">
    <TextBlock><TextLine>
      <String ID="word" CONTENT="{text}" HPOS="10" VPOS="10"
              WIDTH="100" HEIGHT="20"/>
    </TextLine></TextBlock>
  </Page></Layout>
</alto>
"""


def _records(caplog):
    return [
        record.getMessage()
        for record in caplog.records
        if record.name.startswith(ANALYSIS_LOGGER)
    ]


@pytest.fixture
def analysis_engine(
    tmp_path,
    write_engine_config,
    read_engine_config,
    fake_alignment_engine,
):
    """Build the page-analysis engine over a config directory and fake aligner."""

    def _build(config=None, alignment_pages=(), *, aligner=None, directory=None):
        target = directory or tmp_path
        write_engine_config(
            target,
            {
                "name": "chapter_page_analysis_engine_yolo_alto",
                **(config or {}),
            },
        )
        return ChapterPageAnalysisEngineYOLOALTO(
            read_engine_config(target),
            alignment_engine=(
                aligner
                if aligner is not None
                else fake_alignment_engine(list(alignment_pages))
            ),
        )

    return _build


def test_candidate_thresholds_default_to_two_and_are_configurable(
    tmp_path,
    analysis_engine,
):
    defaults = analysis_engine(directory=tmp_path / "defaults")
    configured = analysis_engine(
        {
            "toc_candidate_min_title_count": 4,
            "toc_candidate_min_page_number_count": 3,
        },
        directory=tmp_path / "configured",
    )

    assert defaults.toc_candidate_min_title_count == 2
    assert defaults.toc_candidate_min_page_number_count == 2
    assert configured.toc_candidate_min_title_count == 4
    assert configured.toc_candidate_min_page_number_count == 3
    assert defaults.toc_candidate_window_height_multiplier == 10.0
    assert defaults.toc_candidate_min_window_height_fraction == 0.2
    assert defaults.toc_candidate_max_window_height_fraction == 0.5


@pytest.mark.parametrize(
    "option_name,option_value",
    (
        ("toc_candidate_min_title_count", 0),
        ("toc_candidate_min_title_count", 1.5),
        ("toc_candidate_min_page_number_count", True),
    ),
)
def test_candidate_thresholds_must_be_positive_integers(
    analysis_engine,
    option_name,
    option_value,
):
    with pytest.raises(
        ValueError,
        match=f"{option_name} must be a positive integer",
    ):
        analysis_engine({option_name: option_value})


@pytest.mark.parametrize(
    "option_name,option_value",
    (
        ("toc_candidate_window_height_multiplier", 0),
        ("toc_candidate_window_height_multiplier", float("inf")),
        ("toc_candidate_min_window_height_fraction", 0),
        ("toc_candidate_min_window_height_fraction", 1.1),
        ("toc_candidate_max_window_height_fraction", 0),
        ("toc_candidate_max_window_height_fraction", 1.1),
        ("toc_candidate_min_window_height_fraction", 0.6),
    ),
)
def test_candidate_window_settings_are_validated(
    analysis_engine,
    option_name,
    option_value,
):
    with pytest.raises(ValueError):
        analysis_engine({option_name: option_value})


def test_page_dimensions_use_metadata_precedence_then_image(tmp_path, page_image):
    image_path = tmp_path / "page.jpg"
    page_image(image_path, size=(400, 300))
    common = {
        "page_key": "page",
        "position": 0,
        "image_path": image_path,
        "alto_path": tmp_path / "page.xml",
    }

    assert (
        ChapterPageAnalysisEngineYOLOALTO._page_height(
            ChapterPageInput(
                **common,
                image_dimensions=PageDimensions(100, 120),
                alto_dimensions=PageDimensions(100, 240),
            )
        )
        == 120
    )
    assert (
        ChapterPageAnalysisEngineYOLOALTO._page_height(
            ChapterPageInput(
                **common,
                alto_dimensions=PageDimensions(100, 240),
            )
        )
        == 240
    )
    assert (
        ChapterPageAnalysisEngineYOLOALTO._page_height(ChapterPageInput(**common))
        == 300
    )
    assert (
        ChapterPageAnalysisEngineYOLOALTO._page_width(
            ChapterPageInput(
                **common,
                image_dimensions=PageDimensions(110, 120),
                alto_dimensions=PageDimensions(210, 240),
            )
        )
        == 110
    )
    assert (
        ChapterPageAnalysisEngineYOLOALTO._page_width(
            ChapterPageInput(
                **common,
                alto_dimensions=PageDimensions(210, 240),
            )
        )
        == 210
    )
    assert (
        ChapterPageAnalysisEngineYOLOALTO._page_width(ChapterPageInput(**common))
        == 400
    )


def test_page_height_fails_when_image_cannot_be_read(tmp_path):
    image_path = tmp_path / "page.jpg"
    image_path.write_text("not an image", encoding="utf-8")
    page = ChapterPageInput(
        "page",
        0,
        image_path,
        tmp_path / "page.xml",
    )

    with pytest.raises(ValueError, match="Unable to read page height from image"):
        ChapterPageAnalysisEngineYOLOALTO._page_height(page)


def test_keyword_must_start_above_uppermost_detection_bottom(
    tmp_path,
    analysis_engine,
    caplog,
):
    alto_path = tmp_path / "page.xml"
    alto_path.write_text(
        """\
<alto xmlns="http://www.loc.gov/standards/alto/ns-v2#">
  <Layout><Page ID="page" WIDTH="100" HEIGHT="100">
    <TextBlock><TextLine>
      <String ID="word-above" CONTENT="Contents" HPOS="5" VPOS="20"
              WIDTH="60" HEIGHT="15"/>
    </TextLine><TextLine>
      <String ID="word" CONTENT="Obsah" HPOS="10" VPOS="400"
              WIDTH="50" HEIGHT="20"/>
    </TextLine></TextBlock>
  </Page></Layout>
</alto>
""",
        encoding="utf-8",
    )
    page = ChapterPageInput(
        "page",
        0,
        tmp_path / "page.jpg",
        alto_path,
    )
    engine = analysis_engine({"toc_keywords": ["obsah", "contents"]})

    with caplog.at_level(logging.DEBUG, logger=ANALYSIS_LOGGER):
        valid_keyword = engine._contains_toc_keyword(
            page,
            toc_area_top=500,
            topmost_detection_bottom=520,
        )
    invalid_keyword = engine._contains_toc_keyword(
        page,
        toc_area_top=0,
        topmost_detection_bottom=10,
    )

    assert valid_keyword
    assert not invalid_keyword
    valid_logs = [
        message
        for message in _records(caplog)
        if "Valid TOC keyword occurrence" in message
    ]
    assert len(valid_logs) == 2
    assert any("y=20.000" in message for message in valid_logs)
    assert any("y=400.000" in message for message in valid_logs)


def test_uses_chapter_type_label_mapping(analysis_engine):
    engine = analysis_engine(
        {
            "labels": {
                "Level1Title": "primary",
                "Level2Title": "secondary",
                "PageNumber": "page",
                "DestinationTitle": "destination",
            },
        }
    )

    assert engine.labels == {
        ChapterType.LEVEL_1_TITLE: "primary",
        ChapterType.LEVEL_2_TITLE: "secondary",
        ChapterType.PAGE_NUMBER: "page",
        ChapterType.DESTINATION_TITLE: "destination",
    }


def test_candidate_requires_titles_and_numbers_in_same_vertical_window(
    tmp_path,
    analysis_engine,
    alignment_page,
    region,
):
    image = tmp_path / "page.jpg"
    alto = tmp_path / "page.xml"
    image.touch()
    alto.write_text(_alto_xml("Text"), encoding="utf-8")
    page = ChapterPageInput(
        "page",
        0,
        image,
        alto,
        image_dimensions=PageDimensions(1000, 1000),
    )
    engine = analysis_engine(
        alignment_pages=[
            alignment_page(
                "page",
                [
                    region(0, "kapitola", "First", 10, 10),
                    region(1, "jiny nadpis", "Second", 10, 40),
                    region(2, "cislo strany", "10", 500, 800),
                    region(3, "cislo strany", "11", 500, 830),
                ],
            )
        ]
    )

    result = engine.process([page])

    assert result.toc_pages == ()


def test_candidate_window_analysis_runs_only_in_edge_search_areas(
    tmp_path,
    analysis_engine,
    alignment_page,
):
    inputs = []
    alignments = []
    for position in range(8):
        page_key = f"page-{position}"
        inputs.append(
            ChapterPageInput(
                page_key,
                position,
                tmp_path / f"{page_key}.jpg",
                tmp_path / f"{page_key}.xml",
                image_dimensions=PageDimensions(1000, 1000),
            )
        )
        alignments.append(alignment_page(page_key, []))
    engine = analysis_engine(alignment_pages=alignments)

    with mock.patch.object(
        engine,
        "_find_candidate_windows",
        return_value=None,
    ) as find_windows:
        engine.process(inputs)

    assert [
        call.args[1].page_key for call in find_windows.call_args_list
    ] == ["page-0", "page-1", "page-6", "page-7"]


def test_candidate_accepts_clustered_titles_and_numbers(
    tmp_path,
    analysis_engine,
    alignment_page,
    region,
):
    image = tmp_path / "page.jpg"
    alto = tmp_path / "page.xml"
    image.touch()
    alto.write_text(_alto_xml("Text"), encoding="utf-8")
    page = ChapterPageInput(
        "page",
        0,
        image,
        alto,
        image_dimensions=PageDimensions(1000, 1000),
    )
    engine = analysis_engine(
        alignment_pages=[
            alignment_page(
                "page",
                [
                    region(0, "kapitola", "First", 10, 10),
                    region(1, "jiny nadpis", "Second", 10, 40),
                    region(2, "cislo strany", "10", 500, 70),
                    region(3, "cislo strany", "11", 500, 100),
                ],
            )
        ]
    )

    result = engine.process([page])

    assert tuple(selected.page_key for selected in result.toc_pages) == ("page",)


def test_overlapping_candidate_windows_count_detections_once(
    tmp_path,
    analysis_engine,
    alignment_page,
    region,
    caplog,
):
    engine = analysis_engine()
    page = ChapterPageInput(
        "page",
        0,
        tmp_path / "page.jpg",
        tmp_path / "page.xml",
        image_dimensions=PageDimensions(1000, 1000),
    )
    page_alignment = alignment_page(
        "page",
        [
            region(0, "kapitola", "First", 10, 10),
            region(1, "jiny nadpis", "Second", 10, 40),
            region(2, "cislo strany", "10", 500, 70),
            region(3, "cislo strany", "11", 500, 100),
            region(4, "kapitola", "Third", 10, 130),
            region(5, "cislo strany", "12", 500, 160),
        ],
    )
    page_alignment.alto_height = 1000

    with caplog.at_level(logging.DEBUG, logger=ANALYSIS_LOGGER):
        windows = engine._find_candidate_windows(page_alignment, page)

    assert windows is not None
    assert windows.qualifying_window_count == 2
    assert windows.title_count == 3
    assert windows.page_number_count == 3
    assert windows.visual_score == 6
    assert windows.toc_area_top == 10
    assert windows.toc_area_bottom == 180
    assert windows.topmost_detection_bottom == 30
    assert "cumulative_visual_score=6" in "\n".join(_records(caplog))


def test_separate_qualifying_windows_accumulate_unique_detections(
    tmp_path,
    analysis_engine,
    alignment_page,
    region,
):
    engine = analysis_engine()
    page = ChapterPageInput(
        "page",
        0,
        tmp_path / "page.jpg",
        tmp_path / "page.xml",
        image_dimensions=PageDimensions(1000, 1000),
    )
    page_alignment = alignment_page(
        "page",
        [
            region(0, "kapitola", "First", 10, 10),
            region(1, "jiny nadpis", "Second", 10, 40),
            region(2, "cislo strany", "10", 500, 70),
            region(3, "cislo strany", "11", 500, 100),
            region(4, "kapitola", "Third", 10, 600),
            region(5, "jiny nadpis", "Fourth", 10, 630),
            region(6, "cislo strany", "12", 500, 660),
            region(7, "cislo strany", "13", 500, 690),
        ],
    )
    page_alignment.alto_height = 1000

    windows = engine._find_candidate_windows(page_alignment, page)

    assert windows is not None
    assert windows.title_count == 4
    assert windows.page_number_count == 4
    assert windows.visual_score == 8


def test_candidate_window_has_minimum_page_height_fraction(
    tmp_path,
    analysis_engine,
    alignment_page,
    region,
):
    image = tmp_path / "page.jpg"
    alto = tmp_path / "page.xml"
    image.touch()
    alto.write_text(_alto_xml("Text"), encoding="utf-8")
    page = ChapterPageInput(
        "page",
        0,
        image,
        alto,
        image_dimensions=PageDimensions(1000, 1000),
    )
    engine = analysis_engine(
        {
            "toc_candidate_window_height_multiplier": 1,
            "toc_candidate_min_window_height_fraction": 0.2,
        },
        alignment_pages=[
            alignment_page(
                "page",
                [
                    region(0, "kapitola", "First", 10, 10, height=2),
                    region(1, "jiny nadpis", "Second", 10, 40, height=2),
                    region(2, "cislo strany", "10", 500, 150, height=2),
                    region(3, "cislo strany", "11", 500, 180, height=2),
                ],
            )
        ],
    )

    result = engine.process([page])

    assert tuple(selected.page_key for selected in result.toc_pages) == ("page",)


def test_candidate_window_has_maximum_page_height_fraction(
    tmp_path,
    analysis_engine,
    alignment_page,
    region,
):
    engine = analysis_engine(
        {
            "toc_candidate_window_height_multiplier": 10,
            "toc_candidate_min_window_height_fraction": 0.1,
            "toc_candidate_max_window_height_fraction": 0.3,
        }
    )
    page = ChapterPageInput(
        "page",
        0,
        tmp_path / "page.jpg",
        tmp_path / "page.xml",
        image_dimensions=PageDimensions(1000, 1000),
    )
    page_alignment = alignment_page(
        "page",
        [
            region(0, "kapitola", "First", 10, 0, height=100),
            region(1, "jiny nadpis", "Second", 10, 100, height=100),
            region(2, "cislo strany", "10", 500, 200, height=100),
            region(3, "cislo strany", "11", 500, 360, height=100),
        ],
    )

    windows = engine._find_candidate_windows(page_alignment, page)

    assert windows is None


def test_selects_best_consecutive_group_and_collects_destination_evidence(
    tmp_path,
    analysis_engine,
    alignment_page,
    region,
    fake_alignment_engine,
    caplog,
):
    inputs = []
    alignments = []
    for position in range(12):
        page_key = f"page-{position}"
        image = tmp_path / f"{page_key}.jpg"
        alto = tmp_path / f"{page_key}.xml"
        image.touch()
        alto.write_text(
            _alto_xml("Obsah" if position == 1 else "Text"),
            encoding="utf-8",
        )
        inputs.append(
            ChapterPageInput(
                page_key,
                position,
                image,
                alto,
                image_dimensions=PageDimensions(1000, 1000),
            )
        )
        regions = []
        if position in {0, 1, 2, 11}:
            regions = [
                *[
                    region(index, "kapitola", f"Title {index}", 10, index * 30)
                    for index in range(2)
                ],
                *[
                    region(
                        index + 2,
                        "cislo strany",
                        str(index),
                        500,
                        index * 30,
                    )
                    for index in range(2)
                ],
                region(
                    4,
                    "nadpis v textu",
                    f"Rejected candidate {position}",
                    10,
                    120,
                ),
            ]
        elif position == 5:
            regions = [
                region(0, "nadpis v textu", "Destination", 10, 10),
                region(
                    1,
                    "cislo strany",
                    "wrong",
                    500,
                    900,
                    confidence=0.5,
                ),
                region(
                    2,
                    "cislo strany",
                    "005",
                    500,
                    930,
                    confidence=0.95,
                ),
            ]
        elif position == 6:
            regions = [
                region(0, "cislo strany", "XIV", 500, 900),
            ]
        alignments.append(alignment_page(page_key, regions))
    aligner = fake_alignment_engine(alignments)
    engine = analysis_engine(aligner=aligner)

    with caplog.at_level(logging.INFO, logger=ANALYSIS_LOGGER):
        result = engine.process(inputs)

    assert tuple(page.page_key for page in result.toc_pages) == (
        "page-1",
        "page-2",
    )
    log_output = "\n".join(_records(caplog))
    assert "Analyzing 12 page(s) for TOC candidates" in log_output
    assert (
        "Selected consecutive TOC block: pages=['page-1', 'page-2']" in log_output
    )
    assert (
        "Page analysis selected 2 TOC page(s), 3 destination "
        "title(s), and 4 destination page number(s)" in log_output
    )
    assert tuple(
        item.title.text for item in result.destination_chapters
    ) == (
        "Rejected candidate 0",
        "Destination",
        "Rejected candidate 11",
    )
    assert {
        item.page_key for item in result.destination_page_numbers
    } == {
        "page-0",
        "page-5",
        "page-6",
        "page-11",
    }
    page_numbers = {
        item.page_key: item.text for item in result.destination_page_numbers
    }
    assert page_numbers["page-5"] == "005"
    assert page_numbers["page-6"] == "XIV"
    assert aligner.call_count == 1


def test_toc_group_ties_prefer_shorter_then_earlier_group():
    def candidate(position, score):
        return _TocCandidate(
            page=ChapterPageInput(
                f"page-{position}",
                position,
                Path(f"page-{position}.jpg"),
                Path(f"page-{position}.xml"),
            ),
            visual_score=score,
            contains_keyword=False,
        )

    shorter = ChapterPageAnalysisEngineYOLOALTO._best_group(
        (
            candidate(0, 2),
            candidate(1, 2),
            candidate(5, 4),
        )
    )
    earlier = ChapterPageAnalysisEngineYOLOALTO._best_group(
        (
            candidate(2, 4),
            candidate(7, 4),
        )
    )

    assert tuple(item.page.position for item in shorter) == (5,)
    assert tuple(item.page.position for item in earlier) == (2,)

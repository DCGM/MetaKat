import logging
import types
from unittest import mock

import pytest

from metakat.chapter.engines.core.chapter_core_engine_pipeline import (
    ChapterPipelineCoreEngine,
)
from metakat.chapter.engines.core.chapter_page_analysis.models import (
    ChapterPageAnalysisResult,
    DestinationChapterEvidence,
)
from metakat.chapter.engines.core.models import (
    ChapterPageInput,
    ChapterResult,
    TocBase,
    TocResult,
)
from metakat.common.models import PageDimensions

PIPELINE_LOGGER = "metakat.chapter.engines.core.chapter_core_engine_pipeline"


def _messages(caplog):
    return "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.name.startswith(PIPELINE_LOGGER)
    )


@pytest.fixture
def pipeline_inputs(tmp_path):
    """Touch the TOC and destination page files the pipeline expects on disk."""

    def _build(*stems):
        images = [tmp_path / f"{stem}.jpg" for stem in stems]
        altos = [tmp_path / f"{stem}.xml" for stem in stems]
        for path in (*images, *altos):
            path.touch()
        return (
            [str(path) for path in images],
            [str(path) for path in altos],
            images,
            altos,
        )

    return _build


def test_wrapper_prunes_titleless_results_and_splices_children(evidence, caplog):
    child = ChapterResult(
        toc_page_key="toc",
        title=evidence("Child", "toc"),
        page_start_key="page-2",
    )
    titleless = ChapterResult(
        toc_page_key="toc",
        title=None,
        page_start_key="page-1",
        children=(child,),
    )
    destination_titled = ChapterResult(
        toc_page_key="toc",
        title=None,
        title_destination_page=evidence(
            "Destination title",
            "page-3",
        ),
        page_start_key="page-3",
    )

    with caplog.at_level(logging.INFO, logger=PIPELINE_LOGGER):
        result = ChapterPipelineCoreEngine._prune_titleless_chapters(
            TocResult((titleless, destination_titled))
        )

    assert result.chapters == (child, destination_titled)
    assert "Pruned 1 titleless chapter entry" in _messages(caplog)


def test_wrapper_uses_internal_page_numbers_when_none_are_supplied(
    tmp_path,
    write_engine_config,
    read_engine_config,
    pipeline_inputs,
    evidence,
    physical_page_number,
    caplog,
):
    write_engine_config(tmp_path, {"name": "chapter_core_engine_pipeline"})
    image_paths, alto_paths, _, _ = pipeline_inputs("toc", "destination")
    page_number = physical_page_number("001", "destination")
    analysis_engine = types.SimpleNamespace(
        process=lambda pages: ChapterPageAnalysisResult(
            (pages[0],),
            (
                DestinationChapterEvidence(evidence("TOC title", "toc")),
                DestinationChapterEvidence(
                    evidence("Destination title", "destination")
                ),
            ),
            (page_number,),
        )
    )
    extraction_engine = types.SimpleNamespace(process=lambda pages: TocBase(()))
    expected = TocResult(())
    aligned_inputs = []
    alignment_engine = types.SimpleNamespace(
        process=lambda **kwargs: (aligned_inputs.append(kwargs) or expected)
    )
    engine = ChapterPipelineCoreEngine(
        read_engine_config(tmp_path),
        chapter_page_analysis_engine=analysis_engine,
        chapter_extraction_engine=extraction_engine,
        chapter_alignment_engine=alignment_engine,
    )

    with caplog.at_level(logging.INFO, logger=PIPELINE_LOGGER):
        result = engine.process(image_paths, alto_paths)

    assert result is expected
    assert not hasattr(aligned_inputs[0]["pages"][0], "page_number")
    assert tuple(page.page_key for page in aligned_inputs[0]["pages"]) == (
        "toc",
        "destination",
    )
    assert tuple(page.page_key for page in aligned_inputs[0]["toc_pages"]) == (
        "toc",
    )
    assert aligned_inputs[0]["destination_page_numbers"] == (page_number,)
    log_output = _messages(caplog)
    assert "Starting chapter page analysis stage" in log_output
    assert "Completed chapter page analysis stage" in log_output
    assert "Starting chapter extraction stage" in log_output
    assert "Completed chapter extraction stage" in log_output
    assert "Starting chapter alignment stage" in log_output
    assert "Completed chapter alignment stage" in log_output


def test_wrapper_prefers_supplied_page_numbers_and_validates_them(
    tmp_path,
    write_engine_config,
    read_engine_config,
    pipeline_inputs,
    physical_page_number,
):
    write_engine_config(tmp_path, {"name": "chapter_core_engine_pipeline"})
    image_paths, alto_paths, _, _ = pipeline_inputs("toc", "destination")
    analysis_engine = types.SimpleNamespace(
        process=lambda pages: ChapterPageAnalysisResult(
            (pages[0],),
            (),
            (physical_page_number("1", "destination"),),
        )
    )
    extraction_engine = types.SimpleNamespace(process=lambda pages: TocBase(()))
    aligned_inputs = []
    alignment_engine = types.SimpleNamespace(
        process=lambda **kwargs: (aligned_inputs.append(kwargs) or TocResult(()))
    )
    engine = ChapterPipelineCoreEngine(
        read_engine_config(tmp_path),
        chapter_page_analysis_engine=analysis_engine,
        chapter_extraction_engine=extraction_engine,
        chapter_alignment_engine=alignment_engine,
    )

    engine.process(
        image_paths,
        alto_paths,
        page_numbers=(
            physical_page_number("I", "toc"),
            physical_page_number("2", "destination"),
        ),
        image_dimensions=(
            PageDimensions(10, 20),
            PageDimensions(100, 200),
        ),
        alto_dimensions=(
            PageDimensions(9, 18),
            PageDimensions(90, 180),
        ),
    )
    assert aligned_inputs[0]["destination_page_numbers"] == (
        physical_page_number("2", "destination"),
    )
    assert aligned_inputs[0]["pages"][1].image_dimensions == PageDimensions(
        100, 200
    )
    assert aligned_inputs[0]["pages"][1].alto_dimensions == PageDimensions(90, 180)

    engine.process(image_paths, alto_paths, page_numbers=())
    assert aligned_inputs[1]["destination_page_numbers"] == ()

    with pytest.raises(TypeError, match="PhysicalPageNumberEvidence"):
        engine.process(image_paths, alto_paths, page_numbers=(None,))
    with pytest.raises(ValueError, match="same length"):
        engine.process(image_paths, alto_paths, image_dimensions=())
    with pytest.raises(TypeError, match="PageDimensions or None"):
        engine.process(
            image_paths,
            alto_paths,
            image_dimensions=((100, 200), None),
        )


def test_wrapper_stops_when_page_analysis_finds_no_toc(
    tmp_path,
    write_engine_config,
    read_engine_config,
    pipeline_inputs,
    caplog,
):
    write_engine_config(tmp_path, {"name": "chapter_core_engine_pipeline"})
    image_paths, alto_paths, _, _ = pipeline_inputs("page")
    analysis_engine = types.SimpleNamespace(
        process=mock.Mock(return_value=ChapterPageAnalysisResult((), (), ()))
    )
    extraction_engine = types.SimpleNamespace(process=mock.Mock())
    alignment_engine = types.SimpleNamespace(process=mock.Mock())
    engine = ChapterPipelineCoreEngine(
        read_engine_config(tmp_path),
        chapter_page_analysis_engine=analysis_engine,
        chapter_extraction_engine=extraction_engine,
        chapter_alignment_engine=alignment_engine,
    )

    with caplog.at_level(logging.INFO, logger=PIPELINE_LOGGER):
        result = engine.process(image_paths, alto_paths)

    assert result == TocResult(chapters=())
    extraction_engine.process.assert_not_called()
    alignment_engine.process.assert_not_called()
    log_output = _messages(caplog)
    assert "No TOC pages were selected during page analysis" in log_output
    assert "Starting chapter extraction stage" not in log_output
    assert "Starting chapter alignment stage" not in log_output


def test_external_empty_page_numbers_supply_the_missing_capability(
    tmp_path,
    write_engine_config,
    read_engine_config,
    pipeline_inputs,
):
    write_engine_config(tmp_path, {"name": "chapter_core_engine_pipeline"})
    image_paths, alto_paths, _, _ = pipeline_inputs("toc")
    received_inputs = []
    extraction_engine = types.SimpleNamespace(
        process=mock.Mock(return_value=TocBase(()))
    )
    alignment_engine = types.SimpleNamespace(
        process=lambda **kwargs: (received_inputs.append(kwargs) or TocResult(()))
    )
    engine = ChapterPipelineCoreEngine(
        read_engine_config(tmp_path),
        chapter_page_analysis_engine=types.SimpleNamespace(
            process=lambda pages: ChapterPageAnalysisResult(
                toc_pages=(pages[0],)
            )
        ),
        chapter_extraction_engine=extraction_engine,
        chapter_alignment_engine=alignment_engine,
    )

    engine.process(image_paths, alto_paths, page_numbers=())

    extraction_engine.process.assert_called_once()
    assert received_inputs[0]["destination_chapters"] is None
    assert received_inputs[0]["destination_page_numbers"] == ()


def test_wrapper_rejects_unavailable_destination_capabilities(
    tmp_path,
    write_engine_config,
    read_engine_config,
    pipeline_inputs,
):
    write_engine_config(tmp_path, {"name": "chapter_core_engine_pipeline"})
    image_paths, alto_paths, _, _ = pipeline_inputs("toc")
    extraction_engine = types.SimpleNamespace(process=mock.Mock())
    alignment_engine = types.SimpleNamespace(process=mock.Mock())
    engine = ChapterPipelineCoreEngine(
        read_engine_config(tmp_path),
        chapter_page_analysis_engine=types.SimpleNamespace(
            process=lambda pages: ChapterPageAnalysisResult(
                toc_pages=(pages[0],)
            )
        ),
        chapter_extraction_engine=extraction_engine,
        chapter_alignment_engine=alignment_engine,
    )

    with pytest.raises(ValueError, match="three-stage chapter pipeline requires"):
        engine.process(image_paths, alto_paths)

    engine.chapter_page_analysis_engine = types.SimpleNamespace(
        process=lambda pages: ChapterPageAnalysisResult(toc_pages=())
    )
    with pytest.raises(ValueError, match="three-stage chapter pipeline requires"):
        engine.process(image_paths, alto_paths)

    extraction_engine.process.assert_not_called()
    alignment_engine.process.assert_not_called()


def test_wrapper_passes_all_pages_and_filters_toc_evidence(
    tmp_path,
    write_engine_config,
    read_engine_config,
    pipeline_inputs,
    evidence,
    physical_page_number,
):
    write_engine_config(tmp_path, {"name": "chapter_core_engine_pipeline"})
    image_paths, alto_paths, _, _ = pipeline_inputs("toc", "destination")

    analysis_engine = types.SimpleNamespace(
        process=lambda pages: ChapterPageAnalysisResult(
            (pages[0],),
            (
                DestinationChapterEvidence(evidence("TOC title", "toc")),
                DestinationChapterEvidence(
                    evidence("Destination title", "destination")
                ),
            ),
            (
                physical_page_number("i", "toc"),
                physical_page_number("1", "destination"),
            ),
        )
    )
    extraction_engine = types.SimpleNamespace(process=lambda pages: TocBase(()))
    received_inputs = []
    alignment_engine = types.SimpleNamespace(
        process=lambda **kwargs: (received_inputs.append(kwargs) or TocResult(()))
    )
    engine = ChapterPipelineCoreEngine(
        read_engine_config(tmp_path),
        chapter_page_analysis_engine=analysis_engine,
        chapter_extraction_engine=extraction_engine,
        chapter_alignment_engine=alignment_engine,
    )

    engine.process(image_paths, alto_paths)
    engine.process(
        image_paths,
        alto_paths,
        page_numbers=(
            physical_page_number("II", "toc"),
            physical_page_number("2", "destination"),
        ),
    )

    for inputs in received_inputs:
        assert tuple(page.page_key for page in inputs["pages"]) == (
            "toc",
            "destination",
        )
        assert tuple(page.page_key for page in inputs["toc_pages"]) == ("toc",)
    assert received_inputs[0]["destination_page_numbers"] == (
        physical_page_number("1", "destination"),
    )
    for inputs in received_inputs:
        assert tuple(
            item.title.page_key for item in inputs["destination_chapters"]
        ) == ("destination",)
    assert received_inputs[1]["destination_page_numbers"] == (
        physical_page_number("2", "destination"),
    )


# The two rejected analysis results need the page paths, so they are built
# inside the test from a parametrized selector.
@pytest.mark.parametrize(
    "case,message",
    (
        ("unknown-page-key", "unknown page_key"),
        ("duplicate-page-key", "duplicate page_key"),
    ),
)
def test_wrapper_rejects_unknown_and_duplicate_destination_evidence(
    tmp_path,
    write_engine_config,
    read_engine_config,
    pipeline_inputs,
    evidence,
    physical_page_number,
    case,
    message,
):
    write_engine_config(tmp_path, {"name": "chapter_core_engine_pipeline"})
    image_paths, alto_paths, images, altos = pipeline_inputs("toc", "destination")
    extraction_engine = types.SimpleNamespace(process=lambda pages: TocBase(()))
    alignment_engine = types.SimpleNamespace(process=mock.Mock())
    toc_pages = (ChapterPageInput("toc", 0, images[0], altos[0]),)

    if case == "unknown-page-key":
        analysis = ChapterPageAnalysisResult(
            toc_pages=toc_pages,
            destination_chapters=(
                DestinationChapterEvidence(evidence("Unknown", "unknown")),
            ),
            destination_page_numbers=(),
        )
    else:
        analysis = ChapterPageAnalysisResult(
            toc_pages=toc_pages,
            destination_chapters=(),
            destination_page_numbers=(
                physical_page_number("1", "destination"),
                physical_page_number("2", "destination"),
            ),
        )

    engine = ChapterPipelineCoreEngine(
        read_engine_config(tmp_path),
        chapter_page_analysis_engine=types.SimpleNamespace(
            process=lambda pages, result=analysis: result
        ),
        chapter_extraction_engine=extraction_engine,
        chapter_alignment_engine=alignment_engine,
    )
    with pytest.raises(ValueError, match=message):
        engine.process(image_paths, alto_paths)

    alignment_engine.process.assert_not_called()

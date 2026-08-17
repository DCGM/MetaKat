import types
from unittest import mock

import pytest

from metakat.chapter.engines.core.chapter_core_engine import ChapterCoreEngine
from metakat.chapter.engines.core.chapter_core_engine_pipeline import (
    ChapterPipelineCoreEngine,
)
from metakat.chapter.engines.core.chapter_page_analysis.models import (
    ChapterPageAnalysisResult,
)
from metakat.chapter.engines.core.definitions import load_chapter_core_engine
from metakat.chapter.engines.core.models import (
    ChapterBase,
    ChapterResult,
    TocBase,
    TocResult,
)
from metakat.schemas.base_objects import ChapterType


class _ConcreteCoreEngine(ChapterCoreEngine):
    def process(
        self,
        images,
        alto_files,
        page_numbers=None,
        image_dimensions=None,
        alto_dimensions=None,
    ):
        return TocResult(())


# Parametrized over the removed name rather than split into a second test, so
# the name this test had is preserved; the enum assertion is cheap to repeat.
@pytest.mark.parametrize(
    "removed_name",
    ("Chapter", "Subchapter", "DestinationChapter"),
)
def test_chapter_type_uses_level_based_title_names(removed_name):
    assert {chapter_type.value for chapter_type in ChapterType} == {
        "PageNumber",
        "Level1Title",
        "Level2Title",
        "Subtitle",
        "PartNumber",
        "DestinationTitle",
    }

    with pytest.raises(ValueError):
        ChapterType(removed_name)


def test_shared_toc_models_represent_base_and_aligned_results(evidence):
    base_chapter = ChapterBase(
        toc_page_key="toc-1",
        title=evidence("Root", "toc-1"),
    )
    base_toc = TocBase(chapters=(base_chapter,))
    child = ChapterResult(
        toc_page_key="toc-2",
        title=evidence("Child", "toc-2"),
        page_start_key="destination-2",
    )
    chapter = ChapterResult(
        toc_page_key="toc-1",
        title=evidence("Root", "toc-1"),
        page_start_key="destination-1",
        children=(child,),
    )
    toc = TocResult(chapters=(chapter,))

    assert base_toc.chapters == (base_chapter,)
    assert isinstance(chapter, ChapterBase)
    assert toc.chapters[0].children == (child,)


def test_extraction_model_does_not_accept_alignment_fields(evidence):
    with pytest.raises(TypeError, match="page_start_key"):
        ChapterBase(
            toc_page_key="toc",
            title=evidence("Title", "toc"),
            page_start_key="destination",
        )


def test_page_analysis_evidence_capabilities_default_to_unavailable():
    result = ChapterPageAnalysisResult(toc_pages=())

    assert result.destination_chapters is None
    assert result.destination_page_numbers is None


def test_base_retains_name_and_arbitrary_config(
    tmp_path,
    write_engine_config,
    read_engine_config,
):
    write_engine_config(
        tmp_path,
        {
            "name": "test_chapter_core",
            "engine_specific": {"value": 42},
        },
    )

    engine = _ConcreteCoreEngine(read_engine_config(tmp_path))

    assert engine.name == "test_chapter_core"
    assert engine.config["engine_specific"]["value"] == 42
    assert not hasattr(engine, "id2label")


# The two subTest loops this replaces checked different messages, so both the
# rejected value and its expected message are parameters.
@pytest.mark.parametrize(
    "invalid,message",
    (
        (None, "must be an object"),
        ([], "must be an object"),
        ("config", "must be an object"),
        ({"name": None}, "non-empty string"),
        ({"name": ""}, "non-empty string"),
        ({"name": 12}, "non-empty string"),
    ),
)
def test_base_validates_common_config_shape_and_name(invalid, message):
    with pytest.raises(ValueError, match=message):
        _ConcreteCoreEngine(invalid)


def test_loader_validates_name_before_dispatch(
    tmp_path,
    write_engine_config,
    read_engine_config,
):
    write_engine_config(tmp_path, {"name": "unknown_chapter_engine"})

    with pytest.raises(ValueError, match="Unknown chapter core"):
        load_chapter_core_engine(read_engine_config(tmp_path))


def test_loader_dispatches_pipeline_from_directory_config(
    tmp_path,
    write_engine_config,
    read_engine_config,
):
    write_engine_config(tmp_path, {"name": "chapter_core_engine_pipeline"})
    stage = types.SimpleNamespace()

    with mock.patch.object(
        ChapterPipelineCoreEngine,
        "_load_stage",
        return_value=stage,
    ):
        engine = load_chapter_core_engine(read_engine_config(tmp_path))

    assert isinstance(engine, ChapterPipelineCoreEngine)
    assert engine.chapter_page_analysis_engine is stage
    assert engine.chapter_extraction_engine is stage
    assert engine.chapter_alignment_engine is stage


def test_pipeline_rejects_old_stage_configuration_names(
    tmp_path,
    write_engine_config,
    read_engine_config,
):
    write_engine_config(
        tmp_path,
        {
            "name": "chapter_core_engine_pipeline",
            "stages": {
                "toc_page_analysis": "page-analysis",
                "toc_extraction": "extraction",
                "toc_alignment": "alignment",
            },
        },
    )

    with pytest.raises(ValueError, match="requires an object at 'page_analysis'"):
        ChapterPipelineCoreEngine(read_engine_config(tmp_path))


def test_pipeline_rejects_old_registered_stage_engine_names(
    tmp_path,
    write_engine_config,
    read_engine_config,
):
    write_engine_config(
        tmp_path,
        {
            "name": "chapter_core_engine_pipeline",
            "page_analysis": {
                "name": "toc_page_analysis_engine_yolo_alto",
            },
        },
    )

    with pytest.raises(ValueError, match="Unknown page_analysis engine"):
        ChapterPipelineCoreEngine(read_engine_config(tmp_path))

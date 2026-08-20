import logging
import types
from unittest import mock
from uuid import uuid4

import pytest

from metakat.chapter.engines.bind.chapter_bind_engine_base import (
    ChapterBindEngineBase,
)
from metakat.chapter.engines.core.chapter_page_number_parsers import (
    ArabicRomanChapterPageNumberParser,
)
from metakat.chapter.engines.core.models import ChapterResult, TocResult
from metakat.common.models import BoundingBox, PageDimensions
from metakat.page_number.engines.core.page_number_parsers import (
    DecoratedPageNumberParser,
)
from metakat.schemas.base_objects import (
    HierarchyType,
    MetakatChapter,
    MetakatIO,
    MetakatIssue,
    MetakatPage,
    MetakatPageDimensions,
    MetakatVolume,
)

BIND_LOGGER = "metakat.chapter.engines.bind.chapter_bind_engine_base"


@pytest.fixture
def bind_engine():
    """The bind engine with __init__ bypassed, as these tests drive it."""
    return object.__new__(ChapterBindEngineBase)


def _pages(count, parent_id, *, batch_id=None, page_index=lambda index: index):
    batch_id = batch_id or uuid4()
    return [
        MetakatPage(
            id=uuid4(),
            batch_id=batch_id,
            batch_index=index,
            pageIndex=page_index(index),
            parent_id=parent_id,
        )
        for index in range(count)
    ]


def _messages(caplog):
    return "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.name.startswith(BIND_LOGGER)
    )


def test_process_passes_existing_page_numbers_to_core(bind_engine):
    batch_id = uuid4()
    page_number_detection_id = uuid4()
    first = MetakatPage(
        id=uuid4(),
        batch_id=batch_id,
        batch_index=0,
        pageNumber=("XIV", 0.9, page_number_detection_id),
        imageDim=MetakatPageDimensions(width=100, height=200),
        altoDim=MetakatPageDimensions(width=90, height=180),
    )
    second = MetakatPage(
        id=uuid4(),
        batch_id=batch_id,
        batch_index=1,
        altoDim=MetakatPageDimensions(width=95, height=190),
    )
    metakat_io = MetakatIO(
        batch_id=batch_id,
        elements=[first, second],
        page_to_image_mapping={
            first.id: "first.jpg",
            second.id: "second.jpg",
        },
        page_to_alto_mapping={
            first.id: "first.xml",
            second.id: "second.xml",
        },
        detection_to_bbox={
            page_number_detection_id: (10, 20, 30, 40),
        },
    )
    bind_engine.core_engine = types.SimpleNamespace(
        process=mock.Mock(return_value=TocResult(()))
    )

    result = bind_engine.process("/batch", metakat_io)

    bind_engine.core_engine.process.assert_called_once_with(
        ["/batch/first.jpg", "/batch/second.jpg"],
        ["/batch/first.xml", "/batch/second.xml"],
        page_numbers=(
            DecoratedPageNumberParser.create(
                page_key="first",
                text="XIV",
                confidence=0.9,
                bbox=BoundingBox(10, 20, 30, 40),
            ),
        ),
        image_dimensions=(PageDimensions(100, 200), None),
        alto_dimensions=(
            PageDimensions(90, 180),
            PageDimensions(95, 190),
        ),
    )
    dummy = next(
        element for element in result.elements if element.type == "volume"
    )
    result_pages = [
        element for element in result.elements if element.type == "page"
    ]
    assert all(page.parent_id == dummy.id for page in result_pages)


def test_process_omits_page_numbers_when_none_are_available(bind_engine):
    batch_id = uuid4()
    page = MetakatPage(
        id=uuid4(),
        batch_id=batch_id,
        batch_index=0,
    )
    metakat_io = MetakatIO(
        batch_id=batch_id,
        elements=[page],
        page_to_image_mapping={page.id: "page.jpg"},
        page_to_alto_mapping={page.id: "page.xml"},
    )
    bind_engine.core_engine = types.SimpleNamespace(
        process=mock.Mock(return_value=TocResult(()))
    )

    bind_engine.process("/batch", metakat_io)

    bind_engine.core_engine.process.assert_called_once_with(
        ["/batch/page.jpg"],
        ["/batch/page.xml"],
        page_numbers=None,
    )


def test_issues_are_processed_as_independent_documents(bind_engine, evidence):
    batch_id = uuid4()
    periodical = MetakatVolume(
        id=uuid4(),
        hierarchy=HierarchyType.PERIODICAL,
    )
    first_issue = MetakatIssue(id=uuid4(), parent_id=periodical.id)
    second_issue = MetakatIssue(id=uuid4(), parent_id=periodical.id)
    pages = [
        MetakatPage(
            id=uuid4(),
            batch_id=batch_id,
            batch_index=index,
            pageIndex=index,
            parent_id=(first_issue.id if index < 2 else second_issue.id),
        )
        for index in range(4)
    ]
    metakat_io = MetakatIO(
        batch_id=batch_id,
        elements=[periodical, first_issue, second_issue, *pages],
        page_to_image_mapping={
            pages[0].id: "issue-1/shared.jpg",
            pages[1].id: "issue-1/second.jpg",
            pages[2].id: "issue-2/shared.jpg",
            pages[3].id: "issue-2/second.jpg",
        },
        page_to_alto_mapping={
            pages[0].id: "issue-1/shared.xml",
            pages[1].id: "issue-1/second.xml",
            pages[2].id: "issue-2/shared.xml",
            pages[3].id: "issue-2/second.xml",
        },
    )
    core_result = TocResult(
        (
            ChapterResult(
                toc_page_key="shared",
                title=evidence("Chapter", "shared"),
                page_start_key="shared",
            ),
        )
    )
    bind_engine.core_engine = types.SimpleNamespace(
        process=mock.Mock(side_effect=(core_result, core_result))
    )

    result = bind_engine.process("/batch", metakat_io)

    assert bind_engine.core_engine.process.call_args_list == [
        mock.call(
            [
                "/batch/issue-1/shared.jpg",
                "/batch/issue-1/second.jpg",
            ],
            [
                "/batch/issue-1/shared.xml",
                "/batch/issue-1/second.xml",
            ],
            page_numbers=None,
        ),
        mock.call(
            [
                "/batch/issue-2/shared.jpg",
                "/batch/issue-2/second.jpg",
            ],
            [
                "/batch/issue-2/shared.xml",
                "/batch/issue-2/second.xml",
            ],
            page_numbers=None,
        ),
    ]
    chapter_parents = {
        element.parent_id
        for element in result.elements
        if element.type == "chapter"
    }
    assert chapter_parents == {first_issue.id, second_issue.id}
    for issue in (first_issue, second_issue):
        issue_index = result.elements.index(issue)
        chapter = result.elements[issue_index + 1]
        assert chapter.type == "chapter"
        assert chapter.parent_id == issue.id


def test_leaf_volumes_keep_chapter_parents_and_ends_separate(
    bind_engine,
    evidence,
):
    batch_id = uuid4()
    first_volume = MetakatVolume(
        id=uuid4(),
        hierarchy=HierarchyType.MONOGRAPH,
    )
    second_volume = MetakatVolume(
        id=uuid4(),
        hierarchy=HierarchyType.MULTIPART,
    )
    first_page = MetakatPage(
        id=uuid4(),
        batch_id=batch_id,
        batch_index=0,
        pageIndex=5,
        parent_id=first_volume.id,
    )
    second_page = MetakatPage(
        id=uuid4(),
        batch_id=batch_id,
        batch_index=1,
        pageIndex=50,
        parent_id=second_volume.id,
    )
    metakat_io = MetakatIO(
        batch_id=batch_id,
        elements=[first_volume, second_volume, first_page, second_page],
        page_to_image_mapping={
            first_page.id: "first/page.jpg",
            second_page.id: "second/page.jpg",
        },
        page_to_alto_mapping={
            first_page.id: "first/page.xml",
            second_page.id: "second/page.xml",
        },
    )

    def process(images, alto_files, page_numbers=None):
        assert len(images) == 1
        return TocResult(
            (
                ChapterResult(
                    toc_page_key="page",
                    title=evidence("Chapter", "page"),
                    page_start_key="page",
                    page_end_key="page",
                ),
            ),
        )

    bind_engine.core_engine = types.SimpleNamespace(
        process=mock.Mock(side_effect=process)
    )

    result = bind_engine.process("/batch", metakat_io)

    chapters = {
        element.parent_id: element
        for element in result.elements
        if element.type == "chapter"
    }
    assert bind_engine.core_engine.process.call_count == 2
    assert chapters[first_volume.id].pageIndexEnd == 5
    assert chapters[second_volume.id].pageIndexEnd == 50


def test_pages_with_ineligible_non_null_parents_are_ignored(bind_engine, caplog):
    batch_id = uuid4()
    volume = MetakatVolume(
        id=uuid4(),
        hierarchy=HierarchyType.MULTIPART,
    )
    existing_chapter = MetakatChapter(id=uuid4(), parent_id=volume.id)
    page = MetakatPage(
        id=uuid4(),
        batch_id=batch_id,
        batch_index=0,
        parent_id=existing_chapter.id,
    )
    unknown_parent_page = MetakatPage(
        id=uuid4(),
        batch_id=batch_id,
        batch_index=1,
        parent_id=uuid4(),
    )
    metakat_io = MetakatIO(
        batch_id=batch_id,
        elements=[volume, existing_chapter, page, unknown_parent_page],
        page_to_image_mapping={
            page.id: "page.jpg",
            unknown_parent_page.id: "unknown.jpg",
        },
        page_to_alto_mapping={
            page.id: "page.xml",
            unknown_parent_page.id: "unknown.xml",
        },
    )
    bind_engine.core_engine = types.SimpleNamespace(
        process=mock.Mock(return_value=TocResult(()))
    )

    with caplog.at_level(logging.WARNING, logger=BIND_LOGGER):
        result = bind_engine.process("/batch", metakat_io)

    bind_engine.core_engine.process.assert_not_called()
    assert (
        "Ignoring 2 page(s) whose direct parent is not an eligible"
        in _messages(caplog)
    )
    assert sum(element.type == "volume" for element in result.elements) == 1


def test_orphan_pages_are_persisted_under_one_dummy_monograph(bind_engine):
    batch_id = uuid4()
    volume = MetakatVolume(
        id=uuid4(),
        hierarchy=HierarchyType.MONOGRAPH,
    )
    grouped_page = MetakatPage(
        id=uuid4(),
        batch_id=batch_id,
        batch_index=0,
        parent_id=volume.id,
    )
    orphan_page = MetakatPage(
        id=uuid4(),
        batch_id=batch_id,
        batch_index=1,
        parent_id=None,
    )
    metakat_io = MetakatIO(
        batch_id=batch_id,
        elements=[volume, grouped_page, orphan_page],
        page_to_image_mapping={
            grouped_page.id: "grouped.jpg",
            orphan_page.id: "orphan.jpg",
        },
        page_to_alto_mapping={
            grouped_page.id: "grouped.xml",
            orphan_page.id: "orphan.xml",
        },
    )
    bind_engine.core_engine = types.SimpleNamespace(
        process=mock.Mock(side_effect=(TocResult(()), TocResult(())))
    )

    result = bind_engine.process("/batch", metakat_io)

    result_pages = {
        element.id: element
        for element in result.elements
        if element.type == "page"
    }
    volumes = [
        element for element in result.elements if element.type == "volume"
    ]
    dummy = next(item for item in volumes if item.id != volume.id)
    assert len(volumes) == 2
    assert dummy.hierarchy == HierarchyType.MONOGRAPH
    assert result_pages[grouped_page.id].parent_id == volume.id
    assert result_pages[orphan_page.id].parent_id == dummy.id
    assert bind_engine.core_engine.process.call_count == 2


def test_empty_input_creates_no_dummy_and_does_not_call_core(bind_engine):
    metakat_io = MetakatIO(batch_id=uuid4())
    bind_engine.core_engine = types.SimpleNamespace(process=mock.Mock())

    result = bind_engine.process("/batch", metakat_io)

    bind_engine.core_engine.process.assert_not_called()
    assert result.elements == []


def test_duplicate_page_stems_are_rejected_only_within_a_document(bind_engine):
    batch_id = uuid4()
    volume = MetakatVolume(
        id=uuid4(),
        hierarchy=HierarchyType.MONOGRAPH,
    )
    pages = [
        MetakatPage(
            id=uuid4(),
            batch_id=batch_id,
            batch_index=index,
            parent_id=volume.id,
        )
        for index in range(2)
    ]
    metakat_io = MetakatIO(
        batch_id=batch_id,
        elements=[volume, *pages],
        page_to_image_mapping={
            pages[0].id: "first/page.jpg",
            pages[1].id: "second/page.jpg",
        },
        page_to_alto_mapping={
            pages[0].id: "first/page.xml",
            pages[1].id: "second/page.xml",
        },
    )
    bind_engine.core_engine = types.SimpleNamespace(process=mock.Mock())

    with pytest.raises(ValueError, match="unique stems"):
        bind_engine.process("/batch", metakat_io)

    bind_engine.core_engine.process.assert_not_called()


def test_pages_without_alto_are_not_passed_to_the_core(
    bind_engine,
    evidence,
):
    batch_id = uuid4()
    volume = MetakatVolume(
        id=uuid4(),
        hierarchy=HierarchyType.MONOGRAPH,
    )
    first = MetakatPage(
        id=uuid4(),
        batch_id=batch_id,
        batch_index=0,
        pageIndex=1,
        parent_id=volume.id,
    )
    missing_alto = MetakatPage(
        id=uuid4(),
        batch_id=batch_id,
        batch_index=1,
        pageIndex=10,
        parent_id=volume.id,
    )
    metakat_io = MetakatIO(
        batch_id=batch_id,
        elements=[volume, first, missing_alto],
        page_to_image_mapping={
            first.id: "first.jpg",
            missing_alto.id: "last.jpg",
        },
        page_to_alto_mapping={first.id: "first.xml"},
    )
    bind_engine.core_engine = types.SimpleNamespace(
        process=mock.Mock(
            return_value=TocResult(
                (
                    ChapterResult(
                        toc_page_key="first",
                        title=evidence("Chapter", "first"),
                        page_start_key="first",
                    ),
                ),
            )
        )
    )

    result = bind_engine.process("/batch", metakat_io)

    bind_engine.core_engine.process.assert_called_once_with(
        ["/batch/first.jpg"],
        ["/batch/first.xml"],
        page_numbers=None,
    )
    assert any(element.type == "chapter" for element in result.elements)


def test_page_with_parent_cycle_is_ignored(bind_engine, caplog):
    batch_id = uuid4()
    first_id = uuid4()
    second_id = uuid4()
    first = MetakatChapter(id=first_id, parent_id=second_id)
    second = MetakatChapter(id=second_id, parent_id=first_id)
    page = MetakatPage(
        id=uuid4(),
        batch_id=batch_id,
        batch_index=0,
        parent_id=first.id,
    )
    metakat_io = MetakatIO(
        batch_id=batch_id,
        elements=[first, second, page],
        page_to_image_mapping={page.id: "page.jpg"},
        page_to_alto_mapping={page.id: "page.xml"},
    )
    bind_engine.core_engine = types.SimpleNamespace(
        process=mock.Mock(return_value=TocResult(()))
    )

    with caplog.at_level(logging.WARNING, logger=BIND_LOGGER):
        result = bind_engine.process("/batch", metakat_io)

    assert (
        "Ignoring 1 page(s) whose direct parent is not an eligible"
        in _messages(caplog)
    )
    bind_engine.core_engine.process.assert_not_called()
    assert not any(element.type == "volume" for element in result.elements)
    result_page = next(
        element for element in result.elements if element.type == "page"
    )
    assert result_page.parent_id == first.id


def test_recursive_result_binds_schema_and_detection_provenance(
    bind_engine,
    evidence,
):
    volume_id = uuid4()
    batch_id = uuid4()
    pages = [
        MetakatPage(
            id=uuid4(),
            batch_id=batch_id,
            batch_index=index,
            pageIndex=page_index,
            parent_id=volume_id,
        )
        for index, page_index in enumerate((3, 10, 20))
    ]
    result = TocResult(
        chapters=(
            ChapterResult(
                toc_page_key="toc",
                title=evidence("Chapter", "toc"),
                subtitle=evidence("Subtitle", "toc", y=30),
                page_number=ArabicRomanChapterPageNumberParser.create(
                    evidence("10", "toc", x=500)
                ),
                title_destination_page=evidence("CHAPTER", "destination"),
                page_start_key="destination",
                page_end_key="last",
                children=(
                    ChapterResult(
                        toc_page_key="toc",
                        title=evidence("Child", "toc", y=50),
                        page_start_key="last",
                        page_end_key="last",
                    ),
                ),
            ),
        ),
    )
    page_by_key = {
        "toc": pages[0],
        "destination": pages[1],
        "last": pages[2],
    }

    elements, bbox_by_id, page_by_detection = (
        bind_engine.extract_metakat_elements_from_pipeline(
            result,
            page_by_key,
            container_id=volume_id,
        )
    )

    chapters = [element for element in elements if element.type == "chapter"]
    assert len(chapters) == 2
    root, child = chapters
    assert root.parent_id == volume_id
    assert child.parent_id == root.id
    assert root.pageIndexToc == 3
    assert root.pageIndexStart == 10
    assert root.pageIndexEnd == 20
    assert child.pageIndexEnd == 20
    assert root.pageNumber[0] == "10"
    assert root.subTitle[0] == "Subtitle"
    assert root.title_destination_page[0] == "CHAPTER"
    assert root.id not in bbox_by_id
    assert page_by_detection[root.title[2]] == pages[0].id
    assert page_by_detection[root.title_destination_page[2]] == pages[1].id
    assert len(bbox_by_id) == 5


def test_explicit_container_parents_all_chapter_roots(bind_engine, evidence):
    container_id = uuid4()
    page = MetakatPage(
        id=uuid4(),
        batch_id=uuid4(),
        batch_index=0,
        pageIndex=0,
    )
    result = TocResult(
        chapters=(
            ChapterResult(
                toc_page_key="page",
                title=evidence("One", "page"),
            ),
            ChapterResult(
                toc_page_key="page",
                title=evidence("Two", "page"),
            ),
        ),
    )

    elements, _, _ = bind_engine.extract_metakat_elements_from_pipeline(
        result,
        {"page": page},
        container_id=container_id,
    )

    chapters = [element for element in elements if element.type == "chapter"]
    assert not any(isinstance(element, MetakatVolume) for element in elements)
    assert all(chapter.parent_id == container_id for chapter in chapters)


def test_titleless_chapter_uses_destination_title_evidence(bind_engine, evidence):
    volume_id = uuid4()
    page = MetakatPage(
        id=uuid4(),
        batch_id=uuid4(),
        batch_index=0,
        pageIndex=7,
        parent_id=volume_id,
    )
    result = TocResult(
        chapters=(
            ChapterResult(
                toc_page_key="page",
                title=None,
                page_number=ArabicRomanChapterPageNumberParser.create(
                    evidence("10", "page")
                ),
                title_destination_page=evidence(
                    "Destination title",
                    "page",
                ),
                page_start_key="page",
            ),
        ),
    )

    elements, bbox_by_id, page_by_detection = (
        bind_engine.extract_metakat_elements_from_pipeline(
            result,
            {"page": page},
            container_id=volume_id,
        )
    )

    chapter = next(element for element in elements if element.type == "chapter")
    assert chapter.title is None
    assert chapter.title_destination_page[0] == "Destination title"
    assert chapter.pageIndexStart == 7
    assert len(bbox_by_id) == 2
    assert page_by_detection[chapter.title_destination_page[2]] == page.id

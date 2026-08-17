import json
import sys
from unittest import mock
from uuid import uuid4

import pymupdf
import pytest

from metakat.schemas.base_objects import (
    MetakatChapter,
    MetakatIO,
    MetakatIssue,
    MetakatPage,
    MetakatVolume,
    PageType,
)
from metakat.tools import create_interactive_pdf as pdf_module


def test_single_document_creates_chapter_outline_and_toc_link(tmp_path, page_image):
    page_image(tmp_path / "first.jpg")
    page_image(tmp_path / "second.jpg")
    volume = MetakatVolume(id=uuid4(), title=("Volume", 0.9, uuid4()))
    pages = [
        MetakatPage(
            id=uuid4(),
            batch_id=uuid4(),
            batch_index=index,
            pageIndex=index,
            parent_id=volume.id,
        )
        for index in range(2)
    ]
    title_detection = uuid4()
    chapter = MetakatChapter(
        id=uuid4(),
        parent_id=volume.id,
        pageIndexToc=0,
        pageIndexStart=1,
        title=("Chapter one", 0.8, title_detection),
    )
    metakat_io = MetakatIO(
        batch_id=uuid4(),
        elements=[volume, *pages, chapter],
        page_to_image_mapping={
            pages[0].id: "first.jpg",
            pages[1].id: "second.jpg",
        },
        detection_to_bbox={title_detection: (10, 20, 30, 40)},
        detection_to_page_mapping={title_detection: pages[0].id},
    )
    output = tmp_path / "output.pdf"

    pdf_module.create_interactive_pdf(tmp_path, metakat_io, output)

    document = pymupdf.open(output)
    try:
        assert document.page_count == 2
        assert document.get_toc() == [[1, "Chapter one", 2]]
        links = document[0].get_links()
        assert len(links) == 1
        assert links[0]["page"] == 1
        expected = pymupdf.Rect(
            10 * document[0].rect.width / 100,
            20 * document[0].rect.height / 200,
            40 * document[0].rect.width / 100,
            60 * document[0].rect.height / 200,
        )
        actual = links[0]["from"]
        for actual_value, expected_value in zip(actual, expected):
            assert actual_value == pytest.approx(expected_value, abs=5e-4)
    finally:
        document.close()


def test_multiple_documents_allow_repeated_local_page_indices(tmp_path, page_image):
    issues = []
    pages = []
    chapters = []
    image_mapping = {}
    for document_index, label in enumerate(("Issue A", "Issue B")):
        issue = MetakatIssue(
            id=uuid4(),
            title=(label, 0.9, uuid4()),
        )
        issues.append(issue)
        document_pages = []
        for local_index in range(2):
            batch_index = document_index * 2 + local_index
            filename = f"page-{batch_index}.jpg"
            page_image(tmp_path / filename)
            page = MetakatPage(
                id=uuid4(),
                batch_id=uuid4(),
                batch_index=batch_index,
                pageIndex=local_index,
                parent_id=issue.id,
            )
            pages.append(page)
            document_pages.append(page)
            image_mapping[page.id] = filename
        issue.page_id = document_pages[0].id
        chapters.append(
            MetakatChapter(
                id=uuid4(),
                parent_id=issue.id,
                pageIndexStart=1,
                title=(
                    f"Chapter {document_index + 1}",
                    0.8,
                    uuid4(),
                ),
            )
        )
    metakat_io = MetakatIO(
        batch_id=uuid4(),
        elements=[*issues, *pages, *chapters],
        page_to_image_mapping=image_mapping,
    )
    output = tmp_path / "output.pdf"

    pdf_module.create_interactive_pdf(tmp_path, metakat_io, output)

    document = pymupdf.open(output)
    try:
        assert document.get_toc() == [
            [1, "monograph | Issue A", 1],
            [2, "Chapter 1", 2],
            [1, "monograph | Issue B", 3],
            [2, "Chapter 2", 4],
        ]
    finally:
        document.close()


def test_chapter_outline_label_combines_all_available_fields(tmp_path, page_image):
    page_image(tmp_path / "toc.jpg")
    page_image(tmp_path / "destination.jpg")
    volume = MetakatVolume(id=uuid4())
    pages = [
        MetakatPage(
            id=uuid4(),
            batch_id=uuid4(),
            batch_index=index,
            pageIndex=index,
            parent_id=volume.id,
        )
        for index in range(2)
    ]
    part_detection = uuid4()
    title_detection = uuid4()
    destination_detection = uuid4()
    page_number_detection = uuid4()
    chapter = MetakatChapter(
        id=uuid4(),
        parent_id=volume.id,
        pageIndexToc=0,
        pageIndexStart=1,
        partNumber=("I", 0.9, part_detection),
        title=("TOC title", 0.9, title_detection),
        title_destination_page=(
            "Destination title",
            0.8,
            destination_detection,
        ),
        pageNumber=("12", 0.9, page_number_detection),
    )
    output = tmp_path / "output.pdf"

    pdf_module.create_interactive_pdf(
        tmp_path,
        MetakatIO(
            batch_id=uuid4(),
            elements=[volume, *pages, chapter],
            page_to_image_mapping={
                pages[0].id: "toc.jpg",
                pages[1].id: "destination.jpg",
            },
            detection_to_bbox={
                part_detection: (5, 20, 10, 20),
                title_detection: (20, 20, 40, 20),
                page_number_detection: (80, 20, 10, 20),
                destination_detection: (10, 30, 70, 20),
            },
            detection_to_page_mapping={
                part_detection: pages[0].id,
                title_detection: pages[0].id,
                page_number_detection: pages[0].id,
                destination_detection: pages[1].id,
            },
        ),
        output,
    )

    document = pymupdf.open(output)
    try:
        assert document.get_toc() == [[1, "I | TOC title | 12", 2]]
        source_link = document[0].get_links()[0]
        assert source_link["page"] == 1
        assert "title_destination_page: Destination title (0.80)" in (
            document.xref_get_key(source_link["xref"], "Contents")[1]
        )
        destination_link = document[1].get_links()[0]
        assert destination_link["page"] == 0
        assert "title: TOC title (0.90)" in (
            document.xref_get_key(destination_link["xref"], "Contents")[1]
        )
        notes = list(document[1].annots())
        assert len(notes) == 1
        assert notes[0].info["subject"] == "Chapter metadata"
        assert "pageNumber: 12 (0.90)" in notes[0].info["content"]
    finally:
        document.close()


def test_chapter_outline_uses_destination_title_as_title_fallback():
    chapter = MetakatChapter(
        id=uuid4(),
        parent_id=uuid4(),
        pageIndexStart=0,
        partNumber=("I", 0.9, uuid4()),
        title_destination_page=("Destination title", 0.8, uuid4()),
        pageNumber=("12", 0.9, uuid4()),
    )

    assert pdf_module._chapter_label(chapter) == "I | Destination title | 12"


def test_page_and_bibliographic_sticky_notes_are_added(tmp_path, page_image):
    page_image(tmp_path / "first.jpg")
    page_image(tmp_path / "second.jpg")
    title_detection = uuid4()
    author_detection = uuid4()
    page_number_detection = uuid4()
    volume = MetakatVolume(
        id=uuid4(),
        title=("Book title", 0.95, title_detection),
        author=[("Book author", 0.85, author_detection)],
    )
    pages = [
        MetakatPage(
            id=uuid4(),
            batch_id=uuid4(),
            batch_index=index,
            pageIndex=index,
            parent_id=volume.id,
            pageType=(PageType.TITLE_PAGE, 0.9 - index * 0.1),
            pageNumber=("i", 0.8, page_number_detection)
            if index == 0
            else None,
        )
        for index in range(2)
    ]
    output = tmp_path / "output.pdf"

    pdf_module.create_interactive_pdf(
        tmp_path,
        MetakatIO(
            batch_id=uuid4(),
            elements=[volume, *pages],
            page_to_image_mapping={
                pages[0].id: "first.jpg",
                pages[1].id: "second.jpg",
            },
            detection_to_bbox={
                page_number_detection: (80, 170, 10, 15),
                title_detection: (10, 30, 70, 20),
                author_detection: (10, 70, 70, 20),
            },
            detection_to_page_mapping={
                page_number_detection: pages[0].id,
                title_detection: pages[1].id,
                author_detection: pages[1].id,
            },
        ),
        output,
    )

    document = pymupdf.open(output)
    try:
        first_notes = [annot.info for annot in document[0].annots()]
        second_notes = [annot.info for annot in document[1].annots()]
        assert {note["subject"] for note in first_notes} == {
            "Page type",
            "Page number",
            "Complete bibliographic information",
        }
        aggregate = next(
            note
            for note in first_notes
            if note["subject"] == "Complete bibliographic information"
        )
        assert "title: Book title (0.95)" in aggregate["content"]
        assert "author: Book author (0.85)" in aggregate["content"]
        assert [note["subject"] for note in second_notes].count(
            "Bibliographic detection"
        ) == 2
        assert "Page type" in {note["subject"] for note in second_notes}
    finally:
        document.close()


def test_multiple_links_on_one_toc_page_are_preserved(tmp_path, page_image):
    volume = MetakatVolume(id=uuid4())
    pages = []
    image_mapping = {}
    for index in range(3):
        filename = f"page-{index}.jpg"
        page_image(tmp_path / filename)
        page = MetakatPage(
            id=uuid4(),
            batch_id=uuid4(),
            batch_index=index,
            pageIndex=index,
            parent_id=volume.id,
        )
        pages.append(page)
        image_mapping[page.id] = filename
    detections = [uuid4(), uuid4()]
    chapters = [
        MetakatChapter(
            id=uuid4(),
            parent_id=volume.id,
            pageIndexToc=0,
            pageIndexStart=index + 1,
            title=(f"Chapter {index + 1}", 0.9, detections[index]),
        )
        for index in range(2)
    ]
    metakat_io = MetakatIO(
        batch_id=uuid4(),
        elements=[volume, *pages, *chapters],
        page_to_image_mapping=image_mapping,
        detection_to_bbox={
            detections[0]: (10, 20, 30, 20),
            detections[1]: (10, 60, 30, 20),
        },
        detection_to_page_mapping={
            detection: pages[0].id for detection in detections
        },
    )
    output = tmp_path / "output.pdf"

    pdf_module.create_interactive_pdf(tmp_path, metakat_io, output)

    document = pymupdf.open(output)
    try:
        assert len(document[0].get_links()) == 2
        assert {link["page"] for link in document[0].get_links()} == {1, 2}
    finally:
        document.close()


def test_chapter_hierarchy_is_preserved_in_outline(tmp_path, page_image):
    volume = MetakatVolume(id=uuid4())
    pages = []
    image_mapping = {}
    for index in range(2):
        filename = f"page-{index}.jpg"
        page_image(tmp_path / filename)
        page = MetakatPage(
            id=uuid4(),
            batch_id=uuid4(),
            batch_index=index,
            pageIndex=index,
            parent_id=volume.id,
        )
        pages.append(page)
        image_mapping[page.id] = filename
    parent = MetakatChapter(
        id=uuid4(),
        parent_id=volume.id,
        pageIndexStart=0,
        title=("Parent", 0.9, uuid4()),
    )
    child = MetakatChapter(
        id=uuid4(),
        parent_id=parent.id,
        pageIndexStart=1,
        title=("Child", 0.9, uuid4()),
    )
    metakat_io = MetakatIO(
        batch_id=uuid4(),
        elements=[volume, *pages, parent, child],
        page_to_image_mapping=image_mapping,
    )
    output = tmp_path / "output.pdf"

    pdf_module.create_interactive_pdf(tmp_path, metakat_io, output)

    document = pymupdf.open(output)
    try:
        assert document.get_toc() == [[1, "Parent", 1], [2, "Child", 2]]
    finally:
        document.close()


def test_duplicate_page_index_inside_document_is_rejected(tmp_path, page_image):
    volume = MetakatVolume(id=uuid4())
    pages = [
        MetakatPage(
            id=uuid4(),
            batch_id=uuid4(),
            batch_index=index,
            pageIndex=0,
            parent_id=volume.id,
        )
        for index in range(2)
    ]
    mapping = {}
    for index, page in enumerate(pages):
        filename = f"page-{index}.jpg"
        page_image(tmp_path / filename)
        mapping[page.id] = filename
    metakat_io = MetakatIO(
        batch_id=uuid4(),
        elements=[volume, *pages],
        page_to_image_mapping=mapping,
    )

    with pytest.raises(ValueError, match="Duplicate pageIndex 0"):
        pdf_module.create_interactive_pdf(
            tmp_path,
            metakat_io,
            tmp_path / "output.pdf",
        )


def test_standalone_main_loads_metakat_json(tmp_path):
    metakat_path = tmp_path / "metakat.json"
    metakat_path.write_text(
        json.dumps(MetakatIO(batch_id=uuid4()).model_dump(mode="json")),
        encoding="utf-8",
    )
    output = tmp_path / "output.pdf"
    with (
        mock.patch.object(
            sys,
            "argv",
            [
                "create_interactive_pdf",
                "--batch-dir",
                str(tmp_path),
                "--metakat-json",
                str(metakat_path),
                "--output-metakat-pdf",
                str(output),
            ],
        ),
        mock.patch.object(pdf_module, "create_interactive_pdf") as create,
    ):
        pdf_module.main()

    assert create.call_args.args[0] == str(tmp_path)
    assert isinstance(create.call_args.args[1], MetakatIO)
    assert create.call_args.args[2] == str(output)

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from uuid import uuid4

import pymupdf
from PIL import Image

import metakat.process_batch as process_batch_module
from metakat.schemas.base_objects import (
    MetakatChapter,
    MetakatIO,
    MetakatIssue,
    MetakatPage,
    MetakatVolume,
    PageType,
)
from metakat.tools import create_interactive_pdf as pdf_module


def _image(path: Path, size: tuple[int, int] = (100, 200)) -> None:
    Image.new("RGB", size, color="white").save(path)


class InteractivePDFTest(unittest.TestCase):
    def test_single_document_creates_chapter_outline_and_toc_link(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _image(root / "first.jpg")
            _image(root / "second.jpg")
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
            output = root / "output.pdf"

            pdf_module.create_interactive_pdf(root, metakat_io, output)

            document = pymupdf.open(output)
            try:
                self.assertEqual(document.page_count, 2)
                self.assertEqual(document.get_toc(), [[1, "Chapter one", 2]])
                links = document[0].get_links()
                self.assertEqual(len(links), 1)
                self.assertEqual(links[0]["page"], 1)
                expected = pymupdf.Rect(
                    10 * document[0].rect.width / 100,
                    20 * document[0].rect.height / 200,
                    40 * document[0].rect.width / 100,
                    60 * document[0].rect.height / 200,
                )
                actual = links[0]["from"]
                for actual_value, expected_value in zip(actual, expected):
                    self.assertAlmostEqual(actual_value, expected_value, places=3)
            finally:
                document.close()

    def test_multiple_documents_allow_repeated_local_page_indices(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
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
                    _image(root / filename)
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
            output = root / "output.pdf"

            pdf_module.create_interactive_pdf(root, metakat_io, output)

            document = pymupdf.open(output)
            try:
                self.assertEqual(
                    document.get_toc(),
                    [
                        [1, "monograph | Issue A", 1],
                        [2, "Chapter 1", 2],
                        [1, "monograph | Issue B", 3],
                        [2, "Chapter 2", 4],
                    ],
                )
            finally:
                document.close()

    def test_chapter_outline_label_combines_all_available_fields(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _image(root / "toc.jpg")
            _image(root / "destination.jpg")
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
            output = root / "output.pdf"

            pdf_module.create_interactive_pdf(
                root,
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
                self.assertEqual(
                    document.get_toc(),
                    [[1, "I | TOC title | 12", 2]],
                )
                source_link = document[0].get_links()[0]
                self.assertEqual(source_link["page"], 1)
                self.assertIn(
                    "title_destination_page: Destination title (0.80)",
                    document.xref_get_key(
                        source_link["xref"],
                        "Contents",
                    )[1],
                )
                destination_link = document[1].get_links()[0]
                self.assertEqual(destination_link["page"], 0)
                self.assertIn(
                    "title: TOC title (0.90)",
                    document.xref_get_key(
                        destination_link["xref"],
                        "Contents",
                    )[1],
                )
                notes = list(document[1].annots())
                self.assertEqual(len(notes), 1)
                self.assertEqual(notes[0].info["subject"], "Chapter metadata")
                self.assertIn(
                    "pageNumber: 12 (0.90)",
                    notes[0].info["content"],
                )
            finally:
                document.close()

    def test_chapter_outline_uses_destination_title_as_title_fallback(self):
        chapter = MetakatChapter(
            id=uuid4(),
            parent_id=uuid4(),
            pageIndexStart=0,
            partNumber=("I", 0.9, uuid4()),
            title_destination_page=("Destination title", 0.8, uuid4()),
            pageNumber=("12", 0.9, uuid4()),
        )

        self.assertEqual(
            pdf_module._chapter_label(chapter),
            "I | Destination title | 12",
        )

    def test_page_and_bibliographic_sticky_notes_are_added(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _image(root / "first.jpg")
            _image(root / "second.jpg")
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
            output = root / "output.pdf"

            pdf_module.create_interactive_pdf(
                root,
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
                self.assertEqual(
                    {note["subject"] for note in first_notes},
                    {
                        "Page type",
                        "Page number",
                        "Complete bibliographic information",
                    },
                )
                aggregate = next(
                    note
                    for note in first_notes
                    if note["subject"]
                    == "Complete bibliographic information"
                )
                self.assertIn("title: Book title (0.95)", aggregate["content"])
                self.assertIn("author: Book author (0.85)", aggregate["content"])
                self.assertEqual(
                    [note["subject"] for note in second_notes].count(
                        "Bibliographic detection"
                    ),
                    2,
                )
                self.assertIn("Page type", {
                    note["subject"] for note in second_notes
                })
            finally:
                document.close()

    def test_multiple_links_on_one_toc_page_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            volume = MetakatVolume(id=uuid4())
            pages = []
            image_mapping = {}
            for index in range(3):
                filename = f"page-{index}.jpg"
                _image(root / filename)
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
            output = root / "output.pdf"

            pdf_module.create_interactive_pdf(root, metakat_io, output)

            document = pymupdf.open(output)
            try:
                self.assertEqual(len(document[0].get_links()), 2)
                self.assertEqual(
                    {link["page"] for link in document[0].get_links()},
                    {1, 2},
                )
            finally:
                document.close()

    def test_chapter_hierarchy_is_preserved_in_outline(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            volume = MetakatVolume(id=uuid4())
            pages = []
            image_mapping = {}
            for index in range(2):
                filename = f"page-{index}.jpg"
                _image(root / filename)
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
            output = root / "output.pdf"

            pdf_module.create_interactive_pdf(root, metakat_io, output)

            document = pymupdf.open(output)
            try:
                self.assertEqual(
                    document.get_toc(),
                    [[1, "Parent", 1], [2, "Child", 2]],
                )
            finally:
                document.close()

    def test_duplicate_page_index_inside_document_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
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
                _image(root / filename)
                mapping[page.id] = filename
            metakat_io = MetakatIO(
                batch_id=uuid4(),
                elements=[volume, *pages],
                page_to_image_mapping=mapping,
            )

            with self.assertRaisesRegex(ValueError, "Duplicate pageIndex 0"):
                pdf_module.create_interactive_pdf(
                    root,
                    metakat_io,
                    root / "output.pdf",
                )

    def test_standalone_main_loads_metakat_json(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            metakat_path = root / "metakat.json"
            metakat_path.write_text(
                json.dumps(MetakatIO(batch_id=uuid4()).model_dump(mode="json")),
                encoding="utf-8",
            )
            output = root / "output.pdf"
            with (
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "create_interactive_pdf",
                        "--batch-dir",
                        str(root),
                        "--metakat-json",
                        str(metakat_path),
                        "--output-metakat-pdf",
                        str(output),
                    ],
                ),
                mock.patch.object(pdf_module, "create_interactive_pdf") as create,
            ):
                pdf_module.main()

            self.assertEqual(create.call_args.args[0], str(root))
            self.assertIsInstance(create.call_args.args[1], MetakatIO)
            self.assertEqual(create.call_args.args[2], str(output))

    def test_process_batch_calls_exporter_only_when_requested(self):
        metakat_io = MetakatIO(batch_id=uuid4())
        with (
            mock.patch.object(
                process_batch_module,
                "init_io",
                return_value=(metakat_io, None),
            ),
            mock.patch.object(
                process_batch_module,
                "create_interactive_pdf",
            ) as create,
        ):
            process_batch_module.process_batch(
                batch_dir="batch",
                engine_config={},
            )
            create.assert_not_called()
            process_batch_module.process_batch(
                batch_dir="batch",
                engine_config={},
                output_metakat_pdf="output.pdf",
            )

        create.assert_called_once_with("batch", metakat_io, "output.pdf")


if __name__ == "__main__":
    unittest.main()

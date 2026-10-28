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
    Value,
)
from metakat.tools import create_interactive_pdf as pdf_module


def _v(text, confidence, value_id=None):
    return Value(text=text, confidence=confidence, id=value_id or uuid4())


def _run(scan_index):
    """A pageIndexStart/pageIndexEnd list holding one run."""
    return [(scan_index, uuid4())]


@pytest.mark.parametrize("downscale", [False, True])
def test_document_start_markers_are_page_content_and_have_bookmarks(tmp_path, page_image, downscale):
    volume = MetakatVolume(id=uuid4())
    pages = []
    mapping = {}
    for index, flag in enumerate((None, True, False, True)):
        page = MetakatPage(
            id=uuid4(), batch_id=uuid4(), batch_index=index, pageIndex=index,
            parent_id=volume.id,
            startsDocument=None if flag is None else (flag, 0.0 if index == 3 else 0.95),
        )
        pages.append(page)
        filename = f"page-{index}.jpg"
        page_image(tmp_path / filename)
        mapping[page.id] = filename
    chapter = MetakatChapter(
        id=uuid4(), parent_id=volume.id, pageIndexStart=_run(2),
        titleTocPage=[_v("Existing chapter", 0.9)],
    )
    metakat_io = MetakatIO(
        batch_id=uuid4(), elements=[volume, *pages, chapter], page_to_image_mapping=mapping,
    )
    before = metakat_io.model_dump_json()
    output = tmp_path / "marked.pdf"
    pdf_module.create_interactive_pdf(tmp_path, metakat_io, output, downscale=downscale)
    with pymupdf.open(output) as document:
        assert document.get_toc() == [
            [1, "Existing chapter", 3],
            [1, "Detected document starts", 2],
            [2, "Document start 1 | scan page 2", 2],
            [2, "Document start 2 | scan page 4", 4],
        ]
        for index, page in enumerate(document):
            marked = index in (1, 3)
            assert ("DOCUMENT START" in page.get_text()) is marked
            frames = [drawing for drawing in page.get_drawings()
                      if drawing["color"] is not None and
                      drawing["color"] == pytest.approx(pdf_module._DOCUMENT_START_COLOR)]
            assert bool(frames) is marked
            if marked:
                assert frames[0]["width"] == pytest.approx(min(page.rect.width, page.rect.height) * 0.012)
                assert page.rect.contains(frames[0]["rect"])
    assert metakat_io.model_dump_json() == before


def test_single_document_creates_chapter_outline_and_toc_link(tmp_path, page_image):
    page_image(tmp_path / "first.jpg")
    page_image(tmp_path / "second.jpg")
    volume = MetakatVolume(id=uuid4(), title=[_v("Volume", 0.9)])
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
        pageIndexStart=_run(1),
        pageIndexTocPage=0,
        titleTocPage=[_v("Chapter one", 0.8, title_detection)],
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
            title=[_v(label, 0.9)],
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
        document_pages[0].representative = True
        chapters.append(
            MetakatChapter(
                id=uuid4(),
                parent_id=issue.id,
                pageIndexStart=_run(1),
                titleTocPage=[_v(f"Chapter {document_index + 1}", 0.8)],
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
        pageIndexStart=_run(1),
        title=[_v("Destination title", 0.8, destination_detection)],
        pageIndexTocPage=0,
        titleTocPage=[_v("TOC title", 0.9, title_detection)],
        partNumberTocPage=[_v("I", 0.9, part_detection)],
        pageNumberStartTocPage=[_v("12", 0.9, page_number_detection)],
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
        assert "title: Destination title (0.80)" in (
            document.xref_get_key(source_link["xref"], "Contents")[1]
        )
        destination_link = document[1].get_links()[0]
        assert destination_link["page"] == 0
        assert "titleTocPage: TOC title (0.90)" in (
            document.xref_get_key(destination_link["xref"], "Contents")[1]
        )
        notes = list(document[1].annots())
        assert len(notes) == 1
        assert notes[0].info["subject"] == "Chapter metadata"
        assert "pageNumberStartTocPage: 12 (0.90)" in notes[0].info["content"]
    finally:
        document.close()


def test_chapter_outline_uses_destination_title_as_title_fallback():
    chapter = MetakatChapter(
        id=uuid4(),
        parent_id=uuid4(),
        pageIndexStart=_run(0),
        title=[_v("Destination title", 0.8)],
        partNumberTocPage=[_v("I", 0.9)],
        pageNumberStartTocPage=[_v("12", 0.9)],
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
        title=[_v("Book title", 0.95, title_detection)],
        author=[_v("Book author", 0.85, author_detection)],
    )
    pages = [
        MetakatPage(
            id=uuid4(),
            batch_id=uuid4(),
            batch_index=index,
            pageIndex=index,
            parent_id=volume.id,
            pageType=(PageType.TITLE_PAGE, 0.9 - index * 0.1),
            pageNumber=_v("i", 0.8, page_number_detection)
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
            pageIndexStart=_run(index + 1),
            pageIndexTocPage=0,
            titleTocPage=[_v(f"Chapter {index + 1}", 0.9, detections[index])],
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
        pageIndexStart=_run(0),
        titleTocPage=[_v("Parent", 0.9)],
    )
    child = MetakatChapter(
        id=uuid4(),
        parent_id=parent.id,
        pageIndexStart=_run(1),
        titleTocPage=[_v("Child", 0.9)],
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


def test_downscale_preserves_geometry_and_interaction_and_reduces_pdf(tmp_path):
    from PIL import Image,ImageDraw,ImageFont
    image=Image.new('RGB',(2400,3200),'white')
    draw=ImageDraw.Draw(image)
    font=ImageFont.load_default(size=48)
    for y in range(100,3000,70):
        draw.text((100,y),'Readable scanned text for an interactive PDF 123456789',fill='black',font=font)
    image_path=tmp_path/'scan.jpg'
    image.save(image_path,quality=98)
    original_bytes=image_path.read_bytes()
    alto=tmp_path/'scan.xml'
    alto.write_text('<alto><Layout><Page WIDTH="2400" HEIGHT="3200"><TextBlock>'
                    '<TextLine HEIGHT="60"><String HEIGHT="48"/></TextLine>'
                    '</TextBlock></Page></Layout></alto>')
    volume=MetakatVolume(id=uuid4(),title=[_v('Book',.9)])
    pages=[MetakatPage(id=uuid4(),batch_id=uuid4(),batch_index=i,pageIndex=i,parent_id=volume.id)
           for i in range(2)]
    reading=_v('Chapter one',.9)
    chapter=MetakatChapter(id=uuid4(),parent_id=volume.id,pageIndexStart=_run(1),
                           pageIndexTocPage=0,titleTocPage=[reading])
    io=MetakatIO(batch_id=uuid4(),elements=[volume,*pages,chapter],
        page_to_image_mapping={p.id:'scan.jpg' for p in pages},
        page_to_alto_mapping={p.id:'scan.xml' for p in pages},
        detection_to_bbox={reading.id:(100,200,500,48)},
        detection_to_page_mapping={reading.id:pages[0].id})
    before=io.model_dump_json()
    full=tmp_path/'full.pdf';small=tmp_path/'small.pdf'
    pdf_module.create_interactive_pdf(tmp_path,io,full)
    pdf_module.create_interactive_pdf(tmp_path,io,small,downscale=True,max_image_dimension=1600)
    assert image_path.read_bytes()==original_bytes
    assert io.model_dump_json()==before
    assert small.stat().st_size < full.stat().st_size*.7
    with pymupdf.open(full) as a,pymupdf.open(small) as b:
        assert a.get_toc()==b.get_toc() and a.page_count==b.page_count==2
        for original,reduced in zip(a,b):
            assert tuple(original.rect)==pytest.approx(tuple(reduced.rect))
            raster=reduced.get_images(full=True)[0]
            assert raster[2:4]==(1200,1600)
            assert len(original.get_links())==len(reduced.get_links())
            for old,new in zip(original.get_links(),reduced.get_links()):
                assert old['page']==new['page']
                assert tuple(old['from'])==pytest.approx(tuple(new['from']))
            old_notes=list(original.annots() or []);new_notes=list(reduced.annots() or [])
            assert len(old_notes)==len(new_notes)
            for old,new in zip(old_notes,new_notes):
                assert old.info['content']==new.info['content']
                assert tuple(old.rect)==pytest.approx(tuple(new.rect))


def test_downscale_uses_smallest_text_and_never_upscales(tmp_path):
    from PIL import Image
    assert pdf_module._downscale_ratio(4000,6000,8,16,2400)==1
    assert pdf_module._downscale_ratio(4000,6000,32,16,2400)==.5
    assert pdf_module._downscale_ratio(4000,6000,None,16,2400)==.5
    assert pdf_module._downscale_ratio(400,600,32,16,2400)==1
    path=tmp_path/'tiny.png';Image.new('RGB',(400,600),'white').save(path)
    assert pdf_module._reduced_image(path,400,600,1) is None


def test_ocr_readability_geometry_in_alto_and_page_xml(tmp_path):
    alto=tmp_path/'alto.xml'
    alto.write_text('<alto xmlns="urn:alto"><Layout><Page HEIGHT="2000">'
                    '<TextLine HEIGHT="50"><String HEIGHT="30"/><String HEIGHT="10"/>'
                    '</TextLine></Page></Layout></alto>')
    assert pdf_module._ocr_text_height(alto,4000)==100
    alto.write_text('<alto><Page HEIGHT="2000"><TextLine><String HEIGHT="30"/>'
                    '<String HEIGHT="3"/></TextLine></Page></alto>')
    assert pdf_module._ocr_text_height(alto,4000)==60
    page=tmp_path/'page.xml'
    page.write_text('<PcGts><Page imageHeight="2000"><TextRegion><TextLine>'
                    '<Coords points="10,20 100,20 100,50 10,50"/>'
                    '</TextLine></TextRegion></Page></PcGts>')
    assert pdf_module._ocr_text_height(page,4000)==60
    page.write_text('invalid XML')
    assert pdf_module._ocr_text_height(page,4000) is None


def test_all_detections_contribute_and_multiline_boxes_are_not_single_lines():
    volume=MetakatVolume(id=uuid4())
    a,b,c=uuid4(),uuid4(),uuid4();page_id=uuid4()
    volume.title=[_v('A title',.9,a),_v('A long paragraph '*100,.8,b)]
    io=MetakatIO(batch_id=uuid4(),elements=[volume],
        detection_to_bbox={a:(0,0,500,60),b:(0,100,500,600),c:(0,0,10,3)},
        detection_to_page_mapping={a:page_id,b:page_id,c:page_id})
    # Even an unlabelled tiny detection constrains downscaling.
    assert pdf_module._detection_text_heights(io)[page_id]==3
    io.detection_to_bbox.pop(c)
    assert pdf_module._detection_text_heights(io)[page_id] < 60


@pytest.mark.parametrize('settings',[{'min_text_pixels':0},{'min_text_pixels':float('nan')},
                                      {'max_image_dimension':0}])
def test_invalid_downscale_settings_are_rejected(tmp_path,settings):
    with pytest.raises(ValueError):
        pdf_module.create_interactive_pdf(tmp_path,MetakatIO(batch_id=uuid4()),tmp_path/'bad.pdf',
                                          downscale=True,**settings)


def test_standalone_downscale_options_are_forwarded(tmp_path):
    metakat_path=tmp_path/'metakat.json'
    metakat_path.write_text(MetakatIO(batch_id=uuid4()).model_dump_json())
    with (mock.patch.object(sys,'argv',['create_interactive_pdf','--batch-dir',str(tmp_path),
            '--metakat-json',str(metakat_path),'--output-metakat-pdf',str(tmp_path/'out.pdf'),
            '--downscale','--min-text-pixels','20','--max-image-dimension','2000']),
          mock.patch.object(pdf_module,'create_interactive_pdf') as create):
        pdf_module.main()
    assert create.call_args.kwargs == {'downscale':True,'min_text_pixels':20.,'max_image_dimension':2000}

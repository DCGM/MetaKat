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


def test_all_page_types_and_sides_have_colors():
    assert set(pdf_module._PAGE_TYPE_COLORS) == {p.value for p in PageType}
    assert len(set(pdf_module._PAGE_TYPE_COLORS.values())) == len(PageType)
    assert len(set(pdf_module._SIDE_COLORS.values())) == 3


@pytest.mark.parametrize("downscale", [False, True])
@pytest.mark.parametrize("starts", [None, False, True])
def test_full_width_panel_bands_and_stacked_confidences(tmp_path, page_image, downscale, starts):
    volume = MetakatVolume(id=uuid4())
    page = MetakatPage(
        id=uuid4(), batch_id=uuid4(), batch_index=0, parent_id=volume.id,
        pageType=(PageType.TITLE_PAGE, .914), side=("left", .876),
        pageNumber=_v("[123]", .999),
        startsDocument=None if starts is None else (starts, .95),
    )
    page_image(tmp_path / "page.jpg")
    io = MetakatIO(batch_id=page.batch_id, elements=[volume, page],
                   page_to_image_mapping={page.id: "page.jpg"})
    original = io.model_dump_json()
    output = tmp_path / "panel.pdf"
    pdf_module.create_interactive_pdf(tmp_path, io, output, downscale=downscale)
    with pymupdf.open(output) as doc:
        rendered = doc[0]
        image = rendered.get_image_rects(rendered.get_images()[0][0])[0]
        bands = [d for d in rendered.get_drawings()
                 if d["fill"] is not None and d["rect"].height < image.height]
        assert len(bands) == 4
        colors = (pdf_module._boundary_color(page.startsDocument),
                  pdf_module._PAGE_TYPE_COLORS["titlePage"], pdf_module._side_color(page.side),
                  pdf_module._PAGE_NUMBER_COLORS["bracketed"])
        labels = (None, "titlePage", "left", "[123]")
        confidences = ("--" if starts is None else "0.95" if starts else "0.05", "0.91", "0.88", "1.00")
        for index, (band, color, label, confidence) in enumerate(zip(bands, colors, labels, confidences)):
            assert band["fill"] == pytest.approx(color)
            assert band["rect"].x0 == pytest.approx(image.x1)
            assert band["rect"].x1 == pytest.approx(rendered.rect.x1)
            assert band["rect"].height == pytest.approx(bands[0]["rect"].height)
            conf, = rendered.search_for(confidence)
            assert conf.y0 > band["rect"].y1
            if label:
                text, = rendered.search_for(label)
                assert band["rect"].y1 < text.y0 < text.y1 < conf.y0
            if index < 3:
                assert conf.y1 < bands[index+1]["rect"].y0
        _, font_size, padding = pdf_module._panel_dimensions(pymupdf, min(image.width, image.height))
        assert bands[0]["rect"].y0 == pytest.approx(0)
        last_confidence, = rendered.search_for("1.00")
        assert last_confidence.y1 < rendered.rect.y1
        assert rendered.rect.y1 - last_confidence.y1 < font_size + padding
        assert sum(band["rect"].height for band in bands) > image.height * .7
        assert not any(d["color"] is not None for d in rendered.get_drawings())
        assert doc.get_toc() == ([[1, "Detected document starts", 1],
                                  [2, "Document start 1 | scan page 1", 1]] if starts else [])
    assert io.model_dump_json() == original


def test_number_sequence_states_account_for_gaps_and_reset_at_boundaries():
    batch = uuid4()
    def pages(numbers, starts=()):
        return [MetakatPage(id=uuid4(), batch_id=batch, batch_index=i,
                           pageNumber=None if n is None else _v(n, .9),
                           startsDocument=(i in starts, .9)) for i, n in enumerate(numbers)]
    for numbers, boundaries, expected in (
        (["10", "11", "[12]", "15", "16"], (),
         ["normal", "normal", "bracketed", "discontinuity", "normal"]),
        (["10", None, "12", "1", "2"], (3,),
         ["normal", "missing", "normal", "normal", "normal"]),
        (["i", "ii", "1", "2", "[5]"], (),
         ["normal", "normal", "normal", "normal", "discontinuity"]),
    ):
        sample = pages(numbers, boundaries)
        states = pdf_module._page_number_states(sample)
        assert [states[p.id] for p in sample] == expected


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
        image_rect = document[0].get_image_rects(document[0].get_images()[0][0])[0]
        expected = pymupdf.Rect(
            10 * image_rect.width / 100,
            20 * document[0].rect.height / 200,
            40 * image_rect.width / 100,
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
        # Type confidences are printed in the panel rather than separate notes.
        for page in document:
            assert page.search_for("titlePage")
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
    assert create.call_args.kwargs == {'downscale':True,'min_text_pixels':20.,'max_image_dimension':2000,
                                       'page_height':pdf_module.DEFAULT_PAGE_HEIGHT}


@pytest.mark.parametrize("downscale", [False, True])
def test_metadata_strip_keeps_labels_outside_unchanged_image(tmp_path, page_image, downscale):
    page_image(tmp_path / "page.jpg")
    volume = MetakatVolume(id=uuid4())
    page = MetakatPage(
        id=uuid4(), batch_id=uuid4(), batch_index=0, parent_id=volume.id,
        pageType=(PageType.TITLE_PAGE, .9), pageNumber=_v("123", .8),
        startsDocument=(True, .9),
    )
    io = MetakatIO(batch_id=page.batch_id, elements=[volume, page],
                   page_to_image_mapping={page.id: "page.jpg"})
    output = tmp_path / "strip.pdf"
    with pymupdf.open(tmp_path / "page.jpg") as source:
        original = source[0].rect
    pdf_module.create_interactive_pdf(tmp_path, io, output, page_height=None,
                                      downscale=downscale, max_image_dimension=50)
    with pymupdf.open(output) as document:
        rendered = document[0]
        image_rect = rendered.get_image_rects(rendered.get_images()[0][0])[0]
        assert tuple(image_rect) == pytest.approx(tuple(original))
        assert rendered.rect.width > original.width
        assert rendered.rect.height == pytest.approx(original.height)
        # the strip is as wide as the panel text needs: proportional to the page, but never below the minimum font
        assert rendered.rect.width - original.width == pytest.approx(
            pdf_module._panel_dimensions(pymupdf, min(original.width, original.height))[0])
        strip = pymupdf.Rect(original.x1, 0, rendered.rect.x1, rendered.rect.y1)
        drawings = rendered.get_drawings()
        assert any(d["fill"] == (1., 1., 1.) and strip.contains(d["rect"])
                   for d in drawings)
        for drawing in drawings:
            assert drawing["color"] is None
            assert strip.contains(drawing["rect"])
        for text in ("titlePage", "123"):
            labels = rendered.search_for(text)
            assert len(labels) == 1
            assert labels[0].x0 >= original.x1
            assert rendered.rect.contains(labels[0])


@pytest.mark.parametrize("probability, expected", [
    (0., (1., 1., 1.)), (.25, (1., .5, .5)), (.5, (1., 0., 0.)),
    (.75, (.5, .175, .4)), (1., pdf_module._DOCUMENT_START_COLOR),
])
def test_boundary_color_uses_probability_not_decision_confidence(probability, expected):
    from metakat.schemas.base_objects import BooleanValue
    for flag in (False, True):
        value = BooleanValue(val=flag, confidence=probability if flag else 1.-probability)
        assert pdf_module._boundary_color(value) == pytest.approx(expected)
    assert pdf_module._boundary_color(None) == pdf_module._UNKNOWN_COLOR


@pytest.mark.parametrize("side", ["left", "right", "singlePage"])
@pytest.mark.parametrize("confidence", [0., .5, 1.])
def test_side_color_fades_from_white_to_full_color(side, confidence):
    from types import SimpleNamespace
    expected = tuple(1. - confidence * (1. - channel) for channel in pdf_module._SIDE_COLORS[side])
    assert pdf_module._side_color(SimpleNamespace(text=side, confidence=confidence)) == pytest.approx(expected)
    assert pdf_module._side_color(None) == pdf_module._UNKNOWN_COLOR


@pytest.mark.parametrize('downscale', [False, True])
def test_textless_pages_inherit_height_from_their_own_document(tmp_path, downscale):
    from PIL import Image
    volumes = [MetakatVolume(id=uuid4()), MetakatVolume(id=uuid4())]
    pages, images, altos = [], {}, {}
    for doc_index, volume in enumerate(volumes):
        dpi = 96 if doc_index == 0 else 144
        for kind, width in ((PageType.NORMAL_PAGE, 300), (PageType.BLANK, 300), (PageType.SPINE, 30)):
            index = len(pages)
            page = MetakatPage(id=uuid4(), batch_id=uuid4(), batch_index=index,
                               parent_id=volume.id, pageType=(kind, .9))
            name = f'page-{index}.jpg'
            Image.new('RGB', (width, 600), 'white').save(tmp_path / name,
                                                       dpi=(dpi, dpi) if kind == PageType.NORMAL_PAGE else (300, 300))
            images[page.id] = name
            if kind == PageType.NORMAL_PAGE:
                xml = f'page-{index}.xml'
                (tmp_path / xml).write_text('<alto><Page HEIGHT="600"><TextLine HEIGHT="30"/></Page></alto>')
                altos[page.id] = xml
            pages.append(page)
    io = MetakatIO(batch_id=uuid4(), elements=[*volumes, *pages],
                   page_to_image_mapping=images, page_to_alto_mapping=altos)
    before = io.model_dump_json()
    output = tmp_path / 'document-scales.pdf'
    pdf_module.create_interactive_pdf(tmp_path, io, output, downscale=downscale, page_height=None)
    with pymupdf.open(output) as doc:
        image_rects = [p.get_image_rects(p.get_images()[0][0])[0] for p in doc]
        for start in (0, 3):
            reference = image_rects[start]
            assert image_rects[start+1].height == pytest.approx(reference.height, abs=.001)
            assert image_rects[start+2].height == pytest.approx(reference.height, abs=.001)
            assert image_rects[start+1].width == pytest.approx(reference.width, abs=.001)
            assert image_rects[start+2].width == pytest.approx(reference.height * .05, abs=.001)
        assert image_rects[0].height != pytest.approx(image_rects[3].height)
    assert io.model_dump_json() == before


def _scan_batch(tmp_path, sizes):
    """Pages of different pixel sizes and resolutions, one document each, in one batch."""
    from PIL import Image
    volumes = [MetakatVolume(id=uuid4()) for _ in sizes]
    pages, images = [], {}
    for index, ((width, height), volume) in enumerate(zip(sizes, volumes)):
        page = MetakatPage(id=uuid4(), batch_id=uuid4(), batch_index=index, pageIndex=0,
                           parent_id=volume.id, pageType=(PageType.NORMAL_PAGE, .9))
        name = f'scan-{index}.jpg'
        Image.new('RGB', (width, height), 'white').save(tmp_path / name, dpi=(72 * (index + 1),) * 2)
        images[page.id] = name
        pages.append(page)
    return MetakatIO(batch_id=uuid4(), elements=[*volumes, *pages], page_to_image_mapping=images), pages


@pytest.mark.parametrize('downscale', [False, True])
def test_every_page_has_the_same_height_whatever_the_scan(tmp_path, downscale):
    io, pages = _scan_batch(tmp_path, [(3000, 4000), (400, 550), (800, 600), (200, 1000)])
    output = tmp_path / 'uniform.pdf'
    pdf_module.create_interactive_pdf(tmp_path, io, output, downscale=downscale)
    with pymupdf.open(output) as document:
        for page, (width, height) in zip(document, [(3000, 4000), (400, 550), (800, 600), (200, 1000)]):
            image = page.get_image_rects(page.get_images()[0][0])[0]
            assert image.height == pytest.approx(pdf_module.DEFAULT_PAGE_HEIGHT, abs=.01)
            assert image.width == pytest.approx(pdf_module.DEFAULT_PAGE_HEIGHT * width / height, abs=.01)
            assert page.rect.height == pytest.approx(pdf_module.DEFAULT_PAGE_HEIGHT, abs=.01)


def test_page_height_can_be_set_or_switched_off(tmp_path):
    io, pages = _scan_batch(tmp_path, [(3000, 4000), (400, 550)])
    output = tmp_path / 'height.pdf'
    pdf_module.create_interactive_pdf(tmp_path, io, output, page_height=700.)
    with pymupdf.open(output) as document:
        assert [round(p.rect.height) for p in document] == [700, 700]
    pdf_module.create_interactive_pdf(tmp_path, io, output, page_height=None)
    with pymupdf.open(output) as document:
        assert document[0].rect.height != pytest.approx(document[1].rect.height)
    for bad in (0, -1, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='page_height'):
            pdf_module.create_interactive_pdf(tmp_path, io, output, page_height=bad)


def test_panel_text_is_never_smaller_than_the_minimum_font_and_stays_readable_on_uniform_pages(tmp_path):
    assert pdf_module._panel_dimensions(pymupdf, 20)[1] == pdf_module.MINIMUM_PANEL_FONT
    assert pdf_module._panel_dimensions(pymupdf, 1000)[1] == pytest.approx(18.)
    # the narrowest ordinary page (aspect 0.5) at the default height
    short_edge = pdf_module.DEFAULT_PAGE_HEIGHT * .5
    assert pdf_module._panel_dimensions(pymupdf, short_edge)[1] >= 10.


def test_standalone_page_height_zero_means_the_native_size(tmp_path):
    metakat_path = tmp_path / 'metakat.json'
    metakat_path.write_text(MetakatIO(batch_id=uuid4()).model_dump_json())
    with (mock.patch.object(sys, 'argv', ['create_interactive_pdf', '--batch-dir', str(tmp_path),
            '--metakat-json', str(metakat_path), '--output-metakat-pdf', str(tmp_path / 'out.pdf'),
            '--page-height', '0']),
          mock.patch.object(pdf_module, 'create_interactive_pdf') as create):
        pdf_module.main()
    assert create.call_args.kwargs['page_height'] is None

import io

import pymupdf
from PIL import Image

from metakat.chapter.download_articles.common.first_page import extract_first_page
from metakat.chapter.download_articles.common.models import CatalogItem
from metakat.chapter.download_articles.common.oai import parse_records
from metakat.chapter.download_articles.common.preview import render_previews
from metakat.chapter.download_articles.common.selection import select_by_period, select_items
from metakat.chapter.download_articles.common.store import ArticleStore

OAI_PAGE = b"""<?xml version="1.0"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords>
<record><header><identifier>oai:x:1</identifier><datestamp>2022</datestamp><setSpec>s1</setSpec></header>
<metadata><oai_dc:dc xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/" xmlns:dc="http://purl.org/dc/elements/1.1/">
<dc:title>An article</dc:title><dc:title></dc:title><dc:date>2011</dc:date>
</oai_dc:dc></metadata></record>
<record><header status="deleted"><identifier>oai:x:2</identifier></header></record>
<resumptionToken completeListSize="2" cursor="0">token-1</resumptionToken>
</ListRecords></OAI-PMH>"""


def _item(item_id, journal, year, item_type="Article"):
    return CatalogItem(library="test", item_id=item_id, record_id=item_id, journal_id=journal, year=year,
                       item_type=item_type, pdf_urls=[f"https://x/{item_id}.pdf"])


def _pdf_with_scan(scan_size=(1240, 1754)):
    buffer = io.BytesIO()
    Image.new("RGB", scan_size, "white").save(buffer, format="JPEG")
    document = pymupdf.open()
    page = document.new_page(width=595, height=842)
    page.insert_image(page.rect, stream=buffer.getvalue())
    return document.tobytes()


def test_parse_records_skips_deleted_and_empty_values():
    records, token = parse_records(OAI_PAGE)
    assert token == "token-1"
    assert [r.identifier for r in records] == ["oai:x:1"]
    assert records[0].dc["title"] == ["An article"] and records[0].sets == ["s1"]


def test_selection_covers_every_journal_and_prefers_types():
    items = [_item("a1", "A", 1950, "Reviews"), _item("a2", "A", 1990), _item("b1", "B", 2000, "Reviews"),
             _item("c1", "C", 2000, "Advertisement")]
    selected = select_items(items, type_preference=("Article", "Reviews"))
    assert {i.item_id for i in selected} == {"a2", "b1"}


def test_selection_spreads_years_and_respects_earlier_picks():
    items = [_item(str(y), "A", y) for y in range(1950, 2021, 10)]
    first = select_items(items, per_journal=1)
    assert [i.year for i in first] == [1990]
    more = select_items(items, per_journal=3, min_year_gap=20, already_selected={("A", None): [1990]})
    years = sorted(i.year for i in more)
    assert all(abs(y - 1990) >= 20 for y in years)
    assert all(b - a >= 20 for a, b in zip(years, years[1:]))


def test_selection_covers_every_title_of_a_renamed_journal():
    old, new = _item("o1", "A", 1950), _item("n1", "A", 1990)
    old.journal_title, new.journal_title = "Old title", "New title"
    assert {i.item_id for i in select_items([old, new])} == {"o1", "n1"}


def test_period_selection_adds_first_last_and_one_per_period():
    items = [_item(str(y), "A", y) for y in range(1950, 1973) if y != 1966]
    items += [_item("1957r", "A", 1957, "Reviews")]
    selected = select_by_period(items, 5, type_preference=("Article", "Reviews"),
                                already_selected={("A", None): [1961]})
    assert sorted(i.year for i in selected) == [1950, 1957, 1967, 1972]
    assert next(i for i in selected if i.year == 1957).item_type == "Article"


def test_period_selection_keeps_first_year_once_its_only_item_is_stored():
    items = [_item(str(y), "A", y) for y in range(1999, 2010)]
    first = select_by_period(items, 5)
    assert sorted(i.year for i in first) == [1999, 2006, 2009]
    stored = {i.item_id for i in first}
    again = select_by_period(items, 5, already_selected={("A", None): [i.year for i in first]}, stored_ids=stored)
    assert again == []


def test_period_selection_prefers_items_without_a_request():
    items = [_item(str(y), "A", y) for y in range(1950, 1965)] + [_item("1950c", "A", 1950), _item("1956c", "A", 1956)]
    selected = select_by_period(items, 5, cheap=lambda item: item.item_id.endswith("c"))
    # First and last years stay, the first one with its cheap item; the 1955-1959 period takes the
    # cheap 1956 over the middle 1957.
    assert sorted(i.item_id for i in selected) == ["1950c", "1956c", "1964"]


def test_first_page_keeps_scan_bytes_at_native_resolution():
    data, extension, image = extract_first_page(_pdf_with_scan())
    assert extension == "jpg" and image.method == "embedded"
    assert (image.width, image.height) == (1240, 1754)
    assert Image.open(io.BytesIO(data)).size == (1240, 1754)
    assert round(image.dpi) == 150


def test_first_page_keeps_scan_placed_inside_margins():
    buffer = io.BytesIO()
    Image.new("L", (2000, 3000), 255).save(buffer, format="PNG")
    document = pymupdf.open()
    page = document.new_page(width=595, height=842)
    page.insert_image(pymupdf.Rect(60, 60, 535, 772), stream=buffer.getvalue())
    _, extension, image = extract_first_page(document.tobytes())
    assert image.method == "embedded" and (image.width, image.height) == (2000, 3000)


def test_first_page_renders_born_digital_page():
    document = pymupdf.open()
    document.new_page(width=595, height=842).insert_text((72, 72), "Title")
    data, extension, image = extract_first_page(document.tobytes(), default_dpi=144)
    assert extension == "jpg" and image.method == "rendered"
    assert (image.width, image.height) == (1190, 1684)


def test_store_writes_pdf_image_and_metadata(tmp_path):
    store = ArticleStore(tmp_path, "lib")
    item = _item("x1", "A", 1960)
    store.write_catalog([item])
    store.write_selection([item])
    assert store.read_selection() == ["x1"]
    article = store.store(item, _pdf_with_scan(), item.pdf_urls[0])
    assert store.is_stored("x1")
    assert (store.dir / article.image.file).exists() and (store.dir / article.pdf_file).exists()
    assert store.stored_years() == {("A", None): [1960]}


def test_store_keeps_page_image_as_served_and_previews_render(tmp_path):
    store = ArticleStore(tmp_path, "lib")
    buffer = io.BytesIO()
    Image.new("L", (1800, 2700), 255).save(buffer, format="JPEG")
    for year in range(1950, 1975, 5):
        article = store.store_image(_item(f"p{year}", "A", year), buffer.getvalue(), "jpg", "https://x/page")
    assert article.pdf_file is None and article.image.method == "page image"
    assert (article.image.width, article.image.height) == (1800, 2700)
    store.store(_item("b1", "B", 2000), _pdf_with_scan(), None)
    previews = render_previews(store)
    names = sorted(p.relative_to(store.dir).as_posix() for p in previews)
    assert names == ["previews/overview_01.jpg", "previews/timelines/01_A.jpg",
                     "previews/timelines/02_short_lived.jpg"]

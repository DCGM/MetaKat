import io

import pymupdf
from PIL import Image

from metakat.chapter.download_articles.first_page import extract_first_page
from metakat.chapter.download_articles.models import CatalogItem
from metakat.chapter.download_articles.oai import parse_records
from metakat.chapter.download_articles.selection import select_items
from metakat.chapter.download_articles.sources.muni_digilib import MuniDigilibSource, catalog_from_records
from metakat.chapter.download_articles.store import ArticleStore

MUNI_PAGE = b"""<?xml version="1.0"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords>
<record><header><identifier>oai:digilib.phil.muni.cz:node-56</identifier><datestamp>2023</datestamp></header>
<metadata><oai_dc:dc xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/" xmlns:dc="http://purl.org/dc/elements/1.1/">
<dc:title>Neograeca Bohemica</dc:title><dc:date>2014 --</dc:date>
<dc:identifier>https://hdl.handle.net/11222.digilib/142018</dc:identifier>
</oai_dc:dc></metadata></record>
<record><header><identifier>oai:digilib.phil.muni.cz:node-2639</identifier><datestamp>2022</datestamp></header>
<metadata><oai_dc:dc xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/" xmlns:dc="http://purl.org/dc/elements/1.1/">
<dc:title>Dimitrios Vikelas</dc:title><dc:title></dc:title><dc:creator>Moutafidou, Ariadni</dc:creator>
<dc:date>2011</dc:date><dc:type>Article</dc:type>
<dc:identifier>https://hdl.handle.net/11222.digilib/142365</dc:identifier><dc:language>cze</dc:language>
<dc:relation></dc:relation>
<dc:relation>https://digilib.phil.muni.cz/cs/handle/11222.digilib/142364</dc:relation>
<dc:relation>https://digilib.phil.muni.cz/en/handle/11222.digilib/142018</dc:relation>
<dc:relation>https://x/pdf/142365.pdf;https://x/pdf_secondary/142365-source-enhanced.pdf;https://x/pdf_secondary/142365-source.pdf</dc:relation>
<dc:rights>embargoed access</dc:rights>
</oai_dc:dc></metadata></record>
<record><header status="deleted"><identifier>oai:digilib.phil.muni.cz:node-1</identifier></header></record>
<resumptionToken completeListSize="3" cursor="0">token-1</resumptionToken>
</ListRecords></OAI-PMH>"""


def _item(item_id, journal, year, item_type="Article"):
    return CatalogItem(library="test", item_id=item_id, record_id=item_id, journal_id=journal, year=year,
                       item_type=item_type, pdf_urls=[f"https://x/{item_id}.pdf"])


def test_parse_records_skips_deleted_and_empty_values():
    records, token = parse_records(MUNI_PAGE)
    assert token == "token-1"
    assert [r.identifier for r in records] == ["oai:digilib.phil.muni.cz:node-56", "oai:digilib.phil.muni.cz:node-2639"]
    assert records[1].dc["title"] == ["Dimitrios Vikelas"]


def test_muni_catalog_links_journal_and_prefers_source_scan():
    records, _ = parse_records(MUNI_PAGE)
    [item] = catalog_from_records(records)
    assert item.item_id == "142365"
    assert item.journal_id == "142018" and item.journal_title == "Neograeca Bohemica"
    assert item.year == 2011 and item.volume is None and item.issue is None
    assert item.pdf_urls[0].endswith("142365-source.pdf")
    assert item.pdf_urls[-1].endswith("142365-source-enhanced.pdf")
    assert not MuniDigilibSource().is_available(item)


def test_selection_covers_every_journal_and_prefers_types():
    items = [_item("a1", "A", 1950, "Reviews"), _item("a2", "A", 1990), _item("b1", "B", 2000, "Reviews"),
             _item("c1", "C", 2000, "Advertisement")]
    selected = select_items(items, type_preference=("Article", "Reviews"))
    assert {i.item_id for i in selected} == {"a2", "b1"}


def test_selection_spreads_years_and_respects_earlier_picks():
    items = [_item(str(y), "A", y) for y in range(1950, 2021, 10)]
    first = select_items(items, per_journal=1)
    assert [i.year for i in first] == [1990]
    more = select_items(items, per_journal=3, min_year_gap=20, already_selected={"A": [1990]})
    years = sorted(i.year for i in more)
    assert all(abs(y - 1990) >= 20 for y in years)
    assert all(b - a >= 20 for a, b in zip(years, years[1:]))


def _pdf_with_scan(scan_size=(1240, 1754)):
    buffer = io.BytesIO()
    Image.new("RGB", scan_size, "white").save(buffer, format="JPEG")
    document = pymupdf.open()
    page = document.new_page(width=595, height=842)
    page.insert_image(page.rect, stream=buffer.getvalue())
    return document.tobytes()


def test_first_page_keeps_scan_bytes_at_native_resolution():
    data, extension, image = extract_first_page(_pdf_with_scan())
    assert extension == "jpg" and image.method == "embedded"
    assert (image.width, image.height) == (1240, 1754)
    assert Image.open(io.BytesIO(data)).size == (1240, 1754)
    assert round(image.dpi) == 150


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
    assert store.stored_years() == {"A": [1960]}

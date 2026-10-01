import io

import pymupdf
from PIL import Image

from metakat.chapter.download_articles.first_page import extract_first_page
from metakat.chapter.download_articles.models import CatalogItem
from metakat.chapter.download_articles.oai import parse_records
from metakat.chapter.download_articles.selection import select_items
from metakat.chapter.download_articles.sources.dml_cz import DmlCzSource
from metakat.chapter.download_articles.sources.dml_cz import catalog_from_records as dml_catalog_from_records
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


DML_PAGE = b"""<?xml version="1.0"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords>
<record><header><identifier>oai:oai.dml.cz:10338.dmlcz/102514</identifier><setSpec>hdl_10338.dmlcz_100003</setSpec></header>
<metadata><article xmlns="http://jats.nlm.nih.gov" xmlns:xlink="http://www.w3.org/1999/xlink" xml:lang="cs"><front>
<journal-meta><journal-id journal-id-type="dmlcz-id">100003</journal-id>
<journal-title-group><journal-title>Aplikace matematiky</journal-title></journal-title-group></journal-meta>
<article-meta><article-id pub-id-type="dmlcz-id">102514</article-id>
<article-categories><subj-group subj-group-type="dmlcz-article-type"><subject>math</subject></subj-group></article-categories>
<title-group><article-title>Reseni biharmonickeho problemu</article-title>
<trans-title-group xml:lang="en"><trans-title>A solution of the biharmonic problem</trans-title></trans-title-group></title-group>
<contrib-group content-type="authors"><contrib contrib-type="author"><name><surname>Babuska</surname><given-names>Ivo</given-names></name></contrib></contrib-group>
<pub-date><year>1956</year></pub-date><volume>01</volume><issue>1</issue><fpage>34</fpage><lpage>43</lpage>
<ext-link ext-link-type="eudml-fulltext:application/pdf" xlink:href="http://dml.cz/bitstream/handle/10338.dmlcz/102514/AplMat.pdf">PDF</ext-link>
<self-uri content-type="application/pdf" xlink:href="http://dml.cz/bitstream/handle/10338.dmlcz/102514/AplMat.pdf">Full (PDF)</self-uri>
<kwd-group><unstructured-kwd-group>partial differential equations</unstructured-kwd-group></kwd-group>
<kwd-group kwd-group-type="msc"><kwd>35-00</kwd></kwd-group>
</article-meta></front></article></metadata></record>
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


def test_dml_catalog_reads_journal_volume_issue_and_pdf():
    records, token = parse_records(DML_PAGE)
    assert token is None
    [item] = dml_catalog_from_records(records)
    assert (item.item_id, item.journal_id, item.journal_title) == ("102514", "100003", "Aplikace matematiky")
    assert (item.volume, item.issue, item.year, item.pages) == ("01", "1", 1956, "34-43")
    assert item.item_type == "math" and item.authors == ["Babuska, Ivo"] and item.languages == ["cs"]
    assert item.pdf_urls == ["https://dml.cz/bitstream/handle/10338.dmlcz/102514/AplMat.pdf"]
    assert item.record["trans_titles"] == ["A solution of the biharmonic problem"]
    assert item.record["keywords"] == ["partial differential equations"] and item.record["msc"] == ["35-00"]


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


def test_first_page_keeps_scan_placed_inside_margins():
    buffer = io.BytesIO()
    Image.new("L", (2000, 3000), 255).save(buffer, format="PNG")
    document = pymupdf.open()
    page = document.new_page(width=595, height=842)
    page.insert_image(pymupdf.Rect(60, 60, 535, 772), stream=buffer.getvalue())
    _, extension, image = extract_first_page(document.tobytes())
    assert image.method == "embedded" and (image.width, image.height) == (2000, 3000)


def test_dml_skips_its_cover_sheet():
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "Terms of use: ... The Czech Digital Mathematics Library http://dml.cz")
    document.new_page().insert_text((72, 72), "Title of the article")
    assert DmlCzSource().title_page_index(document.tobytes()) == 1
    assert DmlCzSource().title_page_index(_pdf_with_scan()) == 0


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


def test_selection_covers_every_title_of_a_renamed_journal():
    old, new = _item("o1", "A", 1950), _item("n1", "A", 1990)
    old.journal_title, new.journal_title = "Old title", "New title"
    assert {i.item_id for i in select_items([old, new])} == {"o1", "n1"}

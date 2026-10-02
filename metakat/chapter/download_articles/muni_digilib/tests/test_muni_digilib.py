import urllib.error

import pymupdf

from metakat.chapter.download_articles.common.models import CatalogItem
from metakat.chapter.download_articles.common.oai import parse_records
from metakat.chapter.download_articles.common.store import ArticleStore
from metakat.chapter.download_articles.muni_digilib import source as source_module
from metakat.chapter.download_articles.muni_digilib.source import MuniDigilibSource, catalog_from_records

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
</ListRecords></OAI-PMH>"""


def test_muni_catalog_links_journal_and_prefers_source_scan():
    records, _ = parse_records(MUNI_PAGE)
    [item] = catalog_from_records(records)
    assert item.item_id == "142365"
    assert item.journal_id == "142018" and item.journal_title == "Neograeca Bohemica"
    assert item.year == 2011 and item.volume is None and item.issue is None
    assert item.pdf_urls[0].endswith("142365-source.pdf")
    assert item.pdf_urls[-1].endswith("142365-source-enhanced.pdf")
    assert not MuniDigilibSource().is_available(item)


def _record(handle, title, relations=(), pdf=None):
    relation = "".join(f"<dc:relation>https://digilib.phil.muni.cz/handle/11222.digilib/{h}</dc:relation>"
                       for h in relations)
    if pdf:
        relation += f"<dc:relation>https://x/pdf/{pdf}</dc:relation>"
    return (f'<record><header><identifier>oai:x:{handle}</identifier><datestamp>2023</datestamp></header>'
            f'<metadata><oai_dc:dc xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/" '
            f'xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>{title}</dc:title><dc:date>2001</dc:date>'
            f'<dc:identifier>https://hdl.handle.net/11222.digilib/{handle}</dc:identifier>{relation}'
            f'</oai_dc:dc></metadata></record>')


def test_review_belongs_to_its_journal_not_to_the_reviewed_book():
    page = ('<?xml version="1.0"?><OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords>'
            + _record("j", "Religio") + _record("b", "A reviewed book")
            + _record("a1", "Article", ["j"], "a1.pdf") + _record("a2", "Article", ["j"], "a2.pdf")
            + _record("r", "Review", ["b", "j"], "r.pdf") + "</ListRecords></OAI-PMH>")
    records, _ = parse_records(page.encode())
    items = {item.item_id: item for item in catalog_from_records(records)}
    assert {item.journal_title for item in items.values()} == {"Religio"}


def test_ojs_journals_kept_here_join_the_catalog_with_the_known_file_first(tmp_path):
    store = ArticleStore(tmp_path, "journals.muni.cz")
    article = CatalogItem(library="journals.muni.cz", item_id="theatralia-1", record_id="1", journal_id="theatralia",
                          journal_title="Theatralia", year=2015, item_type="article",
                          pdf_urls=["https://journals.phil.muni.cz/theatralia/article/download/1/2"])
    known = article.model_copy(update={"item_id": "religio-1", "journal_id": "religio", "journal_title": "Religio"})
    other = article.model_copy(update={"item_id": "other-1", "journal_id": "other", "journal_title": "Other"})
    store.write_catalog([article, article.model_copy(update={"item_id": "theatralia-2"}), known, other])
    store.mark_unavailable(article, "hosted at digilib.phil.muni.cz: https://digilib.phil.muni.cz/x/T1.pdf")
    store.mark_unavailable(known, "hosted at digilib.phil.muni.cz: https://digilib.phil.muni.cz/x/R1.pdf")
    source = MuniDigilibSource()
    source.root = tmp_path
    items = {item.item_id: item for item in source.ojs_items({"religio"})}
    assert sorted(items) == ["theatralia-1", "theatralia-2"]
    assert items["theatralia-1"].library == "digilib.phil.muni.cz"
    assert items["theatralia-1"].pdf_urls[0] == "https://digilib.phil.muni.cz/x/T1.pdf"
    assert items["theatralia-2"].pdf_urls == article.pdf_urls


def test_saved_files_are_found_by_link_name_or_handle_and_title_page_skips_the_cover(tmp_path):
    two_pages = pymupdf.open()
    two_pages.new_page()
    two_pages.new_page()
    (tmp_path / "AH2026-1-12.pdf").write_bytes(two_pages.tobytes())
    (tmp_path / "128139.pdf").write_bytes(two_pages.tobytes())
    (tmp_path / "108207-source.pdf").write_bytes(two_pages.tobytes())
    source = MuniDigilibSource()

    def item(url):
        return CatalogItem(library="digilib.phil.muni.cz", item_id="x", record_id="x", pdf_urls=[url])

    new = source.local_pdf(item("https://digilib.phil.muni.cz/en/_flysystem/fedora/pdf/AH2026-1-12.pdf"), tmp_path)
    old = source.local_pdf(item("https://digilib.phil.muni.cz/bitstream/handle/11222.digilib/128139/2_AH_35.pdf"),
                           tmp_path)
    scan = source.local_pdf(item("https://x/pdf_secondary/108207-source.pdf"), tmp_path)
    assert old.url.endswith("2_AH_35.pdf")
    assert source.title_page_index(new.data, new.url) == 1 and source.title_page_index(old.data, old.url) == 1
    assert source.title_page_index(scan.data, scan.url) == 0


def test_picked_ojs_links_are_resolved_to_the_file_here(monkeypatch):
    item = CatalogItem(library="digilib.phil.muni.cz", item_id="t-2", record_id="2",
                       pdf_urls=["https://journals.phil.muni.cz/theatralia/article/download/2/3"])
    direct = CatalogItem(library="digilib.phil.muni.cz", item_id="h", record_id="h",
                         pdf_urls=["https://digilib.phil.muni.cz/x/H.pdf"])

    def redirect(url, follow_redirects=True):
        raise urllib.error.HTTPError(url, 302, "Found", {"Location": "https://digilib.phil.muni.cz/x/T2.pdf"}, None)

    monkeypatch.setattr(source_module, "http_get", redirect)
    assert MuniDigilibSource().resolve_links([item, direct])
    assert item.pdf_urls == ["https://digilib.phil.muni.cz/x/T2.pdf",
                             "https://journals.phil.muni.cz/theatralia/article/download/2/3"]
    assert direct.pdf_urls == ["https://digilib.phil.muni.cz/x/H.pdf"]

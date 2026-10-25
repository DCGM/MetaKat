import json
import re
import urllib.error
import urllib.parse

from metakat.chapter.download_articles.common import crossref
from metakat.chapter.download_articles.common.crossref import CrossrefSource, item_from_work
from metakat.chapter.download_articles.common.models import CatalogItem
from metakat.chapter.download_articles.common.store import ArticleStore


def _work(doi, issns, title="ORBIS SCHOLAE", year=2015, **extra):
    return {
        "DOI": doi, "title": ["Characteristics of the Didactic Material"], "container-title": [title],
        "ISSN": issns, "volume": "8", "issue": "2", "page": "9-26", "issued": {"date-parts": [[year, 4, 1]]},
        "author": [{"family": "Seoane", "given": "Ana"}, {"name": "Editorial board"}],
        "link": [{"URL": "https://karolinum.cz/data/clanek/3356/OS_2_2014_03_Seoane.pdf", "content-type": "unspecified"},
                 {"URL": "https://karolinum.cz/doi/10.14712/x.html", "content-type": "unspecified"}],
        "resource": {"primary": {"URL": "http://www.karolinum.cz/doi/10.14712/23363177.2015.64"}},
        "type": "journal-article", **extra,
    }


def test_crossref_work_gives_journal_volume_issue_year_and_pdf():
    item = item_from_work(_work("10.14712/23363177.2015.64", ["2336-3177", "1802-4637"]), "cuni")
    assert item.item_id == "10.14712_23363177.2015.64" and item.record["doi"] == ["10.14712/23363177.2015.64"]
    assert (item.journal_title, item.volume, item.issue, item.pages) == ("ORBIS SCHOLAE", "8", "2", "9-26")
    assert (item.year, item.date) == (2015, "2015-04-01")
    assert item.authors == ["Seoane, Ana", "Editorial board"]
    assert item.pdf_urls == ["https://karolinum.cz/data/clanek/3356/OS_2_2014_03_Seoane.pdf"]
    assert item.landing_url == "http://www.karolinum.cz/doi/10.14712/23363177.2015.64"


def test_similarity_checking_link_is_a_pdf_candidate_under_any_address():
    link = [{"URL": "https://editorial.upce.cz/1804-8048/30/03/1556/pdf", "content-type": "unspecified",
             "intended-application": "similarity-checking"},
            {"URL": "https://editorial.upce.cz/1804-8048/30/03/1556", "content-type": "unspecified",
             "intended-application": "text-mining"}]
    item = item_from_work(_work("10.46585/sp30031556", ["1804-8048"], link=link), "sites")
    assert item.pdf_urls == ["https://editorial.upce.cz/1804-8048/30/03/1556/pdf"]


class _Publisher(CrossrefSource):
    name = "publisher"
    prefixes = ("10.1",)
    exclude_libraries = ("earlier",)


def test_catalog_pages_with_the_cursor_groups_issns_and_leaves_known_dois(tmp_path, monkeypatch):
    pages = {
        "*": [_work("10.1/A", ["2336-3177", "1802-4637"]), _work("10.1/B", ["1802-4637"], title="Orbis Scholae")],
        "c1": [_work("10.1/C", ["2336-3177"], year=2016), _work("10.1/known", ["9999-0000"], title="Other")],
        "c2": [_work("10.1/D", ["1212-8570"], title="Studia theologica")],
        "c3": [],
    }
    following = {"*": "c1", "c1": "c2", "c2": "c3", "c3": "c4"}

    def fake_get(url):
        cursor = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["cursor"][0]
        return json.dumps({"message": {"items": pages[cursor], "next-cursor": following[cursor]}}).encode()

    monkeypatch.setattr(crossref, "http_get", fake_get)
    ArticleStore(tmp_path, "earlier").write_catalog([CatalogItem(library="earlier", item_id="k", record_id="k",
                                                                 record={"doi": ["10.1/KNOWN"]})])
    source = _Publisher()
    source.root = tmp_path
    items = source.build_catalog()
    assert sorted(item.record_id for item in items) == ["10.1/A", "10.1/B", "10.1/C", "10.1/D"]
    journals = {item.record_id: (item.journal_id, item.journal_title) for item in items}
    # Two of three works spell the title in capitals.
    assert {journals[doi] for doi in ("10.1/A", "10.1/B", "10.1/C")} == {("1802-4637+2336-3177", "ORBIS SCHOLAE")}
    assert journals["10.1/D"] == ("1212-8570", "Studia theologica")


def test_titles_lose_markup_and_spellings_differing_in_punctuation_are_one_journal():
    works = [_work("10.3/1", ["1213-1962"], title="Transactions of the VŠB - Technical University"),
             _work("10.3/2", ["1213-1962"], title="Transactions of the VŠB – Technical University"),
             _work("10.3/3", ["1213-1962"], title="Transactions of the VŠB – Technical University"),
             _work("10.3/4", ["2570-7434"], title="Business &amp; IT")]
    works[0]["title"] = ["The <i>Ostrava</i> &amp; Opava"]
    items = [item_from_work(work, "sites") for work in works]
    crossref.unify_journal_titles(items)
    assert items[0].title == "The Ostrava & Opava" and items[3].journal_title == "Business & IT"
    assert {item.journal_title for item in items[:3]} == {"Transactions of the VŠB – Technical University"}


def test_journal_is_asked_for_by_its_next_issn_when_crossref_lacks_one(tmp_path, monkeypatch):
    def fake_get(url):
        if "/journals/1805-9600/" in url:
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
        assert "/journals/1210-2512/" in url
        return json.dumps({"message": {"items": [_work("10.2/R", ["1210-2512"], title="Radioengineering")],
                                       "next-cursor": None}}).encode()

    class Sites(CrossrefSource):
        name = "sites"
        issns = (("1805-9600", "1210-2512"),)

    monkeypatch.setattr(crossref, "http_get", fake_get)
    source = Sites()
    source.root = tmp_path
    assert [item.record_id for item in source.build_catalog()] == ["10.2/R"]


def test_citation_pdf_url_in_either_attribute_order_is_made_absolute():
    page = ('<head><meta name="citation_title" content="T">'
            '<meta content="/bitstreams/1/download?x=1&amp;y=2" name="citation_pdf_url"/></head>')
    assert crossref.citation_pdf_url(page, "https://dspace.zcu.cz/items/1") == \
        "https://dspace.zcu.cz/bitstreams/1/download?x=1&y=2"
    assert crossref.citation_pdf_url('<meta name="citation_pdf_url" content="https://a.cz/1.pdf">', "") == \
        "https://a.cz/1.pdf"
    assert crossref.citation_pdf_url("<meta name='description' content='x'>", "https://a.cz") is None


def test_article_without_a_pdf_link_is_downloaded_from_its_landing_page(monkeypatch):
    class Landing(CrossrefSource):
        name = "landing"
        landing_pdf = True

    work = _work("10.4/L", ["1804-1930"], link=[])
    work["resource"] = {"primary": {"URL": "https://online.agris.cz/article/1"}}
    item = item_from_work(work, "landing")
    pages = {"https://online.agris.cz/article/1": b'<meta name="citation_pdf_url" content="/files/1.pdf">',
             "https://online.agris.cz/files/1.pdf": b"%PDF-1.7"}
    monkeypatch.setattr(crossref, "http_get", lambda url: pages[url])
    monkeypatch.setattr(CrossrefSource, "download_pdf", lambda self, url: pages[url])
    assert Landing().is_available(item) and not _Publisher().is_available(item)
    download = Landing().download(item)
    assert (download.url, download.data) == ("https://online.agris.cz/files/1.pdf", b"%PDF-1.7")
    # A landing page that is the PDF itself.
    pages["https://online.agris.cz/article/1"] = b"%PDF-1.4"
    assert Landing().download(item).data == b"%PDF-1.4"
    # A page without citation_pdf_url: the site's own file, not a cited work.
    Landing.landing_pdf_link = re.compile(r"https://eujem\.cz/wp-content/uploads/[^\"]+\.pdf")
    pages["https://online.agris.cz/article/1"] = (b'<a href="https://cdn.mises.org/Road.pdf">'
                                                   b'<a href="https://eujem.cz/wp-content/uploads/2026/04.pdf">')
    pages["https://eujem.cz/wp-content/uploads/2026/04.pdf"] = b"%PDF-1.5"
    assert Landing().download(item).url == "https://eujem.cz/wp-content/uploads/2026/04.pdf"

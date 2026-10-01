import json
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

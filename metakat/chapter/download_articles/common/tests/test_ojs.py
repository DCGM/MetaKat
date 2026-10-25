import email.message
import urllib.error

import pytest

from metakat.chapter.download_articles.common import ojs
from metakat.chapter.download_articles.common.models import CatalogItem
from metakat.chapter.download_articles.common.oai import OaiRecord, parse_records
from metakat.chapter.download_articles.common.ojs import OjsSource, item_from_record
from metakat.chapter.download_articles.common.source import DownloadBlocked
from metakat.chapter.download_articles.common.store import ArticleStore

OJS_PAGE = """<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords>
<record><header status="deleted"><identifier>oai:ojs.journals.muni.cz:article/1</identifier>
<setSpec>religio:Studie</setSpec></header></record>
<record><header><identifier>oai:ojs.journals.muni.cz:article/22442</identifier>
<setSpec>religio:Studie</setSpec></header><metadata>
<oai_dc:dc xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/" xmlns:dc="http://purl.org/dc/elements/1.1/">
<dc:title xml:lang="cs">Náboženství a moc</dc:title><dc:title xml:lang="en">Religion and Power</dc:title>
<dc:creator>Novák, Jan</dc:creator><dc:creator>Novák, Jan</dc:creator>
<dc:date>2013-07-04</dc:date><dc:type>info:eu-repo/semantics/article</dc:type>
<dc:identifier>https://journals.phil.muni.cz/religio/article/view/22442</dc:identifier>
<dc:identifier>10.5817/Rel2004-1-1</dc:identifier>
<dc:source xml:lang="cs-CZ">Religio; Vol 12 No 1-2 (2004): Tematické číslo; 5-26</dc:source>
<dc:source>2336-4475</dc:source><dc:source>1210-3640</dc:source>
<dc:language>ces</dc:language>
<dc:relation>https://journals.phil.muni.cz/religio/article/view/22442/17883</dc:relation>
<dc:relation>https://example.org/related</dc:relation>
</oai_dc:dc></metadata></record>
<record><header><identifier>oai:ojs.journals.muni.cz:article/22443</identifier>
<setSpec>religio:Recenze</setSpec></header><metadata>
<oai_dc:dc xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/" xmlns:dc="http://purl.org/dc/elements/1.1/">
<dc:title>A book</dc:title><dc:source>Religio; No 2 (2006)</dc:source>
</oai_dc:dc></metadata></record>
<record><header><identifier>oai:ojs.journals.muni.cz:article/9</identifier>
<setSpec>citedby:ART</setSpec></header><metadata>
<oai_dc:dc xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/" xmlns:dc="http://purl.org/dc/elements/1.1/">
<dc:title>Not a journal</dc:title></oai_dc:dc></metadata></record>
</ListRecords></OAI-PMH>""".encode()

JOURNALS = {"religio": "Religio"}
SETS = {"religio": "Religio", "religio:Studie": "Studie", "religio:Recenze": "Recenze", "citedby:ART": "Articles"}


def test_ojs_record_gives_journal_volume_issue_year_section_and_galley():
    records, _ = parse_records(OJS_PAGE)
    items = [item for record in records if (item := item_from_record(record, "lib", JOURNALS, SETS))]
    assert len(items) == 2
    article, review = items
    assert (article.item_id, article.journal_id, article.journal_title) == ("religio-22442", "religio", "Religio")
    assert (article.volume, article.issue, article.year, article.pages) == ("12", "1-2", 2004, "5-26")
    assert article.title == "Náboženství a moc" and article.authors == ["Novák, Jan"]
    assert article.item_type == "article" and article.record["section"] == ["Studie"]
    assert article.record["issn"] == ["2336-4475", "1210-3640"] and article.record["doi"] == ["10.5817/Rel2004-1-1"]
    assert article.pdf_urls == ["https://journals.phil.muni.cz/religio/article/download/22442/17883"]
    assert article.landing_url == "https://journals.phil.muni.cz/religio/article/view/22442"
    # The upload date is not the publication year; without "(year)" in the citation the year stays unknown.
    assert (review.item_type, review.issue, review.year, review.pdf_urls) == ("other", "2", 2006, [])


class _Platform(OjsSource):
    name = "platform"
    oai_urls = ("https://journals.example.cz/index/oai",)


def _redirect(location):
    headers = email.message.Message()
    headers["Location"] = location
    return urllib.error.HTTPError("u", 302, "Found", headers, None)


def test_galleys_follow_redirects_within_the_platform_only(monkeypatch):
    responses = {
        "https://journals.example.cz/a/article/download/1/2": _redirect("/a/article/download/1/2/file.pdf"),
        "https://journals.example.cz/a/article/download/1/2/file.pdf": b"%PDF-1.4 ...",
        "https://journals.example.cz/b/article/download/3/4": _redirect("https://digilib.example.cz/x.pdf"),
    }

    def fake_get(url, follow_redirects=True):
        assert follow_redirects is False
        response = responses[url]
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(ojs, "http_get", fake_get)
    source = _Platform()
    assert source.download_pdf("https://journals.example.cz/a/article/download/1/2").startswith(b"%PDF")
    with pytest.raises(DownloadBlocked, match="hosted at digilib.example.cz"):
        source.download_pdf("https://journals.example.cz/b/article/download/3/4")


def test_journal_hosted_elsewhere_is_no_longer_selected(tmp_path):
    store = ArticleStore(tmp_path, "platform")

    def item(journal, number):
        return CatalogItem(library="platform", item_id=f"{journal}-{number}", record_id=str(number),
                           journal_id=journal, journal_title=journal, year=2000 + number,
                           pdf_urls=[f"https://journals.example.cz/{journal}/article/download/{number}/1"])

    for number in (1, 2):
        store.mark_unavailable(item("elsewhere", number), "hosted at digilib.example.cz: https://digilib.example.cz/x.pdf")
    store.mark_unavailable(item("once", 1), "hosted at www.ceeol.com: https://www.ceeol.com/x")
    source = _Platform()
    source.root = tmp_path
    assert not source.is_available(item("elsewhere", 3))
    assert source.is_available(item("once", 2))
    assert not source.is_available(item("once", 3).model_copy(update={"title": "Obsah"}))


def test_control_characters_in_records_are_dropped():
    page = OJS_PAGE.replace(b"Religion and Power", b"Religion\x0b and&#x1; Power&#10;")
    records, _ = parse_records(page)
    assert records[0].dc["title"] == ["Náboženství a moc", "Religion and Power"]


def test_older_ojs_citation_gives_issue_and_year():
    record = OaiRecord("oai:antropoweb.cz:article/7", None, ["webzin:ST"],
                       {"source": ["AntropoWebzin; AntropoWebzin 1/2013; 3-16"], "date": ["2013-04-30"]})
    item = item_from_record(record, "lib", {"webzin": "AntropoWebzin"}, {"webzin:ST": "Studie"})
    assert (item.volume, item.issue, item.year, item.pages) == (None, "1", 2013, "3-16")

from metakat.chapter.download_articles.national_museum import source
from metakat.chapter.download_articles.national_museum.source import NationalMuseumSource, items_of_issue

ISSUE = """
<div class="pbl__cont">
    <h3 class="pbl__hdr">2009/40/1</h3>
    <div class="perex">
        ISSN : 0024-7774 (print), 1804-6460 (online) <br>Vedoucí redaktor : Benda Petr </div>
</div>
<p class="xs">Nalezeno článků: 1</p>
<div class="articleList"><article class="article"><div class="articleIn"><div class="articleCont">
    <h3 class="articleHdr"><a class="articleHdr__link" href="/periodicke-publikace/lnsr/40-1/first-record-of-a-mole"
       title="First record of a mole in Syria [První nález krtka v&nbsp;Syrii]">First record</a></h3>
    <div class="articleInfo">
        Ján Obuch, Petr Benda            </div>
</div></div></article></div>
"""

ARTICLE = """<p><a href="/file/cd951f0ff92b7f97673a4111527c0980/16433/Benda2009lynx.pdf" target="_blank"
  title="Benda2009lynx.pdf" class="btn btn-xl btIco">"""


def test_issue_page_gives_articles_with_year_volume_issue_issns_and_authors():
    [item] = items_of_issue(ISSUE, "/periodicke-publikace/lnsr/40-1", "Lynx, nová série", "national_museum")
    assert (item.journal_id, item.journal_title, item.year, item.volume, item.issue) == \
        ("lnsr", "Lynx, nová série", 2009, "40", "1")
    assert item.title == "First record of a mole in Syria [První nález krtka v Syrii]"
    assert item.authors == ["Ján Obuch", "Petr Benda"]
    assert item.record["issn"] == ["0024-7774", "1804-6460"]
    assert item.landing_url == "https://publikace.nm.cz/periodicke-publikace/lnsr/40-1/first-record-of-a-mole"
    assert item.item_id.startswith("lnsr_40-1_")


def test_article_pdf_is_the_file_its_page_links(monkeypatch):
    [item] = items_of_issue(ISSUE, "/periodicke-publikace/lnsr/40-1", "Lynx", "national_museum")
    pages = {item.landing_url: ARTICLE.encode(),
             "https://publikace.nm.cz/file/cd951f0ff92b7f97673a4111527c0980/16433/Benda2009lynx.pdf": b"%PDF-1.4"}
    monkeypatch.setattr(source, "http_get", lambda url: pages[url])
    monkeypatch.setattr(NationalMuseumSource, "download_pdf", lambda self, url: pages[url])
    download = NationalMuseumSource().download(item)
    assert download.url.endswith("/16433/Benda2009lynx.pdf") and download.data == b"%PDF-1.4"


def test_article_without_a_file_does_not_start_a_moving_wall(monkeypatch):
    [item] = items_of_issue(ISSUE, "/periodicke-publikace/lnsr/40-1", "Lynx", "national_museum")
    monkeypatch.setattr(source, "http_get", lambda url: b"<p>abstract only</p>")
    try:
        NationalMuseumSource().download(item)
    except source.DownloadBlocked as error:
        assert not NationalMuseumSource().starts_wall(str(error))
    else:
        raise AssertionError("no PDF, no download")
    assert NationalMuseumSource().starts_wall("HTTP Error 403: Forbidden")

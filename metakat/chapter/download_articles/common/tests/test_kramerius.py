import pytest

from metakat.chapter.download_articles.common.kramerius import (
    first_page_pid,
    is_newspaper,
    item_from_docs,
    mods_start_page,
)
from metakat.chapter.download_articles.common.models import CatalogItem
from metakat.chapter.download_articles.common.source import DownloadBlocked
from metakat.chapter.download_articles.common.store import ArticleStore
from metakat.chapter.download_articles.cbvk.source import CbvkSource
from metakat.chapter.download_articles.nkp.source import NkpSource


class FakeKramerius:
    def __init__(self, structure, mods="", pages=()):
        self._structure, self._mods, self._pages = structure, mods, list(pages)

    def structure(self, pid):
        return self._structure

    def mods(self, pid):
        return self._mods

    def search(self, query, fields, rows, start=0, sort=None, filtered=True):
        return self._pages


def test_newspapers_are_told_by_issues_per_volume_or_genre():
    assert is_newspaper({"issues_per_volume": 290.5})
    assert is_newspaper({"issues_per_volume": 4.0, "genres": ["Periodical", "Noviny"]})
    assert not is_newspaper({"issues_per_volume": 12.0, "genres": ["Časopisy"]})
    assert not is_newspaper({"issues_per_volume": None})


def test_first_page_follows_page_links_in_issue_order():
    kramerius = FakeKramerius({"children": {"foster": [{"pid": "p2", "relation": "isOnPage"},
                                                       {"pid": "p1", "relation": "isOnPage"}]}},
                              pages=[{"pid": "p1", "rels_ext_index.sort": 4}, {"pid": "p2", "rels_ext_index.sort": 5}])
    assert first_page_pid(kramerius, "uuid:a") == "p1"


def test_first_page_of_unlinked_article_comes_from_mods_start_page():
    mods = "<mods:part><mods:extent><mods:start>[72a]</mods:start><mods:end>[72d]</mods:end></mods:extent></mods:part>"
    assert mods_start_page(mods) == "72a"
    pages = [{"pid": "p71", "page.number": "[71]"}, {"pid": "p72", "page.number": "[72a]", "rels_ext_index.sort": 3}]
    kramerius = FakeKramerius({"children": {"own": [], "foster": []}}, mods, pages)
    assert first_page_pid(kramerius, "uuid:a", "uuid:issue") == "p72"
    with pytest.raises(DownloadBlocked):
        first_page_pid(FakeKramerius({"children": {}}, "<mods:start>9</mods:start>", pages), "uuid:a", "uuid:issue")


def test_libraries_leave_shared_articles_to_earlier_libraries(tmp_path):
    ArticleStore(tmp_path, "knav").write_catalog([CatalogItem(library="knav", item_id="a1", record_id="uuid:a1")])
    ArticleStore(tmp_path, "mzk").write_catalog([CatalogItem(library="mzk", item_id="a2", record_id="uuid:a2")])
    source = CbvkSource()
    source.root = tmp_path
    assert source.excluded_ids() == {"a1", "a2"}
    source = NkpSource()
    source.root = tmp_path
    assert NkpSource.article_query.startswith("model:internalpart")
    with pytest.raises(FileNotFoundError):
        source.excluded_ids()


def test_cdk_item_keeps_parent_and_periodical_structure():
    doc = {"pid": "uuid:a", "own_pid_path": "uuid:r/uuid:v/uuid:i/uuid:a", "own_parent.pid": "uuid:i",
           "own_model_path": "periodical/periodicalvolume/periodicalitem/internalpart", "root.title": "Letec",
           "title.search": "Křídla", "accessibility": "public", "date_range_start.year": 1936}
    parents = {"uuid:v": {"part.number.str": "18"}, "uuid:i": {"part.number.str": "3", "date.str": "03.1936"}}
    item = item_from_docs(doc, parents, "nkp", "https://api", "https://landing/", {"issues_per_volume": 12.5})
    assert (item.journal_id, item.volume, item.issue, item.date) == ("r", "18", "3", "03.1936")
    assert item.record["parent"] == ["uuid:i"] and item.record["issues_per_volume"] == ["12.5"]
    assert item.landing_url == "https://landing/uuid:a" and NkpSource().is_available(item)

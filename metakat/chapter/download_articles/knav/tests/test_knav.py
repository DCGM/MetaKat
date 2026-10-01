import json

from metakat.chapter.download_articles.common.source import DownloadBlocked
from metakat.chapter.download_articles.knav.source import KnavSource, index_local_pdfs, read_download_errors

ROOT, VOLUME, ISSUE = "uuid:root", "uuid:vol", "uuid:issue"
PARENTS = {
    VOLUME: {"pid": VOLUME, "model": "periodicalvolume", "part.number.str": "41", "date_range_start.year": 1994},
    ISSUE: {"pid": ISSUE, "model": "periodicalitem", "part.number.str": "4", "date.str": "12.1994"},
}


def _doc(pid, **fields):
    doc = {"pid": pid, "own_pid_path": f"{ROOT}/{VOLUME}/{ISSUE}/{pid}",
           "own_model_path": "periodical/periodicalvolume/periodicalitem/article",
           "root.title": "Folia parasitologica", "title.search": "An article", "date_range_start.year": 1994}
    doc.update(fields)
    return doc


def test_item_from_docs_reads_volume_issue_and_earlier_downloads():
    source = KnavSource(pick_dirs=(), pick_logs=())
    source.local, source.errors = {"a1": "/pick/issue/a1.pdf"}, {"a2": "404", "a3": "403"}
    pdf = source.item_from_docs(
        _doc("uuid:a1", **{"ds.img_full.mime": "application/pdf", "authors.search": ["Lukeš, Š."]}), PARENTS)
    assert (pdf.item_id, pdf.journal_id, pdf.journal_title) == ("a1", "root", "Folia parasitologica")
    assert (pdf.volume, pdf.issue, pdf.year, pdf.date) == ("41", "4", 1994, "12.1994")
    assert pdf.item_type == "pdf" and pdf.pdf_urls[0].endswith("/items/uuid:a1/image")
    assert pdf.authors == ["Lukeš, Š."] and pdf.record["local_pdf"] == ["/pick/issue/a1.pdf"]
    assert source.is_available(pdf) and source.is_cheap(pdf)

    scan = source.item_from_docs(_doc("uuid:a2", accessibility="public"), PARENTS)
    assert scan.item_type == "scan" and scan.pdf_urls == []
    assert source.is_available(scan) and not source.is_cheap(scan)

    refused = source.item_from_docs(_doc("uuid:a3", **{"ds.img_full.mime": "application/pdf"}), PARENTS)
    contents = source.item_from_docs(_doc("uuid:a4", **{"title.search": "Obsah"}), PARENTS)
    private = source.item_from_docs(_doc("uuid:a5", accessibility="private"), PARENTS)
    assert not any(source.is_available(item) for item in (refused, contents, private))


def test_earlier_downloads_are_indexed_from_folders_and_logs(tmp_path):
    (tmp_path / "issue").mkdir()
    (tmp_path / "issue" / "a1.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "issue" / "empty.pdf").write_bytes(b"")
    assert index_local_pdfs([tmp_path]) == {"a1": str(tmp_path / "issue" / "a1.pdf")}

    log = tmp_path / "public.log"
    log.write_text("[1/3] Processing: a1\n  Skipping (exists)\n"
                   "[2/3] Processing: a2\ncurl: (22) The requested URL returned error: 404\n  attempt 1 failed\n  FAILED\n"
                   "[3/3] Processing: a3\ncurl: (22) The requested URL returned error: 403\n  FAILED\n")
    assert read_download_errors([log]) == {"a2": "404", "a3": "403"}


class _Kramerius:
    def __init__(self, docs):
        self.docs = docs

    def search_all(self, query, fields, filtered=True):
        return self.docs


def _source(tmp_path, volumes=()):
    periodicals = tmp_path / "periodicals.tsv"
    periodicals.write_text("periodical\tperiodical_pid\nFolia parasitologica\tuuid:root\n", encoding="utf-8")
    source = KnavSource(pick_dirs=(), pick_logs=(), unsegmented_periodicals=periodicals)
    source.root = tmp_path
    source.kramerius = _Kramerius(list(volumes))
    return source


def test_volumes_of_unsegmented_periodicals_are_items_in_years_without_articles(tmp_path):
    volume = {"pid": "uuid:v1993", "root.title": "Folia parasitologica", "part.number.str": "40",
              "date.str": "1993", "date_range_start.year": 1993, "accessibility": "public"}
    licensed = dict(volume, pid="uuid:v1995", **{"date_range_start.year": 1995, "licenses.facet": ["dnnto"]})
    source = _source(tmp_path, [volume, dict(volume, pid="uuid:v1994", **{"date_range_start.year": 1994}), licensed])
    article = source.item_from_docs(_doc("uuid:a1"), PARENTS)
    item, restricted = source.volume_items([article])
    assert (item.item_id, item.record_id, item.item_type) == ("v1993", "uuid:v1993", "volume")
    assert (item.journal_id, item.journal_title, item.volume, item.year) == ("root", "Folia parasitologica", "40", 1993)
    assert source.is_available(item) and not source.is_cheap(item)
    assert restricted.rights == ["dnnto"] and not source.is_available(restricted)


def test_the_article_of_a_volume_is_found_through_its_contents_page(tmp_path):
    source = _source(tmp_path)

    def page(pid, number, page_type, sort):
        return {"pid": pid, "page.number": number, "page.type": page_type, "own_parent.pid": ISSUE,
                "rels_ext_index.sort": sort, "own_pid_path": f"{ROOT}/{VOLUME}/{ISSUE}/{pid}", "accessibility": "public",
                "own_model_path": "periodical/periodicalvolume/periodicalitem/page", "root.title": "Folia parasitologica"}

    cache = tmp_path / "knav" / "toc"
    cache.mkdir(parents=True)
    pages = [page(f"uuid:p{n}", f"[{n}]" if n == 5 else str(n), "NormalPage", n) for n in range(1, 7)]
    pages.append(page("uuid:toc", "(7)", "TableOfContents", 7))
    (cache / "vol.json").write_text(json.dumps({"parents": list(PARENTS.values()), "pages": pages[::-1]}))
    # Entries on pages 1, 2, 5 and 6: the article on 2 and the numbered section on 5 are the longest
    # (3 and 1 pages to the next start), and a section is not an article.
    titles = {1: "Article 1", 2: "Article 2", 5: "1.2 Methods", 6: "Article 6"}
    words = [[0, 40 * n, 100, 40 * n + 20, title] for n, title in titles.items()]
    words += [[500, 40 * n, 520, 40 * n + 20, str(n)] for n in titles]
    (cache / "toc.json").write_text(json.dumps(words))

    item = source.volume_item("root", {"pid": VOLUME, "root.title": "Folia parasitologica", "part.number.str": "41",
                                       "date_range_start.year": 1994})
    source.locate_article(item)
    assert item.record["first_page"] == ["uuid:p2"] and item.record["page_number"] == ["2"]
    assert item.record["toc_entries"] == ["Article 2"] and item.record["contents_pages"] == ["uuid:toc"]
    assert item.record["article_start_pages_found"] == ["4"] and item.record["pages_to_next_start"] == ["3"]
    assert (item.issue, item.date) == ("4", "12.1994")


def test_contents_pages_are_recognised_among_the_last_pages_when_none_is_typed(tmp_path):
    source = _source(tmp_path)
    cache = tmp_path / "knav" / "toc"
    cache.mkdir(parents=True)
    pages = [{"pid": f"uuid:p{n}", "page.number": str(n), "page.type": "NormalPage", "own_parent.pid": VOLUME,
              "rels_ext_index.sort": n, "own_pid_path": f"{ROOT}/{VOLUME}/uuid:p{n}",
              "own_model_path": "periodical/periodicalvolume/page"} for n in range(1, 41)]
    (cache / "vol.json").write_text(json.dumps({"parents": list(PARENTS.values()), "pages": pages}))
    contents = [[0, 40 * n, 100, 40 * n + 20, f"Article {n}"] for n in range(1, 40, 6)]
    contents += [[500, 40 * n, 520, 40 * n + 20, str(n)] for n in range(1, 40, 6)]
    (cache / "p40.json").write_text(json.dumps(contents))
    for n in [*range(1, 9), *range(34, 40)]:
        # Other pages cite a page or two, as running text does.
        (cache / f"p{n}.json").write_text(json.dumps([[0, 0, 100, 20, "see"], [500, 0, 520, 20, "13"]]))
    # A bibliography cites many pages, in no order.
    bibliography = [[[0, 40 * k, 100, 40 * k + 20, f"Work {k}"], [500, 40 * k, 520, 40 * k + 20, str(n)]]
                    for k, n in enumerate((30, 4, 22, 9, 35, 2, 17))]
    (cache / "p33.json").write_text(json.dumps([word for line in bibliography for word in line]))
    item = source.volume_item("root", {"pid": VOLUME, "date_range_start.year": 1994})
    source.locate_article(item)
    assert item.record["contents_pages"] == ["uuid:p40"] and item.record["article_start_pages_found"] == ["7"]


def test_a_volume_without_matching_contents_is_no_refusal_of_knav(tmp_path):
    source = _source(tmp_path)
    cache = tmp_path / "knav" / "toc"
    cache.mkdir(parents=True)
    (cache / "vol.json").write_text(json.dumps({"parents": list(PARENTS.values()), "pages": []}))
    item = source.volume_item("root", {"pid": VOLUME, "date_range_start.year": 1994})
    try:
        source.locate_article(item)
        raise AssertionError("no article can be found")
    except DownloadBlocked as error:
        assert not source.starts_wall(str(error))
    assert source.starts_wall("HTTP Error 403: Forbidden")



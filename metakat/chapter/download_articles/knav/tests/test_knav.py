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

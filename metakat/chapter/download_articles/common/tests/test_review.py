import csv
import io

from PIL import Image

from metakat.chapter.download_articles.common.models import CatalogItem
from metakat.chapter.download_articles.common.preview import sheet_layout
from metakat.chapter.download_articles.common.review import (
    APPROVED,
    REJECTED,
    ReviewLog,
    Session,
    load_sheet,
    sheet_path,
    stored_journals,
)
from metakat.chapter.download_articles.common.store import ArticleStore


def _store_with_journals(tmp_path):
    store = ArticleStore(tmp_path, "lib")
    buffer = io.BytesIO()
    Image.new("L", (600, 900), 255).save(buffer, format="JPEG")
    for journal, years in (("B", (1990, 1960)), ("A", (1950,)), ("C", (2000, 2005, 2010))):
        for year in years:
            item = CatalogItem(library="lib", item_id=f"{journal}{year}", record_id=f"{journal}{year}",
                               journal_id=journal, journal_title=f"Journal {journal}", year=year, title="Art")
            store.store_image(item, buffer.getvalue(), "jpg", None)
    return store


def test_journals_are_ordered_by_first_year_with_pages_in_year_order(tmp_path):
    journals = stored_journals(_store_with_journals(tmp_path))
    assert [j.title for j in journals] == ["Journal A", "Journal B", "Journal C"]
    assert [a.item.year for a in journals[1].articles] == [1960, 1990]


def test_keys_record_verdicts_and_a_new_session_resumes(tmp_path):
    store = _store_with_journals(tmp_path)
    journals = stored_journals(store)
    session = Session(journals, ReviewLog.load(store))
    session.handle("y")
    session.handle("n")
    session.handle("b")
    assert session.journal.title == "Journal B"
    session.handle("q")
    assert session.done

    with open(store.dir / "review.csv", newline="", encoding="utf-8") as file:
        rows = {row["journal_title"]: row for row in csv.DictReader(file)}
    assert rows["Journal A"]["verdict"] == APPROVED and rows["Journal B"]["verdict"] == REJECTED
    assert rows["Journal B"]["samples"] == "2" and rows["Journal B"]["first_year"] == "1960"

    resumed = Session(journals, ReviewLog.load(store))
    assert resumed.journal.title == "Journal C"
    resumed.handle(" ")
    assert resumed.done and ReviewLog.load(store).verdict(journals[2]) is None
    assert Session(journals, ReviewLog.load(store), review_all=True).journal.title == "Journal A"


def test_clearing_a_verdict_and_finished_libraries(tmp_path):
    store = _store_with_journals(tmp_path)
    journals = stored_journals(store)
    session = Session(journals, ReviewLog.load(store))
    for key in "yyy":
        session.handle(key)
    assert session.done and Session(journals, ReviewLog.load(store)).done
    again = Session(journals, ReviewLog.load(store), review_all=True)
    again.handle("u")
    assert Session(journals, ReviewLog.load(store)).journal.title == "Journal A"


def test_sheet_fits_its_tiles_and_is_cached(tmp_path):
    (width, height), boxes = sheet_layout(30)
    assert len(boxes) == 30 and width > height
    assert all(x + w <= width and y + h <= height for x, y, w, h in boxes)

    store = _store_with_journals(tmp_path)
    journal = stored_journals(store)[2]
    sheet = load_sheet(store, journal)
    assert sheet.size == sheet_layout(3)[0] and sheet_path(store, journal).exists()

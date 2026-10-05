import csv
import io

from PIL import Image

from metakat.chapter.download_articles.common.models import CatalogItem
from metakat.chapter.download_articles.common.preview import sheet_layout
from metakat.chapter.download_articles.common.review import (
    APPROVED,
    BY_AUTO,
    BY_ITEM,
    BY_JOURNAL,
    MARK_COLOURS,
    REJECTED,
    REVIEW_COLOURS,
    ItemReviewLog,
    ReviewLog,
    HEADER_HEIGHT,
    MIN_HEIGHT,
    Session,
    _compose,
    _key_name,
    close_round,
    current_round,
    load_sheet,
    mark_verdicts,
    parse_xrandr,
    resized_height,
    review_journals,
    sheet_path,
    stored_journals,
    tile_at,
    to_replace,
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


def _session(store, review_all=False):
    return Session(stored_journals(store), ReviewLog.load(store), ItemReviewLog.load(store), review_all)


def _position(session):
    return session.journal.title, None if session.article is None else session.article.item.item_id


def _rows(path, key):
    with open(path, newline="", encoding="utf-8") as file:
        return {row[key]: row for row in csv.DictReader(file)}


def test_journals_are_ordered_by_first_year_with_pages_in_year_order(tmp_path):
    journals = stored_journals(_store_with_journals(tmp_path))
    assert [j.title for j in journals] == ["Journal A", "Journal B", "Journal C"]
    assert [a.item.year for a in journals[1].articles] == [1960, 1990]


def test_approved_journal_is_reviewed_pick_by_pick_and_rejected_one_takes_its_picks(tmp_path):
    store = _store_with_journals(tmp_path)
    session = _session(store)
    session.handle("y")                       # Journal A approved -> its only pick
    assert _position(session) == ("Journal A", "A1950")
    session.handle("y")                       # the last pick approved -> Journal A's sheet again
    assert _position(session) == ("Journal A", None)
    session.handle(" ")                       # the next journal left to review
    assert _position(session) == ("Journal B", None)
    session.handle("n")                       # Journal B rejected with both picks -> Journal C
    assert _position(session) == ("Journal C", None)
    session.handle("y")
    session.handle("n")                       # C2000 rejected
    session.handle("b")
    assert _position(session) == ("Journal C", "C2000")
    session.handle("b")                       # before the first pick: the journal again
    assert _position(session) == ("Journal C", None)
    session.handle("q")
    assert session.done

    journals = _rows(store.dir / "review.csv", "journal_title")
    assert journals["Journal A"]["verdict"] == APPROVED and journals["Journal B"]["verdict"] == REJECTED
    assert journals["Journal B"]["samples"] == "2" and journals["Journal B"]["first_year"] == "1960"
    picks = _rows(store.dir / "review_items.csv", "item_id")
    assert (picks["A1950"]["verdict"], picks["A1950"]["by"]) == (APPROVED, BY_ITEM)
    assert all((picks[i]["verdict"], picks[i]["by"]) == (REJECTED, BY_JOURNAL) for i in ("B1960", "B1990"))
    assert (picks["C2000"]["verdict"], picks["C2000"]["by"]) == (REJECTED, BY_ITEM)
    assert picks["B1990"]["year"] == "1990" and picks["B1990"]["image"] == "images/B1990.jpg"


def test_new_session_resumes_inside_an_approved_journal(tmp_path):
    store = _store_with_journals(tmp_path)
    session = _session(store)
    for key in "yy n":                        # A and its pick approved, B rejected
        session.handle(key)
    session.handle("y")                       # C approved, no pick reviewed
    session.handle("s")                       # C2000 skipped
    session.handle("y")                       # C2005 approved
    session.handle("q")

    resumed = _session(store)
    assert _position(resumed) == ("Journal C", "C2000")
    resumed.handle("y")
    assert _position(resumed) == ("Journal C", "C2010")
    resumed.handle("j")                       # leave the picks: C was the last journal and stays shown
    assert not resumed.done and _position(resumed) == ("Journal C", None)
    assert _position(_session(store)) == ("Journal C", "C2010")


def test_a_reviewed_library_stays_open_for_checking_until_quit_or_enter(tmp_path):
    store = _store_with_journals(tmp_path)
    session = _session(store)
    for key in "yy nn":                       # A approved with its pick, B and C rejected
        session.handle(key)
    assert session.complete and not session.done and _position(session) == ("Journal C", None)
    reopened = _session(store)
    assert not reopened.done and _position(reopened) == ("Journal A", None)
    reopened.handle("y")                      # every pick judged: A's pick again
    assert _position(reopened) == ("Journal A", "A1950")
    reopened.handle("n")                      # changed to rejected; the last pick: A's sheet again
    assert _position(reopened) == ("Journal A", None) and not reopened.done
    reopened.handle(" ")                      # nothing left to review: the next journal in order
    assert _position(reopened) == ("Journal B", None)
    reopened.handle("enter")
    assert reopened.done and not reopened.quit
    quitting = _session(store)
    quitting.handle("esc")
    assert quitting.done and quitting.quit
    assert _position(_session(store, review_all=True)) == ("Journal A", None)


def test_reapproving_a_rejected_journal_reopens_its_picks(tmp_path):
    store = _store_with_journals(tmp_path)
    session = _session(store)
    session.handle("n")                       # A rejected with its pick
    assert session.items.verdict(session.journals[0].articles[0]) == REJECTED
    again = _session(store, review_all=True)
    again.handle("y")
    assert _position(again) == ("Journal A", "A1950")
    assert again.items.verdict(again.article) is None
    again.handle("u")
    again.handle("b")
    again.handle("u")                         # clear the journal verdict too
    assert _position(_session(store)) == ("Journal A", None)


def test_approving_a_journal_with_every_pick_judged_goes_through_its_picks_again(tmp_path):
    store = _store_with_journals(tmp_path)
    session = _session(store)
    session.handle("n")                       # A rejected
    session.handle("y")                       # B approved, both picks judged
    session.handle("y")
    session.handle("n")
    assert _position(session) == ("Journal B", None)          # after its last pick: B's sheet
    session.handle("y")                       # every pick has a verdict: all of them again, not the next journal
    assert _position(session) == ("Journal B", "B1960")
    session.handle("y")
    assert _position(session) == ("Journal B", "B1990")
    session.handle("y")
    assert _position(session) == ("Journal B", None)


def test_clearing_a_journal_clears_its_picks_too(tmp_path):
    store = _store_with_journals(tmp_path)
    session = _session(store)
    session.handle("right")                   # B's sheet
    session.handle("y")
    session.handle("y")
    session.handle("n")                       # B's last pick: B's sheet again
    assert _position(session) == ("Journal B", None)
    session.handle("u")
    assert session.log.verdict(session.journal) is None
    assert all(session.items.verdict(a) is None for a in session.journal.articles)
    session.handle("y")
    assert _position(session) == ("Journal B", "B1960")


def test_arrows_list_journals_and_picks_in_order(tmp_path):
    store = _store_with_journals(tmp_path)
    session = _session(store)
    session.handle("left")
    assert _position(session) == ("Journal A", None)          # stays at the first
    session.handle("right")
    session.handle("right")
    session.handle("right")
    assert _position(session) == ("Journal C", None)          # stays at the last
    session.handle("left")
    session.handle("y")
    session.handle("right")
    assert _position(session) == ("Journal B", "B1990")
    session.handle("right")                                   # after the last pick: the next journal's sheet
    assert _position(session) == ("Journal C", None)


def test_clicking_a_page_reviews_the_picks_from_it_on_in_order(tmp_path):
    store = _store_with_journals(tmp_path)
    session = _session(store)
    session.handle("right")
    session.handle("right")                                   # C's sheet: 2000, 2005, 2010
    _, boxes = sheet_layout(3)
    x, y, width, height = boxes[1]
    assert tile_at(session.journal, (x + width / 2, y + height / 2)) == 1
    assert tile_at(session.journal, (0, 0)) is None           # the sheet's title, no page
    session.items.set([session.journal.articles[2]], REJECTED)
    session.open_pick(1)
    assert _position(session) == ("Journal C", "C2005") and session.log.verdict(session.journal) is None
    session.handle("y")                                       # the next pick in order, judged or not
    assert _position(session) == ("Journal C", "C2010")
    session.handle("left")
    assert _position(session) == ("Journal C", "C2005")
    session.handle("esc")                                     # on a pick: back to the journal's sheet
    assert _position(session) == ("Journal C", None) and not session.done
    session.handle("esc")                                     # on a sheet: quit
    assert session.done and session.quit


def test_arrow_keys_are_not_read_as_letters():
    assert [_key_name(code) for code in (65361, 65363, 0x1000012, 2555904)] == ["left", "right", "left", "right"]
    assert _key_name(0x100000 | 65361) == "left" and _key_name(0x100000 | ord("y")) == "y"
    assert _key_name(65505) is None and _key_name(ord("Q")) == "q" and _key_name(-1) is None
    assert [_key_name(ord(c)) for c in ",<.>"] == ["left", "left", "right", "right"]


def test_sheet_fits_its_tiles_and_is_cached(tmp_path):
    (width, height), boxes = sheet_layout(30)
    assert len(boxes) == 30 and width > height
    assert all(x + w <= width and y + h <= height for x, y, w, h in boxes)

    store = _store_with_journals(tmp_path)
    journal = stored_journals(store)[2]
    sheet = load_sheet(store, journal)
    assert sheet.size == sheet_layout(3)[0] and sheet_path(store, journal).exists()


def test_sheet_marks_every_tile_with_its_pick_verdict(tmp_path):
    store = _store_with_journals(tmp_path)
    journal = stored_journals(store)[2]                       # C: 2000, 2005, 2010
    items = ItemReviewLog.load(store)
    items.set([journal.articles[0]], APPROVED)
    items.set([journal.articles[1]], REJECTED)
    sheet = load_sheet(store, journal)
    scale = 0.5
    marked = mark_verdicts(sheet.resize((round(sheet.width * scale), round(sheet.height * scale))), scale, journal,
                           items)
    _, boxes = sheet_layout(3)
    corners = [marked.getpixel((round((x + w) * scale) - 3, round(y * scale) + 1)) for x, y, w, _ in boxes]
    assert corners == [MARK_COLOURS[APPROVED, BY_ITEM], MARK_COLOURS[REJECTED, BY_ITEM], MARK_COLOURS[None, None]]
    assert load_sheet(store, journal).getpixel((boxes[0][0] + boxes[0][2] - 3, boxes[0][1] + 1)) != \
        MARK_COLOURS[APPROVED, BY_ITEM]                     # the cached sheet stays unmarked


def test_closed_round_leaves_out_its_rejections_and_its_rejected_picks_get_replacements(tmp_path):
    store = _store_with_journals(tmp_path)
    session = _session(store)
    for key in "yy nyny":                     # A and its pick approved, B rejected with both picks,
        session.handle(key)                   # C approved: C2000 rejected, C2005 approved, C2010 open
    log, items = ReviewLog.load(store), ItemReviewLog.load(store)
    assert current_round(log, items) == 1 and review_journals(store, log, items) == stored_journals(store)

    assert close_round(log, items) == 1
    log, items = ReviewLog.load(store), ItemReviewLog.load(store)
    assert current_round(log, items) == 2
    rows = _rows(store.dir / "review_items.csv", "item_id")
    assert rows["C2000"]["round"] == rows["B1960"]["round"] == "1" and "C2010" not in rows
    shown = review_journals(store, log, items)
    assert [(j.title, [a.item.item_id for a in j.articles]) for j in shown] == [
        ("Journal A", ["A1950"]), ("Journal C", ["C2005", "C2010"])]
    assert review_journals(store, log, items, closed=True) == stored_journals(store)
    assert _position(Session(shown, log, items)) == ("Journal C", "C2010")   # the open pick
    assert [a.item.item_id for a in to_replace(store)] == ["C2000"]          # not B's, rejected with B

    # A replacement picked but not stored yet still leaves C2000 to replace; once stored it does not,
    # and the next review shows it without a verdict.
    rejected = stored_journals(store)[2].articles[0].item
    new = rejected.model_copy(update={"item_id": "C2001", "record_id": "C2001", "year": 2001})
    store.write_replacements([(rejected, new)])
    assert store.replacements() == {"C2000": "C2001"} and len(to_replace(store)) == 1
    store.store_image(new, (store.dir / "images" / "C2000.jpg").read_bytes(), "jpg", None)
    assert to_replace(store) == []
    shown = review_journals(store, log, items)
    assert [a.item.item_id for a in shown[1].articles] == ["C2001", "C2005", "C2010"]

    # A verdict changed later belongs to the round going on: the rejected pick stays shown until it closes.
    items.set([shown[1].articles[1]], REJECTED)
    assert [a.item.item_id for a in review_journals(store, log, items)[1].articles] == ["C2001", "C2005", "C2010"]
    assert close_round(log, items) == 2
    assert [a.item.item_id for a in review_journals(store, log, items)[1].articles] == ["C2001", "C2010"]
    assert _rows(store.dir / "review_items.csv", "item_id")["C2005"]["round"] == "2"


def test_r_toggles_the_review_mark_and_verdicts_keep_it(tmp_path):
    store = _store_with_journals(tmp_path)
    session = _session(store)
    session.handle("y")                       # Journal A approved -> its pick
    session.handle("r")
    assert _position(session) == ("Journal A", "A1950")       # stays on the pick
    assert session.items.is_review(session.article) and session.items.review_by(session.article) == BY_ITEM
    session.handle("y")                       # approved, still a review
    rows = _rows(store.dir / "review_items.csv", "item_id")
    assert (rows["A1950"]["verdict"], rows["A1950"]["review"], rows["A1950"]["review_by"]) == (APPROVED, "yes", BY_ITEM)
    session.open_pick(0)
    session.handle("u")                       # the verdict cleared, the mark stays
    assert session.items.verdict(session.article) is None and session.items.is_review(session.article)
    session.handle("r")
    assert session.items.is_review(session.article) is False
    assert _rows(store.dir / "review_items.csv", "item_id")["A1950"]["review"] == "no"


def test_guessed_review_marks_are_visited_and_confirmed_by_a_verdict(tmp_path):
    store = _store_with_journals(tmp_path)
    session = _session(store)
    for key in "yy yyy yyyy":                 # everything approved
        session.handle(key)
    assert session.complete
    items = ItemReviewLog.load(store)
    c2005 = stored_journals(store)[2].articles[1]
    items.set_review([c2005], True, BY_AUTO, note="title")
    resumed = _session(store)
    assert not resumed.complete and _position(resumed) == ("Journal C", "C2005")
    assert resumed.items.review_note(resumed.article) == "title"
    resumed.handle("y")                       # seen: the guess stands, no longer a guess
    assert resumed.items.is_review(c2005) and resumed.items.review_by(c2005) == BY_ITEM and resumed.complete

    items = ItemReviewLog.load(store)
    items.set_review([c2005], True, BY_AUTO)
    confirmed = _session(store)
    confirmed.handle("r")                     # r on a guess confirms it rather than switching it off
    assert confirmed.items.is_review(c2005) and confirmed.items.review_by(c2005) == BY_ITEM

    items = ItemReviewLog.load(store)
    items.set_review([c2005], True, BY_AUTO)
    wrong = _session(store)
    wrong.handle("r")
    wrong.handle("r")                         # a wrong guess: confirmed, then unmarked, then judged
    wrong.handle("y")
    assert wrong.items.is_review(c2005) is False and wrong.items.review_by(c2005) == BY_ITEM and wrong.complete


def test_sheet_marks_reviews_in_the_other_corner(tmp_path):
    store = _store_with_journals(tmp_path)
    journal = stored_journals(store)[2]                       # C: 2000, 2005, 2010
    items = ItemReviewLog.load(store)
    items.set_review([journal.articles[0]], True)
    items.set_review([journal.articles[1]], True, BY_AUTO)
    items.set_review([journal.articles[2]], False)
    sheet = load_sheet(store, journal)
    marked = mark_verdicts(sheet, 1.0, journal, items)
    _, boxes = sheet_layout(3)
    corners = [marked.getpixel((x + 3, y + 1)) for x, y, _, _ in boxes]
    assert corners[:2] == [REVIEW_COLOURS[BY_ITEM], REVIEW_COLOURS[BY_AUTO]]
    assert corners[2] == sheet.getpixel((boxes[2][0] + 3, boxes[2][1] + 1))     # not a review: no mark


def test_undecided_shows_only_picks_left_to_decide(tmp_path):
    store = _store_with_journals(tmp_path)
    session = _session(store)
    for key in "yy yyy":                      # A and B approved with all their picks
        session.handle(key)
    log, items = ReviewLog.load(store), ItemReviewLog.load(store)
    b1990 = stored_journals(store)[1].articles[1]
    items.set_review([b1990], True, BY_AUTO)  # a guess not yet confirmed
    shown = review_journals(store, log, items, undecided=True)
    assert [(j.title, [a.item.item_id for a in j.articles]) for j in shown] == [
        ("Journal B", ["B1990"]), ("Journal C", ["C2000", "C2005", "C2010"])]
    assert _position(Session(shown, log, items)) == ("Journal B", "B1990")


def test_window_starts_as_high_as_the_primary_screen():
    output = ("Screen 0: minimum 16 x 16, current 4480 x 1440, maximum 32767 x 32767\n"
              "DP-1 connected 2560x1440+1920+0 (normal left inverted right x axis y axis) 600mm x 340mm\n"
              "   2560x1440     59.95*+\n"
              "HDMI-1 connected primary 1920x1200+0+0 (normal left inverted right x axis y axis) 520mm x 320mm\n"
              "DP-2 disconnected (normal left inverted right x axis y axis)\n")
    assert parse_xrandr(output) == (1920, 1200)
    assert parse_xrandr(output.replace(" primary", "")) == (2560, 1440)
    assert parse_xrandr("") is None


def test_a_window_resized_by_hand_gives_the_height_of_the_pages_after():
    image = (1100, 1000)
    assert resized_height((1100, 700), image) == 700          # made lower
    assert resized_height((550, 1000), image) == 500          # made narrower: the image keeps its aspect ratio
    assert resized_height((200, 100), image) == MIN_HEIGHT
    canvas = _compose(Image.new("RGB", (800, 400)), ["header", ""], None, height=900)
    assert canvas.shape[:2] == (900, 1100) and 400 + HEADER_HEIGHT < 900
    # A maximized window: the page fills its whole image area.
    assert _compose(Image.new("RGB", (800, 400)), ["header", ""], None, 1130, 1920).shape[:2] == (1130, 1920)

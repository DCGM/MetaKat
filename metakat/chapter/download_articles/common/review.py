"""Review the sampled journals of library folders, first as a whole and then pick by pick.

    python -m metakat.chapter.download_articles.common.review /mnt/kolosus/data/smart_digiline/articles/knav [...]

A window first shows every stored title page of one journal in year order:

    y  approve the journal and review its picks one by one: those without a verdict, or all of
       them again when every one has a verdict
    n  reject the journal and all its picks, go to the next journal
    ←/→ (or ,/.)  previous/next journal, reviewed or not    space  next journal left to review
    u  clear the verdicts of the journal and all its picks    Enter  next library    q / Esc  quit
    click a page to review the picks from it on: shown enlarged, y / n go on to the next pick in order
    the corner of each page shows its verdict: green approved, red rejected, grey none

The picks of an approved journal are then shown one at a time, enlarged:

    y  approve the pick    n  reject the pick    (after the last pick: the journal sheet again)
    space  skip    ←/b  back (before the first pick: the
    journal)    →  next pick (after the last: the next journal)    u  clear verdict
    j  leave the picks, go to the next journal    Esc  back to the journal sheet    q  quit

Verdicts are saved after every key: journals into ``<library>/review.csv``, picks into
``<library>/review_items.csv`` (``by`` tells a pick's own verdict from one inherited from a rejected
journal). A new session starts at the first journal without a verdict, or inside an approved
journal at its first pick without a verdict; ``--all`` goes through everything again. The window
stays open when everything is reviewed (the header says ALL REVIEWED), so that verdicts can still be
checked and changed; only q (or Esc on a journal sheet) quits, and Enter goes on to the next library folder given. Journal
sheets are cached in ``<library>/previews/journals/``.

The review goes in rounds, until only approved picks are left:

    python -m metakat.chapter.download_articles.common.review --close-round .../articles/knav [...]
    python -m metakat.chapter.download_articles select --source knav --replace
    python -m metakat.chapter.download_articles fetch  --source knav

``--close-round`` stamps every verdict given so far with the round's number (column ``round``). From
then on a review leaves out what a closed round rejected: rejected journals, rejected picks and
journals left with no pick (``--closed`` shows them again). ``select --replace`` picks another item of
the same journal and year (or the closest one) for every pick rejected on its own, and ``fetch``
stores them; the next review shows the approved picks and the new ones, which have no verdict. A
verdict changed later belongs to the round then going on.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from metakat.chapter.download_articles.common.models import StoredArticle
from metakat.chapter.download_articles.common.preview import _font, _slug, journal_sheet, sheet_layout
from metakat.chapter.download_articles.common.selection import journal_key
from metakat.chapter.download_articles.common.store import ArticleStore

REVIEW_FILE, ITEM_REVIEW_FILE = "review.csv", "review_items.csv"
JOURNAL_FIELDS = ["journal_id", "journal_title", "samples", "first_year", "last_year", "verdict", "reviewed_at",
                  "round"]
ITEM_FIELDS = ["item_id", "journal_id", "journal_title", "year", "volume", "issue", "title", "image", "verdict",
               "by", "reviewed_at", "round"]
APPROVED, REJECTED = "approved", "rejected"
BY_ITEM, BY_JOURNAL = "item", "journal"
HEADER_HEIGHT = 64
WINDOW = "MetaKat journal review"


@dataclass
class Journal:
    key: tuple[str | None, str | None]
    articles: list[StoredArticle]

    @property
    def title(self) -> str:
        return self.key[1] or self.key[0] or "?"

    @property
    def years(self) -> list[int]:
        return [a.item.year for a in self.articles if a.item.year]


def stored_journals(store: ArticleStore) -> list[Journal]:
    """The journals with stored title pages, by first year, each with its pages in year order."""
    by_key: dict = defaultdict(list)
    for article in store.stored():
        by_key[journal_key(article.item)].append(article)
    journals = [Journal(key, sorted(articles, key=lambda a: (a.item.year or 0, a.item.item_id)))
                for key, articles in by_key.items()]
    return sorted(journals, key=lambda j: (min(j.years, default=0), j.title))


def review_journals(store: ArticleStore, log: ReviewLog, items: ItemReviewLog, closed: bool = False) -> list[Journal]:
    """The journals and picks a review shows: those of ``stored_journals`` without the rejections of closed
    rounds (``close_round``), and without journals left with no pick; ``closed`` shows them all."""
    journals = []
    for journal in stored_journals(store):
        if not closed and log.verdict(journal) == REJECTED and log.closed(journal):
            continue
        articles = [a for a in journal.articles if closed or not (items.verdict(a) == REJECTED and items.closed(a))]
        if articles:
            journals.append(Journal(journal.key, articles))
    return journals


def current_round(log: ReviewLog, items: ItemReviewLog) -> int:
    """The number of the review round going on: one after the last closed one."""
    return 1 + max((int(row.get("round") or 0) for review_log in (log, items) for row in review_log.rows.values()),
                   default=0)


def close_round(log: ReviewLog, items: ItemReviewLog) -> int:
    """Close the current round: every verdict given in it gets the round's number and is final, so that
    later rounds no longer show its rejections. A verdict changed later belongs to the round then going on
    again. Picks without a verdict stay open. Returns the number of the closed round."""
    number = current_round(log, items)
    for review_log in (log, items):
        opened = [row for row in review_log.rows.values() if row.get("verdict") and not row.get("round")]
        for row in opened:
            row["round"] = number
        if opened:
            review_log.save()
    return number


def to_replace(store: ArticleStore) -> list[StoredArticle]:
    """The picks rejected on their own (not with their journal) in a closed round, in journals not rejected,
    that have no stored replacement yet (``ArticleStore.replacements``)."""
    log, items = ReviewLog.load(store), ItemReviewLog.load(store)
    replacements = store.replacements()
    return [article for journal in stored_journals(store) if log.verdict(journal) != REJECTED
            for article in journal.articles
            if items.verdict(article) == REJECTED and items.by(article) == BY_ITEM and items.closed(article)
            and not store.is_stored(replacements.get(article.item.item_id, ""))]


class _CsvLog:
    """A CSV of verdicts keyed by some of its columns, rewritten whole after every change."""

    fields: list[str]

    def __init__(self, path: Path):
        self.path = path
        self.rows: dict = {}
        if path.exists():
            with open(path, newline="", encoding="utf-8") as file:
                for row in csv.DictReader(file):
                    self.rows[self._key(row)] = row

    def _key(self, row: dict):
        raise NotImplementedError

    def _closed(self, key) -> bool:
        """Whether the verdict under ``key`` belongs to a closed round."""
        return bool((self.rows.get(key) or {}).get("round"))

    def save(self) -> None:
        temporary = self.path.with_suffix(".csv.tmp")
        with open(temporary, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.fields)
            writer.writeheader()
            writer.writerows(sorted(self.rows.values(), key=lambda r: [str(r.get(f, "")) for f in self.fields]))
        temporary.replace(self.path)


class ReviewLog(_CsvLog):
    """``review.csv``: one row per journal with its verdict."""

    fields = JOURNAL_FIELDS

    @classmethod
    def load(cls, store: ArticleStore) -> ReviewLog:
        return cls(store.dir / REVIEW_FILE)

    def _key(self, row: dict):
        return row["journal_id"] or None, row["journal_title"] or None

    def verdict(self, journal: Journal) -> str | None:
        return (self.rows.get(journal.key) or {}).get("verdict") or None

    def closed(self, journal: Journal) -> bool:
        return self._closed(journal.key)

    def set(self, journal: Journal, verdict: str | None) -> None:
        years = journal.years
        self.rows[journal.key] = {
            "journal_id": journal.key[0] or "", "journal_title": journal.key[1] or "",
            "samples": len(journal.articles), "first_year": min(years, default=""),
            "last_year": max(years, default=""), "verdict": verdict or "",
            "reviewed_at": time.strftime("%Y-%m-%dT%H:%M:%S") if verdict else "",
        }
        self.save()


class ItemReviewLog(_CsvLog):
    """``review_items.csv``: one row per stored pick with its verdict and who gave it."""

    fields = ITEM_FIELDS

    @classmethod
    def load(cls, store: ArticleStore) -> ItemReviewLog:
        return cls(store.dir / ITEM_REVIEW_FILE)

    def _key(self, row: dict):
        return row["item_id"]

    def verdict(self, article: StoredArticle) -> str | None:
        return (self.rows.get(article.item.item_id) or {}).get("verdict") or None

    def by(self, article: StoredArticle) -> str | None:
        return (self.rows.get(article.item.item_id) or {}).get("by") or None

    def closed(self, article: StoredArticle) -> bool:
        return self._closed(article.item.item_id)

    def set(self, articles: list[StoredArticle], verdict: str | None, by: str = BY_ITEM) -> None:
        for article in articles:
            item = article.item
            self.rows[item.item_id] = {
                "item_id": item.item_id, "journal_id": item.journal_id or "", "journal_title": item.journal_title or "",
                "year": item.year or "", "volume": item.volume or "", "issue": item.issue or "",
                "title": item.title or "", "image": article.image.file, "verdict": verdict or "",
                "by": by if verdict else "", "reviewed_at": time.strftime("%Y-%m-%dT%H:%M:%S") if verdict else "",
            }
        self.save()


class Session:
    """Which journal or pick is shown and what a key does; kept apart from the window so it can be tested.

    ``item`` is None while the journal is shown as a whole, else the index of the pick shown.
    """

    def __init__(self, journals: list[Journal], log: ReviewLog, items: ItemReviewLog, review_all: bool = False):
        self.journals, self.log, self.items, self.review_all = journals, log, items, review_all
        # The session ends only on q / Esc (``quit``) or Enter (the next library), never by itself: with
        # everything reviewed, the journals are still shown to be checked or changed.
        self.index, self.item, self.done, self.quit = 0, None, not journals, False
        # True while going through every pick of a journal approved again with all its picks judged.
        self.revisit = False
        if journals and not review_all:
            start = next((k for k, j in enumerate(journals) if self._unfinished(j)), None)
            if start is not None:
                self.index = start
                if log.verdict(self.journal) == APPROVED:
                    self.item = self._first_open_item()

    @property
    def complete(self) -> bool:
        """Whether every journal and every pick of an approved journal has a verdict."""
        return not any(self._unfinished(j) for j in self.journals)

    @property
    def journal(self) -> Journal:
        return self.journals[self.index]

    @property
    def article(self) -> StoredArticle | None:
        return None if self.item is None else self.journal.articles[self.item]

    def _unfinished(self, journal: Journal) -> bool:
        verdict = self.log.verdict(journal)
        return not verdict or (verdict == APPROVED and any(not self.items.verdict(a) for a in journal.articles))

    def _first_open_item(self) -> int:
        return next((k for k, a in enumerate(self.journal.articles) if not self.items.verdict(a)), 0)

    def handle(self, key: str) -> None:
        if key == "esc" and self.item is not None:
            self.item, self.revisit = None, False         # from a pick back to its journal's sheet
        elif key in ("q", "esc"):
            self.done = self.quit = True
        elif key == "enter":
            self.done = True
        elif self.item is None:
            self._handle_journal(key)
        else:
            self._handle_item(key)

    def _handle_journal(self, key: str) -> None:
        journal = self.journal
        if key == "y":
            self.log.set(journal, APPROVED)
            # Picks rejected only together with the journal are open for review again.
            self.items.set([a for a in journal.articles if self.items.by(a) == BY_JOURNAL], None)
            # The picks without a verdict; when every pick has one, all of them again.
            self.revisit = all(self.items.verdict(a) for a in journal.articles)
            self.item = self._first_open_item()
        elif key == "n":
            self.log.set(journal, REJECTED)
            self.items.set(journal.articles, REJECTED, by=BY_JOURNAL)
            self._next_journal()
        elif key == "u":
            # The journal is reviewed from scratch: its verdict and those of all its picks are cleared.
            self.log.set(journal, None)
            self.items.set(journal.articles, None)
        elif key in (" ", "s"):
            self._next_journal()
        elif key in ("b", "left"):
            self._show_journal(self.index - 1)
        elif key == "right":
            self._show_journal(self.index + 1)

    def _handle_item(self, key: str) -> None:
        if key in ("y", "n"):
            self.items.set([self.article], APPROVED if key == "y" else REJECTED)
            self._next_item()
        elif key == "u":
            self.items.set([self.article], None)
        elif key in (" ", "s"):
            self._next_item()
        elif key in ("b", "left"):
            self.item = None if self.item == 0 else self.item - 1
        elif key == "right":
            if self.item + 1 < len(self.journal.articles):
                self.item += 1
            else:
                self._show_journal(self.index + 1)
        elif key == "j":
            self._next_journal()

    def open_pick(self, item: int) -> None:
        """Review the picks of the shown journal from ``item`` on, in order (a page clicked on its sheet);
        the journal's own verdict stays as it is."""
        self.item, self.revisit = item, True

    def _show_journal(self, index: int) -> None:
        """The sheet of the journal at ``index`` (clamped), whatever its verdicts."""
        self.index, self.item, self.revisit = max(0, min(index, len(self.journals) - 1)), None, False

    def _next_item(self) -> None:
        """The next pick of the journal (without a verdict, unless reviewing all or revisiting), else the
        journal's sheet again, to see its verdicts before going on."""
        following = range(self.item + 1, len(self.journal.articles))
        item = next((k for k in following if self.review_all or self.revisit
                     or not self.items.verdict(self.journal.articles[k])), None)
        if item is None:
            self.item, self.revisit = None, False
        else:
            self.item = item

    def _next_journal(self) -> None:
        """The next journal with something left to review, after this one or else before it; an approved
        one opens at its first pick without a verdict. With nothing left (or reviewing all), the next
        journal in order, as a whole; the last one stays."""
        self.item, self.revisit = None, False
        others = [*range(self.index + 1, len(self.journals)), *range(self.index)]
        index = None if self.review_all else next((k for k in others if self._unfinished(self.journals[k])), None)
        if index is None:
            self.index = min(self.index + 1, len(self.journals) - 1)
            return
        self.index = index
        if self.log.verdict(self.journal) == APPROVED:
            self.item = self._first_open_item()


def sheet_path(store: ArticleStore, journal: Journal) -> Path:
    digest = hashlib.sha1(f"{journal.key[0]}|{journal.key[1]}".encode()).hexdigest()[:8]
    ids = hashlib.sha1(" ".join(a.item.item_id for a in journal.articles).encode()).hexdigest()[:6]
    return store.dir / "previews" / "journals" / f"{_slug(journal.title)}_{digest}_{ids}.jpg"


def load_sheet(store: ArticleStore, journal: Journal) -> Image.Image:
    """The journal's sheet, rendered once and cached; a new sample set gets a new file."""
    path = sheet_path(store, journal)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        journal_sheet(store, journal.title, journal.articles).save(path, quality=85)
    with Image.open(path) as image:
        return image.convert("RGB")


# Corner marks of the tiles on a journal sheet: a pick's own verdict, one inherited from its rejected
# journal, or none yet (skipped or not reached).
MARK_COLOURS = {(APPROVED, BY_ITEM): (40, 170, 70), (REJECTED, BY_ITEM): (215, 40, 40),
                (REJECTED, BY_JOURNAL): (170, 110, 110), (None, None): (160, 160, 160)}
MARK_LEGEND = "corner: green approved · red rejected · dull red with journal · grey no verdict"


def mark_verdicts(sheet: Image.Image, scale: float, journal: Journal, items: ItemReviewLog) -> Image.Image:
    """A copy of the (scaled) journal sheet with every tile's pick verdict as a triangle in its top right corner."""
    marked = sheet.copy()
    draw = ImageDraw.Draw(marked)
    _, boxes = sheet_layout(len(journal.articles))
    for article, (x, y, width, _) in zip(journal.articles, boxes):
        verdict = items.verdict(article)
        colour = MARK_COLOURS.get((verdict, items.by(article) if verdict else None), MARK_COLOURS[None, None])
        right, top, size = (x + width) * scale, y * scale, max(16.0, 0.22 * width * scale)
        draw.polygon([(right - size, top), (right, top), (right, top + size)], fill=colour, outline="white")
    return marked


def load_page(store: ArticleStore, article: StoredArticle, max_width: int, max_height: int) -> Image.Image:
    with Image.open(store.dir / article.image.file) as image:
        return _fit(image.convert("RGB"), max_width, max_height - HEADER_HEIGHT)[0]


def _header(lines: list[str], width: int, verdict: str | None) -> Image.Image:
    colour = {APPROVED: (30, 120, 50), REJECTED: (160, 40, 40)}.get(verdict, (45, 45, 45))
    header = Image.new("RGB", (width, HEADER_HEIGHT), colour)
    draw = ImageDraw.Draw(header)
    for row, (text, size) in enumerate(zip(lines, (24, 18))):
        draw.text((12, 6 + row * 32), text, font=_font(size, bold=row == 0), fill="white")
    return header


def _compose(image: Image.Image, lines: list[str], verdict: str | None) -> np.ndarray:
    canvas = Image.new("RGB", (max(image.width, 1100), image.height + HEADER_HEIGHT), "white")
    canvas.paste(_header(lines, canvas.width, verdict), (0, 0))
    canvas.paste(image, (0, HEADER_HEIGHT))
    return np.asarray(canvas)[:, :, ::-1].copy()


def _fit(image: Image.Image, max_width: int, max_height: int) -> tuple[Image.Image, float]:
    scale = min(1.0, max_width / image.width, max_height / image.height)
    if scale < 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    return image, scale


# Arrow key codes of ``cv2.waitKeyEx`` with the GTK, Qt and Windows backends. Their low byte must not
# be read as a letter: GTK's left arrow (65361) ends in 81, "Q".
ARROW_KEYS = {65361: "left", 65363: "right", 0x1000012: "left", 0x1000014: "right", 2424832: "left",
              2555904: "right"}


def _key_name(code: int) -> str | None:
    if code < 0:
        return None
    # GTK sets a bit for Num Lock (0x100000) on every key.
    for key in (code, code & 0xFFFF):
        if key in ARROW_KEYS:
            return ARROW_KEYS[key]
    if code & 0xFFFF > 0xFF:
        return None                 # other special keys (Shift, F1, ...)
    code &= 0xFF
    # "," and "." (the "<" and ">" keys) move like the arrows.
    if chr(code) in ",<":
        return "left"
    if chr(code) in ".>":
        return "right"
    if code == 27:
        return "esc"
    if code in (10, 13):
        return "enter"
    if code == 8:
        return "b"
    return chr(code).lower() if 32 <= code < 127 else None


def _wait_key(cv2, clicks: list) -> str | None:
    """The next key, or None when the window was clicked."""
    while not clicks:
        key = _key_name(cv2.waitKeyEx(50))
        if key:
            return key
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return "q"
    return None


def review(directory: Path, review_all: bool, max_width: int, max_height: int, closed: bool = False) -> bool:
    """Review one library folder; returns False when the reviewer quit."""
    import cv2

    store = ArticleStore(directory.parent, directory.name)
    log, items = ReviewLog.load(store), ItemReviewLog.load(store)
    journals = review_journals(store, log, items, closed)
    round_number = current_round(log, items)
    session = Session(journals, log, items, review_all)
    if session.done:
        print(f"{directory}: no journals to show (none stored, or all rejected in closed rounds)")
        return True

    clicks: list[tuple[int, int]] = []
    _open_window(cv2, clicks)
    # Where the image of a re-created window is to be put back, once it is shown.
    anchor: tuple[int, int] | None = None
    page_height = max_height
    with ThreadPoolExecutor(max_workers=2) as prefetch:
        sheets: dict = {}
        pages: dict = {}

        def sheet_of(index):
            if index not in sheets:
                sheets[index] = prefetch.submit(load_sheet, store, journals[index])
            return sheets[index]

        def page_of(index, item):
            if (index, item) not in pages:
                pages[(index, item)] = prefetch.submit(load_page, store, journals[index].articles[item],
                                                       max_width, page_height)
            return pages[(index, item)]

        while not session.done:
            journal, index = session.journal, session.index
            # Keep the images of the shown and the neighbouring journals only.
            for cache in (sheets, pages):
                for key in [k for k in cache if abs((k if isinstance(k, int) else k[0]) - index) > 1]:
                    del cache[key]
            reviewed = sum(1 for j in journals if log.verdict(j))
            position = (f"{directory.name} round {round_number}  [{index + 1}/{len(journals)}, "
                        f"{'ALL REVIEWED' if session.complete else f'{reviewed} reviewed'}]")
            if session.item is None:
                sheet = sheet_of(index).result()
                if index + 1 < len(journals):
                    sheet_of(index + 1)
                if journal.articles:
                    page_of(index, 0)
                shown, scale = _fit(sheet, max_width, max_height - HEADER_HEIGHT)
                shown = mark_verdicts(shown, scale, journal, items)
                verdict = log.verdict(journal)
                lines = [f"{position}  JOURNAL {(verdict or 'no verdict').upper()}  ·  {MARK_LEGEND}",
                         "y review picks · n reject journal and picks · ←/→ or ,/. journals · space next open · "
                         "u clear journal and picks · Enter next library · q quit · click a page to review it"]
            else:
                article, item = session.article, session.item
                shown, scale = page_of(index, item).result(), None
                if item + 1 < len(journal.articles):
                    page_of(index, item + 1)
                elif index + 1 < len(journals):
                    sheet_of(index + 1)
                verdict = items.verdict(article)
                data = article.item
                lines = [f"{position}  pick {item + 1}/{len(journal.articles)}  {(verdict or 'no verdict').upper()}"
                         f"  ·  {data.year or '?'}  v{data.volume or '?'}/{data.issue or '?'}  {data.title or ''}"[:150],
                         f"{journal.title[:70]}  ·  y approve · n reject · ←/→ picks · space skip · u clear · "
                         f"j next journal · Esc journal sheet · q quit"]
            cv2.imshow(WINDOW, _compose(shown, lines, verdict))
            if anchor is not None:
                _place_window(cv2, anchor)
                anchor = None

            key = _wait_key(cv2, clicks)
            if key is None:
                x, y = clicks.pop()
                clicks.clear()
                if scale is not None:
                    pick = tile_at(journal, (x / scale, (y - HEADER_HEIGHT) / scale))
                    if pick is not None:
                        session.open_pick(pick)
                anchor = _reopen_window(cv2, clicks)
                continue
            session.handle(key)
            if session.quit:
                cv2.destroyAllWindows()
                return False
    cv2.destroyAllWindows()
    journal_counts = {v: sum(1 for j in journals if log.verdict(j) == v) for v in (APPROVED, REJECTED)}
    picks = [a for j in journals for a in j.articles]
    pick_counts = {v: sum(1 for a in picks if items.verdict(a) == v) for v in (APPROVED, REJECTED)}
    print(f"{directory}: journals {journal_counts[APPROVED]} approved, {journal_counts[REJECTED]} rejected of "
          f"{len(journals)}; picks {pick_counts[APPROVED]} approved, {pick_counts[REJECTED]} rejected of "
          f"{len(picks)} -> {log.path}, {items.path}")
    return True


def _open_window(cv2, clicks: list) -> None:
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, lambda event, x, y, *_: clicks.append((x, y))
                         if event == cv2.EVENT_LBUTTONDOWN else None)


def _reopen_window(cv2, clicks: list) -> tuple[int, int]:
    """A new window in place of the shown one; returns where its image was.

    With OpenCV's Qt backend a click gives the image view the keyboard focus, and the view then takes
    the arrow keys for scrolling (letters still reach the window, OpenCV issue #1695). Only a new
    window has the focus again.
    """
    x, y, _, _ = cv2.getWindowImageRect(WINDOW)
    cv2.destroyWindow(WINDOW)
    cv2.waitKey(1)
    _open_window(cv2, clicks)
    return x, y


def _place_window(cv2, image_position: tuple[int, int]) -> None:
    """Move the window so that its image is at ``image_position`` (the window frame lies above it)."""
    cv2.moveWindow(WINDOW, *image_position)
    cv2.waitKey(1)
    x, y, _, _ = cv2.getWindowImageRect(WINDOW)
    if (x, y) != image_position:
        cv2.moveWindow(WINDOW, 2 * image_position[0] - x, 2 * image_position[1] - y)


def tile_at(journal: Journal, point: tuple[float, float]) -> int | None:
    """The index of the pick whose tile on the journal's sheet (unscaled) contains ``point``."""
    _, boxes = sheet_layout(len(journal.articles))
    return next((k for k, (x, y, width, height) in enumerate(boxes)
                 if x <= point[0] < x + width and y <= point[1] < y + height), None)


def close(directory: Path) -> None:
    """Close the review round of one library folder and tell what it decided."""
    store = ArticleStore(directory.parent, directory.name)
    log, items = ReviewLog.load(store), ItemReviewLog.load(store)
    number = close_round(log, items)
    picks = [a for j in stored_journals(store) for a in j.articles]
    counts = {v: sum(1 for a in picks if items.verdict(a) == v) for v in (APPROVED, REJECTED)}
    print(f"{directory}: round {number} closed; picks {counts[APPROVED]} approved, {counts[REJECTED]} rejected, "
          f"{len(picks) - sum(counts.values())} without a verdict of {len(picks)}; "
          f"{len(to_replace(store))} rejected picks to replace (select --replace)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directories", nargs="+", type=Path, help="Library folders, e.g. .../articles/knav")
    parser.add_argument("--all", action="store_true", help="Go through every journal and pick again.")
    parser.add_argument("--closed", action="store_true",
                        help="Show the rejections of closed rounds too, to change them.")
    parser.add_argument("--close-round", action="store_true",
                        help="Close the review round of every folder (no window): its verdicts become final and "
                             "its rejections are no longer shown.")
    parser.add_argument("--max-width", type=int, default=1900, help="Window size limit in pixels.")
    parser.add_argument("--max-height", type=int, default=1050)
    args = parser.parse_args()
    for directory in args.directories:
        if args.close_round:
            close(directory.resolve())
        elif not review(directory.resolve(), args.all, args.max_width, args.max_height, args.closed):
            break


if __name__ == "__main__":
    main()

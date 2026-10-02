"""Review the sampled journals of library folders, first as a whole and then pick by pick.

    python -m metakat.chapter.download_articles.common.review /mnt/kolosus/data/smart_digiline/articles/knav [...]

A window first shows every stored title page of one journal in year order:

    y  approve the journal and review its picks one by one
    n  reject the journal and all its picks, go to the next journal
    space  skip    b  previous journal    u  clear verdict    q / Esc  quit
    click a page to see it enlarged, any key returns

The picks of an approved journal are then shown one at a time, enlarged:

    y  approve the pick    n  reject the pick    space  skip    b  back (before the first pick: the journal)
    u  clear verdict       j  leave the picks, go to the next journal    q / Esc  quit

Verdicts are saved after every key: journals into ``<library>/review.csv``, picks into
``<library>/review_items.csv`` (``by`` tells a pick's own verdict from one inherited from a rejected
journal). A new session starts at the first journal without a verdict, or inside an approved
journal at its first pick without a verdict; ``--all`` goes through everything again. Journal sheets
are cached in ``<library>/previews/journals/``.
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
JOURNAL_FIELDS = ["journal_id", "journal_title", "samples", "first_year", "last_year", "verdict", "reviewed_at"]
ITEM_FIELDS = ["item_id", "journal_id", "journal_title", "year", "volume", "issue", "title", "image", "verdict",
               "by", "reviewed_at"]
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
        self.index, self.item, self.done = 0, None, not journals
        if journals and not review_all:
            start = next((k for k, j in enumerate(journals) if self._unfinished(j)), None)
            if start is None:
                self.done = True
            else:
                self.index = start
                if log.verdict(self.journal) == APPROVED:
                    self.item = self._first_open_item()

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
        if key in ("q", "esc"):
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
            self.item = self._first_open_item()
        elif key == "n":
            self.log.set(journal, REJECTED)
            self.items.set(journal.articles, REJECTED, by=BY_JOURNAL)
            self._next_journal()
        elif key == "u":
            self.log.set(journal, None)
            self.items.set([a for a in journal.articles if self.items.by(a) == BY_JOURNAL], None)
        elif key in (" ", "s"):
            self._next_journal()
        elif key == "b":
            self.index = max(0, self.index - 1)

    def _handle_item(self, key: str) -> None:
        if key in ("y", "n"):
            self.items.set([self.article], APPROVED if key == "y" else REJECTED)
            self._next_item()
        elif key == "u":
            self.items.set([self.article], None)
        elif key in (" ", "s"):
            self._next_item()
        elif key == "b":
            self.item = None if self.item == 0 else self.item - 1
        elif key == "j":
            self._next_journal()

    def _next_item(self) -> None:
        """The next pick of the journal (without a verdict, unless reviewing all), else the next journal."""
        following = range(self.item + 1, len(self.journal.articles))
        item = next((k for k in following
                     if self.review_all or not self.items.verdict(self.journal.articles[k])), None)
        if item is None:
            self._next_journal()
        else:
            self.item = item

    def _next_journal(self) -> None:
        """The next journal (with something left to review, unless reviewing all); an approved one
        opens at its first pick without a verdict."""
        self.item = None
        following = range(self.index + 1, len(self.journals))
        index = next((k for k in following if self.review_all or self._unfinished(self.journals[k])), None)
        if index is None:
            self.done = True
            return
        self.index = index
        if not self.review_all and self.log.verdict(self.journal) == APPROVED:
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


def _key_name(code: int) -> str | None:
    if code < 0:
        return None
    code &= 0xFF
    if code == 27:
        return "esc"
    if code == 8:
        return "b"
    return chr(code).lower() if 32 <= code < 127 else None


def _wait_key(cv2, clicks: list) -> str | None:
    """The next key, or None when the window was clicked."""
    while not clicks:
        key = _key_name(cv2.waitKey(50))
        if key:
            return key
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return "q"
    return None


def review(directory: Path, review_all: bool, max_width: int, max_height: int) -> bool:
    """Review one library folder; returns False when the reviewer quit."""
    import cv2

    store = ArticleStore(directory.parent, directory.name)
    journals = stored_journals(store)
    log, items = ReviewLog.load(store), ItemReviewLog.load(store)
    session = Session(journals, log, items, review_all)
    if session.done:
        print(f"{directory}: {len(journals)} journals, nothing to review")
        return True

    clicks: list[tuple[int, int]] = []
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, lambda event, x, y, *_: clicks.append((x, y))
                         if event == cv2.EVENT_LBUTTONDOWN else None)
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
            position = f"{directory.name}  [{index + 1}/{len(journals)}, {reviewed} reviewed]"
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
                         "y approve and review picks · n reject journal and picks · space skip · b back · u clear · "
                         "q quit · click enlarges"]
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
                         f"{journal.title[:70]}  ·  y approve · n reject · space skip · b back · u clear · "
                         f"j next journal · q quit"]
            cv2.imshow(WINDOW, _compose(shown, lines, verdict))

            key = _wait_key(cv2, clicks)
            if key is None:
                x, y = clicks.pop()
                clicks.clear()
                if scale is not None:
                    _enlarge(cv2, store, journal, (x / scale, (y - HEADER_HEIGHT) / scale), max_width, max_height,
                             clicks)
                continue
            session.handle(key)
            if key in ("q", "esc"):
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


def _enlarge(cv2, store: ArticleStore, journal: Journal, point: tuple[float, float], max_width: int,
             max_height: int, clicks: list) -> None:
    _, boxes = sheet_layout(len(journal.articles))
    for article, (x, y, width, height) in zip(journal.articles, boxes):
        if x <= point[0] < x + width and y <= point[1] < y + height:
            item = article.item
            lines = [f"{item.year}  v{item.volume or '?'}/{item.issue or '?'}  {item.title or ''}"[:120],
                     "any key or click returns"]
            cv2.imshow(WINDOW, _compose(load_page(store, article, max_width, max_height), lines, None))
            _wait_key(cv2, clicks)
            clicks.clear()
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directories", nargs="+", type=Path, help="Library folders, e.g. .../articles/knav")
    parser.add_argument("--all", action="store_true", help="Go through every journal and pick again.")
    parser.add_argument("--max-width", type=int, default=1900, help="Window size limit in pixels.")
    parser.add_argument("--max-height", type=int, default=1050)
    args = parser.parse_args()
    for directory in args.directories:
        if not review(directory.resolve(), args.all, args.max_width, args.max_height):
            break


if __name__ == "__main__":
    main()

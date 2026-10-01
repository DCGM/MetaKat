"""Review the sampled journals of library folders one by one and record which ones to trust.

    python -m metakat.chapter.download_articles.common.review /mnt/kolosus/data/smart_digiline/articles/knav [...]

A window shows every stored title page of one journal in year order. Keys:

    y  approve          n  reject          space  skip (no verdict)
    b  back             u  clear verdict   q / Esc  quit
    click a page to see it enlarged, any key returns

Verdicts are saved after every key into ``<library>/review.csv``; a new session starts at the first
journal without a verdict (``--all`` goes through every journal). Journal sheets are cached in
``<library>/previews/journals/``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from metakat.chapter.download_articles.common.models import StoredArticle
from metakat.chapter.download_articles.common.preview import _font, _slug, journal_sheet, sheet_layout
from metakat.chapter.download_articles.common.selection import journal_key
from metakat.chapter.download_articles.common.store import ArticleStore

REVIEW_FILE = "review.csv"
FIELDS = ["journal_id", "journal_title", "samples", "first_year", "last_year", "verdict", "reviewed_at"]
APPROVED, REJECTED = "approved", "rejected"
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


@dataclass
class ReviewLog:
    """``review.csv`` of a library folder: one row per journal with its verdict."""

    path: Path
    rows: dict = field(default_factory=dict)

    @classmethod
    def load(cls, store: ArticleStore) -> ReviewLog:
        log = cls(store.dir / REVIEW_FILE)
        if log.path.exists():
            with open(log.path, newline="", encoding="utf-8") as file:
                for row in csv.DictReader(file):
                    log.rows[(row["journal_id"] or None, row["journal_title"] or None)] = row
        return log

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

    def save(self) -> None:
        temporary = self.path.with_suffix(".csv.tmp")
        with open(temporary, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(sorted(self.rows.values(), key=lambda r: (r["journal_title"], r["journal_id"])))
        temporary.replace(self.path)


class Session:
    """Which journal is shown and what a key does; kept apart from the window so it can be tested."""

    def __init__(self, journals: list[Journal], log: ReviewLog, review_all: bool = False):
        self.journals, self.log = journals, log
        unreviewed = [k for k, j in enumerate(journals) if not log.verdict(j)]
        self.index = 0 if review_all or not unreviewed else unreviewed[0]
        self.done = not journals or (not review_all and not unreviewed)

    @property
    def journal(self) -> Journal:
        return self.journals[self.index]

    def handle(self, key: str) -> None:
        if key in ("y", "n"):
            self.log.set(self.journal, APPROVED if key == "y" else REJECTED)
            self._move(1)
        elif key == "u":
            self.log.set(self.journal, None)
        elif key in (" ", "s"):
            self._move(1)
        elif key == "b":
            self._move(-1)
        elif key in ("q", "esc"):
            self.done = True

    def _move(self, step: int) -> None:
        if self.index + step >= len(self.journals):
            self.done = True
        else:
            self.index = max(0, self.index + step)


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


def _header(text: str, width: int, verdict: str | None) -> Image.Image:
    colour = {APPROVED: (30, 120, 50), REJECTED: (160, 40, 40)}.get(verdict, (45, 45, 45))
    header = Image.new("RGB", (width, HEADER_HEIGHT), colour)
    ImageDraw.Draw(header).text((12, 16), text, font=_font(26, bold=True), fill="white")
    return header


def _to_bgr(image: Image.Image) -> np.ndarray:
    return np.asarray(image)[:, :, ::-1].copy()


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


def review(directory: Path, review_all: bool, max_width: int, max_height: int) -> bool:
    """Review one library folder; returns False when the reviewer quit."""
    import cv2

    store = ArticleStore(directory.parent, directory.name)
    journals = stored_journals(store)
    log = ReviewLog.load(store)
    session = Session(journals, log, review_all)
    if session.done:
        print(f"{directory}: {len(journals)} journals, nothing to review")
        return True

    clicks: list[tuple[int, int]] = []
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, lambda event, x, y, *_: clicks.append((x, y))
                         if event == cv2.EVENT_LBUTTONDOWN else None)
    with ThreadPoolExecutor(max_workers=1) as prefetch:
        pending = {}
        while not session.done:
            journal, index = session.journal, session.index
            future = pending.pop(index, None)
            sheet = future.result() if future else load_sheet(store, journal)
            if index + 1 < len(journals) and index + 1 not in pending:
                pending[index + 1] = prefetch.submit(load_sheet, store, journals[index + 1])

            shown, scale = _fit(sheet, max_width, max_height - HEADER_HEIGHT)
            verdict = log.verdict(journal)
            reviewed = sum(1 for j in journals if log.verdict(j))
            text = (f"{directory.name}  [{index + 1}/{len(journals)}, {reviewed} reviewed]  "
                    f"{(verdict or 'no verdict').upper()}  ·  y approve · n reject · space skip · b back · "
                    f"u clear · q quit · click enlarges")
            canvas = Image.new("RGB", (max(shown.width, 900), shown.height + HEADER_HEIGHT), "white")
            canvas.paste(_header(text, canvas.width, verdict), (0, 0))
            canvas.paste(shown, (0, HEADER_HEIGHT))
            cv2.imshow(WINDOW, _to_bgr(canvas))

            key = None
            while key is None and not clicks:
                key = _key_name(cv2.waitKey(50))
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    key = "q"
            if clicks:
                x, y = clicks.pop()
                clicks.clear()
                _enlarge(cv2, store, journal, (x / scale, (y - HEADER_HEIGHT) / scale), max_width, max_height)
                continue
            session.handle(key)
            if key in ("q", "esc"):
                cv2.destroyAllWindows()
                return False
    cv2.destroyAllWindows()
    counts = {v: sum(1 for j in journals if log.verdict(j) == v) for v in (APPROVED, REJECTED)}
    print(f"{directory}: {counts[APPROVED]} approved, {counts[REJECTED]} rejected, "
          f"{len(journals) - sum(counts.values())} without verdict -> {log.path}")
    return True


def _enlarge(cv2, store: ArticleStore, journal: Journal, point: tuple[float, float], max_width: int,
             max_height: int) -> None:
    _, boxes = sheet_layout(len(journal.articles))
    for article, (x, y, width, height) in zip(journal.articles, boxes):
        if x <= point[0] < x + width and y <= point[1] < y + height:
            with Image.open(store.dir / article.image.file) as image:
                page, _ = _fit(image.convert("RGB"), max_width, max_height - HEADER_HEIGHT)
            item = article.item
            text = f"{item.year}  v{item.volume or '?'}/{item.issue or '?'}  {item.title or ''}"[:120]
            canvas = Image.new("RGB", (max(page.width, 900), page.height + HEADER_HEIGHT), "white")
            canvas.paste(_header(text + "  ·  any key returns", canvas.width, None), (0, 0))
            canvas.paste(page, (0, HEADER_HEIGHT))
            cv2.imshow(WINDOW, _to_bgr(canvas))
            while cv2.waitKey(50) < 0 and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1:
                pass
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directories", nargs="+", type=Path, help="Library folders, e.g. .../articles/knav")
    parser.add_argument("--all", action="store_true", help="Go through every journal, not only unreviewed ones.")
    parser.add_argument("--max-width", type=int, default=1900, help="Window size limit in pixels.")
    parser.add_argument("--max-height", type=int, default=1050)
    args = parser.parse_args()
    for directory in args.directories:
        if not review(directory.resolve(), args.all, args.max_width, args.max_height):
            break


if __name__ == "__main__":
    main()

"""Guess which picks of the current review round are reviews of someone's work (a book, an exhibition,
...) and, with ``--write``, store the guesses as review marks for ``common/review.py``.

    python -m metakat.chapter.download_articles.common.guess_reviews /mnt/kolosus/data/smart_digiline/articles/knav [...] [--write]

The evidence:

* metadata: a ``type``, ``section`` or catalogued Kramerius genre (``genres``, DIKDA) naming reviews (Recenze, Reviews, Buchbesprechungen, Muzejní
  kritika, ...; not "Recenzovaný článek" / "peer-reviewed", which mean refereed, nor "review article",
  a survey), the item type ``review`` of dml.cz, the keyword ``recenze`` of KNAV;
* KNAV: the article's genre in Kramerius (Recenze, Recensions, Reviews, Anotace), asked for 50 picks a
  request and cached in ``<library>/kramerius_genres.json``;
* the record title: an ISBN, a page count ("240 s.", "123 pp.") or "Recenze:", "Review of" at its start;
* the OCR text of the title page (``<library>/txt/<item_id>.txt``, one line of text per line) when it
  exists: an ISBN, a heading such as RECENZE, Reviews, Comptes-rendus, Knihy near the top (not
  "Review" alone, which journals use for review articles, nor "Anotace" or "Literatura", which in
  Czech articles head the abstract and the references), or a citation line near the top: a year with a
  page count or "Place: Publisher, year" (not a "Received: ..." line).

A guess is written as ``review = yes``, ``review_by = auto`` and the evidence in ``review_note``; marks
set or confirmed in the review (``review_by = item``) are never changed. Without ``--write`` the guesses
are only printed.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.parse
from collections import Counter
from pathlib import Path

from metakat.chapter.download_articles.common import http
from metakat.chapter.download_articles.common.models import CatalogItem, StoredArticle
from metakat.chapter.download_articles.common.review import BY_AUTO, ItemReviewLog, ReviewLog, review_journals
from metakat.chapter.download_articles.common.store import ArticleStore

KNAV = "knav"
KNAV_API = "https://kramerius.lib.cas.cz/search/api/client/v7.0"
GENRE_CACHE = "kramerius_genres.json"

NOT_REVIEW = re.compile(r"recenzovan|peer.?review|review[- ]?(articles?|papers?|stud(y|ies))|review \(only|"
                        r"reviewed", re.I)
REVIEW_VALUE = re.compile(r"recenz|rezension|recension|buchbesprech|book ?reviews?|knižní recenze|^reviews?\b|"
                          r"\breviews?$|anotac|annotation|panorama knih|muzejní kritika|^book$", re.I)
REVIEW_KEYWORDS = {"recenze", "recenzia", "review", "reviews", "book review", "rezension"}
REVIEW_GENRES = {"recenze", "recensions", "reviews", "review", "anotace", "annotations"}
# A page count: an abbreviation needs its dot ("240 s.", "123pp."), a word a space ("640 pages"), so that
# "1950s." or "1989 s sebou" are not counts.
PAGES = (r"(\b\d{2,4}\s+(s|S|str|p|pp|с)|\b\d{2,4}(str|p|pp))\.(?=[\s,;:)\]]|$)|"
         r"\b\d{2,4}\s+(stran|stron|stránek|pages|Seiten|páginas|pagine)\b")
TITLE = re.compile(rf"\bISBN\b|{PAGES}|\b(pp|s)\.\s*\d|^\W*(recenze|review of|rezension|book review)\b", re.I)
HEADING = re.compile(r"^\W*(\d+\W+)?(recenze|recenzia|recenzja|rezension(en)?|recension[se]?|reviews|book reviews?|"
                     r"comptes?[- ]rendus?|nové knihy|z nové literatury|knihy|buchbesprechung(en)?|рецензи[яи])"
                     r"(\W+\d+)?\W*$", re.I)
HEADING_WORDS = 4
DATES = re.compile(r"received|accepted|revised|published|submitted|přijato|doručeno|online", re.I)
CITATION = re.compile(rf"(19|20)\d\d.{{0,60}}{PAGES}|{PAGES}.{{0,30}}(19|20)\d\d|"
                      r"\b[A-ZÁ-Ž][a-zá-žäöüß]+(\s[a-zA-ZÁ-Ž][a-zá-žäöüß]+)?\s*:\s*[^,:]{2,60},\s*(19|20)\d\d\b")
# How many lines from the top a heading (a few more) or a citation line may be.
TOP_LINES = 12


def metadata_evidence(item: CatalogItem) -> list[str]:
    found = []
    for field in ("type", "section", "genres"):
        for value in item.record.get(field, []):
            if REVIEW_VALUE.search(value) and not NOT_REVIEW.search(value):
                found.append(f"{field}: {value}")
    if item.item_type and REVIEW_VALUE.search(item.item_type) and not NOT_REVIEW.search(item.item_type):
        found.append(f"type: {item.item_type}")
    found += [f"keyword: {keyword}" for keyword in item.record.get("keywords", [])
              if keyword.lower() in REVIEW_KEYWORDS]
    if item.title and TITLE.search(item.title):
        found.append("title")
    return found


def ocr_evidence(lines: list[str]) -> list[str]:
    found = []
    if any(re.search(r"\bISBN\b", line) for line in lines):
        found.append("OCR: ISBN")
    heading = next((line.strip() for line in lines[:TOP_LINES + 3]
                    if HEADING.match(line.strip()) and len(line.split()) <= HEADING_WORDS), None)
    if heading:
        found.append(f"OCR heading: {heading[:30]}")
    citation = next((line.strip() for line in lines[:TOP_LINES] if CITATION.search(line) and not DATES.search(line)),
                    None)
    if citation:
        found.append(f"OCR citation: {citation[:60]}")
    return found


def knav_genres(store: ArticleStore, picks: list[StoredArticle]) -> dict[str, list[str]]:
    """Kramerius genres of the KNAV article picks (record id -> genres), 50 a request; cached."""
    cache = store.dir / GENRE_CACHE
    genres = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    pids = sorted({a.item.record_id for a in picks if a.item.item_type != "volume"} - set(genres))
    for start in range(0, len(pids), 50):
        batch = pids[start:start + 50]
        params = {"q": "pid:(" + " OR ".join(f'"{pid}"' for pid in batch) + ")", "fl": "pid,genres.facet",
                  "rows": len(batch), "wt": "json"}
        docs = json.loads(http.http_get(f"{KNAV_API}/search?{urllib.parse.urlencode(params)}"))["response"]["docs"]
        found = {doc["pid"]: doc.get("genres.facet", []) for doc in docs}
        genres.update({pid: found.get(pid, []) for pid in batch})
    cache.write_text(json.dumps(genres, ensure_ascii=False), encoding="utf-8")
    return genres


def guess(store: ArticleStore, picks: list[StoredArticle]) -> list[tuple[StoredArticle, list[str]]]:
    """The picks that look like reviews, each with its evidence."""
    genres = knav_genres(store, picks) if store.dir.name == KNAV else {}
    guessed = []
    for article in picks:
        evidence = metadata_evidence(article.item)
        review_genres = [g for g in genres.get(article.item.record_id, []) if g.lower() in REVIEW_GENRES]
        if review_genres:
            evidence.append("genre: " + "/".join(review_genres))
        txt = store.dir / "txt" / f"{article.item.item_id}.txt"
        if txt.exists():
            evidence += ocr_evidence(txt.read_text(encoding="utf-8").splitlines())
        if evidence:
            guessed.append((article, evidence))
    return guessed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directories", nargs="+", type=Path, help="Library folders, e.g. .../articles/knav")
    parser.add_argument("--write", action="store_true", help="Store the guesses as review marks.")
    args = parser.parse_args()
    http.set_min_interval(2.0)
    for directory in args.directories:
        directory = directory.resolve()
        store = ArticleStore(directory.parent, directory.name)
        log, items = ReviewLog.load(store), ItemReviewLog.load(store)
        picks = [a for j in review_journals(store, log, items) for a in j.articles]
        guessed = guess(store, picks)
        without_ocr = sum(1 for a in picks if not (store.dir / "txt" / f"{a.item.item_id}.txt").exists())
        for article, evidence in guessed:
            print(f"  {(items.verdict(article) or 'open'):8} {'; '.join(evidence)[:100]:100} | "
                  f"{(article.item.journal_title or '')[:28]:28} {article.item.year} | {(article.item.title or '')[:60]}")
        kinds = Counter(kind for _, evidence in guessed for kind in {e.split(":")[0] for e in evidence})
        writable = [(a, e) for a, e in guessed if items.review_by(a) in (None, BY_AUTO)]
        if args.write:
            for article, evidence in writable:
                items.set_review([article], True, BY_AUTO, note="; ".join(evidence)[:200])
        print(f"{directory}: {len(guessed)} of {len(picks)} picks look like reviews {dict(kinds)}; "
              f"{len(writable)} {'written' if args.write else 'to write'} as guesses; {without_ocr} without OCR text")


if __name__ == "__main__":
    main()

"""Pair the picks of a library with title pages of the other kind: for every year of a journal with an
article pick, a review from the same journal and year, and for every year with a review pick, an
article, so that reviews are sampled as widely as articles.

    python -m metakat.chapter.download_articles.common.pair_reviews plan   --source knav
    python -m metakat.chapter.download_articles.common.pair_reviews stage  --source knav
    (OCR of <root>/_pairing/<library>/images into page_xml/, alto/ and txt/ beside it)
    python -m metakat.chapter.download_articles.common.pair_reviews settle --source knav

The picks are those the current review round shows that are approved or open; a pick is a review when
its review mark says so (set, confirmed or guessed). Only journals with some sign of reviews get review
slots: catalog items whose metadata names reviews (a review section or type, the KNAV genre Recenze,
...) or whose title reads like a cited book (ISBN, page count, "Recenze:"), or a review among the
picks. Journals without such a sign only get articles for their review picks.

Candidates for a slot come from the same journal and year, never items stored, tried or refused
before, nor items whose title names a non-article (``Source.is_unlikely``) or KNAV volumes. For a
review the best evidence comes first: metadata, then a book citation in the title, then an
"Author Name: Title" title, then (only where the catalog does not mark reviews completely, and for
KNAV only in years without any genre) items without any sign, tried on the strength of their OCR. For
an article, items without any sign of a review, preferring those that cost no request.

``stage`` fetches the next candidate of every open slot (at most ``MAX_ATTEMPTS`` per slot) into the
staging folder ``<root>/_pairing/<library>/``, which has the layout of a library folder.
``settle`` checks every staged candidate against its OCR text (``txt/<item_id>.txt``, as for
``guess_reviews``): a review needs review metadata or a sign in the text (ISBN, a review heading, a
citation near the top), an article needs neither a review heading nor a citation. A candidate that
passes is moved into the library as a new pick, a review with its review mark guessed
(``review_by = auto``); every candidate and its outcome is listed in ``<library>/pair_attempts.tsv``.
Libraries that serve files to people only get a ``selection.html`` in the staging folder; ``stage
--pdf-dir`` then takes the files saved by hand.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from metakat.chapter.download_articles.common import http
from metakat.chapter.download_articles.common.cli import DEFAULT_ROOT, STORED, available_items, fetch_item
from metakat.chapter.download_articles.common.guess_reviews import REVIEW_GENRES, metadata_evidence, ocr_evidence
from metakat.chapter.download_articles.common.models import CatalogItem
from metakat.chapter.download_articles.common.review import (APPROVED, BY_AUTO, ItemReviewLog, ReviewLog,
                                                             review_journals)
from metakat.chapter.download_articles.common.selection import JournalKey, journal_key
from metakat.chapter.download_articles.common.source import Source
from metakat.chapter.download_articles.common.store import ArticleStore

logger = logging.getLogger(__name__)

REVIEW, ARTICLE = "review", "article"
STAGING = "_pairing"
ATTEMPTS_FILE = "pair_attempts.tsv"
ATTEMPT_FIELDS = ["item_id", "want", "for_year", "journal_id", "journal_title", "score", "evidence", "outcome",
                  "tried_at", "title"]
PENDING, PROMOTED, MISSING, WRONG_KIND = "pending", "promoted", "missing", "wrong kind"
MAX_ATTEMPTS = 3
# Catalog titles citing a book that make a journal one with reviews when nothing else does.
MIN_CITED = 2
# Libraries whose catalogs give every item a section or type, so that a year without review items has
# no reviews and is not probed with items that show no sign.
COMPLETE_METADATA = {"digilib.phil.muni.cz", "journals.muni.cz", "ojs.cvut.cz", "ojs_sites", "dml.cz"}
KNAV, KNAV_GENRE_CACHE = "knav", "kramerius_journal_genres.json"

# Evidence scores of a review candidate.
METADATA, CITED, AUTHOR_TITLE, NONE = 3, 2, 1, 0
_NAME = r"[A-ZÀ-ÖØ-ÞĀ-Ž][\w'’.-]*"
_PERSON = rf"{_NAME}(\s+{_NAME}){{1,3}}"
AUTHOR_TITLE_PATTERN = re.compile(
    rf"^\W*{_PERSON}((\s*[-–,&]\s*|\s+(a|and|und|et|i)\s+){_PERSON})*"
    r"(\s*(et al\.|a kol\.|\((eds?|Hrsg|red)\.?\)))?\s*[:,]\s+\S")


@dataclass
class Slot:
    key: JournalKey
    year: int
    want: str
    candidates: list[tuple[CatalogItem, int, list[str]]] = field(default_factory=list)
    tried: int = 0


@dataclass
class JournalPlan:
    key: JournalKey
    evidence: Counter
    eligible: bool
    articles: int = 0
    reviews: int = 0
    slots: list[Slot] = field(default_factory=list)


def review_score(item: CatalogItem, genres: dict[str, list[str]]) -> tuple[int, list[str]]:
    """How strongly the catalog says the item is a review, and why."""
    evidence = [e for e in metadata_evidence(item) if e != "title"]
    review_genres = [g for g in genres.get(item.record_id, []) if g.lower() in REVIEW_GENRES]
    if review_genres:
        evidence.append("genre: " + "/".join(review_genres))
    if evidence:
        return METADATA, evidence
    if "title" in metadata_evidence(item):
        return CITED, ["title cites a book"]
    if item.title and AUTHOR_TITLE_PATTERN.match(item.title):
        return AUTHOR_TITLE, ["title: author: book"]
    return NONE, []


def knav_journal_genres(source: Source, store: ArticleStore, journal_ids: set[str]) -> dict[str, list[str]]:
    """Kramerius genres of every article with a genre in the journals (record id -> genres); cached."""
    cache = store.dir / KNAV_GENRE_CACHE
    data = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {"journals": [], "genres": {}}
    for journal_id in sorted(journal_ids - set(data["journals"])):
        docs = source.kramerius.search_all(f'root.pid:"uuid:{journal_id}" AND model:article AND genres.facet:*',
                                           ["pid", "genres.facet"], filtered=False)
        data["genres"].update({doc["pid"]: doc.get("genres.facet", []) for doc in docs})
        data["journals"].append(journal_id)
        cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data["genres"]


def read_attempts(store: ArticleStore) -> list[dict]:
    path = store.dir / ATTEMPTS_FILE
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def write_attempts(store: ArticleStore, rows: list[dict]) -> None:
    path = store.dir / ATTEMPTS_FILE
    with open(path.with_suffix(".tsv.tmp"), "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=ATTEMPT_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(".tsv.tmp").replace(path)


def plan(source: Source, store: ArticleStore, seed: int = 0) -> list[JournalPlan]:
    """The journals of the library with their evidence of reviews and the slots still open."""
    log, items = ReviewLog.load(store), ItemReviewLog.load(store)
    picks = [a for j in review_journals(store, log, items) for a in j.articles
             if items.verdict(a) in (APPROVED, None)]
    catalog = store.read_catalog()
    stored_ids = {a.item.item_id for a in store.stored()}
    attempts = read_attempts(store)
    tried_ids = {row["item_id"] for row in attempts}
    tried_slots = Counter((row["journal_id"], row["journal_title"], row["for_year"], row["want"]) for row in attempts)

    journal_keys = {journal_key(a.item) for a in picks}
    genres: dict[str, list[str]] = {}
    if store.dir.name == KNAV:
        genres = knav_journal_genres(source, store, {key[0] for key in journal_keys if key[0]})

    by_journal: dict[JournalKey, list[CatalogItem]] = defaultdict(list)
    for item in catalog:
        if journal_key(item) in journal_keys:
            by_journal[journal_key(item)].append(item)
    available = {item.item_id for item in available_items(source, store, catalog)}
    rng = random.Random(seed)

    plans = []
    for key in sorted(journal_keys, key=lambda k: (k[1] or "", k[0] or "")):
        journal_picks = [a for a in picks if journal_key(a.item) == key]
        kinds: dict[int, set[str]] = defaultdict(set)
        for article in journal_picks:
            if article.item.year is not None:
                kinds[article.item.year].add(REVIEW if items.is_review(article) else ARTICLE)
        scores = {item.item_id: review_score(item, genres) for item in by_journal[key]}
        evidence = Counter({"metadata": sum(s == METADATA for s, _ in scores.values()),
                            "cited title": sum(s == CITED for s, _ in scores.values()),
                            "author: title": sum(s == AUTHOR_TITLE for s, _ in scores.values()),
                            "review picks": sum(REVIEW in k for k in kinds.values())})
        # One title citing a book may be a chance (a review article quoting its ISBN); two are a pattern.
        eligible = bool(evidence["metadata"] or evidence["cited title"] >= MIN_CITED or evidence["review picks"])
        journal = JournalPlan(key, evidence, eligible, articles=sum(ARTICLE in k for k in kinds.values()),
                              reviews=evidence["review picks"])
        genre_years = {item.year for item in by_journal[key] if genres.get(item.record_id)}
        candidates_by_year: dict[int, list[CatalogItem]] = defaultdict(list)
        for item in sorted(by_journal[key], key=lambda i: i.item_id):
            if (item.year in kinds and item.item_id in available and item.item_id not in stored_ids
                    and item.item_id not in tried_ids and item.item_type != "volume" and not source.is_unlikely(item)):
                candidates_by_year[item.year].append(item)

        for year in sorted(kinds):
            for want in (REVIEW, ARTICLE):
                other = ARTICLE if want == REVIEW else REVIEW
                if want in kinds[year] or other not in kinds[year] or (want == REVIEW and not eligible):
                    continue
                slot = Slot(key, year, want, tried=tried_slots[(key[0] or "", key[1] or "", str(year), want)])
                pool = candidates_by_year[year][:]
                rng.shuffle(pool)
                if want == REVIEW:
                    probe = (store.dir.name not in COMPLETE_METADATA
                             and not (store.dir.name == KNAV and year in genre_years))
                    ranked = [(i, *scores[i.item_id]) for i in pool if scores[i.item_id][0] > NONE or probe]
                    ranked.sort(key=lambda c: (-c[1], not source.is_cheap(c[0])))
                else:
                    ranked = [(i, NONE, []) for i in pool if scores[i.item_id][0] == NONE]
                    ranked.sort(key=lambda c: not source.is_cheap(c[0]))
                slot.candidates = ranked
                journal.slots.append(slot)
        plans.append(journal)
    return plans


def staging_store(store: ArticleStore) -> ArticleStore:
    return ArticleStore(store.dir.parent / STAGING, store.dir.name)


def stage(source: Source, store: ArticleStore, pdf_dir: Path | None = None, seed: int = 0) -> Counter:
    """Fetch the next candidate of every open slot into the staging folder."""
    staging = staging_store(store)
    attempts = read_attempts(store)
    pending = {row["item_id"] for row in attempts if row["outcome"] == PENDING}
    if pending:
        raise RuntimeError(f"{len(pending)} staged candidates are not settled yet: run settle first")
    counts = Counter()
    chosen: list[tuple[Slot, CatalogItem, int, list[str]]] = []
    for journal in plan(source, store, seed):
        for slot in journal.slots:
            if slot.tried >= MAX_ATTEMPTS:
                counts["slots given up"] += 1
            elif not slot.candidates:
                counts["slots without candidates"] += 1
            else:
                chosen.append((slot, *slot.candidates[0]))
    if source.manual_download:
        staging.write_selection([item for _, item, _, _ in chosen])
        if pdf_dir is None:
            logger.info(f"{len(chosen)} candidates to save by hand: open {staging.selection_html_path}, save the "
                        f"PDFs into one folder and run stage again with --pdf-dir")
            return counts
    refused = store.unavailable()
    for slot, item, score, evidence in chosen:
        outcome = fetch_item(source, staging, item, pdf_dir, refused=refused, refusals=store)
        counts[f"{slot.want} candidates {'staged' if outcome == STORED else 'missing'}"] += 1
        attempts.append({"item_id": item.item_id, "want": slot.want, "for_year": slot.year,
                         "journal_id": slot.key[0] or "", "journal_title": slot.key[1] or "", "score": score,
                         "evidence": "; ".join(evidence), "outcome": PENDING if outcome == STORED else MISSING,
                         "tried_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "title": item.title or ""})
        write_attempts(store, attempts)
    return counts


def settle(store: ArticleStore) -> Counter:
    """Check the staged candidates against their OCR text and move those of the wanted kind into the library."""
    staging = staging_store(store)
    attempts = read_attempts(store)
    items = ItemReviewLog.load(store)
    staged = {a.item.item_id: a for a in staging.stored()}
    counts = Counter()
    for row in attempts:
        if row["outcome"] != PENDING:
            continue
        article = staged.get(row["item_id"])
        txt = staging.dir / "txt" / f"{row['item_id']}.txt"
        if article is None or not txt.exists():
            counts["waiting for OCR"] += 1
            continue
        lines = txt.read_text(encoding="utf-8").splitlines()
        found = ocr_evidence(lines)
        signs = [e for e in found if not e.startswith("OCR: ISBN")]
        if row["want"] == REVIEW:
            right = int(row["score"]) == METADATA or bool(found)
            evidence = "; ".join(filter(None, [row["evidence"], *found]))
        else:
            right = not signs
            evidence = "; ".join(found)
        row["evidence"] = evidence
        row["outcome"] = PROMOTED if right else WRONG_KIND
        counts[f"{row['want']} {row['outcome']}"] += 1
        if right:
            promote(staging, store, row["item_id"])
            if row["want"] == REVIEW:
                items.set_review([article], True, BY_AUTO, note=("pairing; " + evidence)[:200])
    write_attempts(store, attempts)
    return counts


def promote(staging: ArticleStore, store: ArticleStore, item_id: str) -> None:
    """Move a staged candidate's files into the library: PDF, image, metadata and OCR."""
    for folder in ("pdf", "images", "metadata", "page_xml", "alto", "txt"):
        for path in (staging.dir / folder).glob(f"{item_id}.*"):
            (store.dir / folder).mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), store.dir / folder / path.name)


def report(store: ArticleStore, plans: list[JournalPlan]) -> None:
    attempts = read_attempts(store)
    by_journal: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for row in attempts:
        by_journal[(row["journal_id"], row["journal_title"])][f"{row['want']} {row['outcome']}"] += 1
    for journal in plans:
        tried = by_journal[(journal.key[0] or "", journal.key[1] or "")]
        open_slots = Counter(f"{s.want}{'' if s.candidates and s.tried < MAX_ATTEMPTS else ' (none left)'}"
                             for s in journal.slots)
        signs = ", ".join(f"{k} {v}" for k, v in journal.evidence.items() if v) or "no sign of reviews"
        print(f"{(journal.key[1] or journal.key[0] or '?')[:60]:60} | {journal.articles:3} art {journal.reviews:3} rev | "
              f"{'eligible' if journal.eligible else 'no reviews'} ({signs}) | promoted "
              f"{tried[f'{REVIEW} {PROMOTED}']} rev {tried[f'{ARTICLE} {PROMOTED}']} art | "
              f"wrong {tried[f'{REVIEW} {WRONG_KIND}'] + tried[f'{ARTICLE} {WRONG_KIND}']} missing "
              f"{tried[f'{REVIEW} {MISSING}'] + tried[f'{ARTICLE} {MISSING}']} | open {dict(open_slots) or '-'}")


def main():
    from metakat.chapter.download_articles.__main__ import SOURCES

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["plan", "stage", "settle"])
    parser.add_argument("--source", required=True, choices=sorted(SOURCES))
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--pdf-dir", type=Path, help="stage: folder with candidate PDFs saved by hand.")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--delay", type=float, help="Least seconds between two requests (default 2, more where the "
                                                     "library asks for it).")
    args = parser.parse_args()
    logging.basicConfig(format="%(asctime)s - PAIR REVIEWS - %(levelname)s - %(message)s", level=logging.INFO)
    source = SOURCES[args.source]()
    source.root = Path(args.root)
    http.set_min_interval(args.delay if args.delay is not None else max(2.0, source.min_interval))
    store = ArticleStore(args.root, source.name)
    if args.command == "stage":
        logger.info(f"{store.dir.name}: {dict(stage(source, store, args.pdf_dir, args.seed))}")
    elif args.command == "settle":
        logger.info(f"{store.dir.name}: {dict(settle(store))}")
    plans = plan(source, store, args.seed)
    report(store, plans)
    slots = [s for j in plans for s in j.slots]
    print(f"{store.dir.name}: {sum(j.eligible for j in plans)} of {len(plans)} journals with signs of reviews; open "
          f"slots: {sum(s.want == REVIEW for s in slots)} reviews, {sum(s.want == ARTICLE for s in slots)} articles "
          f"({sum(1 for s in slots if s.candidates and s.tried < MAX_ATTEMPTS)} with a candidate left)")


if __name__ == "__main__":
    main()

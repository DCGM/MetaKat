"""Journal articles registered in Crossref, for publishers whose sites have no OAI-PMH interface.

A publisher registers its DOIs under its own prefix, so all its journals are found with
``/prefixes/<prefix>/works``; single journals are found by their ISSNs. The works carry the journal
(``container-title``, ISSNs), volume, issue, pages, the publication date, authors and, as links for
similarity checking or text mining, the URL of the article PDF on the publisher's site, which is where
the file is downloaded from.

A journal is identified by its ISSNs (works list one or both of print and electronic ISSN, so ISSNs
listed together make one journal); its title is the most frequent spelling of its ``container-title``
(publishers deposit e.g. both "ORBIS SCHOLAE" and "Orbis Scholae").
"""
from __future__ import annotations

import json
import logging
import urllib.parse
from collections import Counter, defaultdict

from metakat.chapter.download_articles.common.http import http_get
from metakat.chapter.download_articles.common.models import CatalogItem
from metakat.chapter.download_articles.common.source import NOT_ARTICLE, Source
from metakat.chapter.download_articles.common.store import ArticleStore

logger = logging.getLogger(__name__)

API = "https://api.crossref.org"
ROWS = 1000
FIELDS = ["DOI", "title", "subtitle", "container-title", "ISSN", "volume", "issue", "page", "issued",
          "published-print", "published-online", "author", "link", "resource", "abstract",
          "subject", "license", "type", "publisher"]


class CrossrefSource(Source):
    """Journal articles of one publisher (``prefixes``) or of single journals (``issns``) in Crossref."""

    prefixes: tuple[str, ...] = ()
    # Journals by their ISSNs; each tuple holds the ISSNs of one journal.
    issns: tuple[tuple[str, ...], ...] = ()
    # Libraries harvested before this one: articles whose DOI is in their catalogs are left to them.
    exclude_libraries: tuple[str, ...] = ()

    def build_catalog(self) -> list[CatalogItem]:
        excluded = self.excluded_dois()
        works: dict[str, dict] = {}
        queries = [f"/prefixes/{prefix}/works" for prefix in self.prefixes]
        queries += [f"/journals/{journal[0]}/works" for journal in self.issns]
        for query in queries:
            count = 0
            for work in iter_works(query):
                works.setdefault(work["DOI"].lower(), work)
                count += 1
            logger.info(f"{query}: {count} articles")
        items = [item for doi, work in works.items() if doi not in excluded
                 and (item := item_from_work(work, self.name)) is not None]
        logger.info(f"{len(works)} articles, {len(works.keys() & excluded)} left to {', '.join(self.exclude_libraries)}")
        group_journals(items)
        unify_journal_titles(items)
        for title, count in sorted(Counter(item.journal_title for item in items).items(), key=str):
            logger.info(f"{title}: {count} articles")
        return items

    def excluded_dois(self) -> set[str]:
        dois = set()
        for library in self.exclude_libraries:
            for item in ArticleStore(self.root, library).read_catalog():
                dois.update(doi.lower() for doi in item.record.get("doi", []))
        return dois

    def is_available(self, item: CatalogItem) -> bool:
        return bool(item.pdf_urls) and not (item.title and NOT_ARTICLE.match(item.title))


def iter_works(query: str):
    """Every journal article of a Crossref works query, following the deep-paging cursor."""
    cursor = "*"
    while True:
        params = {"filter": "type:journal-article", "rows": ROWS, "cursor": cursor, "select": ",".join(FIELDS)}
        message = json.loads(http_get(f"{API}{query}?{urllib.parse.urlencode(params)}"))["message"]
        yield from message["items"]
        cursor = message.get("next-cursor")
        if not message["items"] or not cursor:
            return


def item_from_work(work: dict, library: str) -> CatalogItem | None:
    issns = sorted(set(work.get("ISSN", [])))
    if not issns:
        return None
    date = next((work[key]["date-parts"][0] for key in ("issued", "published-print", "published-online")
                 if work.get(key, {}).get("date-parts", [[None]])[0][0]), None)
    links = work.get("link", [])
    pdf_urls = [link["URL"] for link in links if link.get("content-type") == "application/pdf"]
    pdf_urls += [link["URL"] for link in links if link.get("content-type") == "unspecified"
                 and link["URL"].lower().split("?")[0].endswith(".pdf")]
    authors = [", ".join(part for part in (author.get("family"), author.get("given")) if part) or author.get("name", "")
               for author in work.get("author", [])]
    record = {
        "doi": [work["DOI"]],
        "titles": work.get("title", []),
        "subtitles": work.get("subtitle", []),
        "container_titles": work.get("container-title", []),
        "issn": issns,
        "abstracts": [work["abstract"]] if work.get("abstract") else [],
        "subjects": work.get("subject", []),
        "licenses": [license["URL"] for license in work.get("license", [])],
        "publisher": [work["publisher"]] if work.get("publisher") else [],
        "links": [link["URL"] for link in links],
    }
    return CatalogItem(
        library=library,
        item_id=work["DOI"].lower().replace("/", "_"),
        record_id=work["DOI"],
        title=(work.get("title") or [None])[0],
        item_type=work.get("type"),
        journal_id="+".join(issns),
        journal_title=(work.get("container-title") or [None])[0],
        volume=work.get("volume"),
        issue=work.get("issue"),
        year=date[0] if date else None,
        date="-".join(f"{part:02d}" for part in date) if date else None,
        pages=work.get("page"),
        authors=[author for author in authors if author],
        rights=record["licenses"],
        pdf_urls=list(dict.fromkeys(pdf_urls)),
        landing_url=work.get("resource", {}).get("primary", {}).get("URL") or f"https://doi.org/{work['DOI']}",
        record={key: values for key, values in record.items() if values},
    )


def group_journals(items: list[CatalogItem]) -> None:
    """Make ISSNs listed together one journal, identified by all its ISSNs."""
    parent: dict[str, str] = {}

    def find(issn: str) -> str:
        while parent.setdefault(issn, issn) != issn:
            issn = parent[issn]
        return issn

    for item in items:
        first, *others = item.record["issn"]
        find(first)
        for other in others:
            parent[find(other)] = find(first)
    groups: dict[str, set[str]] = defaultdict(set)
    for issn in list(parent):
        groups[find(issn)].add(issn)
    for item in items:
        item.journal_id = "+".join(sorted(groups[find(item.record["issn"][0])]))


def unify_journal_titles(items: list[CatalogItem]) -> None:
    """Give the items of a journal spelled in several letter cases its most frequent spelling."""
    spellings: dict[tuple, Counter] = defaultdict(Counter)
    for item in items:
        spellings[item.journal_id, (item.journal_title or "").casefold()][item.journal_title] += 1
    for item in items:
        item.journal_title = spellings[item.journal_id, (item.journal_title or "").casefold()].most_common(1)[0][0]

"""Articles of the journals of an Open Journal Systems (OJS) platform, harvested over OAI-PMH.

OJS publishes every article in ``oai_dc``. The record's set is ``<journal>:<section>``; ``dc:source``
holds the citation ("Religio; Vol 12 No 1 (2004); 5-26") and the ISSNs; ``dc:relation`` lists the
article's galleys as ``.../article/view/<article>/<galley>``, which are downloaded from
``.../article/download/<article>/<galley>``. A galley may be a link to a file hosted elsewhere (e.g. a
digital library behind a human check, or a paid database); OJS then redirects to it. Such redirects
are not followed: the article is recorded as refused, and a journal whose every attempt went
elsewhere is no longer selected.
"""
from __future__ import annotations

import logging
import re
import urllib.error
from collections import Counter
from functools import cached_property
from urllib.parse import urljoin, urlparse

from metakat.chapter.download_articles.common.http import http_get
from metakat.chapter.download_articles.common.models import CatalogItem
from metakat.chapter.download_articles.common.oai import OaiRecord, iter_records, list_sets
from metakat.chapter.download_articles.common.source import NOT_ARTICLE, DownloadBlocked, Source
from metakat.chapter.download_articles.common.store import ArticleStore

logger = logging.getLogger(__name__)

REDIRECT_CODES = (301, 302, 303, 307, 308)
HOSTED_ELSEWHERE = "hosted at"
# A journal with this many articles hosted elsewhere and none stored is left out of selections.
HOSTED_ELSEWHERE_LIMIT = 2

# Sections whose items are not research articles: picked only from years without an article.
OTHER_SECTION = re.compile(
    r"review|recenz|rezens|book|knih|news|zpráv|zprav|chronic|kronik|obituar|nekrolog|in memoriam|editorial|"
    r"úvodník|uvodnik|letter|dopis|discussion|diskus|polemi|report|announce|oznám|index|rejst|contents|obsah|"
    r"errat|interview|rozhovor|annotation|anotac|miscellan|varia|bibliograf|conference|konferen|jubile|výroč",
    re.IGNORECASE)

# Sets OJS adds besides its journals (DRIVER guidelines).
NOT_JOURNAL_SETS = frozenset({"driver"})

_ISSN = re.compile(r"^\d{4}-\d{3}[\dXx]$")
_GALLEY = re.compile(r"/article/view/[^/]+/[^/]+$")


class OjsSource(Source):
    """Every journal of OJS installations; subclasses set the name and the OAI-PMH addresses.

    One installation may hold many journals (a university platform), or a library may be several
    single-journal installations.
    """

    oai_urls: tuple[str, ...] = ()
    # Journal sets that are not journals (proceedings, book series, ...).
    skip_sets: frozenset[str] = frozenset()
    # Hosts besides the OAI-PMH hosts and the galley's own host that serve the platform's files.
    hosts: tuple[str, ...] = ()
    type_preference = ("article", "other")

    def build_catalog(self) -> list[CatalogItem]:
        items: dict[str, CatalogItem] = {}
        for oai_url in self.oai_urls:
            sets = dict(list_sets(oai_url))
            journals = {spec: name for spec, name in sets.items()
                        if ":" not in spec and spec not in self.skip_sets | NOT_JOURNAL_SETS}
            logger.info(f"{oai_url}: {len(journals)} journals: {', '.join(sorted(journals.values()))}")
            for record in iter_records(oai_url):
                item = item_from_record(record, self.name, journals, sets)
                if item is not None:
                    items.setdefault(item.item_id, item)
        counts = Counter(item.journal_title for item in items.values())
        no_file = Counter(item.journal_title for item in items.values() if not item.pdf_urls)
        for title, count in sorted(counts.items()):
            logger.info(f"{title}: {count} articles, {no_file[title]} without a file")
        return list(items.values())

    def is_available(self, item: CatalogItem) -> bool:
        return (bool(item.pdf_urls) and not (item.title and NOT_ARTICLE.match(item.title))
                and item.journal_id not in self.hosted_elsewhere)

    @cached_property
    def hosted_elsewhere(self) -> set[str]:
        """Journals whose files are kept on other sites: refused that way repeatedly, none stored."""
        if self.root is None:
            return set()
        store = ArticleStore(self.root, self.name)
        refused = Counter(row["journal_id"] for row in store.unavailable().values()
                          if row["reason"].startswith(HOSTED_ELSEWHERE))
        stored = {article.item.journal_id for article in store.stored()}
        journals = {journal for journal, count in refused.items()
                    if count >= HOSTED_ELSEWHERE_LIMIT and journal not in stored}
        if journals:
            logger.info(f"Journals hosted elsewhere, not selected: {', '.join(sorted(journals))}")
        return journals

    def download_pdf(self, url: str) -> bytes:
        """Download a galley, following redirects only within the platform."""
        allowed = {urlparse(url).netloc, *(urlparse(oai_url).netloc for oai_url in self.oai_urls), *self.hosts}
        for _ in range(5):
            try:
                data = http_get(url, follow_redirects=False)
            except urllib.error.HTTPError as error:
                location = error.headers.get("Location") if error.code in REDIRECT_CODES else None
                if not location:
                    raise
                location = urljoin(url, location)
                if urlparse(location).netloc not in allowed:
                    raise DownloadBlocked(f"{HOSTED_ELSEWHERE} {urlparse(location).netloc}: {location}") from None
                url = location
                continue
            if not data.startswith(b"%PDF"):
                raise DownloadBlocked(f"{url} did not return a PDF")
            return data
        raise DownloadBlocked(f"{url}: too many redirects")


def item_from_record(record: OaiRecord, library: str, journals: dict[str, str],
                     sets: dict[str, str]) -> CatalogItem | None:
    """The catalog item of an OJS article record, or None for records outside ``journals``."""
    section_spec = next((s for s in record.sets if s.split(":", 1)[0] in journals), None)
    if section_spec is None:
        return None
    journal_spec = section_spec.split(":", 1)[0]
    article_id = record.identifier.rsplit("/", 1)[-1]
    dc = record.dc
    section = sets.get(section_spec, section_spec.split(":", 1)[-1])

    citation = next((s for s in dc.get("source", []) if ";" in s), None)
    volume = issue = pages = None
    year = None
    if citation:
        parts = [part.strip() for part in citation.split(";")]
        number = parts[1] if len(parts) > 1 else ""
        volume = _first(r"\bVol\.?\s*([^\s(:;]+)", number)
        issue = _first(r"\bNo\.?\s*([^\s(:;]+)", number)
        found_year = _first(r"\((\d{4})\)", number)
        # Older OJS cite an issue as "AntropoWebzin 1/2013".
        if found_year is None and (match := re.search(r"\b(\d{1,2})/((?:18|19|20)\d\d)\b", number)):
            issue = issue or match.group(1)
            found_year = match.group(2)
        year = int(found_year) if found_year else None
        pages = parts[2] if len(parts) > 2 and parts[2] else None

    details = dict(dc)
    details["section"] = [section]
    details["issn"] = [s for s in dc.get("source", []) if _ISSN.match(s)]
    details["doi"] = [i for i in dc.get("identifier", []) if i.startswith("10.")]
    details = {key: values for key, values in details.items() if values}
    landing = next((i for i in dc.get("identifier", []) if i.startswith("http")), None)

    return CatalogItem(
        library=library,
        item_id=f"{journal_spec}-{article_id}",
        record_id=record.identifier,
        title=(dc.get("title") or [None])[0],
        item_type="other" if OTHER_SECTION.search(section) else "article",
        journal_id=journal_spec,
        journal_title=journals[journal_spec],
        volume=volume,
        issue=issue,
        year=year,
        date=(dc.get("date") or [None])[0],
        pages=pages,
        authors=list(dict.fromkeys(dc.get("creator", []))),
        languages=dc.get("language", []),
        rights=dc.get("rights", []),
        pdf_urls=[url.replace("/article/view/", "/article/download/") for url in dc.get("relation", [])
                  if _GALLEY.search(url)],
        landing_url=landing,
        record=details,
    )


def _first(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1) if match else None

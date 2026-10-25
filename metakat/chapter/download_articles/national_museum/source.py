"""Periodicals of the National Museum in Prague, published at publikace.nm.cz.

The site lists the periodicals at ``/periodicke-publikace``; a periodical's page links all its issues
(``/periodicke-publikace/<code>/<volume>-<issue>``); an issue page names its year, volume and issue
("2009/40/1") and ISSNs, and lists its articles with their authors; an article page links the PDF
(``/file/<hash>/<id>/<name>.pdf``). The catalog is built from the issue pages, so the article page is
only read when the article is downloaded.
"""
from __future__ import annotations

import hashlib
import html
import logging
import re
from urllib.parse import urljoin

from metakat.chapter.download_articles.common.http import http_get
from metakat.chapter.download_articles.common.models import CatalogItem
from metakat.chapter.download_articles.common.source import NOT_ARTICLE, Download, DownloadBlocked, Source

logger = logging.getLogger(__name__)

SITE = "https://publikace.nm.cz"
NO_PDF = "links no PDF"
# Listed among the periodicals but published elsewhere.
ELSEWHERE = frozenset({"european-journal-of-taxonomy"})

_PERIODICAL = re.compile(r'<a href="/periodicke-publikace/([a-z0-9-]+)" title="([^"]+)"')
_ISSUE = re.compile(r'href="(/periodicke-publikace/[a-z0-9-]+/\d[^"/]*)"')
_ISSUE_HEADER = re.compile(r'class="pbl__hdr">\s*(\d{4})\s*/\s*([^/<]+?)\s*/\s*([^<]+?)\s*</h3>')
_ISSN = re.compile(r"\b(\d{4}-\d{3}[\dX])\b")
_FOUND = re.compile(r"Nalezeno článků: (\d+)")
_ARTICLE = re.compile(r'<a class="articleHdr__link"\s+href="([^"]+)"\s+title="([^"]*)".*?'
                      r'<div class="articleInfo">(.*?)</div>', re.S)
_PDF = re.compile(r'href="(/file/[0-9a-f]+/\d+/[^"]+\.pdf)"', re.I)


class NationalMuseumSource(Source):
    """Every article of the National Museum's periodicals, from the issue pages of publikace.nm.cz."""

    name = "national_museum"

    def build_catalog(self) -> list[CatalogItem]:
        periodicals = dict(_PERIODICAL.findall(_page(f"{SITE}/periodicke-publikace")))
        items: dict[str, CatalogItem] = {}
        for slug, title in sorted(periodicals.items()):
            if slug in ELSEWHERE:
                continue
            issues = list(dict.fromkeys(_ISSUE.findall(_page(f"{SITE}/periodicke-publikace/{slug}"))))
            count = 0
            for issue_path in issues:
                for item in items_of_issue(_page(SITE + issue_path), issue_path, html.unescape(title), self.name):
                    items.setdefault(item.item_id, item)
                    count += 1
            logger.info(f"{html.unescape(title)}: {len(issues)} issues, {count} articles")
        return list(items.values())

    def is_available(self, item: CatalogItem) -> bool:
        return bool(item.landing_url) and not (item.title and NOT_ARTICLE.match(item.title))

    def starts_wall(self, reason: str) -> bool:
        # Older volumes are listed with abstracts only; newer ones still have their files.
        return NO_PDF not in reason

    def download(self, item: CatalogItem) -> Download:
        page = _page(item.landing_url)
        pdf = _PDF.search(page)
        if pdf is None:
            raise DownloadBlocked(f"{item.landing_url} {NO_PDF}")
        url = urljoin(SITE, pdf.group(1))
        return Download(self.download_pdf(url), url)


def items_of_issue(page: str, issue_path: str, journal_title: str, library: str) -> list[CatalogItem]:
    """The articles an issue page lists, with the issue's year, volume and ISSNs."""
    header = _ISSUE_HEADER.search(page)
    if header is None:
        logger.warning(f"{issue_path}: no year/volume/issue header")
        return []
    year, volume, issue = header.groups()
    perex = page[header.end():page.find("</div>", header.end())]
    issns = sorted(set(_ISSN.findall(perex)))
    code = issue_path.split("/")[2]
    articles = _ARTICLE.findall(page)
    found = _FOUND.search(page)
    if found and int(found.group(1)) != len(articles):
        logger.warning(f"{issue_path}: {found.group(1)} articles found, {len(articles)} listed")
    items = []
    for path, title, authors in articles:
        title = " ".join(html.unescape(title).split())
        items.append(CatalogItem(
            library=library,
            item_id=f"{code}_{issue_path.rsplit('/', 1)[1]}_{hashlib.sha1(path.encode()).hexdigest()[:10]}",
            record_id=path,
            title=title or None,
            item_type="article",
            journal_id=code,
            journal_title=journal_title,
            volume=volume,
            issue=issue,
            year=int(year),
            authors=[a.strip() for a in html.unescape(re.sub(r"<[^>]+>", "", authors)).split(",") if a.strip()],
            landing_url=SITE + path,
            record={"issn": issns, "issue_url": [SITE + issue_path]},
        ))
    return items


def _page(url: str) -> str:
    return http_get(url).decode("utf-8", "replace")

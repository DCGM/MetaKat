from __future__ import annotations

import logging
import re
import urllib.error
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pymupdf

from metakat.chapter.download_articles.common.http import http_get
from metakat.chapter.download_articles.common.models import CatalogItem
from metakat.chapter.download_articles.common.oai import OaiRecord, iter_records
from metakat.chapter.download_articles.common.ojs import HOSTED_ELSEWHERE, REDIRECT_CODES
from metakat.chapter.download_articles.common.source import Download, Source
from metakat.chapter.download_articles.common.store import ArticleStore

logger = logging.getLogger(__name__)

OAI_URL = "https://digilib.phil.muni.cz/oai/request"
HOST = "digilib.phil.muni.cz"
# The Faculty of Arts journals on journals.phil.muni.cz (OJS) keep their files in this library.
OJS_LIBRARY = "journals.muni.cz"
_BITSTREAM = re.compile(r"/bitstream/handle/11222\.digilib/(\d+)/")

# Rights statements of records whose full text is not served.
UNAVAILABLE_RIGHTS = {"embargoed access", "fulltext is not accessible"}


class MuniDigilibSource(Source):
    """Digital Library of the Faculty of Arts, Masaryk University (digilib.phil.muni.cz).

    The OAI-PMH interface publishes oai_dc only, and only part of the library (20 of its 54
    journals in 2026-10). An article record links its journal, never a volume or issue, so only
    the year is known. The Faculty of Arts journals on OJS (journals.phil.muni.cz) that keep their
    files here are taken from the catalog of ``journals.muni.cz``, which recorded them as hosted
    here; a journal the OAI-PMH interface already has is left to it. Every file is behind a
    Cloudflare Turnstile human check, hence downloads are done by hand from ``selection.html``;
    the OJS links of the picked articles are first resolved to the file here (``resolve_links``).
    """

    name = "digilib.phil.muni.cz"
    # OAI-PMH record types, then the OJS ones.
    type_preference = ("Article", "article", "Anniversary Article Obituary", "Editorial", "Reviews", "Chapter",
                       "News", "other")
    manual_download = True

    def build_catalog(self) -> list[CatalogItem]:
        items = catalog_from_records(iter_records(OAI_URL))
        return items + self.ojs_items({_title_key(item.journal_title) for item in items})

    def ojs_items(self, known_journals: set[str]) -> list[CatalogItem]:
        """The articles of the OJS journals recorded as hosted here, the file's address first where known."""
        store = ArticleStore(self.root, OJS_LIBRARY)
        hosted = {item_id: row for item_id, row in store.unavailable().items()
                  if row["reason"].startswith(f"{HOSTED_ELSEWHERE} {HOST}")}
        journals = {row["journal_id"] for row in hosted.values()}
        items = []
        for item in store.read_catalog():
            if item.journal_id not in journals or _title_key(item.journal_title) in known_journals:
                continue
            item = item.model_copy(update={"library": self.name})
            if item.item_id in hosted:
                item.pdf_urls = [hosted[item.item_id]["reason"].split(": ", 1)[1], *item.pdf_urls]
            items.append(item)
        logger.info(f"{len(items)} articles of {len({i.journal_id for i in items})} journals on OJS")
        return items

    def resolve_links(self, items: list[CatalogItem]) -> bool:
        changed = False
        for item in items:
            if not item.pdf_urls or urlparse(item.pdf_urls[0]).netloc == HOST:
                continue
            try:
                http_get(item.pdf_urls[0], follow_redirects=False)
                continue
            except urllib.error.HTTPError as error:
                location = error.headers.get("Location") if error.code in REDIRECT_CODES else None
            except OSError as error:
                logger.warning(f"{item.item_id}: {error}")
                continue
            if location and urlparse(urljoin(item.pdf_urls[0], location)).netloc == HOST:
                item.pdf_urls = [urljoin(item.pdf_urls[0], location), *item.pdf_urls]
                changed = True
            else:
                logger.warning(f"{item.item_id}: {item.pdf_urls[0]} does not lead to {HOST}")
        return changed

    def is_available(self, item: CatalogItem) -> bool:
        return bool(item.pdf_urls) and not UNAVAILABLE_RIGHTS & {r.lower() for r in item.rights}

    def local_pdf(self, item: CatalogItem, pdf_dir: Path | None) -> Download | None:
        found = super().local_pdf(item, pdf_dir)
        if found is not None or pdf_dir is None:
            return found
        # The library serves .../bitstream/handle/11222.digilib/<handle>/<name>.pdf as <handle>.pdf.
        for url in item.pdf_urls:
            if (handle := _BITSTREAM.search(url)) and (path := pdf_dir / f"{handle.group(1)}.pdf").exists():
                return Download(path.read_bytes(), url)
        return None

    def title_page_index(self, pdf_bytes: bytes, url: str | None = None) -> int:
        """The library puts a cover sheet before the article; its original scans (-source.pdf) have none."""
        if url and url.lower().endswith("-source.pdf"):
            return 0
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
            return 1 if document.page_count > 1 else 0


def catalog_from_records(records: Iterable[OaiRecord]) -> list[CatalogItem]:
    records = list(records)
    by_handle = {handle: record for record in records if (handle := _handle(record))}
    # Journals, series and books are the records without a file of their own.
    containers = {handle for handle, record in by_handle.items() if not _pdf_urls(record)}
    # A review links the reviewed book besides its journal; the journal is the container most records link.
    links = Counter(handle for record in records for handle in set(_related_handles(record)) if handle in containers)

    items = []
    for record in records:
        pdf_urls = _pdf_urls(record)
        handle = _handle(record)
        if not pdf_urls or handle is None:
            continue
        journal_handle = max((h for h in _related_handles(record) if h in containers), key=links.__getitem__,
                             default=None)
        journal = by_handle.get(journal_handle)
        dc = record.dc
        date = (dc.get("date") or [None])[0]
        year = re.search(r"\b(1[5-9]\d\d|20\d\d)\b", date or "")
        items.append(CatalogItem(
            library=MuniDigilibSource.name,
            item_id=handle,
            record_id=record.identifier,
            title=(dc.get("title") or [None])[0],
            item_type=(dc.get("type") or [None])[0],
            journal_id=journal_handle,
            journal_title=(journal.dc.get("title") or [None])[0] if journal else None,
            year=int(year.group(1)) if year else None,
            date=date,
            authors=dc.get("creator", []),
            languages=dc.get("language", []),
            rights=dc.get("rights", []),
            pdf_urls=pdf_urls,
            landing_url=f"https://hdl.handle.net/11222.digilib/{handle}",
            record=dc,
        ))
    return items


def _title_key(title: str | None) -> str:
    return re.sub(r"\W+", " ", (title or "").casefold()).strip()


def _handle(record: OaiRecord) -> str | None:
    for identifier in record.dc.get("identifier", []):
        if "hdl.handle.net/" in identifier:
            return identifier.rstrip("/").rsplit("/", 1)[-1]
    return None


def _related_handles(record: OaiRecord) -> list[str]:
    return [value.rstrip("/").rsplit("/", 1)[-1] for value in record.dc.get("relation", []) if "/handle/" in value]


def _pdf_urls(record: OaiRecord) -> list[str]:
    """The record's files, the original scan first, then the published PDF, then the enhanced scan."""
    urls = [url.strip() for value in record.dc.get("relation", []) for url in value.split(";")
            if url.strip().lower().endswith(".pdf")]

    def rank(url: str) -> int:
        if url.endswith("-source.pdf"):
            return 0
        if url.endswith("-source-enhanced.pdf"):
            return 2
        return 1

    return sorted(dict.fromkeys(urls), key=rank)

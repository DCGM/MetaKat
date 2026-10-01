from __future__ import annotations

import re
from collections.abc import Iterable

from metakat.chapter.download_articles.common.models import CatalogItem
from metakat.chapter.download_articles.common.oai import OaiRecord, iter_records
from metakat.chapter.download_articles.common.source import Source

OAI_URL = "https://digilib.phil.muni.cz/oai/request"

# Rights statements of records whose full text is not served.
UNAVAILABLE_RIGHTS = {"embargoed access", "fulltext is not accessible"}


class MuniDigilibSource(Source):
    """Digital Library of the Faculty of Arts, Masaryk University (digilib.phil.muni.cz).

    The OAI-PMH interface publishes oai_dc only, and only part of the library (11 of its 54
    journals in 2026-10). An article record links its journal, never a volume or issue, so only
    the year is known. Every file is behind a Cloudflare Turnstile human check, hence downloads
    are done by hand from ``selection.html``.
    """

    name = "digilib.phil.muni.cz"
    type_preference = ("Article", "Anniversary Article Obituary", "Editorial", "Reviews", "Chapter", "News")
    manual_download = True

    def build_catalog(self) -> list[CatalogItem]:
        return catalog_from_records(iter_records(OAI_URL))

    def is_available(self, item: CatalogItem) -> bool:
        return bool(item.pdf_urls) and not UNAVAILABLE_RIGHTS & {r.lower() for r in item.rights}


def catalog_from_records(records: Iterable[OaiRecord]) -> list[CatalogItem]:
    records = list(records)
    by_handle = {handle: record for record in records if (handle := _handle(record))}
    # Journals, series and books are the records without a file of their own.
    containers = {handle for handle, record in by_handle.items() if not _pdf_urls(record)}

    items = []
    for record in records:
        pdf_urls = _pdf_urls(record)
        handle = _handle(record)
        if not pdf_urls or handle is None:
            continue
        journal_handle = next((h for h in _related_handles(record) if h in containers), None)
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

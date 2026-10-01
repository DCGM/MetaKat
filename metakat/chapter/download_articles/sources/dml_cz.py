from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from collections.abc import Iterable

import pymupdf

from metakat.chapter.download_articles.models import CatalogItem
from metakat.chapter.download_articles.oai import OaiRecord, iter_records, list_sets
from metakat.chapter.download_articles.sources.base import Source

logger = logging.getLogger(__name__)

OAI_URL = "https://dml.cz/dspace-oai/request"
METADATA_PREFIX = "eudml-article2"

_J = "{http://jats.nlm.nih.gov}"
_XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
_XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"


class DmlCzSource(Source):
    """Czech Digital Mathematics Library (dml.cz).

    Articles are harvested in the EuDML JATS format (``eudml-article2``), which carries the journal,
    volume, issue, pages and a link to the article PDF. Harvesting without a set returns only part
    of the journals, so every set is harvested on its own. Proceedings and book collections answer
    the article format with errors and are not harvested.
    """

    name = "dml.cz"
    # Research articles first; "other" also holds volume title pages, "contents" is never picked.
    type_preference = ("math", "physics", "astronomy", "chemistry", "informatics", "history", "politics",
                       "editorial", "review", "news", "other")

    def title_page_index(self, pdf_bytes: bytes) -> int:
        """DML-CZ stamps its PDFs with a cover sheet (citation, persistent URL, terms of use)."""
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
            text = document[0].get_text() if document.page_count > 1 else ""
        return 1 if "Terms of use" in text and "dml.cz" in text else 0

    def build_catalog(self) -> list[CatalogItem]:
        items: dict[str, CatalogItem] = {}
        for spec, name in list_sets(OAI_URL):
            harvested = catalog_from_records(iter_records(OAI_URL, METADATA_PREFIX, set_spec=spec))
            logger.info(f"Set {name} ({spec}): {len(harvested)} articles")
            for item in harvested:
                items.setdefault(item.item_id, item)
        return list(items.values())


def catalog_from_records(records: Iterable[OaiRecord]) -> list[CatalogItem]:
    items = []
    for record in records:
        article = record.metadata
        if article is None or article.tag != f"{_J}article":
            continue
        item = _item(record, article)
        if item is not None:
            items.append(item)
    return items


def _item(record: OaiRecord, article: ET.Element) -> CatalogItem | None:
    journal = article.find(f"{_J}front/{_J}journal-meta")
    meta = article.find(f"{_J}front/{_J}article-meta")
    if meta is None:
        return None

    def text(parent: ET.Element | None, path: str) -> str | None:
        value = parent.findtext(path) if parent is not None else None
        return value.strip() if value and value.strip() else None

    article_id = text(meta, f"{_J}article-id[@pub-id-type='dmlcz-id']")
    if article_id is None:
        return None

    pdf_urls = [link.get(_XLINK_HREF) for link in meta.iterfind(f"{_J}self-uri[@content-type='application/pdf']")]
    pdf_urls += [link.get(_XLINK_HREF) for link in meta.iterfind(f"{_J}ext-link")
                 if link.get("ext-link-type") == "eudml-fulltext:application/pdf"]
    pdf_urls = [url.replace("http://", "https://", 1) for url in dict.fromkeys(u for u in pdf_urls if u)]

    authors = []
    for name in meta.iterfind(f"{_J}contrib-group/{_J}contrib[@contrib-type='author']/{_J}name"):
        surname, given = text(name, f"{_J}surname"), text(name, f"{_J}given-names")
        authors.append(", ".join(part for part in (surname, given) if part))

    year = text(meta, f"{_J}pub-date/{_J}year")
    first_page, last_page = text(meta, f"{_J}fpage"), text(meta, f"{_J}lpage")
    pages = "-".join(p for p in (first_page, last_page) if p) or None

    def flat(element: ET.Element) -> str:
        return " ".join("".join(element.itertext()).split())

    details = {
        "titles": [flat(t) for t in meta.iterfind(f"{_J}title-group/{_J}article-title")],
        "trans_titles": [flat(t) for t in meta.iterfind(f"{_J}title-group/{_J}trans-title-group/{_J}trans-title")],
        "keywords": [flat(k) for group in meta.iterfind(f"{_J}kwd-group")
                     if group.get("kwd-group-type") != "msc" for k in group],
        "msc": [flat(k) for k in meta.iterfind(f"{_J}kwd-group[@kwd-group-type='msc']/{_J}kwd")],
        "abstracts": [flat(a) for a in meta.iterfind(f"{_J}abstract")],
        "volume_id": [v.text for v in meta.iterfind(f"{_J}volume-id") if v.text],
        "issue_id": [v.text for v in meta.iterfind(f"{_J}issue-id") if v.text],
        "issue_title": [v.text for v in meta.iterfind(f"{_J}issue-title") if v.text],
        "doi": [v.text for v in meta.iterfind(f"{_J}article-id[@pub-id-type='doi']") if v.text],
        "sets": record.sets,
    }
    details = {key: values for key, values in details.items() if values}

    return CatalogItem(
        library=DmlCzSource.name,
        item_id=article_id,
        record_id=record.identifier,
        title=(details.get("titles") or [None])[0],
        item_type=text(meta, f"{_J}article-categories/{_J}subj-group[@subj-group-type='dmlcz-article-type']/{_J}subject"),
        journal_id=text(journal, f"{_J}journal-id[@journal-id-type='dmlcz-id']"),
        journal_title=text(journal, f"{_J}journal-title-group/{_J}journal-title"),
        volume=text(meta, f"{_J}volume"),
        issue=text(meta, f"{_J}issue"),
        year=int(year) if year and year.isdigit() else None,
        date=year,
        pages=pages,
        authors=authors,
        languages=[article.get(_XML_LANG)] if article.get(_XML_LANG) else [],
        pdf_urls=pdf_urls,
        landing_url=f"https://dml.cz/handle/10338.dmlcz/{article_id}",
        record=details,
    )

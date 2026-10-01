"""Articles of periodicals in a Kramerius 7 digital library (KNAV, or any library in the Czech Digital Library).

Kramerius catalogues an article as its own record only where the library described the periodical at
article level: as ``model:article`` (KNAV, MZK, ČBVK) or as an ``internalpart`` of an issue (NKP). Kramerius
does not tell newspapers from journals, so a periodical with 40 or more issues per volume, or catalogued
as a newspaper, counts as a newspaper and is left out.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.parse

from metakat.chapter.download_articles.common.http import http_get
from metakat.chapter.download_articles.common.models import CatalogItem
from metakat.chapter.download_articles.common.source import Download, DownloadBlocked, Source
from metakat.chapter.download_articles.common.store import ArticleStore

logger = logging.getLogger(__name__)

PAGE_ROWS = 1000
# Periodicals listed per facet request when counting their volumes, issues and pages.
ROOT_BATCH = 25
NEWSPAPER_ISSUES_PER_VOLUME = 40
NEWSPAPER_GENRES = {"newspaper", "noviny", "regionální noviny", "regionální deníky", "deníky"}

ARTICLE_FIELDS = ["pid", "own_pid_path", "own_model_path", "own_parent.pid", "root.title", "title.search",
                  "titles.search", "authors.search", "languages.facet", "keywords.search", "date.str",
                  "date_range_start.year", "accessibility", "licenses.facet", "ds.img_full.mime", "count_page"]
PARENT_FIELDS = ["pid", "model", "part.number.str", "date.str", "title.search", "date_range_start.year"]

# Contents pages, indexes and similar parts catalogued as articles, never picked.
NOT_ARTICLE = re.compile(
    r"^\W*(obsah|contents?|table of contents|inhalt|sommaire|содержание|tiráž|impressum|rejstřík|index|"
    r"errata|oprava|obálka|cover|reklam\w*|inzer\w*|inserat\w*|anzeigen?|advertisements?)\b", re.IGNORECASE)


class Kramerius:
    """The client API of one Kramerius 7 instance; ``fq`` restricts every search (e.g. to one library)."""

    def __init__(self, api: str, fq: str | None = None):
        self.api = api
        self.fq = fq

    def search(self, query: str, fields: list[str], rows: int, start: int = 0, sort: str | None = None,
               filtered: bool = True) -> list[dict]:
        params = {"q": query, "fl": ",".join(fields), "rows": rows, "start": start}
        if sort:
            params["sort"] = sort
        if filtered and self.fq:
            params["fq"] = self.fq
        return self._get(params)["response"]["docs"]

    def search_all(self, query: str, fields: list[str], filtered: bool = True) -> list[dict]:
        docs, start = [], 0
        while True:
            page = self.search(query, fields, PAGE_ROWS, start, sort="pid asc", filtered=filtered)
            docs += page
            if len(page) < PAGE_ROWS:
                return docs
            start += PAGE_ROWS

    def facet(self, query: str, field: str, filtered: bool = True) -> dict[str, int]:
        params = {"q": query, "rows": 0, "facet": "true", "facet.field": field, "facet.limit": -1,
                  "facet.mincount": 1}
        if filtered and self.fq:
            params["fq"] = self.fq
        values = self._get(params)["facet_counts"]["facet_fields"][field]
        return dict(zip(values[::2], values[1::2]))

    def structure(self, pid: str) -> dict:
        return json.loads(http_get(f"{self.api}/items/{pid}/info/structure"))

    def mods(self, pid: str) -> str:
        return http_get(f"{self.api}/items/{pid}/metadata/mods").decode("utf-8", "replace")

    def image_url(self, pid: str) -> str:
        return f"{self.api}/items/{pid}/image"

    def _get(self, params: dict) -> dict:
        return json.loads(http_get(f"{self.api}/search?{urllib.parse.urlencode(params)}"))


class KrameriusSource(Source):
    """Every article of every periodical in a Kramerius library, except newspapers.

    Born-digital articles are one PDF each; scanned articles have no file of their own, and their
    title page is the image of the first page they are on.
    """

    api: str
    # Restricts every search, e.g. to one library of the Czech Digital Library.
    fq: str | None = None
    # The records that are articles.
    article_query = "model:article AND own_model_path:periodical*"
    # Item pages for people: landing_url + pid.
    landing_url: str
    skip_newspapers = True
    # Libraries sampled before this one (folders under the output root). Libraries share digitised
    # periodicals under the same pids; articles already in their catalogs are left to them.
    exclude_libraries: tuple[str, ...] = ()

    def __init__(self):
        self.kramerius = Kramerius(self.api, self.fq)

    def build_catalog(self) -> list[CatalogItem]:
        excluded = self.excluded_ids()
        counts = self.kramerius.facet(self.article_query, "root.pid")
        roots = periodical_info(self.kramerius, list(counts))
        items, shared = [], 0
        for number, (root, info) in enumerate(roots.items(), 1):
            if self.skip_newspapers and info["newspaper_like"]:
                logger.info(f"Periodical {number}/{len(roots)} {root} {info['title']}: newspaper "
                            f"({info['issues_per_volume']} issues per volume, {counts[root]} articles), left out")
                continue
            parents = {doc["pid"]: doc for doc in self.kramerius.search_all(
                f'root.pid:"{root}" AND model:(periodicalvolume OR periodicalitem OR supplement)', PARENT_FIELDS,
                filtered=False)}
            articles = self.kramerius.search_all(f'root.pid:"{root}" AND ({self.article_query})', ARTICLE_FIELDS)
            own = [doc for doc in articles if doc["pid"].removeprefix("uuid:") not in excluded]
            shared += len(articles) - len(own)
            items += [self.item_from_docs(doc, parents, info) for doc in own]
            logger.info(f"Periodical {number}/{len(roots)} {root} {info['title']}: {len(own)} articles"
                        + (f" ({len(articles) - len(own)} left to {', '.join(self.exclude_libraries)})"
                           if len(own) < len(articles) else ""))
        logger.info(f"{shared} articles left to {', '.join(self.exclude_libraries)}")
        return items

    def excluded_ids(self) -> set[str]:
        excluded = set()
        for library in self.exclude_libraries:
            store = ArticleStore(self.root, library)
            if not store.catalog_path.exists():
                raise FileNotFoundError(f"Catalog {library} first: {store.catalog_path} is missing")
            excluded |= {item.item_id for item in store.read_catalog()}
        return excluded

    def item_from_docs(self, doc: dict, parents: dict[str, dict], periodical: dict | None = None) -> CatalogItem:
        return item_from_docs(doc, parents, self.name, self.api, self.landing_url, periodical)

    def is_available(self, item: CatalogItem) -> bool:
        if item.title and NOT_ARTICLE.match(item.title):
            return False
        # Anonymous users get public items only.
        return item.record.get("accessibility") != ["private"]

    def download(self, item: CatalogItem) -> Download:
        if item.pdf_urls:
            return super().download(item)
        page = first_page_pid(self.kramerius, item.record_id, item.record.get("parent", [None])[0])
        url = self.kramerius.image_url(page)
        data = http_get(url)
        extension = image_extension(data)
        if extension is None:
            raise DownloadBlocked(f"{url} did not return an image")
        return Download(data, url, kind=extension)


def periodical_info(kramerius: Kramerius, roots: list[str]) -> dict[str, dict]:
    """Title, genres and the volume/issue/page counts of each periodical, to tell newspapers apart."""
    info = {root: {"title": None, "genres": [], "volumes": 0, "issues": 0, "pages": 0} for root in roots}
    for start in range(0, len(roots), ROOT_BATCH):
        batch = roots[start:start + ROOT_BATCH]
        in_batch = "root.pid:(" + " OR ".join(f'"{root}"' for root in batch) + ")"
        for model, key in (("periodicalvolume", "volumes"), ("periodicalitem", "issues"), ("page", "pages")):
            for root, count in kramerius.facet(f"model:{model} AND {in_batch}", "root.pid", filtered=False).items():
                if root in info:
                    info[root][key] = count
        query = "pid:(" + " OR ".join(f'"{root}"' for root in batch) + ")"
        for doc in kramerius.search(query, ["pid", "root.title", "genres.facet"], len(batch), filtered=False):
            info[doc["pid"]]["title"] = doc.get("root.title")
            info[doc["pid"]]["genres"] = doc.get("genres.facet", [])
    for root, entry in info.items():
        entry["issues_per_volume"] = round(entry["issues"] / entry["volumes"], 1) if entry["volumes"] else None
        entry["pages_per_issue"] = round(entry["pages"] / entry["issues"], 1) if entry["issues"] else None
        entry["newspaper_like"] = is_newspaper(entry)
    return info


def is_newspaper(periodical: dict) -> bool:
    """Dailies and weeklies: many issues per volume, or catalogued as a newspaper."""
    if any(genre.lower() in NEWSPAPER_GENRES for genre in periodical.get("genres") or []):
        return True
    return (periodical.get("issues_per_volume") or 0) >= NEWSPAPER_ISSUES_PER_VOLUME


def item_from_docs(doc: dict, parents: dict[str, dict], library: str, api: str, landing_url: str,
                   periodical: dict | None = None) -> CatalogItem:
    """A catalog item from a Kramerius article document and the volume/issue documents above it."""
    pid = doc["pid"]
    path = doc.get("own_pid_path", "").split("/")
    models = doc.get("own_model_path", "").split("/")
    volume = issue = None
    year = doc.get("date_range_start.year")
    date = doc.get("date.str")
    for ancestor, model in zip(path, models):
        parent = parents.get(ancestor, {})
        number = parent.get("part.number.str") or parent.get("title.search")
        if model == "periodicalvolume" and volume is None:
            volume = parent.get("part.number.str")
        elif model in ("periodicalitem", "supplement") and issue is None:
            issue = number
        year = year or parent.get("date_range_start.year")
        if model == "periodicalitem":
            date = date or parent.get("date.str")

    periodical = periodical or {}
    record = {
        "titles": doc.get("titles.search", []),
        "keywords": doc.get("keywords.search", []),
        "own_model_path": [doc.get("own_model_path", "")],
        "parent": [doc["own_parent.pid"]] if doc.get("own_parent.pid") else [],
        "accessibility": [doc["accessibility"]] if doc.get("accessibility") else [],
        "licenses": doc.get("licenses.facet", []),
        "mime": [doc["ds.img_full.mime"]] if doc.get("ds.img_full.mime") else [],
        "count_page": [str(doc["count_page"])] if doc.get("count_page") else [],
        "periodical_genres": periodical.get("genres") or [],
        "issues_per_volume": [str(periodical["issues_per_volume"])] if periodical.get("issues_per_volume") else [],
    }
    is_pdf = doc.get("ds.img_full.mime") == "application/pdf"
    return CatalogItem(
        library=library,
        item_id=pid.removeprefix("uuid:"),
        record_id=pid,
        title=doc.get("title.search"),
        item_type="pdf" if is_pdf else "scan",
        journal_id=path[0].removeprefix("uuid:") if path and path[0] else None,
        journal_title=doc.get("root.title"),
        volume=volume,
        issue=issue,
        year=int(year) if year else None,
        date=date,
        authors=doc.get("authors.search", []),
        languages=doc.get("languages.facet", []),
        rights=doc.get("licenses.facet", []),
        pdf_urls=[f"{api}/items/{pid}/image"] if is_pdf else [],
        landing_url=f"{landing_url}{pid}",
        record={key: values for key, values in record.items() if values},
    )


def first_page_pid(kramerius: Kramerius, article_pid: str, parent_pid: str | None = None) -> str:
    """The pid of the first page an article is on.

    By the article's links to its pages, ordered as the pages are in their issue; for articles without
    links (some ČBVK ones), by the start page in the article's MODS among the page numbers of its issue.
    """
    children = kramerius.structure(article_pid).get("children", {})
    pages = [child["pid"] for child in children.get("foster", []) + children.get("own", [])
             if child.get("relation") in ("isOnPage", "hasPage")]
    if len(pages) == 1:
        return pages[0]
    if pages:
        docs = kramerius.search(" OR ".join(f'pid:"{page}"' for page in pages[:50]), ["pid", "rels_ext_index.sort"],
                                len(pages), filtered=False)
        order = {doc["pid"]: doc.get("rels_ext_index.sort", 10**9) for doc in docs}
        return min(pages, key=lambda page: (order.get(page, 10**9), pages.index(page)))

    start = mods_start_page(kramerius.mods(article_pid))
    if start is None or parent_pid is None:
        raise DownloadBlocked(f"{article_pid} is on no page")
    docs = kramerius.search(f'own_parent.pid:"{parent_pid}" AND model:page',
                            ["pid", "page.number", "rels_ext_index.sort"], PAGE_ROWS, filtered=False)
    matches = [doc for doc in docs if normalize_page_number(doc.get("page.number")) == start]
    if not matches:
        raise DownloadBlocked(f"{article_pid}: no page numbered {start} in {parent_pid}")
    return min(matches, key=lambda doc: doc.get("rels_ext_index.sort", 10**9))["pid"]


def mods_start_page(mods: str) -> str | None:
    match = re.search(r"<(?:mods:)?start>(.*?)</(?:mods:)?start>", mods, flags=re.S)
    return normalize_page_number(match.group(1)) if match else None


def normalize_page_number(number) -> str | None:
    if number is None:
        return None
    number = re.sub(r"[\[\]\s.]", "", str(number)).lower()
    return number or None


def image_extension(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8"):
        return "jpg"
    if data.startswith(b"\x89PNG"):
        return "png"
    if data[4:12] in (b"jP  \r\n\x87\n",) or data.startswith(b"\xff\x4f\xff\x51"):
        return "jp2"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "tif"
    return None

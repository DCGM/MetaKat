from __future__ import annotations

import json
import logging
import os
import re
import urllib.parse
from pathlib import Path

from metakat.chapter.download_articles.common.http import http_get
from metakat.chapter.download_articles.common.models import CatalogItem
from metakat.chapter.download_articles.common.source import Download, DownloadBlocked, Source

logger = logging.getLogger(__name__)

API = "https://kramerius.lib.cas.cz/search/api/client/v7.0"
PAGE_ROWS = 1000

# Earlier downloads of born-digital articles (title.automatic_pick): PDFs in <dir>/<issue>/<article>.pdf
# and the logs of the download runs, which tell the refused (403) and PDF-less (404) articles.
AUTOMATIC_PICK_DIRS = (
    Path("/mnt/kolosus/data/smart_digiline/articles/title.automatic_pick/knav/public"),
    Path("/mnt/kolosus/data/smart_digiline/articles/title.automatic_pick/knav/private"),
)
AUTOMATIC_PICK_LOGS = (
    Path.home() / "data/smart_digiline/articles/title.automatic_pick/knav/public.log",
    Path.home() / "data/smart_digiline/articles/title.automatic_pick/knav/private.log",
)

ARTICLE_FIELDS = ["pid", "own_pid_path", "own_model_path", "root.title", "title.search", "titles.search",
                  "authors.search", "languages.facet", "keywords.search", "date.str", "date_range_start.year",
                  "accessibility", "licenses.facet", "ds.img_full.mime", "count_page"]
PARENT_FIELDS = ["pid", "model", "part.number.str", "date.str", "title.search", "date_range_start.year"]

# Contents pages, indexes and similar parts catalogued as articles, never picked.
_NOT_ARTICLE = re.compile(
    r"^\W*(obsah|contents?|table of contents|inhalt|sommaire|содержание|tiráž|impressum|rejstřík|index|"
    r"errata|oprava|obálka|cover)\b", re.IGNORECASE)


class KnavSource(Source):
    """Digital Library of the Czech Academy of Sciences (Kramerius of the Library of the CAS, KNAV).

    Every article of every periodical is catalogued from the Kramerius search index, with the
    numbers of its volume and issue. Born-digital articles are one PDF each; scanned articles have no
    file of their own, and their title page is the image of the first page they are on.

    PDFs downloaded earlier into ``title.automatic_pick/knav`` are reused, and the logs of those
    downloads tell which articles were refused, so neither costs a request. KNAV refuses the newest
    issues of many journals and whole journals under licence, so a refusal is treated as a moving
    wall: the journal's later years are only taken from earlier downloads.
    """

    name = "knav"
    moving_wall = True

    def __init__(self, pick_dirs=AUTOMATIC_PICK_DIRS, pick_logs=AUTOMATIC_PICK_LOGS):
        self.pick_dirs = pick_dirs
        self.pick_logs = pick_logs

    def build_catalog(self) -> list[CatalogItem]:
        local = index_local_pdfs(self.pick_dirs)
        errors = read_download_errors(self.pick_logs)
        logger.info(f"Earlier downloads: {len(local)} PDFs, {len(errors)} errors")

        roots = _facet("model:article AND root.model:periodical", "root.pid")
        items = []
        for number, root in enumerate(roots, 1):
            parents = {doc["pid"]: doc for doc in _search_all(
                f'root.pid:"{root}" AND model:(periodicalvolume OR periodicalitem OR supplement)', PARENT_FIELDS)}
            articles = _search_all(f'root.pid:"{root}" AND model:article', ARTICLE_FIELDS)
            items += [item_from_docs(doc, parents, local, errors) for doc in articles]
            logger.info(f"Periodical {number}/{len(roots)} {root}: {len(articles)} articles")
        return items

    def is_available(self, item: CatalogItem) -> bool:
        if item.title and _NOT_ARTICLE.match(item.title):
            return False
        if item.record.get("previous_download_error") == ["403"]:
            return False
        if item.pdf_urls:
            return True
        # Scanned article: its page images are served to the public only.
        return item.record.get("accessibility") != ["private"]

    def is_cheap(self, item: CatalogItem) -> bool:
        return bool(item.record.get("local_pdf"))

    def local_pdf(self, item: CatalogItem, pdf_dir: Path | None) -> Download | None:
        for path in item.record.get("local_pdf", []):
            if os.path.exists(path):
                return Download(Path(path).read_bytes(), item.pdf_urls[0] if item.pdf_urls else None)
        return super().local_pdf(item, pdf_dir)

    def download(self, item: CatalogItem) -> Download:
        if item.pdf_urls:
            return super().download(item)
        page = first_page_pid(item.item_id)
        url = f"{API}/items/{page}/image"
        data = http_get(url)
        extension = _image_extension(data)
        if extension is None:
            raise DownloadBlocked(f"{url} did not return an image")
        return Download(data, url, kind=extension)


def item_from_docs(doc: dict, parents: dict[str, dict], local: dict[str, str], errors: dict[str, str]) -> CatalogItem:
    """A catalog item from a Kramerius article document and the volume/issue documents above it."""
    pid = doc["pid"]
    uuid = pid.removeprefix("uuid:")
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

    record = {
        "titles": doc.get("titles.search", []),
        "keywords": doc.get("keywords.search", []),
        "own_model_path": [doc.get("own_model_path", "")],
        "accessibility": [doc["accessibility"]] if doc.get("accessibility") else [],
        "licenses": doc.get("licenses.facet", []),
        "mime": [doc["ds.img_full.mime"]] if doc.get("ds.img_full.mime") else [],
        "count_page": [str(doc["count_page"])] if doc.get("count_page") else [],
        "local_pdf": [local[uuid]] if uuid in local else [],
        "previous_download_error": [errors[uuid]] if uuid in errors else [],
    }
    is_pdf = doc.get("ds.img_full.mime") == "application/pdf"
    return CatalogItem(
        library=KnavSource.name,
        item_id=uuid,
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
        pdf_urls=[f"{API}/items/{pid}/image"] if is_pdf else [],
        landing_url=f"https://kramerius.lib.cas.cz/uuid/{pid}",
        record={key: values for key, values in record.items() if values},
    )


def index_local_pdfs(directories) -> dict[str, str]:
    """Article uuid -> path of its PDF downloaded earlier (``<dir>/<issue>/<article>.pdf``)."""
    local = {}
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        for issue in os.scandir(directory):
            if issue.is_dir():
                for entry in os.scandir(issue.path):
                    if entry.name.endswith(".pdf") and entry.stat().st_size > 0:
                        local[entry.name[:-4]] = entry.path
    return local


def read_download_errors(logs) -> dict[str, str]:
    """Article uuid -> HTTP status of its failed earlier download, from the download.sh logs."""
    errors = {}
    for log in logs:
        if not os.path.exists(log):
            continue
        current = None
        with open(log, encoding="utf-8", errors="replace") as file:
            for line in file:
                if "Processing: " in line:
                    current = line.rsplit("Processing: ", 1)[1].strip()
                elif current and (match := re.search(r"returned error: (\d{3})", line)):
                    errors[current] = match.group(1)
    return errors


def first_page_pid(article_uuid: str) -> str:
    """The pid of the first page a scanned article is on, by the pages' order in their issue."""
    structure = json.loads(http_get(f"{API}/items/uuid:{article_uuid}/info/structure"))
    children = structure.get("children", {})
    pages = [child["pid"] for child in children.get("foster", []) + children.get("own", [])
             if child.get("relation") in ("isOnPage", "hasPage")]
    if not pages:
        raise DownloadBlocked(f"uuid:{article_uuid} is on no page")
    if len(pages) == 1:
        return pages[0]
    docs = _search(" OR ".join(f'pid:"{page}"' for page in pages[:50]), ["pid", "rels_ext_index.sort"], len(pages))
    order = {doc["pid"]: doc.get("rels_ext_index.sort", 10**9) for doc in docs}
    return min(pages, key=lambda page: (order.get(page, 10**9), pages.index(page)))


def _image_extension(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8"):
        return "jpg"
    if data.startswith(b"\x89PNG"):
        return "png"
    if data[4:12] in (b"jP  \r\n\x87\n",) or data.startswith(b"\xff\x4f\xff\x51"):
        return "jp2"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "tif"
    return None


def _search(query: str, fields: list[str], rows: int, start: int = 0, sort: str | None = None) -> list[dict]:
    params = {"q": query, "fl": ",".join(fields), "rows": rows, "start": start}
    if sort:
        params["sort"] = sort
    response = json.loads(http_get(f"{API}/search?{urllib.parse.urlencode(params)}"))
    return response["response"]["docs"]


def _search_all(query: str, fields: list[str]) -> list[dict]:
    docs, start = [], 0
    while True:
        page = _search(query, fields, PAGE_ROWS, start, sort="pid asc")
        docs += page
        if len(page) < PAGE_ROWS:
            return docs
        start += PAGE_ROWS


def _facet(query: str, field: str) -> list[str]:
    params = {"q": query, "rows": 0, "facet": "true", "facet.field": field, "facet.limit": -1, "facet.mincount": 1}
    response = json.loads(http_get(f"{API}/search?{urllib.parse.urlencode(params)}"))
    values = response["facet_counts"]["facet_fields"][field]
    return values[::2]

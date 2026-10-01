from __future__ import annotations

import csv
import json
import logging
import os
import re
import urllib.error
from pathlib import Path
from typing import Callable

from metakat.chapter.download_articles.common.http import http_get
from metakat.chapter.download_articles.common.kramerius import NOT_ARTICLE, PARENT_FIELDS, KrameriusSource
from metakat.chapter.download_articles.common.models import CatalogItem
from metakat.chapter.download_articles.common.source import Download, DownloadBlocked
from metakat.chapter.download_articles.common.store import ArticleStore
from metakat.chapter.download_articles.common.toc import Page, Word, alto_words, ocr_words, start_pages, toc_entries

logger = logging.getLogger(__name__)

API = "https://kramerius.lib.cas.cz/search/api/client/v7.0"

# Periodicals with volumes scanned page by page, whose articles have no records of their own, from the
# library's list of periodicals (sheet "Časopisy bez metadat článků"); their articles are found through
# the volumes' contents pages.
UNSEGMENTED_PERIODICALS = Path(__file__).with_name("unsegmented_periodicals.tsv")
VOLUME_FIELDS = ["pid", "root.title", "part.number.str", "date.str", "date_range_start.year", "accessibility",
                 "licenses.facet"]
# A contents entry of a numbered section of a longer text ("1.4 The impact ...", "3. Experience ..."),
# not an article.
SUBSECTION = re.compile(r"^\W*\d+\.(\d|\s)")
# Pages from an article's start to the next one found: shorter are news and notes, longer a gap of
# entries the contents page (or its OCR) misses.
ARTICLE_PAGES = range(3, 61)
# Licences under which KNAV refuses a volume's pages to anonymous users (out of commerce, on site only).
RESTRICTED_LICENSES = {"dnnto", "dnntt", "onsite"}
PAGE_FIELDS = ["pid", "page.number", "page.type", "own_parent.pid", "own_pid_path", "own_model_path",
               "rels_ext_index.sort", "accessibility", "root.title"]

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


class KnavSource(KrameriusSource):
    """Digital Library of the Czech Academy of Sciences (Kramerius of the Library of the CAS, KNAV).

    Every article of every periodical is catalogued from the Kramerius search index, with the
    numbers of its volume and issue. KNAV holds journals only, so no periodical is checked for
    being a newspaper.

    PDFs downloaded earlier into ``title.automatic_pick/knav`` are reused, and the logs of those
    downloads tell which articles were refused, so neither costs a request. KNAV refuses the newest
    issues of many journals and whole journals under licence, so a refusal is treated as a moving
    wall: the journal's later years are only taken from earlier downloads.

    The periodicals in ``unsegmented_periodicals`` also have volumes without article records. Each
    such volume (a year without articles in the catalog) is one item of type "volume", selected like
    any article; when it is fetched, its article is found through its contents pages: of the pages
    the entries start on, the middle one of those 3 to 60 pages before the next start (a main article
    rather than a review or a short note), leaving out numbered sections of longer texts. The volume's page
    list and the words of its contents pages are kept in ``<library>/toc/``, so they are requested once.
    """

    name = "knav"
    api = API
    landing_url = "https://kramerius.lib.cas.cz/uuid/"
    moving_wall = True
    skip_newspapers = False
    type_preference = ("pdf", "scan", "volume")

    def __init__(self, pick_dirs=AUTOMATIC_PICK_DIRS, pick_logs=AUTOMATIC_PICK_LOGS,
                 unsegmented_periodicals: Path | None = UNSEGMENTED_PERIODICALS):
        super().__init__()
        self.pick_dirs = pick_dirs
        self.pick_logs = pick_logs
        self.unsegmented_periodicals = unsegmented_periodicals
        self.local: dict[str, str] = {}
        self.errors: dict[str, str] = {}

    def build_catalog(self) -> list[CatalogItem]:
        self.local = index_local_pdfs(self.pick_dirs)
        self.errors = read_download_errors(self.pick_logs)
        logger.info(f"Earlier downloads: {len(self.local)} PDFs, {len(self.errors)} errors")
        articles = super().build_catalog()
        return articles + self.volume_items(articles)

    def volume_items(self, articles: list[CatalogItem]) -> list[CatalogItem]:
        """One item per volume of the unsegmented periodicals in a year without article records."""
        if self.unsegmented_periodicals is None:
            return []
        with_articles = {(item.journal_id, item.year) for item in articles}
        items = []
        for periodical in read_periodicals(self.unsegmented_periodicals):
            journal_id = periodical.removeprefix("uuid:")
            volumes = self.kramerius.search_all(f'root.pid:"{periodical}" AND model:periodicalvolume', VOLUME_FIELDS,
                                                filtered=False)
            own = [doc for doc in volumes if (journal_id, doc.get("date_range_start.year")) not in with_articles]
            items += [self.volume_item(journal_id, doc) for doc in own]
            logger.info(f"{periodical} {volumes[0].get('root.title') if volumes else ''}: {len(own)} of "
                        f"{len(volumes)} volumes without article records")
        return items

    def volume_item(self, journal_id: str, doc: dict) -> CatalogItem:
        year = doc.get("date_range_start.year")
        return CatalogItem(
            library=self.name,
            item_id=doc["pid"].removeprefix("uuid:"),
            record_id=doc["pid"],
            item_type="volume",
            journal_id=journal_id,
            journal_title=doc.get("root.title"),
            volume=doc.get("part.number.str"),
            year=int(year) if year else None,
            date=doc.get("date.str"),
            rights=doc.get("licenses.facet", []),
            landing_url=f"{self.landing_url}{doc['pid']}",
            record={"accessibility": [doc["accessibility"]]} if doc.get("accessibility") else {},
        )

    def download(self, item: CatalogItem) -> Download:
        if item.item_type == "volume":
            self.locate_article(item)
        return super().download(item)

    def locate_article(self, item: CatalogItem) -> None:
        """Find the article of a volume item through the volume's contents pages, and record its page
        (``record.first_page``), issue and the contents entries pointing to it in the item."""
        cache = ArticleStore(self.root, self.name).dir / "toc"
        cache.mkdir(parents=True, exist_ok=True)
        volume = _cached(cache / f"{item.item_id}.json", lambda: self.volume_structure(item.record_id))
        pages = physical_order(item.record_id, volume["pages"], volume["parents"])
        contents = []
        for page in pages:
            if (page.page_type or "").lower() == "tableofcontents":
                words = _cached(cache / f"{page.pid.removeprefix('uuid:')}.json", lambda: self.page_words(page.pid))
                contents.append((page, toc_entries([Word(*word) for word in words])))
        starts = start_pages(contents, pages)
        if not starts:
            raise DownloadBlocked(f"{item.record_id}: {len(contents)} contents pages, no entry matches a page")
        articles = [start for start in starts if not SUBSECTION.match(start.entries[0].text)] or starts
        articles = [start for start in articles if start.span in ARTICLE_PAGES] or articles
        start = articles[len(articles) // 2]
        page = self.item_from_docs({doc["pid"]: doc for doc in volume["pages"]}[start.page.pid],
                                   {doc["pid"]: doc for doc in volume["parents"]})
        item.issue = page.issue
        item.date = page.date or item.date
        item.record.update({key: values for key, values in {
            "first_page": [start.page.pid],
            "page_number": [start.page.label] if start.page.label else [],
            "toc_entries": [entry.text for entry in start.entries],
            "contents_pages": start.contents_pages,
            "pages_to_next_start": [str(start.span)] if start.span is not None else [],
            "article_start_pages_found": [str(len(starts))],
        }.items() if values})

    def volume_structure(self, volume_pid: str) -> dict:
        uuid = volume_pid.removeprefix("uuid:")
        return {
            "parents": self.kramerius.search_all(
                f"own_pid_path:*{uuid}* AND model:(periodicalvolume OR periodicalitem OR supplement)",
                PARENT_FIELDS + ["rels_ext_index.sort"], filtered=False),
            "pages": self.kramerius.search_all(f"own_pid_path:*{uuid}* AND model:page", PAGE_FIELDS, filtered=False),
        }

    def page_words(self, pid: str) -> list[list]:
        """The words of a page with their boxes: from its ALTO, or by OCR of its image when it has none."""
        try:
            words = alto_words(http_get(f"{self.api}/items/{pid}/ocr/alto"))
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            words = ocr_words(http_get(self.kramerius.image_url(pid)))
        return [[word.x0, word.y0, word.x1, word.y1, word.text] for word in words]

    def item_from_docs(self, doc: dict, parents: dict[str, dict], periodical: dict | None = None) -> CatalogItem:
        item = super().item_from_docs(doc, parents, periodical)
        if item.item_id in self.local:
            item.record["local_pdf"] = [self.local[item.item_id]]
        if item.item_id in self.errors:
            item.record["previous_download_error"] = [self.errors[item.item_id]]
        return item

    def is_available(self, item: CatalogItem) -> bool:
        if item.record.get("previous_download_error") == ["403"]:
            return False
        if item.item_type == "volume" and RESTRICTED_LICENSES & set(item.rights):
            return False
        if item.pdf_urls and not (item.title and NOT_ARTICLE.match(item.title)):
            # KNAV serves the PDFs of many private articles too; it tells which by refusing the rest.
            return True
        return super().is_available(item)

    def is_cheap(self, item: CatalogItem) -> bool:
        return bool(item.record.get("local_pdf"))

    def local_pdf(self, item: CatalogItem, pdf_dir: Path | None) -> Download | None:
        for path in item.record.get("local_pdf", []):
            if os.path.exists(path):
                return Download(Path(path).read_bytes(), item.pdf_urls[0] if item.pdf_urls else None)
        return super().local_pdf(item, pdf_dir)


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


def read_periodicals(path: Path) -> list[str]:
    """The periodical pids of an ``unsegmented_periodicals.tsv`` (columns periodical, periodical_pid)."""
    with open(path, encoding="utf-8", newline="") as file:
        return [row["periodical_pid"] for row in csv.DictReader(file, delimiter="\t")]


def physical_order(volume_pid: str, pages: list[dict], parents: list[dict]) -> list[Page]:
    """A volume's pages in reading order: by the order of the volume's children (issues, or pages
    directly in the volume), then by the page's order within its issue."""
    order = {doc["pid"]: doc.get("rels_ext_index.sort", 0) for doc in parents + pages}

    def key(doc: dict) -> tuple:
        path = doc.get("own_pid_path", "").split("/")
        below = path[path.index(volume_pid) + 1:] if volume_pid in path else [doc["pid"]]
        return tuple(order.get(pid, 0) for pid in below)

    return [Page(doc["pid"], doc.get("page.number"), doc.get("own_parent.pid"), doc.get("page.type"))
            for doc in sorted(pages, key=key)]


def _cached(path: Path, compute: Callable[[], object]):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    value = compute()
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return value

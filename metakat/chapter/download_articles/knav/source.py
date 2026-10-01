from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from metakat.chapter.download_articles.common.kramerius import NOT_ARTICLE, KrameriusSource
from metakat.chapter.download_articles.common.models import CatalogItem
from metakat.chapter.download_articles.common.source import Download

logger = logging.getLogger(__name__)

API = "https://kramerius.lib.cas.cz/search/api/client/v7.0"

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
    """

    name = "knav"
    api = API
    landing_url = "https://kramerius.lib.cas.cz/uuid/"
    moving_wall = True
    skip_newspapers = False

    def __init__(self, pick_dirs=AUTOMATIC_PICK_DIRS, pick_logs=AUTOMATIC_PICK_LOGS):
        super().__init__()
        self.pick_dirs = pick_dirs
        self.pick_logs = pick_logs
        self.local: dict[str, str] = {}
        self.errors: dict[str, str] = {}

    def build_catalog(self) -> list[CatalogItem]:
        self.local = index_local_pdfs(self.pick_dirs)
        self.errors = read_download_errors(self.pick_logs)
        logger.info(f"Earlier downloads: {len(self.local)} PDFs, {len(self.errors)} errors")
        return super().build_catalog()

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

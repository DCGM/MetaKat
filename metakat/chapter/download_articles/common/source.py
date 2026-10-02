from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from metakat.chapter.download_articles.common.http import http_get
from metakat.chapter.download_articles.common.models import CatalogItem

# Contents pages, indexes and similar parts catalogued as articles, never picked.
NOT_ARTICLE = re.compile(
    r"^\W*(obsah|contents?|table of contents|inhalt|sommaire|содержание|tiráž|impressum|rejstřík|index|"
    r"errata|oprava|obálka|cover|reklam\w*|inzer\w*|inserat\w*|anzeigen?|annonc\w*|advertisements?)\b", re.IGNORECASE)
# Other parts often catalogued as articles whose first page is rarely an article's title page; a
# replacement for a rejected pick avoids them (``Source.is_unlikely``).
UNLIKELY_ARTICLE = re.compile(
    r"^\W*(titulní list|title page|(přední |zadní |vnitřní )?obálka|(instructions|information|guidelines|notes?) "
    r"(for|to) (the )?(authors?|contributors|subscribers|readers)|pokyny (pro|k) (autory|autorům|přispěvatel\w*)|"
    r"informace pro (autory|předplatitele|čtenáře)|summary|summaries|editorial|úvodník|úvodem|na úvod|"
    r"předmluva|foreword|preface|vorwort|in memoriam|nekrolog\w*|obituary|personalia)\b", re.IGNORECASE)


class DownloadBlocked(RuntimeError):
    """The library answered a file request with something other than the file, e.g. a login page."""


@dataclass
class Download:
    """What a library serves for an article: its PDF, or directly the image of its title page."""

    data: bytes
    url: str | None
    # "pdf", or the image file extension when the library serves page images (scanned articles).
    kind: str = "pdf"


class Source(ABC):
    """A digital library the article sampler can catalog and download from."""

    # Folder name of the library under the output root.
    name: str
    # Accepted item types, best first; see ``select_items``. Empty accepts every type.
    type_preference: tuple[str, ...] = ()
    # True when the library serves its files only to people (e.g. behind a CAPTCHA). The sampler
    # then never requests files itself and only ingests PDFs saved by hand.
    manual_download: bool = False
    # True when the library withholds its newest volumes: a journal's years from its first refused
    # item on are then not selected, except items that cost no request (``is_cheap``).
    moving_wall: bool = False
    # Least seconds between two requests for a library that refuses faster ones (HTTP 429).
    min_interval: float = 0.0
    # The output root holding every library's folder; set by the command line.
    root: Path | None = None

    @abstractmethod
    def build_catalog(self) -> list[CatalogItem]:
        """Harvest the library's catalog of article-like items."""

    def is_available(self, item: CatalogItem) -> bool:
        """Whether the item's full text is expected to be downloadable at all."""
        return bool(item.pdf_urls)

    def starts_wall(self, reason: str) -> bool:
        """Whether an item recorded as unavailable for ``reason`` starts a moving wall (``moving_wall``);
        failures of the sampler's own, rather than refusals of the library, do not."""
        return True

    def resolve_links(self, items: list[CatalogItem]) -> bool:
        """Before a selection for downloading by hand is written: put the address of the file itself first
        in ``pdf_urls`` of items whose links redirect, so that the saved files are found by name. Returns
        whether any item changed (the catalog is then written again)."""
        return False

    def is_unlikely(self, item: CatalogItem) -> bool:
        """Whether the item's title names a part that is rarely an article (a title leaf, a contents page,
        instructions for authors, ...); replacements for rejected picks avoid such items."""
        return bool(item.title and (NOT_ARTICLE.match(item.title) or UNLIKELY_ARTICLE.match(item.title)))

    def is_cheap(self, item: CatalogItem) -> bool:
        """Whether the item can be stored without asking the library, e.g. it was downloaded before."""
        return False

    def local_pdf(self, item: CatalogItem, pdf_dir: Path | None) -> Download | None:
        """The item's PDF from disk: saved by hand into ``pdf_dir``, or kept from earlier downloads."""
        if pdf_dir is None:
            return None
        for url in item.pdf_urls:
            path = pdf_dir / url.rsplit("/", 1)[-1]
            if path.exists():
                return Download(path.read_bytes(), url)
        return None

    def download(self, item: CatalogItem) -> Download:
        """Download the item; raises ``DownloadBlocked`` or an HTTP error when it is refused."""
        error: Exception | None = None
        for url in item.pdf_urls:
            try:
                return Download(self.download_pdf(url), url)
            except (DownloadBlocked, OSError) as failure:
                error = failure
        raise error or DownloadBlocked(f"{item.item_id} has no file to download")

    def download_pdf(self, url: str) -> bytes:
        data = http_get(url)
        if not data.startswith(b"%PDF"):
            raise DownloadBlocked(f"{url} did not return a PDF")
        return data

    def title_page_index(self, pdf_bytes: bytes, url: str | None = None) -> int:
        """Index of the article's first page in its PDF (served from ``url``), for libraries that prepend
        cover sheets."""
        return 0

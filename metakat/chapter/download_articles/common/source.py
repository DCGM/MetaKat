from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from metakat.chapter.download_articles.common.http import http_get
from metakat.chapter.download_articles.common.models import CatalogItem


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
    # item on are then not selected. Libraries that refuse by document licence keep it False.
    moving_wall: bool = False

    @abstractmethod
    def build_catalog(self) -> list[CatalogItem]:
        """Harvest the library's catalog of article-like items."""

    def is_available(self, item: CatalogItem) -> bool:
        """Whether the item's full text is expected to be downloadable at all."""
        return bool(item.pdf_urls)

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

    def title_page_index(self, pdf_bytes: bytes) -> int:
        """Index of the article's first page in its PDF, for libraries that prepend cover sheets."""
        return 0

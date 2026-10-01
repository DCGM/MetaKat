from __future__ import annotations

from abc import ABC, abstractmethod

from metakat.chapter.download_articles.models import CatalogItem
from metakat.chapter.download_articles.oai import http_get


class DownloadBlocked(RuntimeError):
    """The library answered a file request with something other than the file, e.g. a human check."""


class Source(ABC):
    """A digital library the article sampler can catalog and download from."""

    # Folder name of the library under the output root.
    name: str
    # Accepted item types, best first; see ``select_items``. Empty accepts every type.
    type_preference: tuple[str, ...] = ()
    # True when the library serves its files only to people (e.g. behind a CAPTCHA). The sampler
    # then never requests files itself and only ingests PDFs saved by hand.
    manual_download: bool = False

    @abstractmethod
    def build_catalog(self) -> list[CatalogItem]:
        """Harvest the library's catalog of article-like items."""

    def is_available(self, item: CatalogItem) -> bool:
        """Whether the item's full text is expected to be downloadable at all."""
        return bool(item.pdf_urls)

    def download_pdf(self, url: str) -> bytes:
        data = http_get(url)
        if not data.startswith(b"%PDF"):
            raise DownloadBlocked(f"{url} did not return a PDF")
        return data

    def title_page_index(self, pdf_bytes: bytes) -> int:
        """Index of the article's first page in its PDF, for libraries that prepend cover sheets."""
        return 0

    def local_pdf_names(self, item: CatalogItem) -> list[str]:
        """File names a browser gives the item's PDFs, in ``pdf_urls`` order."""
        return [url.rsplit("/", 1)[-1] for url in item.pdf_urls]

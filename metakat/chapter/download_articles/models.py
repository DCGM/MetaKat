from __future__ import annotations

from pydantic import BaseModel, Field


class CatalogItem(BaseModel):
    """One downloadable article-like unit of a digital library, as described by its catalog.

    Values the library does not provide stay ``None``; nothing is inferred.
    """

    library: str
    item_id: str
    record_id: str
    title: str | None = None
    item_type: str | None = None
    journal_id: str | None = None
    journal_title: str | None = None
    volume: str | None = None
    issue: str | None = None
    year: int | None = None
    date: str | None = None
    pages: str | None = None
    authors: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    rights: list[str] = Field(default_factory=list)
    # Ordered by preference; the first one that downloads is used.
    pdf_urls: list[str] = Field(default_factory=list)
    landing_url: str | None = None
    # The record as harvested, so later questions can be answered without harvesting again.
    record: dict[str, list[str]] = Field(default_factory=dict)


class FirstPageImage(BaseModel):
    """How the stored title page image was obtained from the article PDF."""

    file: str
    # 1-based page of the article PDF the image shows.
    page: int = 1
    width: int
    height: int
    # "embedded": the page's single scanned image, stored byte for byte.
    # "rendered": the page rendered at ``dpi``, the scan's own resolution when it has one.
    method: str
    dpi: float | None = None


class StoredArticle(BaseModel):
    """Metadata written next to every stored title page."""

    item: CatalogItem
    pdf_file: str
    pdf_url: str | None = None
    image: FirstPageImage
    stored_at: str

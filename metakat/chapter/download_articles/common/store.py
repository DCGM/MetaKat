from __future__ import annotations

import csv
import html
import io
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from metakat.chapter.download_articles.common.first_page import extract_first_page
from metakat.chapter.download_articles.common.models import CatalogItem, FirstPageImage, StoredArticle
from metakat.chapter.download_articles.common.selection import JournalKey, journal_key

SELECTION_COLUMNS = ["item_id", "journal_id", "journal_title", "volume", "issue", "year", "item_type", "title", "pdf_url"]


class ArticleStore:
    """One library's folder: its catalog, the current selection and the stored title pages.

    Layout::

        <root>/<library>/catalog.jsonl         every catalog item, one JSON object per line
        <root>/<library>/selection.tsv         the items picked by the last ``select`` run
        <root>/<library>/selection.html        the same items as links, for libraries downloaded by hand
        <root>/<library>/unavailable.tsv       items the library did not serve (e.g. a moving wall)
        <root>/<library>/pdf/<item_id>.pdf     the article PDF as downloaded
        <root>/<library>/images/<item_id>.*    its first page at the provided resolution
        <root>/<library>/metadata/<item_id>.json
    """

    def __init__(self, root: str | Path, library: str):
        self.dir = Path(root) / library
        self.catalog_path = self.dir / "catalog.jsonl"
        self.selection_path = self.dir / "selection.tsv"
        self.selection_html_path = self.dir / "selection.html"
        self.unavailable_path = self.dir / "unavailable.tsv"
        self.pdf_dir = self.dir / "pdf"
        self.images_dir = self.dir / "images"
        self.metadata_dir = self.dir / "metadata"

    def write_catalog(self, items: list[CatalogItem]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.catalog_path, "w", encoding="utf-8") as file:
            for item in items:
                file.write(item.model_dump_json() + "\n")

    def read_catalog(self) -> list[CatalogItem]:
        with open(self.catalog_path, encoding="utf-8") as file:
            return [CatalogItem.model_validate_json(line) for line in file if line.strip()]

    def stored(self) -> list[StoredArticle]:
        if not self.metadata_dir.exists():
            return []
        return [StoredArticle.model_validate_json(path.read_text(encoding="utf-8"))
                for path in sorted(self.metadata_dir.glob("*.json"))]

    def stored_years(self) -> dict[JournalKey, list[int | None]]:
        years: dict[JournalKey, list[int | None]] = defaultdict(list)
        for article in self.stored():
            if article.item.journal_id is not None:
                years[journal_key(article.item)].append(article.item.year)
        return dict(years)

    def write_selection(self, items: list[CatalogItem]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.selection_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file, delimiter="\t")
            writer.writerow(SELECTION_COLUMNS)
            for item in items:
                writer.writerow([item.item_id, item.journal_id, item.journal_title, item.volume, item.issue,
                                 item.year, item.item_type, item.title, item.pdf_urls[0] if item.pdf_urls else ""])

        rows = "\n".join(
            f'<li><a href="{html.escape(item.pdf_urls[0])}">{html.escape(item.item_id)}</a> '
            f'{html.escape(item.journal_title or "")} ({item.year or "?"}): {html.escape(item.title or "")}</li>'
            for item in items if item.pdf_urls
        )
        self.selection_html_path.write_text(
            f'<!doctype html><meta charset="utf-8"><title>Selection</title><ol>\n{rows}\n</ol>\n', encoding="utf-8")

    def read_selection(self) -> list[str]:
        with open(self.selection_path, encoding="utf-8", newline="") as file:
            return [row["item_id"] for row in csv.DictReader(file, delimiter="\t")]

    def stored_pdf(self, item_id: str) -> bytes | None:
        path = self.pdf_dir / f"{item_id}.pdf"
        return path.read_bytes() if path.exists() else None

    def store(self, item: CatalogItem, pdf_bytes: bytes, pdf_url: str | None, page_index: int = 0) -> StoredArticle:
        """Store the PDF, the image of its title page (``page_index``) and the metadata describing both."""
        image_bytes, extension, image = extract_first_page(pdf_bytes, page_index)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = self.pdf_dir / f"{item.item_id}.pdf"
        pdf_path.write_bytes(pdf_bytes)
        image.page = page_index + 1
        return self._store_image(item, image_bytes, extension, image, str(pdf_path.relative_to(self.dir)), pdf_url)

    def store_image(self, item: CatalogItem, image_bytes: bytes, extension: str, url: str) -> StoredArticle:
        """Store a title page the library serves as an image, as served."""
        with Image.open(io.BytesIO(image_bytes)) as decoded:
            width, height = decoded.size
        image = FirstPageImage(file="", width=width, height=height, method="page image")
        return self._store_image(item, image_bytes, extension, image, None, url)

    def _store_image(self, item: CatalogItem, image_bytes: bytes, extension: str, image: FirstPageImage,
                     pdf_file: str | None, url: str | None) -> StoredArticle:
        for directory in (self.images_dir, self.metadata_dir):
            directory.mkdir(parents=True, exist_ok=True)
        for old_image in self.images_dir.glob(f"{item.item_id}.*"):
            old_image.unlink()
        image_path = self.images_dir / f"{item.item_id}.{extension}"
        image_path.write_bytes(image_bytes)
        image.file = str(image_path.relative_to(self.dir))

        article = StoredArticle(
            item=item, pdf_file=pdf_file, pdf_url=url, image=image,
            stored_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        metadata_path = self.metadata_dir / f"{item.item_id}.json"
        metadata_path.write_text(json.dumps(article.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        return article

    def mark_unavailable(self, item: CatalogItem, reason: str) -> None:
        """Remember an item the library would not serve, so later selections avoid it and its period."""
        self.dir.mkdir(parents=True, exist_ok=True)
        new_file = not self.unavailable_path.exists()
        with open(self.unavailable_path, "a", encoding="utf-8", newline="") as file:
            writer = csv.writer(file, delimiter="\t")
            if new_file:
                writer.writerow(["item_id", "journal_id", "journal_title", "year", "reason"])
            writer.writerow([item.item_id, item.journal_id, item.journal_title, item.year, reason])

    def unavailable(self) -> dict[str, dict[str, str]]:
        if not self.unavailable_path.exists():
            return {}
        with open(self.unavailable_path, encoding="utf-8", newline="") as file:
            return {row["item_id"]: row for row in csv.DictReader(file, delimiter="\t")}

    def is_stored(self, item_id: str) -> bool:
        return (self.metadata_dir / f"{item_id}.json").exists()

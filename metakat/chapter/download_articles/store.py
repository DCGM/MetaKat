from __future__ import annotations

import csv
import html
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from metakat.chapter.download_articles.first_page import extract_first_page
from metakat.chapter.download_articles.models import CatalogItem, StoredArticle
from metakat.chapter.download_articles.selection import JournalKey, journal_key

SELECTION_COLUMNS = ["item_id", "journal_id", "journal_title", "volume", "issue", "year", "item_type", "title", "pdf_url"]


class ArticleStore:
    """One library's folder: its catalog, the current selection and the stored title pages.

    Layout::

        <root>/<library>/catalog.jsonl         every catalog item, one JSON object per line
        <root>/<library>/selection.tsv         the items picked by the last ``select`` run
        <root>/<library>/selection.html        the same items as links, for libraries downloaded by hand
        <root>/<library>/pdf/<item_id>.pdf     the article PDF as downloaded
        <root>/<library>/images/<item_id>.*    its first page at the provided resolution
        <root>/<library>/metadata/<item_id>.json
    """

    def __init__(self, root: str | Path, library: str):
        self.dir = Path(root) / library
        self.catalog_path = self.dir / "catalog.jsonl"
        self.selection_path = self.dir / "selection.tsv"
        self.selection_html_path = self.dir / "selection.html"
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
        for directory in (self.pdf_dir, self.images_dir, self.metadata_dir):
            directory.mkdir(parents=True, exist_ok=True)

        image_bytes, extension, image = extract_first_page(pdf_bytes, page_index)
        for old_image in self.images_dir.glob(f"{item.item_id}.*"):
            old_image.unlink()
        image_path = self.images_dir / f"{item.item_id}.{extension}"
        pdf_path = self.pdf_dir / f"{item.item_id}.pdf"
        pdf_path.write_bytes(pdf_bytes)
        image_path.write_bytes(image_bytes)
        image.file = str(image_path.relative_to(self.dir))
        image.page = page_index + 1

        article = StoredArticle(
            item=item, pdf_file=str(pdf_path.relative_to(self.dir)), pdf_url=pdf_url, image=image,
            stored_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        metadata_path = self.metadata_dir / f"{item.item_id}.json"
        metadata_path.write_text(json.dumps(article.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        return article

    def is_stored(self, item_id: str) -> bool:
        return (self.metadata_dir / f"{item_id}.json").exists()

"""Sample article title pages from digital libraries.

    python -m metakat.chapter.download_articles catalog --source muni_digilib
    python -m metakat.chapter.download_articles select  --source muni_digilib --per-journal 1
    python -m metakat.chapter.download_articles fetch   --source muni_digilib [--pdf-dir DIR]

See README.md in this directory.
"""
import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

from metakat.chapter.download_articles.selection import select_items
from metakat.chapter.download_articles.sources import SOURCES, DownloadBlocked
from metakat.chapter.download_articles.store import ArticleStore

DEFAULT_ROOT = "/mnt/kolosus/data/smart_digiline/articles"

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["catalog", "select", "fetch"])
    parser.add_argument("--source", required=True, choices=sorted(SOURCES))
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Output root; the library gets its own folder in it.")
    parser.add_argument("--per-journal", default=1, type=int, help="select: new items per journal.")
    parser.add_argument("--min-year-gap", default=0, type=int,
                        help="select: minimum distance in years from every item already picked in the journal.")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--pdf-dir", type=Path, help="fetch: folder with PDFs saved by hand.")
    parser.add_argument("--delay", default=1.0, type=float, help="fetch: seconds between downloads.")
    parser.add_argument("--logging-level", default=logging.INFO)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(format="%(asctime)s - DOWNLOAD ARTICLES - %(levelname)s - %(message)s",
                        level=args.logging_level)
    logging.Formatter.converter = time.gmtime
    logger.info(" ".join(sys.argv))

    source = SOURCES[args.source]()
    store = ArticleStore(args.root, source.name)

    if args.command == "catalog":
        items = source.build_catalog()
        store.write_catalog(items)
        journals = {item.journal_id for item in items}
        logger.info(f"Catalog: {len(items)} items in {len(journals)} journals -> {store.catalog_path}")
        logger.info(f"Item types: {Counter(item.item_type for item in items).most_common()}")

    elif args.command == "select":
        items = [item for item in store.read_catalog()
                 if source.is_available(item) and not store.is_stored(item.item_id)]
        selected = select_items(items, per_journal=args.per_journal, type_preference=source.type_preference,
                                min_year_gap=args.min_year_gap, already_selected=store.stored_years(),
                                seed=args.seed)
        store.write_selection(selected)
        logger.info(f"Selected {len(selected)} items from {len({i.journal_id for i in selected})} journals "
                    f"-> {store.selection_path}")
        if source.manual_download:
            logger.info(f"{source.name} serves files to people only: open {store.selection_html_path}, save the "
                        f"PDFs into one folder and run fetch with --pdf-dir.")

    elif args.command == "fetch":
        fetch(source, store, args.pdf_dir, args.delay)


def fetch(source, store, pdf_dir, delay):
    catalog = {item.item_id: item for item in store.read_catalog()}
    stored, missing, failed = 0, [], []
    for item_id in store.read_selection():
        item = catalog[item_id]
        if store.is_stored(item_id):
            continue
        pdf_bytes, pdf_url = None, None

        if pdf_dir is not None:
            for name, url in zip(source.local_pdf_names(item), item.pdf_urls):
                if (pdf_dir / name).exists():
                    pdf_bytes, pdf_url = (pdf_dir / name).read_bytes(), url
                    break

        if pdf_bytes is None and not source.manual_download:
            for url in item.pdf_urls:
                try:
                    pdf_bytes, pdf_url = source.download_pdf(url), url
                    break
                except (DownloadBlocked, OSError) as error:
                    logger.warning(f"{item_id}: {error}")
                finally:
                    time.sleep(delay)

        if pdf_bytes is None:
            missing.append(item_id)
            continue
        try:
            article = store.store(item, pdf_bytes, pdf_url)
        except Exception as error:
            logger.error(f"{item_id}: cannot extract the first page: {error}")
            failed.append(item_id)
            continue
        stored += 1
        logger.info(f"{item_id}: {article.image.file} {article.image.width}x{article.image.height} "
                    f"({article.image.method}, {article.image.dpi} dpi)")

    logger.info(f"Stored {stored}, missing PDF {len(missing)}, failed {len(failed)}")
    if missing:
        logger.info(f"Missing: {' '.join(missing)}")


if __name__ == "__main__":
    main()

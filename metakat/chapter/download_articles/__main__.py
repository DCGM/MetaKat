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

from metakat.chapter.download_articles.selection import journal_key, select_by_period, select_items
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
    parser.add_argument("--period", type=int,
                        help="select: instead of --per-journal, give every journal its first and last year and "
                             "one item per this many years, counting items already stored.")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--pdf-dir", type=Path, help="fetch: folder with PDFs saved by hand.")
    parser.add_argument("--delay", default=2.0, type=float,
                        help="fetch: seconds between downloads; downloads are never parallel.")
    parser.add_argument("--reextract", action="store_true",
                        help="fetch: extract the title page again for selected items already stored, from their stored PDF.")
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
        journals = {journal_key(item) for item in items}
        logger.info(f"Catalog: {len(items)} items in {len(journals)} journals -> {store.catalog_path}")
        logger.info(f"Item types: {Counter(item.item_type for item in items).most_common()}")

    elif args.command == "select":
        # Libraries withhold their newest volumes (a moving wall): from the first year a journal's
        # item was refused on, its later years are not picked either.
        walls: dict = {}
        catalog = store.read_catalog()
        unavailable = store.unavailable()
        for item in catalog:
            if item.item_id in unavailable and item.year is not None:
                key = journal_key(item)
                walls[key] = min(walls.get(key, item.year), item.year)
        items = [item for item in catalog
                 if source.is_available(item) and not store.is_stored(item.item_id)
                 and item.item_id not in unavailable
                 and not (item.year is not None and item.year >= walls.get(journal_key(item), item.year + 1))]
        if walls:
            logger.info(f"Moving walls: {', '.join(f'{k[1]} {y}' for k, y in sorted(walls.items(), key=str))}")
        if args.period:
            selected = select_by_period(items, args.period, type_preference=source.type_preference,
                                        already_selected=store.stored_years(), seed=args.seed)
        else:
            selected = select_items(items, per_journal=args.per_journal, type_preference=source.type_preference,
                                    min_year_gap=args.min_year_gap, already_selected=store.stored_years(),
                                    seed=args.seed)
        store.write_selection(selected)
        logger.info(f"Selected {len(selected)} items from {len({journal_key(i) for i in selected})} journals "
                    f"-> {store.selection_path}")
        if source.manual_download:
            logger.info(f"{source.name} serves files to people only: open {store.selection_html_path}, save the "
                        f"PDFs into one folder and run fetch with --pdf-dir.")

    elif args.command == "fetch":
        fetch(source, store, args.pdf_dir, args.delay, args.reextract)


def fetch(source, store, pdf_dir, delay, reextract=False):
    catalog = {item.item_id: item for item in store.read_catalog()}
    previous = {article.item.item_id: article for article in store.stored()}
    stored, missing, failed = 0, [], []
    for item_id in store.read_selection():
        item = catalog[item_id]
        if store.is_stored(item_id) and not reextract:
            continue
        pdf_bytes, pdf_url = store.stored_pdf(item_id), None
        if pdf_bytes is not None and item_id in previous:
            pdf_url = previous[item_id].pdf_url

        if pdf_bytes is None and pdf_dir is not None:
            for name, url in zip(source.local_pdf_names(item), item.pdf_urls):
                if (pdf_dir / name).exists():
                    pdf_bytes, pdf_url = (pdf_dir / name).read_bytes(), url
                    break

        refused = None
        if pdf_bytes is None and not source.manual_download:
            for url in item.pdf_urls:
                try:
                    pdf_bytes, pdf_url = source.download_pdf(url), url
                    break
                except (DownloadBlocked, OSError) as error:
                    logger.warning(f"{item_id}: {error}")
                    if isinstance(error, DownloadBlocked) or getattr(error, "code", None) in (401, 403, 404, 410):
                        refused = str(error)
                finally:
                    time.sleep(delay)

        if pdf_bytes is None:
            missing.append(item_id)
            if refused:
                store.mark_unavailable(item, refused)
            continue
        try:
            article = store.store(item, pdf_bytes, pdf_url, source.title_page_index(pdf_bytes))
        except Exception as error:
            logger.error(f"{item_id}: cannot extract the title page: {error}")
            failed.append(item_id)
            continue
        stored += 1
        logger.info(f"{item_id}: {article.image.file} page {article.image.page} "
                    f"{article.image.width}x{article.image.height} ({article.image.method}, {article.image.dpi} dpi)")

    logger.info(f"Stored {stored}, missing PDF {len(missing)}, failed {len(failed)}")
    if missing:
        logger.info(f"Missing: {' '.join(missing)}")


if __name__ == "__main__":
    main()

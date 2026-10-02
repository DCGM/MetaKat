"""Sample article title pages from digital libraries.

    python -m metakat.chapter.download_articles catalog --source dml_cz
    python -m metakat.chapter.download_articles select  --source dml_cz --period 5
    python -m metakat.chapter.download_articles select  --source dml_cz --replace
    python -m metakat.chapter.download_articles fetch   --source dml_cz [--pdf-dir DIR]
    python -m metakat.chapter.download_articles preview --source dml_cz

See README.md in metakat/chapter/download_articles.
"""
from __future__ import annotations

import argparse
import logging
import socket
import ssl
import sys
import time
from collections import Counter
from pathlib import Path

from metakat.chapter.download_articles.common import http
from metakat.chapter.download_articles.common.preview import render_previews
from metakat.chapter.download_articles.common.review import to_replace
from metakat.chapter.download_articles.common.selection import (journal_key, select_by_period, select_items,
                                                                 select_replacements)
from metakat.chapter.download_articles.common.source import Download, DownloadBlocked, Source
from metakat.chapter.download_articles.common.store import ArticleStore

DEFAULT_ROOT = "/mnt/kolosus/data/smart_digiline/articles"
# Failures meaning the file's host is gone (unknown name, a certificate of another site), not a hiccup.
DEAD_HOST_ERRORS = (socket.gaierror, ssl.SSLCertVerificationError)

logger = logging.getLogger(__name__)


def parse_args(sources: dict[str, type[Source]]):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["catalog", "select", "fetch", "preview"])
    parser.add_argument("--source", required=True, choices=sorted(sources))
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Output root; the library gets its own folder in it.")
    parser.add_argument("--per-journal", default=1, type=int, help="select: new items per journal.")
    parser.add_argument("--min-year-gap", default=0, type=int,
                        help="select: minimum distance in years from every item already picked in the journal.")
    parser.add_argument("--period", type=int,
                        help="select: instead of --per-journal, give every journal its first and last year and "
                             "one item per this many years, counting items already stored.")
    parser.add_argument("--replace", action="store_true",
                        help="select: instead, one replacement for every pick rejected in a closed review round "
                             "(common/review.py --close-round), from its journal and the closest year.")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--pdf-dir", type=Path, help="fetch: folder with PDFs saved by hand.")
    parser.add_argument("--delay", type=float,
                        help="Minimum seconds between two requests to the library; requests are never parallel. "
                             "Default 1 for catalog, 2 for fetch, more for libraries that ask for it.")
    parser.add_argument("--reextract", action="store_true",
                        help="fetch: extract the title page again for selected items already stored, from their stored PDF.")
    parser.add_argument("--logging-level", default=logging.INFO)
    return parser.parse_args()


def main(sources: dict[str, type[Source]]):
    args = parse_args(sources)
    logging.basicConfig(format="%(asctime)s - DOWNLOAD ARTICLES - %(levelname)s - %(message)s",
                        level=args.logging_level)
    logging.Formatter.converter = time.gmtime
    logger.info(" ".join(sys.argv))
    source = sources[args.source]()
    default_delay = max(2.0 if args.command == "fetch" else 1.0, source.min_interval)
    http.set_min_interval(args.delay if args.delay is not None else default_delay)
    source.root = Path(args.root)
    store = ArticleStore(args.root, source.name)

    if args.command == "catalog":
        items = source.build_catalog()
        store.write_catalog(items)
        logger.info(f"Catalog: {len(items)} items in {len({journal_key(i) for i in items})} journals "
                    f"-> {store.catalog_path}")
        logger.info(f"Item types: {Counter(item.item_type for item in items).most_common()}")
    elif args.command == "select":
        select(source, store, args)
    elif args.command == "fetch":
        fetch(source, store, args.pdf_dir, args.reextract)
    elif args.command == "preview":
        for path in render_previews(store):
            logger.info(f"Preview: {path}")


def select(source: Source, store: ArticleStore, args) -> None:
    catalog = store.read_catalog()
    unavailable = store.unavailable()
    # From the first year a journal's item was refused on, a moving wall withholds its later years too.
    walls: dict = {}
    if source.moving_wall:
        for item in catalog:
            if (item.item_id in unavailable and item.year is not None
                    and source.starts_wall(unavailable[item.item_id].get("reason") or "")):
                key = journal_key(item)
                walls[key] = min(walls.get(key, item.year), item.year)
        if walls:
            logger.info(f"Moving walls: {', '.join(f'{k[1]} {y}' for k, y in sorted(walls.items(), key=str))}")

    # Items behind a wall stay selectable when they cost no request (e.g. downloaded before).
    stored_ids = {article.item.item_id for article in store.stored()}
    available = [item for item in catalog
                 if item.item_id in stored_ids
                 or (source.is_available(item) and item.item_id not in unavailable
                     and (source.is_cheap(item)
                          or not (item.year is not None and item.year >= walls.get(journal_key(item), item.year + 1))))]
    if args.replace:
        rejected = [article.item for article in to_replace(store)]
        pairs = select_replacements(available, rejected, type_preference=source.type_preference, seed=args.seed,
                                    cheap=source.is_cheap, unlikely=source.is_unlikely, excluded_ids=stored_ids)
        store.write_replacements(pairs)
        selected = [new for _, new in pairs]
        replaced = {old.item_id for old, _ in pairs}
        for item in rejected:
            if item.item_id not in replaced:
                logger.warning(f"{item.item_id}: nothing left to replace it in {item.journal_title} {item.year}")
        logger.info(f"Replacements for {len(pairs)} of {len(rejected)} rejected picks -> {store.replacements_path}")
    elif args.period:
        selected = select_by_period(available, args.period, type_preference=source.type_preference,
                                    already_selected=store.stored_years(), seed=args.seed, cheap=source.is_cheap,
                                    stored_ids=stored_ids)
    else:
        selected = select_items([item for item in available if item.item_id not in stored_ids],
                                per_journal=args.per_journal, type_preference=source.type_preference,
                                min_year_gap=args.min_year_gap, already_selected=store.stored_years(), seed=args.seed)
    # Files saved by hand are matched by the name of their link, so links must point at the files.
    if source.manual_download and source.resolve_links(selected):
        store.write_catalog(catalog)
    store.write_selection(selected)
    cheap = sum(1 for item in selected if source.is_cheap(item))
    logger.info(f"Selected {len(selected)} items from {len({journal_key(i) for i in selected})} journals, "
                f"{cheap} of them without a request -> {store.selection_path}")
    if source.manual_download:
        logger.info(f"{source.name} serves files to people only: open {store.selection_html_path}, save the "
                    f"PDFs into one folder and run fetch with --pdf-dir.")


def fetch(source: Source, store: ArticleStore, pdf_dir: Path | None, reextract: bool = False) -> None:
    catalog = {item.item_id: item for item in store.read_catalog()}
    previous = {article.item.item_id: article for article in store.stored()}
    # Items refused by an earlier fetch of the same selection are not requested again.
    refused = store.unavailable()
    stored, missing, failed = 0, [], []
    for item_id in store.read_selection():
        item = catalog[item_id]
        if store.is_stored(item_id) and not reextract:
            continue

        download = None
        stored_pdf = store.stored_pdf(item_id)
        if stored_pdf is not None:
            download = Download(stored_pdf, previous[item_id].pdf_url if item_id in previous else None)
        if download is None:
            download = source.local_pdf(item, pdf_dir)
        if download is None and not source.manual_download and item_id not in refused:
            try:
                download = source.download(item)
            except (DownloadBlocked, OSError) as error:
                logger.warning(f"{item_id}: {error}")
                if (isinstance(error, DownloadBlocked) or getattr(error, "code", None) in (401, 403, 404, 410)
                        or isinstance(getattr(error, "reason", None), DEAD_HOST_ERRORS)):
                    store.mark_unavailable(item, str(error))
            except Exception as error:
                # A failure of the sampler's own (e.g. a missing optional dependency), not a refusal: the
                # item stays selectable and the other items are still fetched.
                logger.exception(f"{item_id}: {error!r}")
        if download is None:
            missing.append(item_id)
            continue

        try:
            if download.kind == "pdf":
                article = store.store(item, download.data, download.url, source.title_page_index(download.data, download.url))
            else:
                article = store.store_image(item, download.data, download.kind, download.url)
        except Exception as error:
            logger.error(f"{item_id}: cannot extract the title page: {error}")
            failed.append(item_id)
            continue
        stored += 1
        logger.info(f"{item_id}: {article.image.file} page {article.image.page} "
                    f"{article.image.width}x{article.image.height} ({article.image.method}, {article.image.dpi} dpi)")

    logger.info(f"Stored {stored}, missing {len(missing)}, failed {len(failed)}")
    if missing:
        logger.info(f"Missing: {' '.join(missing)}")

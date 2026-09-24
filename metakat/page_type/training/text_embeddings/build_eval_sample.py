"""Build the labelled page sample used by evaluate_embeddings.py.

Joins three sources, all read sequentially in one pass each (the text LMDB is
never opened, random LMDB reads on the data disk are very slow):

  - the extraction progress.tsv of the text dump: which pages have text, and
    the library and document archive of every page;
  - the archive page-type CSV (pages_all.public.csv): the page type the archive
    currently holds, which is noisy -- specific types are often recorded as a
    less specific one such as NormalPage;
  - the hand-annotated page lists (annotated.pages.*): "<name> <PageType>" lines
    whose name is "uuid:<page_id>.jpg" or "<page_id>.jpg"; legacy names that do
    not carry a page UUID cannot be joined and are counted as unmapped.

The output TSV has one row per sampled page:

  key             "{library}_{page_id}", the key of the text and embedding LMDBs
  doc             the page's document archive (the directory of its page images);
                  for periodicals this groups a whole title, which is the unit of
                  the train/test split so no document is on both sides
  order, doc_pages  the page's position and the page count of its CSV item
                  (empty for annotated pages missing from the CSV)
  archive_label   the CSV page type; "" if missing or not a PageType value
  annotated_label the hand annotation, "" if the page is not annotated
  split           train | test | annotated | background

Every annotated page with text is included (split "annotated"), and every
document holding one goes to the test side. Archive-labelled pages are sampled
per class, up to --per-class-cap pages and --per-doc-cap pages per document and
class; pages without a usable label form the "background" split, shown on the
map only. Sampling and the split use a stable hash of the key/document, so
rebuilding gives the same sample.

Example:
    python -m metakat.page_type.training.text_embeddings.build_eval_sample \\
        --labels-csv /data/smart_digiline/page_types.2026-07-14/pages_all.public.csv \\
        --annotated /data/smart_digiline/page_types.2026-07-14/annotated.pages.all \\
                    /data/smart_digiline/page_types.2026-07-14/annotated.pages.tst \\
        --extract-progress /data/2026-09-16.db_text_dump/progress.tsv \\
        --output /data/2026-09-16.db_text_dump/embeddings/eval_sample.tsv
"""

import argparse
import csv
import hashlib
import heapq
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from metakat.schemas.base_objects import PageType

logger = logging.getLogger(__name__)

PAGE_TYPES = frozenset(t.value for t in PageType)
BACKGROUND = '__background__'
SAMPLE_COLUMNS = ('key', 'library', 'page_id', 'doc', 'order', 'doc_pages', 'archive_label',
                  'annotated_label', 'split')


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description='Build the labelled page sample for embedding evaluation.')
    parser.add_argument('--labels-csv', type=Path, required=True, help='Archive page-type CSV.')
    parser.add_argument('--annotated', type=Path, nargs='*', default=[],
                        help='Hand-annotated page lists; a page in several lists takes the label '
                             'from the last one.')
    parser.add_argument('--extract-progress', type=Path, required=True,
                        help='progress.tsv of the text dump (library, page_id, zip_path, status).')
    parser.add_argument('--output', type=Path, required=True, help='Output sample TSV.')
    parser.add_argument('--per-class-cap', type=int, default=5000,
                        help='Max archive-labelled pages per page type.')
    parser.add_argument('--per-doc-cap', type=int, default=10,
                        help='Max pages of one page type from one document.')
    parser.add_argument('--background-cap', type=int, default=20000,
                        help='Max pages without a usable archive label (map background).')
    parser.add_argument('--test-fraction', type=float, default=0.2,
                        help='Fraction of documents on the test side (annotated documents always are).')
    parser.add_argument('--logging-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    return parser.parse_args(argv)


def stable_hash(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode('utf-8'), digest_size=8).digest(), 'little')


def doc_from_path(path: str) -> str:
    """".../<doc>.images/<page>.jpg" or ".../<doc>.page_xml.zip" -> "<library dir>/<doc>"."""
    if path.endswith('.page_xml.zip'):
        directory = path[:-len('.page_xml.zip')]
    else:
        directory = os.path.dirname(path)
        if directory.endswith('.images'):
            directory = directory[:-len('.images')]
    library_dir, doc = os.path.split(directory)
    return f'{os.path.basename(library_dir)}/{doc}'


def page_id_from_annotated_name(name: str) -> Optional[str]:
    name = name.removeprefix('uuid:').removesuffix('.jpg')
    return name if len(name) == 36 and name.count('-') == 4 else None


def load_annotations(paths: List[Path]) -> Dict[str, str]:
    annotations: Dict[str, str] = {}
    unmapped = Counter()
    conflicts = 0
    for path in paths:
        with path.open(encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if not line.strip():
                    continue
                name, label = line.rsplit(' ', 1)
                page_id = page_id_from_annotated_name(name)
                if page_id is None:
                    unmapped[path.name] += 1
                    continue
                if page_id in annotations and annotations[page_id] != label:
                    conflicts += 1
                annotations[page_id] = label
    logger.info(f'{len(annotations)} annotated page ids; unmapped names {dict(unmapped)}; '
                f'{conflicts} pages annotated differently in two lists (last list wins)')
    return annotations


def scan_extract_progress(path: Path, annotations: Dict[str, str]) -> Tuple[np.ndarray, Dict[str, Tuple[str, str]]]:
    """Returns the sorted hashes of keys with text, and page_id -> (library, doc) of annotated pages with text."""
    hashes = []
    annotated_pages: Dict[str, Tuple[str, str]] = {}
    with path.open(encoding='utf-8') as f:
        for i, line in enumerate(f):
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 4 or parts[3] != 'done':
                continue
            library, page_id, zip_path = parts[0], parts[1], parts[2]
            hashes.append(hash(f'{library}_{page_id}'))
            if page_id in annotations:
                annotated_pages[page_id] = (library, doc_from_path(zip_path))
            if i % 5_000_000 == 0:
                logger.info(f'extract progress: {i} lines')
    text_keys = np.unique(np.array(hashes, dtype=np.int64))
    logger.info(f'{len(text_keys)} pages with text, {len(annotated_pages)} of them annotated')
    return text_keys, annotated_pages


def main(argv=None):
    args = parse_arguments(argv)
    logging.basicConfig(level=args.logging_level,
                        format='EVAL SAMPLE - %(asctime)s - %(levelname)s - %(message)s')
    logger.info(' '.join(sys.argv))

    annotations = load_annotations(args.annotated)
    text_keys, annotated_pages = scan_extract_progress(args.extract_progress, annotations)

    def has_text(key: str) -> bool:
        h = hash(key)
        j = np.searchsorted(text_keys, h)
        return j < len(text_keys) and text_keys[j] == h

    # Per class: a max-heap (by negated priority) of the lowest-priority candidates. Kept at a
    # multiple of the cap so the per-document cap applied afterwards still leaves enough pages.
    heap_size = {True: args.per_class_cap * 4, False: args.background_cap * 4}
    heaps: Dict[str, list] = defaultdict(list)
    doc_pages = Counter()
    annotated_csv: Dict[str, dict] = {}
    counts = Counter()

    csv.field_size_limit(sys.maxsize)
    start = time.time()
    with args.labels_csv.open(newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        col = {name: header.index(name) for name in ('library', 'item_id', 'page_id', 'page_type', 'order',
                                                     'image_path')}
        for i, row in enumerate(reader):
            if i % 2_000_000 == 0:
                logger.info(f'csv: {i} rows, {time.time() - start:.0f}s')
            if len(row) != len(header):
                counts['malformed'] += 1
                continue
            item = f"{row[col['library']]}/{row[col['item_id']]}"
            doc_pages[item] += 1
            key = f"{row[col['library']]}_{row[col['page_id']]}"
            if not has_text(key):
                counts['no_text'] += 1
                continue
            label = row[col['page_type']] if row[col['page_type']] in PAGE_TYPES else ''
            record = {'key': key, 'library': row[col['library']], 'page_id': row[col['page_id']],
                      'doc': doc_from_path(row[col['image_path']]), 'item': item,
                      'order': row[col['order']], 'archive_label': label}
            if row[col['page_id']] in annotations:
                annotated_csv[row[col['page_id']]] = record
                continue
            counts['with_text'] += 1
            group = label or BACKGROUND
            heap = heaps[group]
            entry = (-stable_hash(key), key, record)
            if len(heap) < heap_size[bool(label)]:
                heapq.heappush(heap, entry)
            elif entry[0] > heap[0][0]:
                heapq.heapreplace(heap, entry)
    logger.info(f'CSV done: {dict(counts)}')

    rows = []
    for group, heap in heaps.items():
        cap = args.per_class_cap if group != BACKGROUND else args.background_cap
        per_doc = Counter()
        taken = 0
        for _neg_priority, _key, record in sorted(heap, reverse=True):
            if taken >= cap:
                break
            if per_doc[record['doc']] >= args.per_doc_cap:
                continue
            per_doc[record['doc']] += 1
            taken += 1
            rows.append((record, '', 'background' if group == BACKGROUND else None))

    for page_id, label in annotations.items():
        if page_id in annotated_csv:
            rows.append((annotated_csv[page_id], label, 'annotated'))
        elif page_id in annotated_pages:
            library, doc = annotated_pages[page_id]
            rows.append(({'key': f'{library}_{page_id}', 'library': library, 'page_id': page_id, 'doc': doc,
                          'item': None, 'order': '', 'archive_label': ''}, label, 'annotated'))

    annotated_docs = {record['doc'] for record, _label, split in rows if split == 'annotated'}
    threshold = int(args.test_fraction * 2 ** 64)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    split_counts = Counter()
    temp_path = args.output.with_name(args.output.name + '.tmp')
    with temp_path.open('w', encoding='utf-8', newline='') as out:
        writer = csv.writer(out, delimiter='\t', lineterminator='\n')
        writer.writerow(SAMPLE_COLUMNS)
        for record, annotated_label, split in rows:
            if split is None:
                test_doc = record['doc'] in annotated_docs or stable_hash(record['doc']) < threshold
                split = 'test' if test_doc else 'train'
            split_counts[split] += 1
            pages = doc_pages.get(record['item'], '') if record['item'] else ''
            writer.writerow((record['key'], record['library'], record['page_id'], record['doc'], record['order'],
                             pages, record['archive_label'], annotated_label, split))
    temp_path.replace(args.output)

    logger.info(f'Wrote {sum(split_counts.values())} pages to {args.output}: {dict(split_counts)}')


if __name__ == '__main__':
    main()

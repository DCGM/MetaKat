import argparse
import logging
import os
import random
import sys
import time
from collections import defaultdict
from uuid import UUID

from tools.mods_helper import page_type_classes

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-pages-all', required=True, type=str)
    parser.add_argument('--input-pages-trn', type=str)
    parser.add_argument('--input-pages-tst', type=str)
    parser.add_argument('--output-pages-trn', required=True, type=str)
    parser.add_argument('--output-pages-tst', required=True, type=str)

    parser.add_argument('--images-dir', type=str)
    parser.add_argument('--page-to-doc-mapping', required=True, type=str)
    parser.add_argument('--max-tst-pages-per-type', default=100, type=int)
    parser.add_argument('--logging-level', default=logging.INFO)
    return parser.parse_args()


def main():
    args = parse_args()

    log_formatter = logging.Formatter('%(asctime)s - CREATE TRN TST - %(levelname)s - %(message)s')
    log_formatter.converter = time.gmtime
    handler = logging.StreamHandler()
    handler.setFormatter(log_formatter)
    file_handler = logging.FileHandler(os.path.join(os.path.dirname(args.output_pages_trn), 'create_trn_tst.log'))
    file_handler.setFormatter(log_formatter)
    logger = logging.getLogger()
    logger.handlers = []
    logger.addHandler(handler)
    logger.addHandler(file_handler)
    logger.setLevel(args.logging_level)

    logger.info(' '.join(sys.argv))

    page_to_doc_mapping = {}
    with open(args.page_to_doc_mapping) as f:
        for line in f.readlines():
            page_id, doc_id = line.strip().split()
            page_to_doc_mapping[f'{page_id}.jpg'] = doc_id

    trn_pages = {page_type: [] for page_type in page_type_classes.values()}
    tst_pages = {page_type: [] for page_type in page_type_classes.values()}
    page_to_type_mapping = {}

    existing_pages = set()
    existing_tst_doc = set()

    if args.input_pages_trn:
        with open(args.input_pages_trn) as f:
            for line in f.readlines():
                page_id, page_type = line.strip().split()
                page_to_type_mapping[page_id] = page_type
                trn_pages[page_type].append(page_id)
        for page_type, pages in trn_pages.items():
            existing_pages.update(pages)

    if args.input_pages_tst:
        with open(args.input_pages_tst) as f:
            for line in f.readlines():
                page_id, page_type = line.strip().split()
                page_to_type_mapping[page_id] = page_type
                tst_pages[page_type].append(page_id)
        for page_type, pages in tst_pages.items():
            existing_pages.update(pages)
            for p in pages:
                if p in page_to_doc_mapping:
                    existing_tst_doc.add(page_to_doc_mapping[p])
                else:
                    existing_tst_doc.add(get_doc_id_from_name(p))

    all_pages = existing_pages.copy()
    doc_to_pages_mapping = defaultdict(list)
    with open(args.input_pages_all) as f:
        for line in f.readlines():
            if page_id not in existing_pages:
                page_id, page_type = line.strip().split()
                page_to_type_mapping[page_id] = page_type
                if page_id in page_to_doc_mapping:
                    doc_to_pages_mapping[page_to_doc_mapping[page_id]].append((page_id, page_type))
                else:
                    doc_id = get_doc_id_from_name(page_id)
                    if doc_id == 'default':
                        logger.warning(f'Page {page_id} not in page to doc mapping, adding to default doc')
                    doc_to_pages_mapping[doc_id].append((page_id, page_type))
                all_pages.add(page_id)

    if args.images_dir:
        for page_id in all_pages:
            if not os.path.exists(os.path.join(args.images_dir, page_id)):
                logger.error(f'Image {page_id} does not exist')

    doc_to_pages_mapping = list(doc_to_pages_mapping.items())
    random.shuffle(doc_to_pages_mapping)

    number_of_pages_per_type = get_number_of_pages_per_type(page_to_type_mapping)
    max_number_of_tst_pages_per_type = {page_type: min(int(n / 2), args.max_tst_pages_per_type)
                                        for page_type, n in number_of_pages_per_type.items()}

    lost_pages = 0
    for i, (doc_id, pages) in enumerate(doc_to_pages_mapping):
        added_to_tst = False
        lp = 0
        if len(pages) <= 1 and doc_id not in existing_tst_doc:
            for p in pages:
                if len(tst_pages[p[1]]) < max_number_of_tst_pages_per_type[p[1]]:
                    tst_pages[p[1]].append(p[0])
                    added_to_tst = True
                else:
                    lp += 1
        if not added_to_tst:
            lp = 0
            for page_id, page_type in pages:
                trn_pages[page_type].append(page_id)
        lost_pages += lp

    for page_type, pages in trn_pages.items():
        logger.info(f'{page_type} TRN: {len(pages)}, TST: {len(tst_pages[page_type])}')

    logger.info('')
    logger.info(f'Lost pages: {lost_pages}')

    save_pages_to_file(trn_pages, args.output_pages_trn)
    save_pages_to_file(tst_pages, args.output_pages_tst)


def get_doc_id_from_name(name):
    doc_id = None
    if name.startswith('mc_'):
        try:
            doc_id = name.split('_')[1]
        except IndexError:
            pass
    else:
        try:
            UUID(name.split('.')[0])
            doc_id = name.split('.')[0]
        except IndexError:
            pass
    if doc_id is None:
        doc_id = 'default'
    return doc_id

def save_pages_to_file(pages, output_file):
    lines = []
    with open(output_file, 'w') as f:
        for page_type, pages in pages.items():
            for page_id in pages:
                lines.append(f'{page_id} {page_type}\n')
        random.shuffle(lines)
        f.writelines(lines)


def get_number_of_pages_per_type(page_to_type_mapping):
    pages_per_type = defaultdict(int)
    for page_type in page_to_type_mapping.values():
        pages_per_type[page_type] += 1
    return pages_per_type

if __name__ == '__main__':
    main()

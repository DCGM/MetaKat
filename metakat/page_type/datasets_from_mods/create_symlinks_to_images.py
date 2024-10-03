import argparse
import json
import logging
import os.path
import sys
import time

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pages', required=True, type=str)
    parser.add_argument('--page-to-doc-mapping', required=True, type=str)
    parser.add_argument('--images-dir', required=True, type=str)
    parser.add_argument('--output-dir', required=True, type=str)
    parser.add_argument('--logging-level', default=logging.INFO)
    return parser.parse_args()


def main():
    args = parse_args()

    log_formatter = logging.Formatter('CREATE SYMLINKS - %(asctime)s - %(filename)s - %(levelname)s - %(message)s')
    log_formatter.converter = time.gmtime
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(log_formatter)
    logger = logging.getLogger()
    logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel(args.logging_level)

    logger.info(' '.join(sys.argv))

    private = True if 'private' in args.page_to_doc_mapping else False

    with open(args.pages) as f:
        pages = {x.strip().split()[0] for x in f.readlines()}

    page_to_doc_mapping = {}

    with open(args.page_to_doc_mapping) as f:
        page_to_doc_mapping_lines = f.readlines()
        for i, line in enumerate(page_to_doc_mapping_lines):
            page_id, doc_id = line.strip().split()
            page_id = f'{page_id}.jpg'
            if page_id in pages:
                page_to_doc_mapping[page_id] = doc_id

    os.makedirs(args.output_dir, exist_ok=True)

    for page_id in pages:
        if page_id not in page_to_doc_mapping:
            logger.info(f'No doc_id for {page_id}')
            continue
        dst = os.path.join(args.output_dir, page_id)
        if os.path.exists(dst):
            logger.info(f'Symlink for {page_id} -> {dst} already exists')
            continue
        doc_id = page_to_doc_mapping[page_id]
        page_id_tmp = page_id.replace('.jpg', '')
        if doc_id == page_id_tmp:
            logger.info(f'Page {page_id} is a doc')
            continue
        doc_id = doc_id.replace('uuid:', '')
        if private:
            doc_id = f'{doc_id}.images'
        src = os.path.join(args.images_dir, doc_id, page_id)
        if not os.path.exists(src):
            logger.info(f'Image for {page_id} -> {src} does not exists')
        else:
            os.symlink(src, dst)


def process_instance(val):
    if isinstance(val, dict):
        page_ids = []
        for key, val in val.items():
            page_ids += process_instance(val)
        return page_ids
    elif isinstance(val, list):
        return val


if __name__ == '__main__':
    main()

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
    parser.add_argument('--images-dir', required=True, type=str)
    parser.add_argument('--neighbour-page-mapping', required=True, type=str)
    parser.add_argument('--logging-level', default=logging.INFO)
    return parser.parse_args()


def main():
    args = parse_args()

    log_formatter = logging.Formatter('CREATE ENHANCED IMAGES - %(asctime)s - %(filename)s - %(levelname)s - %(message)s')
    log_formatter.converter = time.gmtime
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(log_formatter)
    logger = logging.getLogger()
    logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel(args.logging_level)

    logger.info(' '.join(sys.argv))

    with open(args.pages) as f:
        pages = [x.strip().split()[0] for x in f.readlines()]

    with open(args.neighbour_page_mapping) as f:
        neighbour_page_mapping = {}
        for line in f.readlines():
            page_id, previous_page_id, previous_pages, next_page_id, next_pages = line.strip().split()
            if page_id in pages:
                neighbour_page_mapping[page_id] = [previous_page_id, previous_pages, next_page_id, next_pages]

    for page_id in pages:
        if page_id not in neighbour_page_mapping:
            logger.info(f'No neighbours for {page_id}')
            continue

    for page_id in pages:
        previous_page_id, _, next_page_id, _ = neighbour_page_mapping[page_id]
        for my_id in [page_id, previous_page_id, next_page_id]:
            if my_id == 'None':
                continue
            if not os.path.exists(os.path.join(args.images_dir, my_id)):
                logger.info(f'Image for {my_id} does not exists')


if __name__ == '__main__':
    main()

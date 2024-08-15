import argparse
import json
import logging
import os.path
import sys
import time

import cv2
import numpy as np

from tools.resize_images import resize_image_with_max_size

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pages', required=True, type=str)
    parser.add_argument('--images-dir', required=True, type=str)
    parser.add_argument('--neighbour-page-mapping', required=True, type=str)
    parser.add_argument('--max-size', default=224, type=int)
    parser.add_argument('--output-dir', required=True, type=str)
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

    os.makedirs(args.output_dir, exist_ok=True)

    for i, page_id in enumerate(pages):
        output_path = os.path.join(args.output_dir, page_id)
        if os.path.exists(output_path):
            continue
        previous_page_id, previous_pages, next_page_id, next_pages = neighbour_page_mapping[page_id]
        main_page_path = os.path.join(args.images_dir, page_id)

        main_page = cv2.imread(main_page_path)

        if previous_page_id == 'None' and previous_pages == '0':
            previous_page = np.full_like(main_page, (255, 0, 0), dtype=np.uint8)
        elif previous_page_id == 'None':
            previous_page = np.zeros_like(main_page)
        else:
            previous_page_path = os.path.join(args.images_dir, previous_page_id)
            previous_page = cv2.imread(previous_page_path)
        if previous_page is None:
            logger.warning(f'Previous page not found or invalid: {previous_page_path}')
            previous_page = np.zeros_like(main_page)

        if next_page_id == 'None' and next_pages == '0':
            next_page = np.full_like(main_page, (0, 255, 0), dtype=np.uint8)
        elif next_page_id == 'None':
            next_page = np.zeros_like(main_page)
        else:
            next_page_path = os.path.join(args.images_dir, next_page_id)
            next_page = cv2.imread(next_page_path)
        if next_page is None:
            logger.warning(f'Next page not found or invalid: {next_page_path}')
            next_page = np.zeros_like(main_page)

        width = max(previous_page.shape[1] + next_page.shape[1], main_page.shape[1])

        bottom_row = np.zeros((max(previous_page.shape[0], next_page.shape[0]),
                               width, 3), dtype=np.uint8)
        bottom_row[:previous_page.shape[0], :previous_page.shape[1]] = previous_page
        bottom_row[:next_page.shape[0], previous_page.shape[1]:previous_page.shape[1] + next_page.shape[1]] = next_page
        top_row = np.zeros((main_page.shape[0], width, 3), dtype=np.uint8)
        start_middle_x = top_row.shape[1] // 2 - main_page.shape[1] // 2
        top_row[:, start_middle_x:start_middle_x + main_page.shape[1]] = main_page
        enhanced_page = np.concatenate([top_row, bottom_row], axis=0)

        enhanced_page = resize_image_with_max_size(enhanced_page, args.max_size)
        cv2.imwrite(output_path, enhanced_page)

        if (i + 1) % 1000 == 0:
            logger.info(f'Processed pages: {i + 1}/{len(pages)}')


if __name__ == '__main__':
    main()

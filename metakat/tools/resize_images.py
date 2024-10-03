import argparse
import glob
import logging
import os
import sys
import time

import cv2

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-images-dir', required=True, type=str)
    parser.add_argument('--output-images-dir', required=True, type=str)
    parser.add_argument('--max-size', default=224, type=int)
    parser.add_argument('--logging-level', default=logging.INFO)
    return parser.parse_args()


def main():
    args = parse_args()

    log_formatter = logging.Formatter('%(asctime)s - RESIZE IMAGE - %(levelname)s - %(message)s')
    log_formatter.converter = time.gmtime
    handler = logging.StreamHandler()
    handler.setFormatter(log_formatter)
    logger = logging.getLogger()
    logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel(args.logging_level)

    logger.info(' '.join(sys.argv))

    os.makedirs(args.output_images_dir, exist_ok=True)

    input_image_paths = glob.glob(os.path.join(args.input_images_dir, '*'))
    for i, image in enumerate(input_image_paths):
        output_image_path = os.path.join(args.output_images_dir, os.path.basename(image))
        #if os.path.exists(output_image_path):
        #    continue
        img = cv2.imread(image)
        img = resize_image_with_max_size(img, args.max_size)
        cv2.imwrite(output_image_path, img)

        if (i + 1) % 1000 == 0:
            logger.info(f'Processed images: {i + 1}/{len(input_image_paths)}')


def resize_image_with_max_size(img, max_size=224):
    height, width, _ = img.shape
    if height >= width:
        new_height = max_size
        new_width = int(width * new_height / height)
    else:
        new_width = max_size
        new_height = int(height * new_width / width)

    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return img


if __name__ == '__main__':
    main()

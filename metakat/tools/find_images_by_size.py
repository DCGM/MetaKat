import argparse
import glob
import logging
import os
import sys
import time

import cv2

logger = logging.getLogger(__name__)

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--width", type=int, default=2)
    parser.add_argument("--height", type=int, default=2)
    parser.add_argument("--log-level", type=int, default=logging.INFO)
    parser.add_argument("--save-image-names", type=bool, default=False)
    return parser.parse_args()

def main():
    args = arg_parse()

    my_logger = setup_logger(args.output_dir, args.log_level)
    my_logger.info(' '.join(sys.argv))

    os.makedirs(args.output_dir, exist_ok=True)

    #subfolder for images
    image_subfolder = os.path.join(args.output_dir, "images")
    os.makedirs(image_subfolder, exist_ok=True)

    input_image_paths = glob.glob(os.path.join(args.input_dir, '*'))

    image_names = []

    for i, image in enumerate(input_image_paths):
        base_name = os.path.basename(image)
        cleaned_name = os.path.splitext(base_name)[0]
        output_name = f"{cleaned_name}.jpg"

        output_image_path = os.path.join(image_subfolder, output_name)

        img = cv2.imread(image)
        height, width = img.shape[:2]

        if height <= args.height or width <= args.width:
            cv2.imwrite(output_image_path, img)

            if args.save_image_names:
                image_names.append(base_name + " h: " + str(height) + " w: " + str(width))

        if (i + 1) % 1000 == 0:
            logger.info(f'Processed images: {i + 1}/{len(input_image_paths)}')

    logger.info("\n")

    for name in image_names:
        logger.info(f'FOUND IMAGE: {name}')
    logger.info(f'TOTAL: {len(image_names)} \n')

    return

def setup_logger(out_file, log_level = logging.INFO):
    log_formatter = logging.Formatter('%(asctime)s - FIND IMAGES - %(levelname)s - %(message)s')
    log_formatter.converter = time.gmtime

    # console log
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)

    # file log
    file_handler = logging.FileHandler(os.path.join(out_file, "find_images.log"))
    file_handler.setFormatter(log_formatter)

    my_logger = logging.getLogger()
    my_logger.handlers = []
    my_logger.addHandler(stream_handler)
    my_logger.addHandler(file_handler)
    my_logger.setLevel(log_level)

    return my_logger

if __name__ == '__main__':
    main()

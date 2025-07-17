"""
File: find_images_by_page_type.py
Author: [Matej Smida]
Date: 2025-05-12
Description: find all images of certain page type from dataset
             for [for debugging and dataset creating].
"""

import argparse
import glob
import logging
import sys
import time
import os

import cv2

logger = logging.getLogger(__name__)

page_types = ["Abstract",
"Advertisement",
"Appendix",
"BackCover",
"BackEndPaper",
"BackEndSheet",
"Bibliography",
"Blank",
"CalibrationTable",
"Cover",
"CustomInclude",
"Dedication",
"Edge",
"Errata",
"FlyLeaf",
"FragmentsOfBookbinding",
"FrontCover",
"FrontEndPaper",
"FrontEndSheet",
"FrontJacket",
"Frontispiece",
"Illustration",
"Impressum",
"Imprimatur",
"Index",
"Jacket",
"ListOfIllustrations",
"ListOfMaps",
"ListOfTables",
"Map",
"NormalPage",
"Obituary",
"Preface",
"SheetMusic",
"Spine",
"Table",
"TableOfContents",
"TitlePage"]

def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pages-file", required=True)
    parser.add_argument("--images-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--page-type", type=str, required=True)
    parser.add_argument("--log-level", type=int, default=logging.INFO)
    parser.add_argument("--save-as-jpg", type=bool, default=False)

    return parser.parse_args()

def setup_logger(out_file, log_level = logging.INFO):
    log_formatter = logging.Formatter("'%(asctime)s - FIND IMAGES - %(levelname)s - %(message)s'")
    log_formatter.converter = time.gmtime

    stream = logging.StreamHandler()
    stream.setFormatter(log_formatter)

    file = logging.FileHandler(os.path.join(out_file, "find_images_by_page_type.log"))
    file.setFormatter(log_formatter)

    my_logger = logging.getLogger()
    my_logger.handlers = [stream, file]
    my_logger.setLevel(log_level)

    return my_logger


def main():
    args = arg_parse()

    my_logger = setup_logger(args.output_file, args.log_level)
    my_logger.info(" ".join(sys.argv))

    if args.page_type.lower() not in list(map(lambda x: x.lower(), page_types)):
        my_logger.error(f"Invalid page type {args.page_type} \n")
        return

    os.makedirs(args.output_file, exist_ok=True)

    image_subfolder = os.path.join(args.output_file, "images")
    os.makedirs(image_subfolder, exist_ok=True)

    input_images = glob.glob(os.path.join(args.images_file, "*"))

    page_type_dic = {}

    with open(args.pages_file, "r") as file:
        for line in file:
            name, p_type = line.strip().split()

            if p_type.lower() == args.page_type.lower():
                page_type_dic[name] = "not_found"

    if len(page_type_dic) == 0:
        my_logger.error(f"Page type {args.page_type} not found in pages file \n")
        return

    for name, value in page_type_dic.items():
        print(name, value)

    for i, image in enumerate(input_images):
        base_name = os.path.basename(image)

        if base_name in page_type_dic:
            page_type_dic[base_name] = "found"

            if args.save_as_jpg:
                cleaned_name = os.path.splitext(base_name)[0]
                output_name = f"{cleaned_name}.jpg"
            else:
                output_name = base_name

            output_image_path = os.path.join(image_subfolder, output_name)

            cv2.imwrite(output_image_path, cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB))

        if (i + 1) % 1000 == 0:
            logger.info(f"Processed {i + 1}/{len(input_images)} images")

    logger.info("\n")

    logger.info(f"Found {len(page_type_dic)} images \n")

    for name, value in page_type_dic.items():
        if value == "not_found":
            logger.warning(f"Image {name} not found in image folder \n")

    return

if __name__ == "__main__":
    main()
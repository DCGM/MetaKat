import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict

from natsort import natsorted

from page_annotator.examples.annotator_api import AnnotatorApi
from tools.mods_helper import page_type_classes


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--api-key', required=True, type=str)
    parser.add_argument('--url', default="https://page.semant.cz/api/", type=str)
    parser.add_argument('--output-dir', type=str)

    parser.add_argument('--logging-level', default=logging.INFO)

    args = parser.parse_args()
    return args


logger = logging.getLogger(__name__)


def main():
    args = parse_args()

    log_formatter = logging.Formatter('%(asctime)s - DOWNLOAD IMAGES - %(levelname)s - %(message)s')
    log_formatter.converter = time.gmtime
    handler = logging.StreamHandler()
    handler.setFormatter(log_formatter)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(args.logging_level)

    logger.info(' '.join(sys.argv))

    api = AnnotatorApi(args.api_key, args.url)
    res, _ = api.get_image_list(camera_list=['pages'])
    annotated = defaultdict(lambda: 0)
    page_types = list(page_type_classes.values())

    out_all = []

    for r in res:
        if len(r['position']) == 1 and r['position'][0]['value']['type'] in page_types and not r['position'][0]['ignore']:
            page_type = r['position'][0]['value']['type']
            jl = r['json'].replace("'", '"').replace("1.0", '"1.0"').replace('False', '"False"').replace('True', '"True"')
            jl = json.loads(jl)
            image_name = jl['original']['img_path']

            for pt in page_types:
                if pt == image_name.split('.')[-2]:
                    image_name = image_name.replace(f'.{pt}.jpg', '')
                    if 'mc_' in image_name:
                        image_name = '.'.join(image_name.split('.')[1:])
                    break

            logger.info(f"{image_name} {page_type}")
            out_all.append(f" {image_name} {page_type}")

            annotated[r['position'][0]['value']['type']] += 1
            pass
        else:
            pass

    logger.info(natsorted(annotated.items(), key=lambda x: x[0], reverse=False))

    with open(os.path.join(args.output_dir, "pages.all"), 'w') as f:
        for line in out_all:
            f.write(f"{line}\n")


if __name__ == '__main__':
    main()

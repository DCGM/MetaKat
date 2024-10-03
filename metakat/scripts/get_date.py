import argparse
import json
import logging
import os

import sys
import time

from metakat.tools.mods_helper import  get_year_from_doc_mods

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mods-dir', required=True, type=str)
    parser.add_argument('--mods-jsonl', required=True, type=str)
    parser.add_argument('--output-file', required=True, type=str)

    parser.add_argument('--logging-level', default=logging.INFO)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    log_formatter = logging.Formatter('GET DATE - %(asctime)s - %(filename)s - %(levelname)s - %(message)s')
    log_formatter.converter = time.gmtime
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(log_formatter)
    logger = logging.getLogger()
    logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel(args.logging_level)

    logger.info(' '.join(sys.argv))

    mods_jsonl = {}
    with open(args.mods_jsonl) as f:
        for line in f.readlines():
            key, val = list(json.loads(line).items())[0]
            mods_jsonl[key] = val

    logger.info(f'Number of top level docs to process: {len(mods_jsonl)}')

    now = time.time()

    uuids_with_dates = []
    for i, (key, val) in enumerate(mods_jsonl.items()):
        uuids_with_dates += process_top_level_doc({key: val}, '', args.mods_dir)
        if (i + 1) % 5000 == 0:
            logger.info(f'{i + 1}/{len(mods_jsonl)} top level docs processed')
    logger.info(f'{len(uuids_with_dates)} bottom level docs processed in {time.time() - now:.2f} s')

    with open(args.output_file, 'w') as f:
        for el in uuids_with_dates:
            f.write(json.dumps(el) + '\n')


def process_top_level_doc(child_level, parent_path, mods_dir):
    uuid_with_dates = []
    if isinstance(child_level, dict):
        for key, val in child_level.items():
            if isinstance(val, list):
                uuid_path = str(os.path.join(parent_path, key))
                mods_path = os.path.join(mods_dir, uuid_path + '.mods')
                if not os.path.exists(mods_path):
                    logger.warning(f'Mods file {mods_path} does not exist')
                    continue
                date = get_year_from_doc_mods(mods_path, original_date=True)
                uuids_with_date = [None, None, None]
                uuids = uuid_path.split('/')
                uuids_with_date[-len(uuids):] = uuids
                uuids_with_date.append(date)
                uuid_with_dates.append(uuids_with_date)
            else:
                uuid_with_dates += process_top_level_doc(val, os.path.join(parent_path, key), mods_dir)
    return uuid_with_dates


if __name__ == '__main__':
    main()
